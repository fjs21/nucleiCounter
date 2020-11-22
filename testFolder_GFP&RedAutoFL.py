import random
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import pyplot as plt
import cv2 as cv

import math
from scipy import optimize

# to enable quiting mid-code
import sys

# start JVM for compatibility with VSI files
import javabridge
import bioformats

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

# function for optimizing morphology steps
from tryMorphology import tryMorphology
tM = tryMorphology()

class analyzeTranswell():

    def __init__(self, folder:int, debug:bool=False):
        
        self.folder = folder
        self.debug = debug

        settings = Settings()      

        # retrieve settings using 'folder'
        self.name = settings.folder_dicts[folder]['name']
        self.root = settings.folder_dicts[folder]['root']
        self.pattern = settings.folder_dicts[folder]['pattern']
        self.files = find(self.pattern, self.root)
        self.dapi_ch = settings.folder_dicts[folder]['dapi_ch']
        self.dapi_gamma = settings.folder_dicts[folder]['dapi_gamma']
        self.o4_ch = settings.folder_dicts[folder]['o4_ch']
        self.o4_gamma = settings.folder_dicts[folder]['o4_gamma']
        self.autoFL_dilate = settings.folder_dicts[folder]['autoFL_dilate']
        self.autoFL_gamma = settings.folder_dicts[folder]['autoFL_gamma']
        self.marker_index = settings.folder_dicts[folder]['marker_index']

        if self.o4_ch is not None:
            self.model = loadKerasModel(settings.kerasModel)

        self.thres = settings.folder_dicts[folder]['thres']
        self.o4_cutoff = 0.5 # default was 0.5

        # start analysis
        print(f"Found {len(self.files)} matching '{self.pattern}' in '{self.root}'")
        print("***************************")

        # select file sample
        if debug:
            print("Running debug on a few images...")
            # select five files at random
            self.files = list(self.files[i] for i in random.sample(list(range(len(self.files))), 1))

            # select five files to do manual count comparisons
            # files = list(files[i] for i in range(3,4)) 
        else:
            print("Starting to analyze images")

        results = []

    def parseFileName(self,imgFile):
        """Extract stage, well and image position from file name."""

        imgFile_split = imgFile.rsplit('_',maxsplit=2)
        imgFile_split.reverse()

        if(imgFile_split[2].upper().find('PRE')>0):
            stage = "PRE"
        elif(imgFile_split[2].upper().find('POST')>0):
            stage = "POST"
        else:
            stage = None

        well_position = imgFile_split[1].split('-')
        well = well_position[0]
        try:
            position = well_position[1]
        except:
            print(f"Error parsing: {imgFile}")
            position = None  

        return [stage, well, position]

    def getContoursArea(self, contours, min=100, max=2000):
        contours_area = []
        areas = []
        # calculate area and filter into new array
        for con in contours:
            area = cv.contourArea(con)
            areas.append(area)
            if min < area < max:
                contours_area.append(con)

        if self.debug:
            print(f"median area: {np.median(area)}")
            print(f"mean area: {np.mean(area)}")

        return(contours_area)

    def getContoursCircles(self, contours, min=0.4, max=1.2):        
        contours_circles = []
        circularities = []
        # check if contour is of circular shape
        for con in contours:
            perimeter = cv.arcLength(con, True)
            area = cv.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            circularities.append(circularity)
            if 0.4 < circularity < 1.2:
                contours_circles.append(con)

        if self.debug:
            print(f"median circularity: {np.median(circularities)}")

        return(contours_circles)

    def optimize_gamma(self, img, target_median: float = 90):
        """ """
        img_min    = np.min(img)
        img_median = np.median(img)
        img_max    = np.max(img)

        def calc_median(gamma):
            min_g = np.power(img_min/img_max, gamma) * img_max
            median_g = np.power(img_median/img_max, gamma) * img_max
            max_g = img_max
            return (median_g - min_g)/(max_g - min_g)*256

        def error(gamma):
            return abs(target_median - calc_median(gamma))

        result = optimize.minimize_scalar(error)
        print(result)
        return result.x

    def setMaxPixelInImage(self, img, max):
        """ Remove top of image but setting max pixel intensity. """
        
        img_new = img
        img_new[img > max] = max
        
        return img_new

    def findAutoFluorescenceMask(self, sCI, gamma, dilate: bool=False):
        """ Determine autoFL threshold image for subsequent steps. """
        self.red = sCI.images[2]

        # gamma = self.optimize_gamma(self.red)
        red = self.setMaxPixelInImage(self.red, max = 2 * np.median(self.red))
        red = sCI.proccessNuclearImage(red, gamma=gamma)

        if self.debug:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax1.imshow(red)     
            ax1.set_title(f"gamma = {gamma}")
        
        if self.debug:
            print(f"autoFL pre-gamma min: {np.min(self.red)}, median: {np.median(self.red)}, max: {np.max(self.red)}")    
            print(f"autoFL post-gamma min: {np.min(red)}, median: {255-np.median(red)}, max: {np.max(red)}")

        red_blur = cv.GaussianBlur(red,(5,5),0)

        th = cv.adaptiveThreshold(red_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,2.5)     
        red_thresh = cv.bitwise_not(th)
        if self.debug:
            ax2 = fig.add_subplot(1,3,2, sharex=ax1, sharey=ax1)
            ax2.imshow(red_thresh)
            ax2.set_title("threshold: c = 2.5")

        th = cv.adaptiveThreshold(red_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,6)     
        red_thresh = cv.bitwise_not(th)
        if self.debug:
            ax3 = fig.add_subplot(1,3,3, sharex=ax1, sharey=ax1)
            ax3.imshow(red_thresh)
            ax3.set_title("threshold: c = 6")
            plt.show()

        if dilate:
            kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
            red_thresh = cv.dilate(red_thresh, kernel)
        
        self.autoFL = red_thresh

    def findNucleiOnTranswell(self, sCI):
        """ Find nuclei using contours and filter on size and circularity."""
        
        self.dapi = sCI.images[sCI.dapi_ch]

        if self.debug:
            print(f"dapi pre-gamma min: {np.min(self.dapi)}, median: {np.median(self.dapi)}, max: {np.max(self.dapi)}")  

        """ Process DAPI image for thresholding. """
        dapi = sCI.proccessNuclearImage(self.dapi, gamma=self.dapi_gamma, debug=self.debug)
        
        dapi_blur = cv.GaussianBlur(dapi,(5,5),0)

        """ Set 'c' based on pre- or post-scrape - visual analysis showed this improved reliability of counts. """
        if self.stage=="PRE":
            dapi_thresh = cv.adaptiveThreshold(dapi_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
                cv.THRESH_BINARY,23,4)
        else:
            dapi_thresh = cv.adaptiveThreshold(dapi_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
                cv.THRESH_BINARY,23,10)

        dapi_only = cv.bitwise_not(cv.bitwise_and(cv.bitwise_not(dapi_thresh), cv.bitwise_not(self.autoFL)))

        if self.debug:
            titles = ['DAPI','DAPI_thresh','autoFL','dapi_only']
            images = [dapi_blur, dapi_thresh, cv.bitwise_not(self.autoFL), dapi_only]
            sCI.showImages(images, titles)      

        # tM.tryDilation(cv.bitwise_not(dapi_only))

        """ Remove small holes present in the Dapi+ nuclei with CLOSE operation. """
        dapi_only = cv.bitwise_not(dapi_only)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        dapi_close = cv.morphologyEx(dapi_only, cv.MORPH_CLOSE, kernel)
            
        """ ERODE threshold image to remove noise. """
        # tM.tryErosion(dapi_close) 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        dapi_erode = cv.erode(dapi_close, kernel)
        if self.debug:
            sCI.showImages([dapi_only, dapi_close, dapi_erode],['dapi_only','dapi_close','dapi_erode'])

        """ Calculate number of nuclei using standard approach. """
        """ Taylor whether or not to use the dapi_close or dapi_erode image based on pre- or post-scrape. """
        if self.stage=="PRE":
            self.nucleiCount, self.output, self.nucleiMask, self.nucleiWatershed, self.nucleiMarkers = sCI.thresholdSegmentation(dapi_erode, dapi, opening_iterations = 2, background_iterations = 10, debug = self.debug)
        else:
            self.nucleiCount, self.output, self.nucleiMask, self.nucleiWatershed, self.nucleiMarkers = sCI.thresholdSegmentation(dapi_close, dapi, opening_iterations = 2, background_iterations = 10, debug = self.debug)
        # print(f"Nuclei Count: {self.nucleiCount}")

        """ Use watershed markers to find contours. """
        # https://stackoverflow.com/questions/50882663/find-contours-after-watershed-opencv
        m1 = self.nucleiMarkers.astype(np.uint8)
        # so the marker image borders are 255 but each cell is given a number above 2
        # threshold of 2 or more will find all cells
        ret, m2 = cv.threshold(m1, 2, 255, cv.THRESH_BINARY)
        # if self.debug:
        #     sCI.showImages([m1,m2],['m1','m2'])

        """ Calculate contours from nuclei and filter based on size and circularity. """
        # from https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection
        contours, hierarchy = cv.findContours(m2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_area = self.getContoursArea(contours, min = 40)
        contours_circles = self.getContoursCircles(contours_area)

        print(f"Final nuclei count: {len(contours_circles)}")

        # print(contours_circles[0])
        self.nucleiImg = cv.drawContours(self.nucleiWatershed, contours_circles, -1, (0,255,0), thickness=1)
        self.nucleiCircles = contours_circles
        self.nucleiCount = len(contours_circles)
        
        if self.debug:
            plt.imshow(self.nucleiImg)
            plt.show()

    def findGFP(self, sCI):
        """ GFP count. """
        # GFP = cv.normalize(src=GFP, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # plt.imshow(GFP)
        # plt.show()
        # sCI.plotHistogram(self.GFP, gamma=0.1)

        self.GFP = sCI.images[1]

        GFP = sCI.proccessNuclearImage(self.GFP, gamma=0.2, debug=False)

        # trying different treshold settings
        # blocksize = larger will focus on larger objects
        # c = larger will look for larger changes in intensity
        img = GFP
        img_blur = cv.GaussianBlur(img,(5,5),0)

        th1 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,27,3)
        th2 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,29,3)
        th3 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,31,3)
        if self.debug:
            titles = ['Original Image (Blur)', 'th1 (blocksize=27)', 'th2 (blocksize=29)', 'th3 (blocksize=31)']
            images = [img_blur, th1, th2, th3]
            sCI.showImages(images, titles)

        # tM.tryClosing(th3)    

        """ invert GFP threshold for subsequent steps - background = white for morphology and bitwise operations. """
        GFP_thresh = cv.bitwise_not(th3)
        
        """ find GFP threshold image that is non-overlapping with autoFL. """
        overlap_thresh = cv.bitwise_and(GFP_thresh, cv.bitwise_not(self.autoFL))
        
        """ Close holes in GFP threshold that are often present in very large nuclei. """       
        # tM.tryClose(overlap_thresh)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        overlap_thresh = cv.morphologyEx(overlap_thresh, cv.MORPH_CLOSE, kernel, iterations = 2)

        """ Show GFP threshold along with calcuated nuclei and GFP+ cells. """
        overlap = cv.cvtColor(overlap_thresh, cv.COLOR_GRAY2BGR)
        overlap = cv.drawContours(overlap, self.nucleiCircles, -1, (255,0,0), 1)
        rgb = sCI.colorImage(blue=sCI.images[0],gamma=0.2)
        GFP = cv.drawContours(rgb, self.nucleiCircles, -1, (255,255,255), 1)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(overlap)
        ax2 = fig.add_subplot(2,2,2, sharex=ax1,sharey=ax1)
        ax2.imshow(GFP)
        
        # nucleiCircles = list(self.nucleiCircles[i] for i in random.sample(list(range(len(self.nucleiCircles))),50))

        """ Using nucleiCircles determine how many pixels overlapping with each nucleus are GFP+ """
        rgb = sCI.colorImage(blue=sCI.images[0],green=sCI.images[1],gamma=0.2)
        img = cv.drawContours(rgb, self.nucleiCircles, -1, (255,0,0), 1)
        self.GFPcount = 0
        GFPpercentages = []
        for nuclei in self.nucleiCircles:
            boundRect = cv.boundingRect(nuclei)
            # print(f"x = {boundRect[0]}, y = {boundRect[1]}, width = {boundRect[2]}, height= {boundRect[3]}")
            cv.rectangle(img, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (255,0,0), 1)
            nucleusSize = 0
            GFPcount = 0
            for x in range(0, int(boundRect[2])):
                for y in range(0, int(boundRect[3])):
                    pt = (int(boundRect[0]) + x, int(boundRect[1]) + y)
                    ppt = cv.pointPolygonTest(nuclei, pt, measureDist=False)            
                    if ppt != -1:
                        """ Pixel inside nucleus """
                        nucleusSize += 1
                        """ Check if Pixel is GFP+ """
                        if overlap_thresh[pt[1],pt[0]] != 0:
                            GFPcount += 1
                            # plt.scatter(pt[0],pt[1],c="purple",s=0.2)
                        # else:
                        #     plt.scatter(pt[0],pt[1],c="tan",s=0.2)
            
            # print(f"nucleusSize: {nucleusSize}. GFPcount: {GFPcount}. %GFP =  {'{:.2%}'.format(GFPcount/nucleusSize*100)}")
            GFPpercentage = (GFPcount/nucleusSize)
            GFPpercentages.append(GFPpercentage)

            """ Count those cells with 50% or more GFP+ pixels."""
            if GFPpercentage > 0.5:
                self.GFPcount += 1
                cv.rectangle(img, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,255), 2)

        print(f"GFP+ cells: {self.GFPcount}")

        self.diagnostic_img = img

        ax3 = fig.add_subplot(2,2,3, sharex=ax1,sharey=ax1)
        ax3.imshow(img)       
        fig.suptitle(self.imgFile)

        # plot histogrm of GFP+ pixels in each nuclei contour
        ax4 = fig.add_subplot(2,2,4)
        ax4.hist(GFPpercentages)
        ax4.set_yscale('log')
        ax4.set_title('distrubition of GFP+ pixels % per nuclei', fontsize=8)

        """ Show diagnostic plot if debug is set. """
        if self.debug:
            plt.show()
        else:
            plt.close(fig)

    def plotResults(self):
        plt.figure(figsize = (10,10))
        # plt.title(f"{self.path}\\{self.imgFile}")
        plt.title(f"File: {self.imgFile}")
        plt.imshow(self.diagnostic_img)
        self.export_pdf.savefig(dpi = 300)
        plt.close()

    def runAnalysis(self):    
        self.results = []

        with PdfPages('results_folder_' + str(self.folder) + '.pdf') as self.export_pdf:

            for file in self.files:

                self.path = file['path'] 
                self.imgFile = file['name']
                print(f"Processing: {self.path}\\{self.imgFile}")

                # parse file names
                self.stage, well, position = self.parseFileName(self.imgFile)

                try:
                    # load images
                    sCI = singleCompositeImage(self.path, self.imgFile, 
                        dapi_ch = self.dapi_ch, dapi_gamma = self.dapi_gamma, 
                        o4_ch=self.o4_ch, o4_gamma = self.o4_gamma,
                        scalefactor=1, debug=False)

                    # find autofluor
                    self.findAutoFluorescenceMask(sCI, gamma = self.autoFL_gamma, dilate = self.autoFL_dilate)

                    # find nuclei
                    self.findNucleiOnTranswell(sCI)

                    # find GFP+ cells
                    self.findGFP(sCI)

                    # plot results
                    self.plotResults()

                    result = {
                        'path': sCI.path,
                        'imgFile': sCI.imgFile,
                        'stage': self.stage,
                        'well': well,
                        'position': position,
                        }


                    if "model" in locals():
                        sCI.processCells()
                        sCI.getPredictions(self.model)
                        sCI.processPredictions(export_pdf, prediction_cutoff = self.o4_cutoff, debug=False)

                        result.update({
                            'nucleiCount': sCI.nucleiCount,
                            'o4pos_count': sCI.o4pos_count,
                            'o4neg_count': sCI.o4neg_count,
                            'o4%': "{:.2%}".format(sCI.o4pos_count/(sCI.o4pos_count+sCI.o4neg_count)),
                            })

                    result.update({
                        'nucleiCount': self.nucleiCount,
                        'GFP_count': self.GFPcount,
                        })
                    
                    self.results.append(result)  
                     
                except:
                    print(f"Failed on path '{self.path}'. Image: {self.imgFile}")
                    raise
            print("Finished processing files and writing PDF to disk.")

    def exportResults(self, name_stem = 'results_folder_'):
        # output results as csv
        import csv
        filename = name_stem + str(self.folder) + '.csv'
        with open(filename,'w',newline='') as f:
            # report analysis settings
            w = csv.writer(f)
            w.writerow([
                'name', self.name
                ])
            w.writerow('')
            # results
            w = csv.DictWriter(f, self.results[0].keys())
            w.writeheader()
            w.writerows(self.results)

    def imageStats(self):

        self.results = []
        for file in self.files:
            self.path = file['path'] 
            self.imgFile = file['name']
            print(f"Processing: {self.path}\\{self.imgFile}")

            # parse file names  
            stage, well, position = self.parseFileName(self.imgFile)
            try:
                # load images
                sCI = singleCompositeImage(self.path, self.imgFile, 
                    dapi_ch = self.dapi_ch, dapi_gamma = self.dapi_gamma, 
                    o4_ch=self.o4_ch, o4_gamma = self.o4_gamma,
                    scalefactor=1, debug=False)

                self.dapi = sCI.images[sCI.dapi_ch]

                dapi_focus = cv.Laplacian(self.dapi, cv.CV_64F).var()
                if self.debug:
                    print(f"dapi pre-gamma min: {np.min(self.dapi)}, median: {np.median(self.dapi)}, max: {np.max(self.dapi)}") 
                    print(f"dapi focus: {dapi_focus}")

                result = {
                    'path': sCI.path,
                    'imgFile': sCI.imgFile,
                    'stage': stage,
                    'well': well,
                    'position': position,
                    'dapi_min': np.min(self.dapi),
                    'dapi_median': np.median(self.dapi),
                    'dapi_max': np.max(self.dapi),
                    'dapi_focus': dapi_focus                 
                }
                self.results.append(result)  

            except:
                    print(f"Failed on path '{self.path}'. Image: {self.imgFile}")
                    raise

        print("Finished gethering stats.")

# run code

javabridge.start_vm(class_path=bioformats.JARS)        

""" JW Transwell experiments for SULF2 paper - run seperately due to memory limitations """
# a = analyzeTranswell(folder = 8, debug = False) # Migration(1)
# a = analyzeTranswell(folder = 9, debug = False) # Migration(2)
# a = analyzeTranswell(folder = 10, debug = False) # Migration(3)
# a = analyzeTranswell(folder = 11, debug = False) # Migration(4)
# a = analyzeTranswell(folder = 12, debug = False) # Migration(5)

""" uncomment the following to run a debug on a specific image."""
# newfiles = []
# for file in a.files:
#     if file['name'] == 'Migration assay(2)_S2KD_post scrape_D2-4_10270.vsi':
#         newfiles.append(file)
# a.files = newfiles
# print(a.files)
# a.debug = True

""" uncomment to run count and export pdf/csv files. """
# a.runAnalysis()
# a.exportResults()

""" uncomment to retrieve some basic image stats. """
# a.imageStats()
# a.exportResults('imageStats_folder_')

""" runs analysis on mulitple experiments. (SLOW!!) """
for i in range(8,13):
    print(f"Running on folder {i}")
    a = analyzeTranswell(folder = i, debug = False)
    a.runAnalysis()
    a.exportResults()

javabridge.kill_vm()

print('All Done')