import random
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import pyplot as plt
import cv2 as cv

import math

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
        self.o4_ch = settings.folder_dicts[folder]['o4_ch']
        self.marker_index = settings.folder_dicts[folder]['marker_index']

        if self.o4_ch is not None:
            self.model = loadKerasModel(settings.kerasModel)

        self.gamma = settings.folder_dicts[folder]['gamma']
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

    def findAutoFluorescenceMask(self, sCI):
        """ Determine autoFL threshold image for subsequent steps. """
        self.red = sCI.images[2]

        red = sCI.proccessNuclearImage(self.red, gamma=1)
        red_blur = cv.GaussianBlur(red,(5,5),0)

        th = cv.adaptiveThreshold(red_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,2)     
        red_thresh = cv.bitwise_not(th)

        self.autoFL = red_thresh

    def findNucleiOnTranswell(self, sCI):
        """ Find nuclei using contours and filter on size and circularity."""
        
        """ Process DAPI image for thresholding. """
        dapi = sCI.proccessNuclearImage(sCI.images[sCI.dapi_ch], gamma=self.gamma, debug=self.debug)
        
        dapi_blur = cv.GaussianBlur(dapi,(5,5),0)

        dapi_thresh = cv.adaptiveThreshold(dapi_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,2)

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
        if self.debug:
            sCI.showImages([dapi_only, dapi_close],['dapi_only','dapi_close'])
    
        """ ERODE threshold image to remove noise. """
        # tM.tryErosion(dapi_close) 
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        dapi_erode = cv.erode(dapi_close, kernel)
        if self.debug:
            sCI.showImages([dapi_only, dapi_close, dapi_erode],['dapi_only','dapi_close','dapi_erode'])

        """ Calculate number of nuclei using standard approach. """
        self.nucleiCount, self.output, self.nucleiMask, self.nucleiWatershed, self.nucleiMarkers = sCI.thresholdSegmentation(dapi_erode, dapi, opening_iterations = 2, background_iterations = 10, debug = False)

        """ Use watershed markers to find countours. """
        # https://stackoverflow.com/questions/50882663/find-contours-after-watershed-opencv
        m1 = self.nucleiMarkers.astype(np.uint8)
        ret, m2 = cv.threshold(m1, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
        if self.debug:
            cv.imshow('DAPI watershed-based threshold',m2)
            cv.waitKey()
            cv.destroyAllWindows()

        """ Calculate contours from nuclei and filter based on size and circularity. """
        # from https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection
        contours, hierarchy = cv.findContours(m2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_area = self.getContoursArea(contours)
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
        rgb = sCI.colorImage(blue=sCI.images[0],green=sCI.images[1],gamma=0.2)
        GFP = cv.drawContours(rgb, self.nucleiCircles, -1, (255,0,0), 1)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(overlap)
        ax2 = fig.add_subplot(2,2,2, sharex=ax1,sharey=ax1)
        ax2.imshow(GFP)
        
        # nucleiCircles = list(self.nucleiCircles[i] for i in random.sample(list(range(len(self.nucleiCircles))),50))

        """ Using nucleiCircles determine how many pixels overlapping with each nucleus are GFP+ """
        img = GFP
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
            plt.show(fig)
        else:
            plt.close(fig)

    def plotResults(self):
        plt.figure(figsize = (10,10))
        plt.title(f"{self.path}\\{self.imgFile}")
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
                stage, well, position = self.parseFileName(self.imgFile)

                try:
                    # load images
                    sCI = singleCompositeImage(self.path, self.imgFile, self.dapi_ch, o4_ch=self.o4_ch, scalefactor=1, debug=False, gamma=self.gamma)

                    # find autofluor
                    self.findAutoFluorescenceMask(sCI)

                    # find nuclei
                    self.findNucleiOnTranswell(sCI)

                    # find GFP+ cells
                    self.findGFP(sCI)

                    # plot results
                    self.plotResults()

                    result = {
                        'path': sCI.path,
                        'imgFile': sCI.imgFile,
                        'stage': stage,
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

    def exportResults(self):
        # output results as csv
        import csv
        filename = 'results_folder_' + str(self.folder) + '.csv'
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

# run code

javabridge.start_vm(class_path=bioformats.JARS)        

# JW Transwell
a = analyzeTranswell(folder = 8, debug = False)
# a = analyzeTranswell(folder = 9, debug = False)

# newfiles = []
# for file in a.files:
#     if file['name'] == 'Migration assay(1)_NCKD_pre scrape + 594_A1-3_9402.vsi':
#         newfiles.append(file)
# a.files = newfiles
# print(a.files)
# a.debug = True

a.runAnalysis()
a.exportResults()

javabridge.kill_vm()

print('All Done')