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

class analyzeGFP():

    def __init__(self, folder:int, debug:bool=False):
        
        self.folder = folder
        self.debug = debug

        settings = Settings()      

        # retrieve settings using 'folder'
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

    def findNucleiOnTranswell(self, sCI):
        """ Find nuclei using contours and filter on size and circularity."""
        sCI.processDAPI(threshold_method=self.thres, gamma=self.gamma, debug=False) # based on manual counts (see OneNote)
        if self.debug:
            sCI.reportResults()

        # use watershed to find countours
        # https://stackoverflow.com/questions/50882663/find-contours-after-watershed-opencv
        m1 = sCI.nucleiMarkers.astype(np.uint8)
        ret, m2 = cv.threshold(m1, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
        # cv.imshow('m2',m2)

        # from https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection
        contours, hierarchy = cv.findContours(m2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contours_area = self.getContoursArea(contours)

        contours_circles = self.getContoursCircles(contours_area)

        print(f"number of passing cells: {len(contours_circles)}")

        # print(contours_circles[0])
        self.nucleiImg = cv.drawContours(sCI.nucleiWatershed, contours_circles, -1, (0,255,0), thickness=1)
        self.nucleiCircles = contours_circles
        self.nucleiCount = len(contours_circles)
        # plt.imshow(self.nucleiImg)
        # plt.show()

    def findGFP(self, sCI):
        """ GFP count. """
        # GFP = cv.normalize(src=GFP, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # plt.imshow(GFP)
        # plt.show()
        # sCI.plotHistogram(self.GFP, gamma=0.1)

        self.GFP = sCI.images[1]

        # finding holes in transwell
        GFP = sCI.proccessNuclearImage(self.GFP, gamma=1, debug=False)
        GFP = cv.bitwise_not(GFP)
        GFP_blur = cv.GaussianBlur(GFP,(5,5),0)

        # trying different treshold settings
        # blocksize = larger will focus on larger objects
        # c = larger will look for larger changes in intensity
        th1 = cv.adaptiveThreshold(GFP_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,2)
        th2 = cv.adaptiveThreshold(GFP_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,3)
        th3 = cv.adaptiveThreshold(GFP_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,4)
        # titles = ['Original Image (Blur)', 'th1', 'th2', 'th3']
        # images = [GFP_blur, th1, th2, th3]
        # sCI.showImages(images, titles)       

        # find outlines of thresholded image
        contours, hierarchy = cv.findContours(th2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        # filter to focus on circular holes
        contours_areas = self.getContoursArea(contours, min=10)
        contours_circles = self.getContoursCircles(contours_areas, min=0.7)

        # show pre/post-filtered contours
        # img = cv.cvtColor(GFP, cv.COLOR_GRAY2BGR)
        # img = cv.drawContours(img, contours, -1, (0,255,0), 1)
        # img = cv.drawContours(img, contours_circles, -1, (255,0,0), 1)
        # plt.imshow(img)
        # plt.title('Pre and Post-filtered contours')
        # plt.show()

        # Calculate threshold from contours
        # create blank matrix
        b = np.ones(shape=[2048, 2048], dtype=np.uint8)
        # Add filled contours representing holes in transwells
        holes_thresh = cv.fillPoly(b, contours_circles, (255,255,255))
        # Dilate to make threshold encompass surronding bright areas
        kernel = np.ones((3,3),np.uint8)
        holes_thresh = cv.dilate(holes_thresh, kernel, iterations=3)

        # Show dilated threshold
        # img = cv.cvtColor(holes_thresh, cv.COLOR_GRAY2BGR)
        # img = cv.drawContours(img,contours_circles, -1, (255,0,0), 1)
        # plt.imshow(img)
        # plt.show()

        # Grab GFP image use gamma to bring out weakly stained cells
        GFP = sCI.proccessNuclearImage(self.GFP, gamma=0.2, debug=False)

        # trying different treshold settings
        # blocksize = larger will focus on larger objects
        # c = larger will look for larger changes in intensity
        img = GFP
        img_blur = cv.GaussianBlur(img,(5,5),0)

        th1 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,2)
        th2 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,2)
        th3 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,23,3)
        # titles = ['Original Image (Blur)', 'th1', 'th2', 'th3']
        # images = [img_blur, th1, th2, th3]
        # sCI.showImages(images, titles)

        # Use threshold of GFP and remove holes
        GFP_thresh = cv.bitwise_not(th3)
        GFP_thresh = cv.erode(GFP_thresh, kernel, iterations = 1)
        overlap_thresh = cv.bitwise_and(GFP_thresh, cv.bitwise_not(holes_thresh))

        # show new threshold
        overlap = cv.cvtColor(overlap_thresh, cv.COLOR_GRAY2BGR)
        overlap = cv.drawContours(overlap, self.nucleiCircles, -1, (255,0,0), 1)
        # overlap = cv.drawContours(overlap, contours_circles, -1, (255,0,0), 1)
        rgb = sCI.colorImage(blue=sCI.images[0],green=sCI.images[1],gamma=0.2)
        GFP = cv.drawContours(rgb, self.nucleiCircles, -1, (255,0,0), 1)
        # plt.subplot(1,2,1),plt.imshow(overlap)
        # plt.subplot(1,2,2),plt.imshow(GFP)
        # plt.show()


        # nucleiCircles = list(self.nucleiCircles[i] for i in random.sample(list(range(len(self.nucleiCircles))),50))

        img = GFP
        self.GFPcount = 0
        GFPpercentages = []
        for nuclei in self.nucleiCircles:
            boundRect = cv.boundingRect(nuclei)
            # print(f"x = {boundRect[0]}, y = {boundRect[1]}, width = {boundRect[2]}, height= {boundRect[3]}")
            # cv.rectangle(img, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (255,0,0), 1)
            nucleusSize = 0
            GFPcount = 0
            for x in range(0, int(boundRect[2])):
                for y in range(0, int(boundRect[3])):
                    pt = (int(boundRect[0]) + x, int(boundRect[1]) + y)
                    ppt = cv.pointPolygonTest(nuclei, pt, measureDist=False)            
                    if ppt == 1:
                        # Pixel inside nucleus
                        nucleusSize += 1
                        # Check if Pixel is GFP+
                        if overlap_thresh[pt[1],pt[0]] != 0:
                            GFPcount += 1
                            # plt.scatter(pt[0],pt[1],c="purple",s=0.2)
                        # else:
                        #     plt.scatter(pt[0],pt[1],c="tan",s=0.2)
            
            # print(f"nucleusSize: {nucleusSize}. GFPcount: {GFPcount}. %GFP =  {'{:.2%}'.format(GFPcount/nucleusSize*100)}")
            GFPpercentage = (GFPcount/nucleusSize)
            GFPpercentages.append(GFPpercentage)

            if GFPpercentage > 0.6:
                self.GFPcount += 1
                # cv.rectangle(img, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,255), 2)
        # plt.imshow(img)
        # self.export_pdf.savefig(dpi=300)
        # plt.close()
        # plt.show()

        # plot histogrm of GFP+ pixels in each nuclei contour
        # plt.hist(GFPpercentages)
        # plt.yscale('log')
        # plt.show()
               
        print(f"GFP+ cells: {self.GFPcount}")

    def plotResults(self, sCI):
        plt.figure(figsize= (20,10))
        plt.suptitle(f"{self.path}\\{self.imgFile}")
        # plt.subplot(1,2,1), plt.imshow(img)
        # plt.subplot(1,2,1), plt.title(f"DAPI+: {sCI.nucleiCount}")
        # plt.subplot(1,2,1), plt.scatter(sCI.centroid_x,sCI.centroid_y,c=c,s=0.5)
        plt.imshow(self.img)
        plt.title(f"DAPI+: {sCI.nucleiCount}")
        # plt.scatter(sCI.centroid_x,sCI.centroid_y,c=c,s=4,cmap=plt.get_cmap('Set1'))            
        plt.show()

        # plt.subplot(1,2,2), plt.imshow(GFP_watershed)
        # plt.subplot(1,2,2), plt.scatter(sCI.centroid_x,sCI.centroid_y,s=0.5)
        # plt.subplot(1,2,2), plt.scatter(GFP_centroid_x,GFP_centroid_y,c="yellow",s=0.5)
        # plt.subplot(1,2,2), plt.title(f"GFP+DAPI+: {GFP_DAPI_markers.max()}")
        # export_pdf.savefig(dpi=300)
        # plt.close()

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

                    # find nuclei
                    self.findNucleiOnTranswell(sCI)

                    # find GFP+ cells
                    self.findGFP(sCI)

                    # plot results
                    # self.plotResults(sCI)

                    if "model" in locals():
                        sCI.processCells()
                        sCI.getPredictions(self.model)
                        sCI.processPredictions(export_pdf, prediction_cutoff = self.o4_cutoff, debug=False)

                        self.results.append({
                            'path': sCI.path,
                            'imgFile': sCI.imgFile,
                            'stage': stage,
                            'well': well,
                            'position': position,
                            'nucleiCount': sCI.nucleiCount,
                            'o4pos_count': sCI.o4pos_count,
                            'o4neg_count': sCI.o4neg_count,
                            'o4%': "{:.2%}".format(sCI.o4pos_count/(sCI.o4pos_count+sCI.o4neg_count)),
                            })

                    else:            
                        self.results.append({
                            'path': sCI.path,
                            'imgFile': sCI.imgFile,
                            'stage': stage,
                            'well': well,
                            'position': position,
                            'nucleiCount': self.nucleiCount,
                            'GFP_count': self.GFPcount,
                            })
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
                'gamma', self.gamma,
                'thres', self.thres,
                ])
            w.writerow('')
            # results
            w = csv.DictWriter(f, self.results[0].keys())
            w.writeheader()
            w.writerows(self.results)

# run code

javabridge.start_vm(class_path=bioformats.JARS)        

# JW Transwell
a = analyzeGFP(folder = 7, debug = False)
a.runAnalysis()
a.exportResults()

javabridge.kill_vm()

print('All Done')