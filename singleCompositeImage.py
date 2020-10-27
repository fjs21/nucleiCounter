import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import math
import os, errno
from pathlib import Path

# to enable VSI file support
import bioformats

from settings import Settings
from commonFunctions import fullPath 

class singleCompositeImage():

    def __init__(self: str, path: str, imgFile: str, dapi_ch: int, o4_ch: int=None, EdU_ch: int=None,
        # to take into account different magnifications across images and training set
        scalefactor: float = 1, 
        # gamma correct (0.5)
        gamma: float = 1.0, 
        debug: bool = False):
        
        self.debug = debug

        self.path = path
        self.imgFile = imgFile
        
        self.dapi_ch = dapi_ch
        self.o4_ch = o4_ch
        self.EdU_ch = EdU_ch
        
        self.scalefactor = scalefactor
        self.gamma = gamma
        self.debug = debug

        self.settings = Settings()

        # load images
        self.images = self.loadImages()
        # standardize scale
        if self.scalefactor != 1:
            self.scaleImages(scalefactor=scalefactor, debug=self.debug)
        # get color image for export
        if(self.o4_ch is None):
            if(self.EdU_ch is not None):
                self.rgb = self.colorImage(blue=self.images[self.dapi_ch], red=self.images[self.EdU_ch], gamma=self.gamma)
            else:
                self.rgb = self.colorImage(blue=self.images[self.dapi_ch], gamma=self.gamma)
        elif(self.o4_ch is not None):
            self.rgb = self.colorImage(blue=self.images[self.dapi_ch], green=self.images[self.o4_ch], gamma=self.gamma)

    def processDAPI(self, threshold_method: str, gamma: float = -1, debug: bool=False):
        """ Process DAPI channel. """

        # if DAPI gamma not set, use global gamma setting
        if (gamma == -1):
            gamma = self.gamma

        self.nucleiImg = self.proccessNuclearImage(self.images[self.dapi_ch], gamma=gamma, debug=debug)
        self.threshold_method = threshold_method
        self.nucleiThresh = self.imageThreshold(self.nucleiImg, self.threshold_method, debug)
        
        self.nucleiCount, self.output, self.nucleiMask, self.nucleiWatershed, self.nucleiMarkers = self.thresholdSegmentation(self.nucleiThresh, self.nucleiImg, debug)
        self.centroids = self.output[3][1:,]
        self.centroid_x = self.centroids[:,0].astype(int)
        self.centroid_y = self.centroids[:,1].astype(int)

    def processCells(self, debug=False):
        self.width = self.settings.width
        self.height = self.settings.height
        # get images
        self.getCells(debug)
        self.predictions = np.empty(shape=[len(self.cells)])
        
    def processMarkers(self, markerFile, marker_index, debug=False):
        self.markerFile = markerFile
        self.marker_index = marker_index
        # get markers
        self.readMarkers(debug)
        # filter markers_X and markers_Y based on markers_type == 2
        self.markers_XY = self.markers[self.markers[:,2] == self.marker_index, :2]
        # find nearest cells to each marker
        self.findNearestNeighbors(debug)
        # assign markers to cells
        self.assignMarkersToCells(debug)

    def loadImages(self, debug: bool = False):
        """IMAGE LOADING"""
        fullpath = fullPath(self.path, self.imgFile)
        if Path(fullpath).suffix == '.tif':
            ret, images = cv.imreadmulti(fullpath, flags = -1)
            if not ret:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fullpath)
        elif Path(fullpath).suffix == '.vsi':
            images = self.openVSI()

        if self.debug or debug:
            print(f"Loaded '{self.imgFile}' from '{self.path}'\n with {len(images)} channels.")
            titles = []
            for i in range(len(images)):
                if (i == self.dapi_ch):
                    titles.append('DAPI')
                elif (i == self.o4_ch):
                    titles.append('O4')
                elif (i == self.EdU_ch):
                    titles.append('EdU')
                else:
                    titles.append(f'channel #{i}')

            self.showImages(images, titles, 'Loaded Images')    

        return images

    def openVSI(self, debug: bool = False):
        """Using bioformats to open .vsi image"""
        fullpath = fullPath(self.path, self.imgFile)
        images = bioformats.load_image(fullpath, rescale=False)
        images = cv.split(images)

        return images

    def scaleImages(self, scalefactor, debug=False):
        """Scale images."""
        print(self.images[0].shape, scalefactor)
        for i in range(len(self.images)):
            height = int(self.images[i].shape[0] * scalefactor)
            width = int(self.images[i].shape[1] * scalefactor)
            dim = (width, height)
            self.images[i] = cv.resize(self.images[i], dim, interpolation = cv.INTER_AREA)
        print(self.images[0].shape)

    def showImages(self, images, titles='', suptitle=''):
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        # doesn't show both only most recent...
        #plt.suptitle(suptitle)
        plt.suptitle("press 'Q' to move to next step", verticalalignment="bottom")

        cols = int(len(images) // 2 + len(images) % 2)
        rows = int(len(images) // cols + len(images) % cols)
        #plt.figure(figsize = (rows,cols))
        # print("cells/rows",cols,rows)
        for i in range(len(images)):
            img = images[i]
            img = self.gammaCorrect(img)
            img = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            plt.subplot(rows,cols,i+1),plt.imshow(img,'gray')
            if titles != '':
                plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.tight_layout()
        plt.show()

    def proccessNuclearImage(self, img, gamma: float = -1, debug: bool = False):
        """Function to proccess a flourescence image with nuclear localized signal (e.g. DAPI)."""

        # gamma correct - load glbbal setting if not set by parameter
        if gamma == -1:
            gamma = self.gamma

        if gamma != 1:
            if debug:
                self.plotHistogram(img, gamma)
            img = self.gammaCorrect(img, gamma)

        # normalize (stretch histogram and convert to 8-bit)
        img = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
 
        # invert image 
        img = cv.bitwise_not(img)

        # FUTURE: consider other normalization strategies

        return img

    def imageThreshold(self, img, threshold_method, debug=False):
        """IMAGE THRESHOLDING."""
        # based on - https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

        #img = cv.medianBlur(img,5)
        img_blur = cv.GaussianBlur(img,(5,5),0)

        ret,th1 = cv.threshold(img_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        th2 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,11,2)
        th3 = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,11,2)
        titles = ['Original Image (Blur)', 'Global Otsu Thresholding',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img_blur, th1, th2, th3]

        if self.debug or debug:
            self.showImages(images, titles)

        return cv.bitwise_not(eval(threshold_method))

    def thresholdSegmentation(self, thresh, img, debug=False):
        """SEGMENTATION and WATERSHED"""
        # based on - https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html
        # 1. noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations=3)

        # 2. sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=3)

        # 3. Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5) # calculates distance from boundary
        
        dt = dist_transform[dist_transform != 0] #remove zeros
        
        if debug:
            print(f"Max distance: {dist_transform.max()}")
            print(f"Median distance: {np.median(dt)}")

        ret, sure_fg = cv.threshold(dist_transform, np.median(dt), 255, 0) # use median distance (assume most cells are singlets)

        # 4. Finding unknown region
        sure_fg = np.uint8(sure_fg) 
        unknown = cv.subtract(sure_bg,sure_fg)

        # 5. Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        # img = self.proccessNuclearImage(self.images[channel])
        img = cv.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        markers = cv.watershed(img,markers)

        img[markers == -1] = [255,0,0]

        titles = ['threshold', 'opening', 'dist_transform', 'sure_fg', 'unknown', 'watershed']
        images = [thresh, opening, dist_transform, sure_fg, unknown, img]

        if self.debug or debug:
            self.showImages(images,titles)

        count = markers.max()-1
        output = cv.connectedComponentsWithStats(sure_fg)

        return([count, output, sure_fg, img, markers])

    # other functions of potential interest - findContours
    # NEXT FILTER ON SIZE, CIRCULARITY - GET X-Y centroid
    # https://www.learnopencv.com/blob-detection-using-opencv-python-c/ - for circularity

    def showCentroids(self):
        """Show centroids side-by-side with image."""
        plt.subplot(1,2,1),plt.imshow(img,'gray')
        plt.subplot(1,2,2),plt.scatter(centroid_x,-centroid_y)
        plt.show()

    def colorImage(self, blue, green='', red='', gamma: float = -1):
        """Creates color imnage showing O4 and DAPI in consistent way. Requires blue image"""

        if gamma == -1:
            gamma = self.gamma

        if isinstance(red, np.ndarray):
            red = cv.normalize(src=red, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        else:
            red = np.zeros(blue.shape, dtype=np.uint8)

        if isinstance(green, np.ndarray):
            # add gama correction to O4 channel
            if gamma != 1.0:
                green = self.gammaCorrect(green)
            green = cv.normalize(src=green, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        else:
            green = np.zeros(blue.shape, dtype=np.uint8)
        
        if gamma != 1.0:
                blue = self.gammaCorrect(blue)
        blue = cv.normalize(src=blue, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        rgb = cv.merge((blue, green, red))

        return rgb

    def getCell(self, rgb, centroid):
        """Find boundaries for cell ROI and return image."""
        x_min = (centroid[0] - self.width/2).astype(int)
        x_max = (centroid[0] + self.width/2).astype(int)
        y_min = (centroid[1] - self.height/2).astype(int)
        y_max = (centroid[1] + self.height/2).astype(int)
        #print(x_min,x_max,y_min,y_max)
        cell = rgb[y_min:y_max,x_min:x_max,:]
        return cell

    def getCells(self, debug=False):
        """Find all cell images."""
        self.cells = []
        debug_once = debug or self.debug
        for i in range(self.centroids.shape[0]):
            self.cell = self.getCell(self.rgb, self.centroids[i])
            if self.cell.shape==(self.width,self.height,3):
                self.cells.append(self.cell)
                if debug_once:
                    self.showCell(i)
                    debug_once = False 
            else:
                self.cells.append(0)

    def readMarkers(self, debug=False):
        fullpath = fullPath(self.path, self.markerFile)
        root = ET.parse(fullpath).getroot()

        # Define marker X, Y, and type
        markers_X = []
        markers_Y = []
        markers_type = []
        #marker_type = 0

        for marker_type_tag in root.iter('Marker_Type'):
            marker_type = marker_type_tag.find('Type').text
            for marker_tag in marker_type_tag.iter('Marker'):
                markers_X.append(int(float(marker_tag.find('MarkerX').text)*self.scalefactor))
                markers_Y.append(int(float(marker_tag.find('MarkerY').text)*self.scalefactor))
                markers_type.append(int(marker_type))

        self.markers = np.zeros((len(markers_X),3), dtype=np.uint16)
        self.markers[:,0] = markers_X
        self.markers[:,1] = markers_Y
        self.markers[:,2] = markers_type
        if debug or self.debug:
            values, counts = np.unique(markers_type, return_counts=True)
            print(f"Markers summary: {values} with counts: {counts}")
            self.showMarkers()

    def showMarkers(self):
        """ Show overlay of centroids and markers on rgb image."""
        plt.imshow(self.rgb)
        plt.scatter(self.centroids[:,0], self.centroids[:,1], s=5, c="blue")
        scatter = plt.scatter(self.markers[:,0], self.markers[:,1], s=5,
            c=self.markers[:,2], cmap=plt.get_cmap('Accent'))
        plt.legend(*scatter.legend_elements(), loc='right', title='types',
            bbox_to_anchor= (0.0, 0.5))
        plt.xticks([]),plt.yticks([])
        plt.show()
        
    # CODE FOR CALCULATION OF NEAREST NEIGHBORS
    def p2p_distance(self, p1, p2):
        """Find distance between two points in 2D space."""
        v = np.array(p2) - np.array(p1)
        v = abs(v)
        # v = hypotenuse
        d = math.sqrt(v[0]**2 + v[1]**2)
        return d

    def findDistances(self, sources, targets):
        """Returns an array of distances between two sets of points. (p1_index, p2_index, d). """
        fD = np.zeros((sources.shape[0],targets.shape[0]))
        for i in range(sources.shape[0]):
            p1 = sources[i,]
            for j in range(targets.shape[0]):
                p2 = targets[j,]
                d = self.p2p_distance(p1, p2)
                fD[i,j] = d
        return fD

    def findNearestNeighbors(self, debug = False):
        self.fd = self.findDistances(self.centroids, self.markers_XY)

        fd_mins = np.amin(self.fd,axis=1)
        result = np.argmin(self.fd,axis=1)
        if debug or self.debug:
            print(f"Assignment of each centroid to nearest marker: {result}") 

        # distance of each marker to nearest centroid - used for classification
        fd_mins_c = np.amin(self.fd,axis=0)
        result_c = np.argmin(self.fd,axis=0)
        if debug or self.debug:
            print(f"assignment of each marker to nearest centroid: {result_c}") 

        self.NN = result_c

    def assignMarkersToCells(self, debug=False):
        """Function to assign markers to specific cells"""
        self.centroids_classification = np.zeros(self.centroids.shape[0]) # set all to O4-
        
        debug_once = debug or self.debug
        for i in range(self.markers_XY.shape[0]):
            if self.fd[self.NN[i],i] < self.settings.fD_cutoff:
                self.centroids_classification[self.NN[i]] = 1 # set to O4+ 
                if debug_once and not isinstance(self.cells[self.NN[i]], int):
                    print(f"{i}: marker_XY={self.markers_XY[i,]}, ", 
                        f"nearest centroid#={self.NN[i]}, at ({self.centroids[self.NN[i]]}), d={self.fd[self.NN[i],i]}")
                    self.showCell(self.NN[i],'An example O4+ cell')
                    debug_once = False
        # check if cell is not usable as image to close to edge - assign to -1
        for i in range(self.centroids.shape[0]):
            if isinstance(self.cells[i], int):
                self.centroids_classification[i] = -1    

        if debug or self.debug:
            plt.imshow(self.rgb)
            plt.scatter(self.centroids[:,0], self.centroids[:,1], s=5, c="blue")
            plt.scatter(self.markers_XY[:,0], self.markers_XY[:,1], s=5, c='green')
            for i in range(self.markers_XY.shape[0]):
                if self.fd[self.NN[i],i] < self.settings.fD_cutoff:
                    plt.plot( (self.markers_XY[i,0], self.centroids[self.NN[i],0]) , (self.markers_XY[i,1], 
                        self.centroids[self.NN[i],1]), c='white' )
                else:
                    plt.plot( (self.markers_XY[i,0], self.centroids[self.NN[i],0]) , (self.markers_XY[i,1],
                        self.centroids[self.NN[i],1]), c='red' )              
            plt.show()

    def reportResults(self):
        # Image details
        print(f"Path: {self.path}")
        print(f"Image File: {self.imgFile}")
        print(f"Shape = {self.images[self.dapi_ch].shape}, Bit depth = {self.images[self.dapi_ch].dtype}") 
        print(f"Nuclei found: {self.nucleiCount}\n")
        
        if hasattr(self, 'markerFile'):
            # Marker counter details
            print(f"Counter File: {self.markerFile} using marker #{self.marker_index}")
            # Summarize marker info
            values, counts = np.unique(self.markers[:,2], return_counts=True)
            print(f"Unique marker values = {values}\n with counts:{counts}\n")
            # Summarize classification
            print("Classification complete (-1: image too close to edge, 0: O4-DAPI+, 1: O4+DAPI+")
            values, counts = np.unique(self.centroids_classification, return_counts=True)
            print(f"values, counts: {values, counts}\n")

    def showCell(self, cell_index, title=''):
        # print(self.cells[cell_index].shape)
        plt.imshow(self.cells[cell_index])
        if title != '':
            plt.title(title)
        plt.xticks([]),plt.yticks([])
        plt.show()

    def saveCellImg(self, cell_index, filename):
        cv.imwrite(filename, self.cells[cell_index])

    def getPredictions(self, model):
        """Find predictions for all cells in image."""
        for i in range(len(self.cells)):
            if not isinstance(self.cells[i],int):
                cell = self.cells[i].astype('float64')
                cell = np.expand_dims(cell, axis=0)
                # Find predictions - might not be the best method for multiple cells
                self.predictions[i] = model.predict(cell)
            else:
                self.predictions[i] = -1

    def classifyCell(self, cell_index, cutoff = 0.5):
        """Return cell classification"""
        if self.predictions[cell_index] == -1:
            return -1
        if self.predictions[cell_index] > cutoff:
            return 1
        else:
            return 0

    def processPredictions(self, export_pdf, prediction_cutoff = 0.5, debug: bool=False):
        from matplotlib.patches import Rectangle

        # bar width on plots to indicate which cells will not be counted
        width = self.settings.width/2
        height = self.settings.height/2

        cell_info = np.zeros((len(self.cells),3))
        # start per image counters for each classification
        self.o4pos_count = 0
        self.o4neg_count = 0
        for i in range(len(self.cells)):
            if not isinstance(self.cells[i], int):
                # total_cellImages += 1
                cell_type = self.classifyCell(i, prediction_cutoff)
                if cell_type == 1:
                    title='O4+'
                    self.o4pos_count += 1
                    cell_info[i,:] = [self.centroid_x[i], self.centroid_y[i], 1]
                else:
                    title='O4-'
                    self.o4neg_count += 1
                    cell_info[i,:] = [self.centroid_x[i], self.centroid_y[i], 0]
                # Show each cell with title label
                if debug:
                    self.showCell(i, title)
        # Generate summary image
        plt.figure(figsize= (10,10))
        plt.imshow(self.rgb)
        plt.title(os.path.join(self.path,self.imgFile)[-80:])
        plt.scatter(cell_info[:,0],cell_info[:,1],c=cell_info[:,2],s=0.2)
        # if marker_index != 0:
        #     if pattern == '*MMStack.ome*.tif':
        #         markerFile = findNewestMarkerFile(self.path)
        #     elif pattern == '*Composite*.tif':
        #         markerFile = findMatchingMarkerFile(self.path, self.imgFile)
        #     if markerFile:
        #         self.processMarkers(markerFile['name'], marker_index)
        #         plt.scatter(self.markers_XY[:,0], self.markers_XY[:,1], s=2, c='green', alpha=0.5)
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((0,0),width,self.rgb.shape[0],fill="white",alpha=0.4,ec=None))
        currentAxis.add_patch(Rectangle((self.rgb.shape[1]-width,0),width,self.rgb.shape[0],fill="white",alpha=0.4,ec=None))
        currentAxis.add_patch(Rectangle((width,0),self.rgb.shape[1]-(2*width),height,fill="white",alpha=0.4,ec=None))
        currentAxis.add_patch(Rectangle((width,self.rgb.shape[0]-height),self.rgb.shape[1]-(2*width),height,fill="white",alpha=0.4,ec=None))
        export_pdf.savefig()
        plt.close()

    def gammaCorrect(self, image, gamma: float=-1):
        """Gamma correct."""
        if gamma == -1:
            gamma = self.gamma
        max_pixel = np.max(image)
        corrected_image = image
        corrected_image = (corrected_image / max_pixel) 
        corrected_image = np.power(corrected_image, gamma)
        corrected_image = corrected_image * max_pixel
        return corrected_image

    def plotHistogram(self, image, gamma: float=-1):
        """Plot a histogram of an image."""
        if gamma == -1:
            gamma = self.gamma
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        img = cv.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        plt.subplot(2,2,1),plt.imshow(img)
        plt.title('Original Image')
        hist = cv.calcHist(img,[0],None,[255],[0,255])
        plt.subplot(2,2,2),plt.plot(hist, color='k')
        plt.xlim([0, 255])
        plt.yscale('log')

        img_g = self.gammaCorrect(image, gamma)
        img_g = cv.normalize(src=img_g, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        plt.subplot(2,2,3),plt.imshow(img_g)
        plt.title('Gamma Corrected Image')
        hist_g = cv.calcHist(img_g,[0],None,[255],[0,255])
        plt.subplot(2,2,4),plt.plot(hist_g, color='k')       
        plt.xlim([0, 255])
        plt.yscale('log')

        plt.suptitle('Gamma Correction')

        plt.tight_layout()
        plt.show()