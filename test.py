import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages

import cv2 as cv

# start JVM for compatibility with VSI files
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

settings = Settings()

# path = 'C:\\Users\\fjs21\\OneDrive\\Research&Lab\\Data\\Transwell Experiments 2020\\Image Data from Pilot Experiment' 
# imgFile = 'Post scrape on C2_C2-1_8365.tif' # Dapi in ch 1
# dapi_ch = 0
# o4_ch = None
# gamma = False
# thres = 'th2'

# path = 'C:\\scratch\\sep 2020 sample'
# imgFile = 'sibk sep 2020 sample_B2-11_8232.vsi' # Dapi in ch 2
# dapi_ch = 1
# o4_ch = 2
# EdU_ch = None
# gamma = True
# thres = "th2"
# model = loadKerasModel(settings.kerasModel)

# path = 'Y:\\People\\Ahmed\\Transwell Experiments\\Exp 3\\Proliferation\\Alone'
# imgFile = 'Transwell 3_Multichannel_20190603_2939.vsi'
# dapi_ch = 0
# o4_ch = None
# EdU_ch = 1
# gamma = 3
# thres = 'th2'

path = 'Y:\\People\\Roopa\\dk si alpha expts\\siBK 0731 sample\\NC\\oxo'
imgFile = 'siBK 07312019_C2-11_7321.vsi'
dapi_ch = 1
o4_ch = 2
EdU_ch = None
dapi_gamma = 0.5
o4_gamma = 0.2
thres = 'th2'
model = loadKerasModel(settings.kerasModel)

# marker_index = 5
width = height = 128

sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch=o4_ch, EdU_ch=EdU_ch, scalefactor=1, debug=True, dapi_gamma = dapi_gamma, o4_gamma = o4_gamma)
sCI.processDAPI(threshold_method=thres)
sCI.processCells()
# sCI.processMarkers(markerFile, marker_index, debug=True)
# print(sCI.centroids_classification[3])
sCI.reportResults()

""" EdU count. """
if (EdU_ch is not None):
    # Count EdU channel
    EdU = sCI.proccessNuclearImage(sCI.images[sCI.EdU_ch], gamma=gamma)
    sCI.threshold_method = 'th1'
    sCI.imageThreshold(EdU, debug=True)

    # kernel = np.ones((3,3),np.uint8)
    # print(kernel)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
    # print(kernel)
    # sCI.thresh = cv.morphologyEx(sCI.thresh,cv.MORPH_CLOSE,kernel, iterations=3)

    EdU_count, EdU_output, EdU_mask = sCI.thresholdSegmentation(EdU_ch, debug=True)
    EdU_DAPI_overlap = cv.bitwise_and(sCI.nucleiMask, EdU_mask)
    ret,EdU_DAPI_markers = cv.connectedComponents(EdU_DAPI_overlap)
    EdU_count2 = EdU_DAPI_markers.max() # count does not use watershed step

    print(f"EdU count: {EdU_count}")
    print(f"EdU count: {EdU_count2}")

else:
    EdU_count = None


if "model" in locals():
    sCI.getPredictions(model)

    with PdfPages('test.pdf') as export_pdf:
        sCI.processPredictions(export_pdf, debug=False)
    print(f"Total cell classified: {sCI.o4pos_count+sCI.o4neg_count}")
    print(f"O4+ cells: {sCI.o4pos_count}")
    print("O4%: " + "{:.2%}".format(sCI.o4pos_count/(sCI.o4pos_count+sCI.o4neg_count)))

javabridge.kill_vm()
print('All Done')

