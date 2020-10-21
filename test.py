import numpy as np 

# start JVM for compatibility with VSI files
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

# path = 'C:\\Users\\fjs21\\OneDrive\\Research&Lab\\Data\\Transwell Experiments 2020\\Image Data from Pilot Experiment' 
# imgFile = 'Post scrape on C2_C2-1_8365.tif' # Dapi in ch 1
# dapi_ch = 0
# o4_ch = None
# gamma = False
# thres = 'th2'

path = 'C:\\scratch\\sep 2020 sample'
imgFile = 'sibk sep 2020 sample_B2-11_8232.vsi' # Dapi in ch 2
dapi_ch = 1
o4_ch = 2
gamma = False
thres = "th2"

# marker_index = 5
width = height = 128

sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch, scalefactor=1, debug=True, gamma=gamma)
sCI.processDAPI(threshold_method=thres, gamma=gamma)
sCI.processCells()
# sCI.processMarkers(markerFile, marker_index, debug=True)
sCI.reportResults()
# print(sCI.centroids_classification[3])

print('All Done')

