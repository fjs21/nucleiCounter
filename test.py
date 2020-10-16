import numpy as np 

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

path = 'C:\\Users\\fjs21\\OneDrive\\Desktop\\data' 

imgFile = 'Post scrape on C2_C2-1_8365.tif' # Dapi in ch 1

dapi_ch = 0
o4_ch = None
# marker_index = 5
width = height = 128

sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch, scalefactor=1, debug=True)
sCI.processDAPI(threshold_method='th3')
sCI.processCells()
# sCI.processMarkers(markerFile, marker_index, debug=True)
sCI.reportResults()
# print(sCI.centroids_classification[3])

print('All Done')