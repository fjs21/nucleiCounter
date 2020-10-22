import random
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import pyplot as plt
import cv2 as cv

# start JVM for compatibility with VSI files
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

settings = Settings()

# manual settings
# folder = 1
# debug = False
# thres = 'th2'
# gamma = True
# model = loadKerasModel(settings.kerasModel)

# IGBFP2 experiment
folder = 2
debug = False
thres = 'th2'
gamma = False

# folder = 3
# debug = False
# thres = 'th2'
# gamma = True
# model = loadKerasModel(settings.kerasModel)

# retrieve settings using 'folder'
root = settings.folder_dicts[folder]['root']
pattern = settings.folder_dicts[folder]['pattern']
files = find(pattern, root)
dapi_ch = settings.folder_dicts[folder]['dapi_ch']
o4_ch = settings.folder_dicts[folder]['o4_ch']
EdU_ch = settings.folder_dicts[folder]['EdU_ch']
marker_index = settings.folder_dicts[folder]['marker_index']

# start analysis
print(f"Found {len(files)} matching '{pattern}' in '{root}'")
print("***************************")
print("Starting to analyze images")

# select file sample
if debug:
    # select five files at random
    # files = list(files[i] for i in random.sample(list(range(len(files))), 5))

    # select five files to do manual count comparisons
    files = list(files[i] for i in range(1,2)) 

results = []

with PdfPages('testFolder.pdf') as export_pdf:

    for file in files:
        path = file['path'] 
        imgFile = file['name']

        # parse file names
        imgFile_split = imgFile.split('_')
        if(imgFile_split[0].upper().find('PRE')>0):
            stage = "PRE"
        else:
            stage = "POST"
        well_position = imgFile_split[1].split('-')
        well = well_position[0]
        position = well_position[1]

        try:
            sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch=o4_ch, EdU_ch=EdU_ch, scalefactor=1, debug=debug, gamma=gamma)
            sCI.processDAPI(threshold_method=thres, gamma=gamma) # based on manual counts (see OneNote)
            if debug:
                sCI.reportResults()

            if (EdU_ch is not None):
                # Count EdU channel
                EdU = sCI.proccessNuclearImage(sCI.images[sCI.EdU_ch], gamma=gamma)
                sCI.threshold_method = thres
                sCI.imageThreshold(EdU, debug=debug)
                EdU_count, EdU_output, EdU_mask = sCI.thresholdSegmentation(EdU_ch, debug=debug)
                EdU_DAPI_overlap = cv.bitwise_and(sCI.nucleiMask, EdU_mask)
                ret,EdU_DAPI_markers = cv.connectedComponents(EdU_DAPI_overlap)
                EdU_count2 = EdU_DAPI_markers.max() # count does not use watershed step
            else:
                EdU_count = None

            mCherry = sCI.proccessNuclearImage(sCI.images[2], gamma=True)
            sCI.threshold_method = thres
            sCI.imageThreshold(mCherry, debug=debug)
            mCherry_count, mCherry_output, mCherry_mask = sCI.thresholdSegmentation(2, debug=debug)

            # plt.subplot(1,3,1),plt.imshow(sCI.nucleiMask)
            # plt.subplot(1,3,2),plt.imshow(mCherry_mask)
            
            mCherry_DAPI_overlap = cv.bitwise_and(sCI.nucleiMask, mCherry_mask)
            ret,mCherry_DAPI_markers = cv.connectedComponents(mCherry_DAPI_overlap)
            # print(f"Number of mCherry+ cells: {markers.max()}")

            mCherry_EdU_overlap = cv.bitwise_and(EdU_DAPI_overlap, mCherry_mask)
            ret,mCherry_EdU_markers = cv.connectedComponents(mCherry_EdU_overlap)
            
            # plt.subplot(1,3,3),plt.imshow(overlap)
            # plt.show()

            if "model" in locals():
                sCI.processCells()
                sCI.getPredictions(model)
                sCI.processPredictions(export_pdf, debug=False)

                results.append({
                    'path': sCI.path,
                    'imgFile': sCI.imgFile,
                    'stage': stage,
                    'well': well,
                    'position': position,
                    'nucleiCount': sCI.nucleiCount,
                    'o4pos_count': sCI.o4pos_count,
                    'o4neg_count': sCI.o4neg_count,
                    'o4%': "{:.2%}".format(sCI.o4pos_count/(sCI.o4pos_count+sCI.o4neg_count)),
                    'EdU_count': EdU_count,
                    'mCherry_count': mCherry_DAPI_markers.max(),
                    'mCherryEdU_count': mCherry_EdU_markers.max(),
                    })

            else:            
                results.append({
                    'path': sCI.path,
                    'imgFile': sCI.imgFile,
                    'stage': stage,
                    'well': well,
                    'position': position,
                    'nucleiCount': sCI.nucleiCount,
                    'EdU_count': EdU_count,
                    'mCherry_count': mCherry_DAPI_markers.max(),
                    'mCherryEdU_count': mCherry_EdU_markers.max(),
                    'EdU_count2': EdU_count2,
                    })
        except:
            print(f"Failed on path '{path}'. Image: {imgFile}")
            raise

# output results as csv
import csv
filename = 'results_folder_' + str(folder) + '.csv'
with open(filename,'w',newline='') as f:
    w = csv.DictWriter(f, results[0].keys())
    w.writeheader()
    w.writerows(results)

javabridge.kill_vm()

print('All Done')