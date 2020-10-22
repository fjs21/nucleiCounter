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

# Ahmed's n = 3 of the transwell experiments
# images severly over saturated
folder = 5
debug = False
thres = 'th2'
gamma = 4

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


# select file sample
if debug:
    print("Running debug on a few images...")
    # select five files at random
    # files = list(files[i] for i in random.sample(list(range(len(files))), 1))

    # select five files to do manual count comparisons
    files = list(files[i] for i in range(3,4)) 
else:
    print("Starting to analyze images")
results = []

def parseFileName(imgFile):
    """Extract stage, well and image position from file name."""
    imgFile_split = imgFile.split('_')
    if(imgFile_split[0].upper().find('PRE')>0):
        stage = "PRE"
    elif(imgFile_split[0].upper().find('POST')>0):
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

with PdfPages('results_folder_' + str(folder) + '.pdf') as export_pdf:

    for file in files:
        path = file['path'] 
        imgFile = file['name']
        print(f"Processing: {path}\\{imgFile}")

        # parse file names
        stage, well, position = parseFileName(imgFile)

        try:
            sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch=o4_ch, EdU_ch=EdU_ch, scalefactor=1, debug=debug, gamma=gamma)
            sCI.processDAPI(threshold_method=thres, gamma=gamma, debug=debug) # based on manual counts (see OneNote)
            if debug:
                sCI.reportResults()

            """ EdU count. """
            if (EdU_ch is not None):
                # Count EdU channel
                EdU = sCI.proccessNuclearImage(sCI.images[sCI.EdU_ch], gamma=gamma)
                sCI.threshold_method = 'th1'
                sCI.imageThreshold(EdU, debug=debug)
            
                # kernel = np.ones((3,3),np.uint8)
                # print(kernel)
                # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
                # print(kernel)
                # sCI.thresh = cv.morphologyEx(sCI.thresh,cv.MORPH_CLOSE,kernel, iterations=3)

                EdU_count, EdU_output, EdU_mask, EdU_watershed = sCI.thresholdSegmentation(EdU_ch, debug=debug)
                EdU_DAPI_overlap = cv.bitwise_and(sCI.nucleiMask, EdU_mask)
                ret,EdU_DAPI_markers = cv.connectedComponents(EdU_DAPI_overlap)
                EdU_count2 = EdU_DAPI_markers.max() # count does not use watershed step

                EdU_centroid_x = EdU_output[3][1:,0].astype(int)
                EdU_centroid_y = EdU_output[3][1:,1].astype(int)

                """ Generate a summary PDF to quickly review DAPI and EdU counts. """
                plt.figure(figsize= (20,10))
                plt.suptitle(f"{path}\\{imgFile}")
                plt.subplot(1,2,1), plt.imshow(sCI.nucleiWatershed)
                plt.subplot(1,2,2), plt.imshow(EdU_watershed)
                plt.subplot(1,2,2), plt.scatter(sCI.centroid_x,sCI.centroid_y,s=0.5)
                plt.subplot(1,2,2), plt.scatter(EdU_centroid_x,EdU_centroid_y,c="yellow",s=0.5)
                export_pdf.savefig(dpi=300)
                plt.close()

            else:
                EdU_count = None
            
            """ mCherry count. """
            # mCherry = sCI.proccessNuclearImage(sCI.images[2], gamma=True)
            # sCI.threshold_method = thres
            # sCI.imageThreshold(mCherry, debug=debug)
            # mCherry_count, mCherry_output, mCherry_mask = sCI.thresholdSegmentation(2, debug=debug)

            # # plt.subplot(1,3,1),plt.imshow(sCI.nucleiMask)
            # # plt.subplot(1,3,2),plt.imshow(mCherry_mask)
            
            # mCherry_DAPI_overlap = cv.bitwise_and(sCI.nucleiMask, mCherry_mask)
            # ret,mCherry_DAPI_markers = cv.connectedComponents(mCherry_DAPI_overlap)
            # # print(f"Number of mCherry+ cells: {markers.max()}")

            # mCherry_EdU_overlap = cv.bitwise_and(EdU_DAPI_overlap, mCherry_mask)
            # ret,mCherry_EdU_markers = cv.connectedComponents(mCherry_EdU_overlap)
            
            # # plt.subplot(1,3,3),plt.imshow(overlap)
            # # plt.show()

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
                    # 'mCherry_count': mCherry_DAPI_markers.max(),
                    # 'mCherryEdU_count': mCherry_EdU_markers.max(),
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
                    # 'mCherry_count': mCherry_DAPI_markers.max(),
                    # 'mCherryEdU_count': mCherry_EdU_markers.max(),
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