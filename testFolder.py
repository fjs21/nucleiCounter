import random
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages

# start JVM for compatibility with VSI files
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

settings = Settings()

# BK siRNA experiments
folder = 6 # July 31, 2019 # 4 (July 2020, Discard) # 3 (Feb 2020) # 1 (Sept 2020) 
debug = False

model = loadKerasModel(settings.kerasModel)

# retrieve settings using 'folder'
root = settings.folder_dicts[folder]['root']
pattern = settings.folder_dicts[folder]['pattern']
files = find(pattern, root)
dapi_ch = settings.folder_dicts[folder]['dapi_ch']
o4_ch = settings.folder_dicts[folder]['o4_ch']
marker_index = settings.folder_dicts[folder]['marker_index']

gamma = settings.folder_dicts[folder]['gamma']
thres = settings.folder_dicts[folder]['thres']
o4_cutoff = 0.5 # default was 0.5

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
            sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch=o4_ch, scalefactor=1, debug=debug, gamma=gamma)
            sCI.processDAPI(threshold_method=thres, gamma=gamma, debug=debug) # based on manual counts (see OneNote)
            if debug:
                sCI.reportResults()

            if "model" in locals():
                sCI.processCells()
                sCI.getPredictions(model)
                sCI.processPredictions(export_pdf, prediction_cutoff = o4_cutoff, debug=False)

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
                    })

            else:            
                results.append({
                    'path': sCI.path,
                    'imgFile': sCI.imgFile,
                    'stage': stage,
                    'well': well,
                    'position': position,
                    'nucleiCount': sCI.nucleiCount,
                    })
        except:
            print(f"Failed on path '{path}'. Image: {imgFile}")
            raise

# output results as csv
import csv
filename = 'results_folder_' + str(folder) + '.csv'
with open(filename,'w',newline='') as f:
    # report analysis settings
    w = csv.writer(f)
    w.writerow([
        'gamma', gamma,
        'thres', thres,
        ])
    w.writerow('')
    # results
    w = csv.DictWriter(f, results[0].keys())
    w.writeheader()
    w.writerows(results)

javabridge.kill_vm()

print('All Done')