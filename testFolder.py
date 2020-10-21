import random
import numpy as np 

# start JVM for compatibility with VSI files
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

settings = Settings()

# manual settings
folder = 0
debug = True
thres = 'th2'
gamma = False

# retrieve settings using 'folder'
root = settings.folder_dicts[folder]['root']
pattern = settings.folder_dicts[folder]['pattern']
files = find(pattern, root)
dapi_ch = settings.folder_dicts[folder]['dapi_ch']
o4_ch = settings.folder_dicts[folder]['o4_ch']
marker_index = settings.folder_dicts[folder]['marker_index']


# start analysis
print(f"Found {len(files)} matching '{pattern}' in '{root}'")
print("***************************")
print("Starting to analyze images")

# select file sample
if debug:
    # select five files at random
    #files = list(files[i] for i in random.sample(list(range(len(files))), 5))

    # select five files to do manual count comparisons
    files = list(files[i] for i in range(7,12)) 

results = []

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
        sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch, scalefactor=1, debug=debug, gamma=gamma)
        sCI.processDAPI(threshold_method=thres, gamma=gamma) # based on manual counts (see OneNote)
        if debug:
            sCI.reportResults()

        results.append({
            'path': sCI.path,
            'imgFile': sCI.imgFile,
            'stage': stage,
            'well': well,
            'position': position,
            'nucleiCount': sCI.nucleiCount
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