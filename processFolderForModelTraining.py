"""

Each image in a experimental folder is processed.

For DAPI & O4 channels

If markerFile is present, it 'processMarkers' and adds to 'd' list object.

At the end it looks for distances between markers and centroids and compares the automated results with the manually
annotated results.

annotations.json

if export flag is set, then cell images are exported to 'output' folder for model training
"""

import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
# import tkinter.filedialog as fileDialog
from tkinter.ttk import Progressbar

import random
import numpy as np
# import os
import json
from matplotlib import pyplot as plt

# start JVM for compatibility with VSI files
# print('JAVA_HOME =', os.environ['JAVA_HOME'])
import javabridge
import bioformats

from settings import Settings
from singleCompositeImage import singleCompositeImage
from commonFunctions import *

javabridge.start_vm(class_path=bioformats.JARS)

settings = Settings()


class Application(tk.Frame):
    def __init__(self, master=None):
        """Setup tk application."""
        super().__init__(master)
        self.progress = None
        self.console = None
        self.console_scrollbar = None
        self.root = None
        self.master = master
        self.master.title("Process folder for Model Training")
        self.pack()
        self.main_container = tk.Frame(master)
        self.main_container.pack(side="top", fill="both", expand=True)
        self.top_frame = tk.Frame(self.main_container)
        self.bottom_frame = tk.Frame(self.main_container, background="grey")
        self.top_frame.pack(side="top", fill="x", expand=False)
        self.bottom_frame.pack(side="bottom", fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        """Creates widgets on initial window."""
        print("Creating widgets...")

        n1 = tk.Label(self.top_frame,
                      text="""Experiment Name""",
                      font=tkFont.Font(family="Calibri", size=14))
        l13 = tk.Label(self.top_frame,
                       text="""Enable debug?""",
                       anchor='e',
                       font=tkFont.Font(family="Calibri", size=12))

        n1.grid(row=0, column=0, sticky='w', pady=2)
        l13.grid(row=1, column=2, sticky='w', pady=2)

        self.name = tk.StringVar()
        experiments = list(settings.experiments)
        print(experiments)
        self.name.set(settings.defaults["name"])
        combo = ttk.Combobox(self.top_frame, values=experiments,
                             width=80, textvariable=self.name,
                             font=tkFont.Font(family="Calibri", size=14))

        # debug mode?
        debug = tk.BooleanVar()
        debug.set(settings.defaults["debug"])
        e12 = tk.Checkbutton(self.top_frame, text='', variable=debug,
                             onvalue=True, offvalue=False,
                             anchor='w')

        combo.grid(row=0, column=1, sticky='w', pady=2)
        e12.grid(row=1, column=3, columnspan=3, sticky='w', pady=2)

        # start button
        button2 = tk.Button(self.bottom_frame,
                            text="Start",
                            command=lambda: self.start_analysis(experiment=self.name.get(),
                                                                debug=debug.get()),
                            font=tkFont.Font(family="Calibri", size=12))

        button2.pack(side="top")
        # define output console area
        tk.Label(self.bottom_frame,
                 text="""Output""",
                 justify=tk.LEFT,
                 anchor='n',
                 font=tkFont.Font(family="Calibri", size=12)).pack(side='left', fill='y')
        # add scroll bar
        self.console_scrollbar = tk.Scrollbar(self.bottom_frame, orient=tk.VERTICAL)
        self.console_scrollbar.pack(side="right", fill='y')
        # add text console widget
        self.console = tk.Text(self.bottom_frame,
                               yscrollcommand=self.console_scrollbar,
                               font=tkFont.Font(family="Calibri", size=12))
        self.console.pack(side="top", fill="both", expand=True)
        self.console.bind("<Key>", lambda e: "break")
        # assign scroll bar to console yview
        self.console_scrollbar.config(command=self.console.yview)
        # add progress bar
        self.progress = Progressbar(self.bottom_frame,
                                    length=200, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side="bottom", fill="y", expand=True)

    def start_analysis(self, experiment, export: bool = True, debug: bool = False):
        """ Run analysis - process folder for presence of count markers"""

        # retrieve analysis settings
        root = settings.experiments[experiment]['root']
        pattern = settings.experiments[experiment]['pattern']
        dapi_ch = settings.experiments[experiment]['dapi_ch']
        dapi_gamma = settings.experiments[experiment].get('dapi_gamma', 1.0)
        o4_ch = settings.experiments[experiment]['o4_ch']
        o4_gamma = settings.experiments[experiment].get('o4_gamma', 1.0)
        scalefactor = settings.experiments[experiment].get('scalefactor', 1.0)
        # which marker was used in cell counter? set marker_index, set 1 if not present.
        # might be better to prompt
        marker_index = settings.experiments[experiment].get('marker_index', 1)

        # create empty results folder
        create_empty_folder(os.path.join(root, 'keras'))

        # fix unicode filenames
        fix_unicode_filenames(experiment)

        # enumerate all image files using pattern
        files = find(pattern, root)

        # start process
        self.console.insert("end", f"\nFound {len(files)} matching '{pattern}' in '{root}'")
        self.console.insert("end", "\n***************************")
        self.console.insert("end", "\nStarting to analyze images")

        # select file sample
        if debug:
            # select two files at random
            files = list(files[i] for i in random.sample(list(range(len(files))), 5))

        # list to store all cell images
        imageSet = []
        # list to store distances
        d = []

        # rest progress bar
        self.progress["value"] = 0
        self.progress.update()
        # calculate increment for progress bar
        fileNumber = len(files)
        currentFileNumber = 0
        progress_increment = 100 / fileNumber

        for file in files:
            # increment currentFileNumber
            currentFileNumber += 1

            # increment progress bar
            self.progress["value"] += progress_increment
            self.progress.update()

            path = file['path']
            imgFile = file['name']

            try:
                sCI = singleCompositeImage(path=path,
                                           imgFile=imgFile,
                                           dapi_ch=dapi_ch,
                                           dapi_gamma=dapi_gamma,
                                           o4_ch=o4_ch,
                                           o4_gamma=o4_gamma,
                                           scalefactor=scalefactor)
                # process image to count DAPI+ nuclei
                sCI.processDAPI(threshold_method='th2')
                # process each cell within the image
                sCI.processCells()
                # append image to imageSet list
                imageSet.append(sCI)

                """ If marker_index has been set find associated markerFile, i.e. XML data file. """
                if marker_index != 0:
                    markerFile = []
                    if pattern == '*MMStack.ome*.tif':
                        markerFile = findNewestMarkerFile(path)
                    elif pattern == '*Composite*.tif':
                        markerFile = findMatchingMarkerFile(path, imgFile)
                    else:
                        markerFile = findMatchingMarkerFile(path, imgFile)

                    if markerFile:
                        sCI.processMarkers(markerFile['name'], marker_index, debug)
                        if sCI.markers_XY is not None:
                            for i in range(sCI.markers_XY.shape[0]):
                                d.append(sCI.fd[sCI.NN[i], i])

                if debug:
                    sCI.reportResults()

            except:
                self.console.insert("end", f"\nFailed on path '{path}'. Image: {imgFile}")
                raise

            self.console.update()

        """ If there are successfully processed markers plot the distribution of distances. """
        if d:
            self.console.insert("end", f"\nPlotting density of distances between {len(d)} markers and their nearest "
                                       f"neighbor nuclei.")
            import pandas as pd

            pd.DataFrame(d).plot(kind='density')
            plt.show()

            from statistics import mean, stdev

            self.console.insert("end", f"\nMean of marker-to-cell distances: {mean(d)}")
            self.console.insert("end", f"\nStDev of marker-to-cell distances: {stdev(d)}")

            import scipy.stats as st

            self.console.insert("end", st.t.interval(0.95, len(d) - 1, loc=np.mean(d), scale=st.sem(d)))

        self.console.insert("end", "\n***************************")

        self.console.insert("end", f"\nNumber of images loaded in imageSet object: {len(imageSet)}")

        self.console.insert("end", "\n***************************")
        self.console.insert("end", "\nExporting data")
        annotations = []

        """ 
         Export annotations.json file and individual cell images
         Output annotations.json file - appropriate to be stored in root folder
        
         List of dictionaries
            {
                "cell": FILENAME "o4pos.0.tif",
                "path": ORIGINAL IMAGE PATH "data\\_6.17.16 Experiment 23 (E23) - CHPG differentiation Rep1 (4 days) - (from Scope folder)\\Original & composite ICC images + cell counter files\\1) 230 uM - new=done\\field_1",
                "imgFile": ORIGINAL IMAGE FILENAME "field 1_MMStack.ome.tif",
                "markerFile": XML MARKER FILE "newCellCounter_composite.xml",
                "cellIndex": INT REFERRING TO CELL 4,
                "centroid": [X,Y coordinates] [
                    1061.7179487179487,
                    70.58974358974359
                    ],
                "classification": set by processFolderForModelTraining.py 0 = O4-, 1 = O4+ 1,
                "annotation": set by manualAnnotation.py -1 BAD, 0 O4-, 1 O4+
            } 
        """

        """
         Export cell_indices.json file
         Not sure the point of cell_indices.json - disabled for now
        
        # filename = 'cell_indices.json'
        # try:
        #     with open(filename, 'r') as f:
        #         indices = json.load(f)
        #     self.console.insert("end", f"\nLoading... {indices}")
        #     o4neg_cell_index = int(indices['o4neg_cell_index'])
        #     o4pos_cell_index = int(indices['o4pos_cell_index'])
        # except:
        """
        cell_index = 0

        # reset progress bar
        self.progress["value"] = 0
        self.progress.update()

        # For each image, iterate over each cell, update the annotations list object and export image for model training
        for sCI in imageSet:

            # increment progress bar
            self.progress["value"] += progress_increment
            self.progress.update()

            # retrieve and count each classification
            # values, counts = np.unique(sCI.centroids_classification, return_counts=True)
            # self.console.insert("end", values)
            # self.console.insert("end", counts) # sometimes only two values returned. which causes next code to fail

            # total_cellImages += counts[1]
            # if len(counts) == 3:
            #     total_cellImages += counts[2]

            if sCI.centroids_classification is None:
                sCI.centroids_classification = np.zeros(sCI.centroids.shape[0])  # set all to O4-

            for i in range(len(sCI.cells)):

                if isinstance(sCI.cells[i], int):
                    # not a valid cell, skipping...
                    continue

                # Are there markers classified for this image?
                if sCI.centroids_classification[i] == -1:
                    filename = 'unknown.{}.tif'.format(str(cell_index))
                    cell_index += 1

                elif int(sCI.centroids_classification[i]) == 0:
                    filename = 'o4neg.{}.tif'.format(str(cell_index))
                    cell_index += 1

                elif int(sCI.centroids_classification[i]) == 1:
                    filename = 'o4pos.{}.tif'.format(str(cell_index))
                    cell_index += 1

                else:
                    raise

                annotations.append({
                    'cell': filename,
                    'path': sCI.path,
                    'imgFile': sCI.imgFile,
                    'markerFile': sCI.markerFile,
                    'cellIndex': i,
                    'centroid': list(sCI.centroids[i, :]),
                    'classification': int(sCI.centroids_classification[i])
                })
                if export:
                    sCI.saveCellImg(i, os.path.join(os.path.join(root, 'keras'), filename))

        self.console.insert("end", f"\nTotal cells: {cell_index}")
        # self.console.insert("end", f"\nNew Total O4+: {o4pos_cell_index}. New Total O4-: {o4neg_cell_index}.")

        filename = fullPath(root, 'annotations.json')
        with open(filename, 'w') as f:
            json.dump(annotations, f)

        """ 
        Disabled for now. 
         if export:
            export cell_indices.json file
            filename = os.path.join(os.path.join(settings.kerasFolder, 'output'), 'cell_indices.json')
            indices = {
                'o4pos_cell_index': o4pos_cell_index,
                'o4neg_cell_index': o4neg_cell_index
            }
            with open(filename, 'w') as f:
                json.dump(indices, f)
        """

        self.console.insert("end", "\nAll Done.")


# Starts application.
root = tk.Tk()
root.geometry('+100+100')
root.resizable(width=False, height=False)
app = Application(master=root)
app.mainloop()
javabridge.kill_vm()
