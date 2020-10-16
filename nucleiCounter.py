import cv2 as cv
import os
import json
import tkinter as tk
import tkinter.font as tkFont
import tkinter.filedialog as fileDialog
from tkinter.ttk import Progressbar 

from commonFunctions import *
#from settings import Settings
from singleCompositeImage import singleCompositeImage

class Application(tk.Frame):
    def __init__(self, master=None):
        """Setup tk application."""
        super().__init__(master)
        self.master = master
        self.master.title("Nuclei counter")
        self.pack()
        self.main_container = tk.Frame(master)
        self.main_container.pack(side = "top", fill = "both", expand = True)
        self.top_frame = tk.Frame(self.main_container)
        self.bottom_frame = tk.Frame(self.main_container, background = "grey")
        self.top_frame.pack(side="top", fill="x", expand = False)
        self.bottom_frame.pack(side="bottom", fill="both", expand = True)
        self.top_left = tk.Frame(self.top_frame, background = "pink")
        self.top_right = tk.Frame(self.top_frame, background = "blue")
        self.top_left.pack(side = "left", fill = "x", expand = True)
        self.top_right.pack(side = "right", fill = "x", expand = True)
        self.top_right_upper = tk.Frame(self.top_right)
        self.top_right_upper.pack(side = "top", fill = "x", expand = True)
        self.create_widgets()

    def create_widgets(self):
        """Creates widgets on initial window."""
        tk.Label(self.top_left,
            text = """1. Select folder to proccess:""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12)).pack(fill='x')
 
        tk.Label(self.top_left,
            text = """2. Define pattern for image files:""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12)).pack(fill='x')

        tk.Label(self.top_left,
            text = """3. Which image slice contains the DAPI image?""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12)).pack(fill='x')

        button = tk.Button(self.top_right_upper,
                          text="Browse",                          
                          command=lambda: self.select_folder(),
                          font = tkFont.Font(family="Calibri", size=12))
        button.pack(side=tk.RIGHT)

        self.root = tk.StringVar()
        tk.Entry(self.top_right_upper, width=80, textvariable = self.root,
            font = tkFont.Font(family="Calibri", size=12)).pack(side='left')


        pattern = tk.StringVar()
        pattern.set("*.tif")

        tk.Entry(self.top_right, width=20, textvariable = pattern,
                          font = tkFont.Font(family="Calibri", size=12)).pack(fill = 'x')



        dapi_ch = tk.IntVar()
        dapi_ch.set(0)
        tk.Entry(self.top_right, width=20, textvariable=dapi_ch,
                          font = tkFont.Font(family="Calibri", size=12)).pack(fill = 'x')

        o4_ch = None
        marker_index = None

        button2 = tk.Button(self.bottom_frame,
                          text="Start",
                          command=lambda: self.start_analysis(self.root.get(), 
                                                            pattern.get(), 
                                                            dapi_ch.get(), 
                                                            o4_ch, marker_index),
                          font = tkFont.Font(family="Calibri", size=12))
        button2.pack(side="top")

        tk.Label(self.bottom_frame,
            text = """Output""",
            justify = tk.LEFT,
            font = tkFont.Font(family="Calibri", size=12)).pack(side='left')

        self.console = tk.Text(self.bottom_frame,
                          font = tkFont.Font(family="Calibri", size=12))
        self.console.pack(side="top", fill="both", expand = True)



    def select_folder(self):
        import os
        self.root.set(os.path.abspath(fileDialog.askdirectory(title='Select source folder containing image files')))
        
    def start_analysis(self, root:str, pattern:str, dapi_ch:int, o4_ch:int, marker_index:int, debug: bool = False):

        # start analysis
        files = find(pattern, root)

        if (len(files)==0):
            self.console.insert("end",f"No files found in '{root}'. Check the input.")
            return
        else:
            self.console.insert("end",f"\nFound {len(files)} matching '{pattern}' in '{root}'")
        self.console.insert("end","\n***************************")
        self.console.insert("end","\nStarting to analyze images")
        self.console.update()

        # select file sample
        if debug:
            # select five files at random
            #files = list(files[i] for i in random.sample(list(range(len(files))), 5))

            # select five files to do manual count comparisons
            files = list(files[i] for i in range(7,12)) 

        results = []

        self.progress = Progressbar(self.bottom_frame,
            length=200, cursor='spider',orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side="bottom", fill="y", expand="True")

        i = 100/len(files)
        for file in files:
            self.progress["value"] += i
            self.progress.update()

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
                sCI = singleCompositeImage(path, imgFile, dapi_ch, o4_ch, scalefactor=1, debug=False)
                sCI.processDAPI(threshold_method='th2') # based on manual counts (see OneNote)
                if debug:
                    sCI.reportResults()
                
                #self.console.insert('end', f"\nimgFile: {sCI.imgFile} found {sCI.nucleiCount} DAPI+ nuclei.")
                #self.console.update()    

                results.append({
                    'path': sCI.path,
                    'imgFile': sCI.imgFile,
                    'stage': stage,
                    'well': well,
                    'position': position,
                    'nucleiCount': sCI.nucleiCount
                    })
            except:
                self.console.insert("end",f"\nFailed on path '{path}'. Image: {imgFile}")
                raise

        # output results as csv
        import csv
        filename = fullPath(root, 'results_nucleiCounter.csv')
        with open(filename,'w',newline='') as f:
            w = csv.DictWriter(f, results[0].keys())
            w.writeheader()
            w.writerows(results)

        self.console.insert("end",'\nAll Done')

# Starts application.
root = tk.Tk()
app = Application(master=root)
app.mainloop()        