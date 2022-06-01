import cv2 as cv
import os
import json

import tkinter as tk
import tkinter.font as tkFont
import tkinter.filedialog as fileDialog
from tkinter.ttk import Progressbar 

from matplotlib.backends.backend_pdf import PdfPages

# start JVM for compatibility with VSI files
#print('JAVA_HOME =', os.environ['JAVA_HOME'])
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

from settings import Settings
from commonFunctions import *
from singleCompositeImage import singleCompositeImage

settings = Settings()

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
        self.create_widgets()

    def create_widgets(self):
        """Creates widgets on initial window."""
        l1 = tk.Label(self.top_frame,
            text = """1. Select folder to proccess:""",
            font = tkFont.Font(family="Calibri", size=12))

 
        l2 = tk.Label(self.top_frame,
            text = """2. Define pattern for image files:""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l3 = tk.Label(self.top_frame,
            text = """Choose slice for each antigen label, use -1 to indicate antigen not present.""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l4 = tk.Label(self.top_frame,
            text = """3. Which image slice contains the DAPI image?""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l5 = tk.Label(self.top_frame,
            text = """4. DAPI gamma = """,
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l6 = tk.Label(self.top_frame,
            text = """5. Which image slice contains the O4 image?""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l7 = tk.Label(self.top_frame,
            text = """6. O4 gamma = """,
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l8 = tk.Label(self.top_frame,
            text = """7. Which image slice contains the EdU image?""",
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l9 = tk.Label(self.top_frame,
            text = """8. EdU gamma = """,
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l10 = tk.Label(self.top_frame,
            text = """9. Scalefactor = """,
            justify = tk.LEFT,
            anchor = 'w',
            font = tkFont.Font(family="Calibri", size=12))

        l11 = tk.Label(self.top_frame,
            text = """Enable debug?""",
            anchor = 'e',
            font = tkFont.Font(family="Calibri", size=12))

        l1.grid(row = 0, column = 0, sticky = 'w', pady = 2)
        l2.grid(row = 1, column = 0, sticky = 'w', pady = 2)
        l3.grid(row = 2, column = 0, sticky = 'w', pady = 2)
        l4.grid(row = 3, column = 0, sticky = 'w', pady = 2)
        l5.grid(row = 3, column = 2, sticky = 'w', pady = 2)
        l6.grid(row = 4, column = 0, sticky = 'w', pady = 2)
        l7.grid(row = 4, column = 2, sticky = 'w', pady = 2)
        l8.grid(row = 5, column = 0, sticky = 'w', pady = 2)
        l9.grid(row = 5, column = 2, sticky = 'w', pady = 2)
        l10.grid(row = 6, column = 0, sticky = 'w', pady = 2)
        l11.grid(row = 6, column = 2, sticky = 'w', pady = 2)

        e1 = tk.Frame(self.top_frame)
        
        tk.Button(e1,
            text="Browse",                          
            command=lambda: self.select_folder(),
            font = tkFont.Font(family="Calibri", size=12)).pack(side=tk.RIGHT)

        self.root = tk.StringVar()
        self.root.set(settings.defaults["root"])
        tk.Entry(e1, width=80, textvariable = self.root,
            font = tkFont.Font(family="Calibri", size=12)).pack(side=tk.LEFT)

        # set file pattern
        pattern = tk.StringVar()
        pattern.set(settings.defaults["pattern"])
        e2 = tk.Entry(self.top_frame, width=20, textvariable = pattern,
                          font = tkFont.Font(family="Calibri", size=12))

        # set DAPI channel
        dapi_ch = tk.IntVar()
        dapi_ch.set(settings.defaults["dapi_ch"])
        e3 = tk.Entry(self.top_frame, width=20, textvariable=dapi_ch,
            font = tkFont.Font(family="Calibri", size=12))

        # set O4 gamma
        dapi_gamma = tk.DoubleVar()
        dapi_gamma.set(settings.defaults["dapi_gamma"])
        e4 = tk.Entry(self.top_frame, width=20, textvariable=dapi_gamma,
            font = tkFont.Font(family="Calibri", size=12))

        # set O4 channel
        o4_ch = tk.IntVar()
        o4_ch.set(settings.defaults["o4_ch"])
        e5 = tk.Entry(self.top_frame, width=20, textvariable=o4_ch,
            font = tkFont.Font(family="Calibri", size=12))

        # set O4 gamma
        o4_gamma = tk.DoubleVar()
        o4_gamma.set(settings.defaults["o4_gamma"])
        e6 = tk.Entry(self.top_frame, width=20, textvariable=o4_gamma,
            font = tkFont.Font(family="Calibri", size=12))

        # set EdU channel
        edu_ch = tk.IntVar()
        edu_ch.set(settings.defaults["edu_ch"])
        e7 = tk.Entry(self.top_frame, width=20, textvariable=edu_ch,
            font = tkFont.Font(family="Calibri", size=12))

        # set EdU gamma
        edu_gamma = tk.DoubleVar()
        edu_gamma.set(settings.defaults["edu_gamma"])
        e8 = tk.Entry(self.top_frame, width=20, textvariable=edu_gamma,
            font = tkFont.Font(family="Calibri", size=12))

        # set scalefactor
        scalefactor = tk.DoubleVar()
        scalefactor.set(settings.defaults["scalefactor"])
        e9 = tk.Entry(self.top_frame, width=20, textvariable=scalefactor,
            font = tkFont.Font(family="Calibri", size=12))

        # debug mode?
        debug = tk.BooleanVar()
        debug.set(settings.defaults["debug"])
        e10 = tk.Checkbutton(self.top_frame, text='', variable=debug,
            onvalue=True, offvalue=False,
            anchor='w')

        e1.grid(row = 0, column = 1, columnspan = 3, sticky = 'w', pady = 2)
        e2.grid(row = 1, column = 1, columnspan = 3, sticky = 'w', pady = 2)
        e3.grid(row = 3, column = 1, sticky = 'w', pady = 2)
        e4.grid(row = 3, column = 3, sticky = 'w', pady = 2)
        e5.grid(row = 4, column = 1, sticky = 'w', pady = 2)
        e6.grid(row = 4, column = 3, sticky = 'w', pady = 2)
        e7.grid(row = 5, column = 1, sticky = 'w', pady = 2)
        e8.grid(row = 5, column = 3, sticky = 'w', pady = 2)
        e9.grid(row = 6, column = 1, sticky = 'w', pady = 2)
        e10.grid(row = 6, column = 3, columnspan = 3, sticky = 'w', pady = 2)

        # start button
        button2 = tk.Button(self.bottom_frame,
            text="Start",
            command=lambda: self.start_analysis(root = self.root.get(), 
                pattern = pattern.get(), 
                dapi_ch = dapi_ch.get(), 
                dapi_gamma = dapi_gamma.get(),
                o4_ch = o4_ch.get(), 
                o4_gamma = o4_gamma.get(),
                edu_ch = edu_ch.get(),
                edu_gamma = edu_gamma.get(),
                scalefactor = scalefactor.get(),
                debug = debug.get()),
            font = tkFont.Font(family="Calibri", size=12))
        button2.pack(side="top")

        # define output console area
        tk.Label(self.bottom_frame,
            text = """Output""",
            justify = tk.LEFT,
            anchor = 'n',
            font = tkFont.Font(family="Calibri", size=12)).pack(side='left', fill='y')
        # add scroll bar
        self.console_scrollbar = tk.Scrollbar(self.bottom_frame, orient=tk.VERTICAL)
        self.console_scrollbar.pack(side="right",fill='y')
        # add text console widget
        self.console = tk.Text(self.bottom_frame,
                        yscrollcommand = self.console_scrollbar,
                        font = tkFont.Font(family="Calibri", size=12))
        self.console.pack(side="top", fill="both", expand = True)
        self.console.bind("<Key>", lambda e: "break")
        # assign scroll bar to console yview
        self.console_scrollbar.config(command=self.console.yview)
        # add progress bar
        self.progress = Progressbar(self.bottom_frame,
            length=200, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side="bottom", fill="y", expand="True")

    def select_folder(self):
        import os
        self.root.set(os.path.abspath(fileDialog.askdirectory(title='Select source folder containing image files')))
        
    def start_analysis(
        self,
        root:str, 
        pattern:str, 
        dapi_ch:int, 
        o4_ch:int = -1, 
        edu_ch:int = -1, 
        dapi_gamma:float = 1.0, 
        o4_gamma:float = 1.0, 
        edu_gamma:float = 1.0, 
        scalefactor:float = 1.0, 
        debug: bool = False):

        # clear console
        self.console.delete(1.0,tk.END)

        # save settings
        settings.updateDefaults(
            root, 
            pattern, 
            dapi_ch, 
            o4_ch, 
            edu_ch, 
            dapi_gamma, 
            o4_gamma, 
            edu_gamma, 
            scalefactor,
            debug)

        # set o4_ch and edu_ch to none if -1
        if (o4_ch == -1):
            o4_ch = None
            self.console.insert("end", "\nSkipping O4 channel & analysis")

        if (edu_ch == -1):
            edu_ch = None
            self.console.insert("end", "\nSkipping EdU channel & analysis")

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

            # select first two files to do manual count comparisons
            self.console.insert("end","\ndebug: selecting first two files to do manual count comparisons") 
            files = list(files[i] for i in range(0,2)) 

        results = []

        if (o4_ch != None):
            model = loadKerasModel(settings.kerasModel)

        # rest progress bar
        self.progress["value"]=0
        self.progress.update()
        # calculate increment for progress bar
        fileNumber = len(files)
        currentFileNumber = 0
        i = 100/fileNumber

        with PdfPages(fullPath(root, 'results_nucleiCounter.pdf')) as export_pdf:

            for file in files:

                # increment currentFileNumber
                currentFileNumber += 1

                # increment progress bar
                self.progress["value"] += i
                self.progress.update()

                path = file['path'] 
                imgFile = file['name']

                # parse file names
                try: 
                    stage, well, position = parseFileName(imgFile)
                except:
                    print(f"\nCould not parseFileName '{path}'. Image: {imgFile}")                    
                    stage = None

                try:
                    sCI = singleCompositeImage(
                        path = path, 
                        imgFile = imgFile, 
                        dapi_ch = dapi_ch, 
                        dapi_gamma = dapi_gamma, 
                        o4_ch = o4_ch, 
                        o4_gamma = o4_gamma,
                        EdU_ch = edu_ch,
                        EdU_gamma = edu_gamma,
                        scalefactor = scalefactor, 
                        debug = debug)
                    sCI.processDAPI(threshold_method='th2') # based on manual counts (see OneNote)

                    if (o4_ch != None):
                        sCI.processCells()
                        sCI.getPredictions(model)
                        sCI.processPredictions(export_pdf, debug = debug)

                    if (edu_ch != None):
                        sCI.countEdUchannel(export_pdf)

                    if debug:
                        sCI.reportResults()
                        self.console.insert('end', f"\nimgFile: {sCI.imgFile} found {sCI.nucleiCount} DAPI+ nuclei.")

                        if (o4_ch != None):
                            self.console.insert('end', f" O4 pos: {sCI.o4pos_count}.")
       
                        if (edu_ch != None):
                            self.console.insert('end', f" EdU pos: {sCI.edupos_count}.")

                        self.console.update()                 

                    # report result of nuclei count
                    result = {
                            'path': sCI.path,
                            'imgFile': sCI.imgFile,
                            'nucleiCount': sCI.nucleiCount}

                    # add details parsed from fileName
                    if (stage != None):
                        result['stage'] = stage
                        result['well'] = well
                        result['position'] = position

                    # add O4 counts
                    if (o4_ch != None):
                        if (sCI.o4pos_count+sCI.o4neg_count)>0:
                            o4_percentage = sCI.o4pos_count/(sCI.o4pos_count+sCI.o4neg_count)
                        else:
                            o4_percentage = 0
                            self.console.insert('end',f"Error calculating O4% in {sCI.imgFile}.")

                        result['o4pos_count'] = sCI.o4pos_count
                        result['o4neg_count'] = sCI.o4neg_count
                        result['o4%'] = "{:.2%}".format(o4_percentage)

                    # add EdU counts
                    if (edu_ch != None):
                        result['edupos_count'] = sCI.edupos_count

                    results.append(result)

                    self.console.insert("end",f"\nCompleted '{imgFile}'. {currentFileNumber} of {fileNumber} files.")
                except:
                    self.console.insert("end",f"\nFailed on path '{path}'. Image: {imgFile}")                    
                    raise

                self.console.update()
                self.console.yview("end")

        # output results as csv
        import csv
        filename = fullPath(root, 'results_nucleiCounter.csv')
        with open(filename,'w',newline='') as f:
            w = csv.DictWriter(f, results[0].keys())
            w.writeheader()
            w.writerows(results)

        self.console.insert("end",f'\nResults saved to {filename}.')
        self.console.insert("end",'\nAll Done')



# Starts application.
root = tk.Tk()
root.geometry('+100+100')
root.resizable(width=False, height=False)
app = Application(master=root)
app.mainloop()        
javabridge.kill_vm()