import sys

# import tkinter as tk
from mttkinter import mtTkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import tkinter.filedialog as fileDialog
from tkinter.ttk import Progressbar

from matplotlib.backends.backend_pdf import PdfPages

# from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue, Empty

from settings import Settings
from commonFunctions import *
from singleCompositeImage_imagej import singleCompositeImage

# start JVM for compatibility with VSI files
# print('JAVA_HOME =', os.environ['JAVA_HOME'])
import javabridge
import bioformats
# javabridge.start_vm(class_path=bioformats.JARS)

# to enable VSI file support using FIJI
import imagej

settings = Settings()

class TextRedirector:
    """ This class allows capturing of stdout into console in GUI """
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insert('end', text)
        self.text_widget.see('end')
        self.text_widget.update()

    def flush(self):
        pass

class Application(tk.Frame):
    def __init__(self, master=None):
        """Setup tk application."""
        super().__init__(master)
        self.name = None
        self.progress = None
        self.console = None
        self.console_scrollbar = None
        self.root = None
        self.master = master
        self.master.title("Sim Lab: In vitro analysis")
        self.pack()
        self.main_container = tk.Frame(master)
        self.main_container.pack(side="top", fill="both", expand=True)
        self.top_frame = tk.Frame(self.main_container)
        self.bottom_frame = tk.Frame(self.main_container, background="grey")
        self.top_frame.pack(side="top", fill="x", expand=False)
        self.bottom_frame.pack(side="bottom", fill="both", expand=True)
        self.create_widgets()

        self.queue = Queue()

    def create_widgets(self):
        """Creates widgets on initial window."""
        n1 = tk.Label(self.top_frame,
                     text="""Experiment Name""",
                     font=tkFont.Font(family="Calibri", size=14))

        l1 = tk.Label(self.top_frame,
                      text="""1. Select folder to process:""",
                      font=tkFont.Font(family="Calibri", size=12))

        l2 = tk.Label(self.top_frame,
                      text="""2. Define pattern for image files:""",
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l3 = tk.Label(self.top_frame,
                      text="""Choose slice for each antigen label, use -1 to indicate antigen not present.""",
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l4 = tk.Label(self.top_frame,
                      text="""3. Which image slice contains the DAPI image?""",
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l5 = tk.Label(self.top_frame,
                      text="""4. DAPI gamma = """,
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l6 = tk.Label(self.top_frame,
                      text="""5. Which image slice contains the O4 image?""",
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l7 = tk.Label(self.top_frame,
                      text="""6. O4 gamma = """,
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l8 = tk.Label(self.top_frame,
                      text="""7. Which image slice contains the EdU (or other nuclear localized) image?""",
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l9 = tk.Label(self.top_frame,
                      text="""8. EdU gamma = """,
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        l10 = tk.Label(self.top_frame,
                       text="""9. Which image slice contains the Gfap image?""",
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(family="Calibri", size=12))

        l11 = tk.Label(self.top_frame,
                       text="""10. Gfap threshold = """,
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(family="Calibri", size=12))

        l12 = tk.Label(self.top_frame,
                       text="""11. Scalefactor (1.0 is equivalent to 1.5385 pixels per 1.0 μm) = """,
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(family="Calibri", size=12))

        l13 = tk.Label(self.top_frame,
                       text="""Enable debug?""",
                       anchor='e',
                       font=tkFont.Font(family="Calibri", size=12))

        prediction_cutoff_label = tk.Label(self.top_frame,
                       text="""O4 Keras Model Prediction cutoff:""",
                       anchor='e',
                       font=tkFont.Font(family="Calibri", size=12))

        n1.grid(row=0, column=0, sticky='w', pady=2)
        l1.grid(row=1, column=0, sticky='w', pady=2)
        l2.grid(row=2, column=0, sticky='w', pady=2)
        l3.grid(row=3, column=0, sticky='w', pady=2)
        l4.grid(row=4, column=0, sticky='w', pady=2)
        l5.grid(row=4, column=2, sticky='w', pady=2)
        l6.grid(row=5, column=0, sticky='w', pady=2)
        l7.grid(row=5, column=2, sticky='w', pady=2)
        l8.grid(row=6, column=0, sticky='w', pady=2)
        l9.grid(row=6, column=2, sticky='w', pady=2)
        l10.grid(row=7, column=0, sticky='w', pady=2)
        l11.grid(row=7, column=2, sticky='w', pady=2)
        l12.grid(row=8, column=0, sticky='w', pady=2)
        l13.grid(row=8, column=2, sticky='w', pady=2)
        prediction_cutoff_label.grid(row=9, column=0, sticky='w', pady=2)

        """Now for the text entry and other boxes"""
        def selection_changed(event):
            selected_value = combo.get()

            self.name.set(selected_value)
            self.root.set(settings.experiments[selected_value]['root'])
            pattern.set(settings.experiments[selected_value]['pattern'])
            dapi_ch.set(settings.experiments[selected_value].get('dapi_ch', 0))
            dapi_gamma.set(settings.experiments[selected_value].get('dapi_gamma', 1.0))
            o4_ch.set(settings.experiments[selected_value].get('o4_ch', -1))
            o4_gamma.set(settings.experiments[selected_value].get('o4_gamma', 1.0))
            edu_ch.set(settings.experiments[selected_value].get('edu_ch', -1))
            edu_gamma.set(settings.experiments[selected_value].get('edu_gamma', 1.0))
            gfap_ch.set(settings.experiments[selected_value].get('gfap_ch', -1))
            gfap_th.set(settings.experiments[selected_value].get('gfap_th', 1000))
            scalefactor.set(settings.experiments[selected_value].get('scalefactor', 1))
            debug.set(settings.experiments[selected_value].get('debug', 0))
            print("Selected:", selected_value)

        self.name = tk.StringVar()
        experiments = list(settings.experiments)
        print(experiments)
        self.name.set(settings.defaults["name"])
        combo = ttk.Combobox(self.top_frame, values=experiments,
                         width=80, textvariable=self.name,
                         font=tkFont.Font(family="Calibri", size=14))
        combo.bind("<<ComboboxSelected>>", selection_changed)

        e1 = tk.Frame(self.top_frame)
        tk.Button(e1,
                  text="Browse",
                  command=lambda: self.select_folder(),
                  font=tkFont.Font(family="Calibri", size=12)).pack(side=tk.RIGHT)

        self.root = tk.StringVar()
        self.root.set(settings.defaults["root"])
        tk.Entry(e1, width=80, textvariable=self.root,
                 font=tkFont.Font(family="Calibri", size=12)).pack(side=tk.LEFT)

        # set file pattern
        pattern = tk.StringVar()
        pattern.set(settings.defaults["pattern"])
        e2 = tk.Entry(self.top_frame, width=20, textvariable=pattern,
                      font=tkFont.Font(family="Calibri", size=12))

        # set DAPI channel
        dapi_ch = tk.IntVar()
        dapi_ch.set(settings.defaults["dapi_ch"])
        e3 = tk.Entry(self.top_frame, width=20, textvariable=dapi_ch,
                      font=tkFont.Font(family="Calibri", size=12))

        # set O4 gamma
        dapi_gamma = tk.DoubleVar()
        dapi_gamma.set(settings.defaults["dapi_gamma"])
        e4 = tk.Entry(self.top_frame, width=20, textvariable=dapi_gamma,
                      font=tkFont.Font(family="Calibri", size=12))

        # set O4 channel
        o4_ch = tk.IntVar()
        o4_ch.set(settings.defaults["o4_ch"])
        e5 = tk.Entry(self.top_frame, width=20, textvariable=o4_ch,
                      font=tkFont.Font(family="Calibri", size=12))

        # set O4 gamma
        o4_gamma = tk.DoubleVar()
        o4_gamma.set(settings.defaults["o4_gamma"])
        e6 = tk.Entry(self.top_frame, width=20, textvariable=o4_gamma,
                      font=tkFont.Font(family="Calibri", size=12))

        # set EdU channel
        edu_ch = tk.IntVar()
        edu_ch.set(settings.defaults["edu_ch"])
        e7 = tk.Entry(self.top_frame, width=20, textvariable=edu_ch,
                      font=tkFont.Font(family="Calibri", size=12))

        # set EdU gamma
        edu_gamma = tk.DoubleVar()
        edu_gamma.set(settings.defaults["edu_gamma"])
        e8 = tk.Entry(self.top_frame, width=20, textvariable=edu_gamma,
                      font=tkFont.Font(family="Calibri", size=12))

        # set Gfap channel
        gfap_ch = tk.IntVar()
        gfap_ch.set(settings.defaults["gfap_ch"])
        e9 = tk.Entry(self.top_frame, width=20, textvariable=gfap_ch,
                      font=tkFont.Font(family="Calibri", size=12))

        # set Gfap threshold
        gfap_th = tk.IntVar()
        gfap_th.set(settings.defaults["gfap_th"])
        e10 = tk.Entry(self.top_frame, width=20, textvariable=gfap_th,
                       font=tkFont.Font(family="Calibri", size=12))

        # set scalefactor
        scalefactor = tk.DoubleVar()
        scalefactor.set(settings.defaults["scalefactor"])
        e11 = tk.Entry(self.top_frame, width=20, textvariable=scalefactor,
                       font=tkFont.Font(family="Calibri", size=12))

        # debug mode?
        debug = tk.BooleanVar()
        debug.set(settings.defaults["debug"])
        e12 = tk.Checkbutton(self.top_frame, text='', variable=debug,
                             onvalue=True, offvalue=False,
                             anchor='w')

        # prediction_cutoff
        prediction_cutoff = tk.DoubleVar()
        if 'prediction_cutoff' in settings.defaults:
            prediction_cutoff.set(settings.defaults['prediction_cutoff'])
        else:
            prediction_cutoff.set(0.5)
        prediction_cutoff_entry = tk.Entry(self.top_frame, width=20,
                                           textvariable=prediction_cutoff,
                                           font=tkFont.Font(family="Calibri", size=12))

        combo.grid(row=0, column=1, columnspan=3, sticky='w', pady=2)
        e1.grid(row=1, column=1, columnspan=3, sticky='w', pady=2)
        e2.grid(row=2, column=1, columnspan=3, sticky='w', pady=2)
        e3.grid(row=4, column=1, sticky='w', pady=2)
        e4.grid(row=4, column=3, sticky='w', pady=2)
        e5.grid(row=5, column=1, sticky='w', pady=2)
        e6.grid(row=5, column=3, sticky='w', pady=2)
        e7.grid(row=6, column=1, sticky='w', pady=2)
        e8.grid(row=6, column=3, sticky='w', pady=2)
        e9.grid(row=7, column=1, sticky='w', pady=2)
        e10.grid(row=7, column=3, sticky='w', pady=2)
        e11.grid(row=8, column=1, sticky='w', pady=2)
        e12.grid(row=8, column=3, columnspan=3, sticky='w', pady=2)
        prediction_cutoff_entry.grid(row=9, column=1, sticky='w', pady=2)

        # start button
        button2 = tk.Button(self.bottom_frame,
                            text="Start",
                            command=lambda: self.start_analysis(name = self.name.get(),
                                                                folder_root=self.root.get(),
                                                                pattern=pattern.get(),
                                                                dapi_ch=dapi_ch.get(),
                                                                dapi_gamma=dapi_gamma.get(),
                                                                o4_ch=o4_ch.get(),
                                                                o4_gamma=o4_gamma.get(),
                                                                edu_ch=edu_ch.get(),
                                                                edu_gamma=edu_gamma.get(),
                                                                gfap_ch=gfap_ch.get(),
                                                                gfap_th=gfap_th.get(),
                                                                scalefactor=scalefactor.get(),
                                                                prediction_cutoff=prediction_cutoff.get(),
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

        text_redirector = TextRedirector(self.console)
        sys.stdout = text_redirector

    def select_folder(self):
        import os
        self.root.set(os.path.abspath(fileDialog.askdirectory(title='Select source folder containing image files')))

    def start_analysis(self,
                       name: str,
                       folder_root: str,
                       pattern: str,
                       dapi_ch: int,
                       o4_ch: int = -1,
                       edu_ch: int = -1,
                       gfap_ch: int = -1,
                       dapi_gamma: float = 1.0,
                       o4_gamma: float = 1.0,
                       edu_gamma: float = 1.0,
                       gfap_th: int = 1000,
                       scalefactor: float = 1.0,
                       prediction_cutoff: float = 0.5,
                       debug: bool = False):

        # self.console.delete(1.0, tk.END)
        print("Starting analysis")

        # save settings
        settings.updateDefaults(
            name,
            folder_root,
            pattern,
            dapi_ch,
            o4_ch,
            edu_ch,
            gfap_ch,
            dapi_gamma,
            o4_gamma,
            edu_gamma,
            gfap_th,
            scalefactor,
            debug)

        if name == "temp":
            print("Temporary analysis. experiments.json will not be updated")
        else:
            settings.saveExperimentalParameters()
            print("Updating list of experiment parameters")

        print(f"O4 channel: {o4_ch}")

        # set o4_ch and edu_ch to none if -1
        if o4_ch == -1:
            o4_ch = None
            print("Skipping O4 channel & analysis")

        if edu_ch == -1:
            edu_ch = None
            print("Skipping EdU channel & analysis")

        if gfap_ch == -1:
            gfap_ch = None
            print("Skipping Gfap channel & analysis")

        # queue.put("Hello")

        # start analysis
        files = find(pattern, folder_root, excluded_subfolder='keras')

        if len(files) == 0:
            print(f"No files found in '{folder_root}'. Check the input.")
            queue.put('done')
            return
        else:
            print(f"\nFound {len(files)} matching '{pattern}' in '{folder_root}'")
        print("***************************")
        print("Starting to analyze images")

        # select file sample
        if debug:
            # select five files at random
            # files = list(files[i] for i in random.sample(list(range(len(files))), 5))

            # select first two files to do manual count comparisons
            print("debug: selecting first two files to do manual count comparisons")
            files = list(files[i] for i in range(0, 2))

        model = None
        if o4_ch is not None:
            if os.path.exists(os.path.join(folder_root, settings.kerasModel)):
                print("Using experiment specific model.")
                model = loadKerasModel(os.path.join(folder_root, settings.kerasModel))
            else:
                print("Using old general model.")
                model = loadKerasModel('o4counter_wAug_5.1.h5')

        results = []

        fileNumber = len(files)
        currentFileNumber = 0
        i = 100 / fileNumber

        with PdfPages(fullPath(folder_root, 'results_nucleiCounter.pdf')) as export_pdf:

            for file in files:

                # increment currentFileNumber
                currentFileNumber += 1

                # increment progress bar
                # self.progress["value"] += i
                # self.progress.update()

                path = file['path']
                imgFile = file['name']

                # parse file names
                try:
                    stage, well, position = parseFileName(imgFile)
                except Exception:
                    print(f"Could not parseFileName '{path}'. Image: {imgFile}")
                    print(Exception)
                    stage = None


                args_dict = {
                    'queue': self.queue,
                    'path': path,
                    'imgFile': imgFile,
                    'dapi_ch': dapi_ch,
                    'dapi_gamma': dapi_gamma,
                    'o4_ch': o4_ch,
                    'o4_gamma': o4_gamma,
                    'edu_ch': edu_ch,
                    'edu_gamma': edu_gamma,
                    'gfap_ch': gfap_ch,
                    'gfap_th': gfap_th,
                    'scalefactor': scalefactor,
                    'prediction_cutoff': prediction_cutoff,
                    'debug': debug}

                """ Process image """
                Thread(target=analysis, args=(args_dict, model), daemon=True).start()
                results.append(self.queue.get(block=True))

        # output results as csv
        import csv
        filename = fullPath(folder_root, 'results_nucleiCounter.csv')
        with open(filename, 'w', newline='') as f:
            w = csv.DictWriter(f, results[0].keys())
            w.writeheader()
            w.writerows(results)

        print(f'Results saved to {filename}.')
        print('All Done')

        # self.update_progress()

    def update_progress(self):
        try:
            data = self.queue.get(block=False)
        except Empty:
            pass
        else:
            if data == 'done':
                print("All Done")
                return
            print(data)
            # self.console.update()
            # self.console.yview("end")
        finally:
            self.console.update()
            self.after(100, self.update_progress)


def analysis(args_dict, model):

    queue = args_dict['queue']
    path = str(args_dict['path'])
    imgFile = str(args_dict['imgFile'])
    dapi_ch = int(args_dict['dapi_ch'])
    o4_ch = args_dict['o4_ch']
    edu_ch = args_dict['edu_ch']
    gfap_ch = args_dict['gfap_ch']
    dapi_gamma = float(args_dict['dapi_gamma'])
    o4_gamma = float(args_dict['o4_gamma'])
    edu_gamma = float(args_dict['edu_gamma'])
    gfap_th = int(args_dict['gfap_th'])
    scalefactor = float(args_dict['scalefactor'])
    prediction_cutoff = float(args_dict['prediction_cutoff'])
    debug = bool(args_dict['debug'])

    # start imagej for image loading

    try:
        ij = imagej.init('sc.fiji:fiji', mode='headless')

        sCI = singleCompositeImage(
            ij=ij,
            path=path,
            imgFile=imgFile,
            dapi_ch=dapi_ch,
            dapi_gamma=dapi_gamma,
            o4_ch=o4_ch,
            o4_gamma=o4_gamma,
            EdU_ch=edu_ch,
            EdU_gamma=edu_gamma,
            gfap_ch=gfap_ch,
            gfap_th=gfap_th,
            scalefactor=scalefactor,
            debug=debug)
        sCI.processDAPI(threshold_method='th2')  # based on manual counts (see OneNote)

        if o4_ch is not None:
            sCI.processCells()
            sCI.getPredictions(model)
            # sCI.processPredictions(export_pdf, prediction_cutoff=prediction_cutoff, debug=debug)

        # if edu_ch is not None:
            # sCI.countEdUchannel(export_pdf)

        # if gfap_ch is not None:
            # sCI.countGfapchannel(export_pdf)

        if debug:
            sCI.reportResults()
            print(f"imgFile: {sCI.imgFile} found {sCI.nucleiCount} DAPI+ nuclei.")

            if o4_ch is not None:
                print(f"O4 pos: {sCI.o4pos_count}.")

            if edu_ch is not None:
                print(f"EdU pos: {sCI.edupos_count}.")

            if gfap_ch is not None:
                print(f"Gfap pos: {sCI.gfappos_count}.")
            # self.console.update()

        # report result of nuclei count
        result = {
            'path': sCI.path,
            'imgFile': sCI.imgFile,
            'nucleiCount': sCI.nucleiCount}

        # add details parsed from fileName
        if stage is not None:
            result['stage'] = stage
            result['well'] = well
            result['position'] = position

        # add O4 counts
        if o4_ch is not None:
            if (sCI.o4pos_count + sCI.o4neg_count) > 0:
                o4_percentage = sCI.o4pos_count / (sCI.o4pos_count + sCI.o4neg_count)
            else:
                o4_percentage = 0
                print(f"Error calculating O4% in {sCI.imgFile}.")

            result['o4pos_count'] = sCI.o4pos_count
            result['o4neg_count'] = sCI.o4neg_count
            result['o4%'] = "{:.2%}".format(o4_percentage)

        # add EdU counts
        if edu_ch is not None:
            result['edupos_count'] = sCI.edupos_count

        # add Gfap counts
        if gfap_ch is not None:
            result['gfappos_count'] = sCI.gfappos_count

        queue.put(result)
        print(f"Completed '{imgFile}'. {currentFileNumber} of {fileNumber} files.")

    except Exception:
        print(f"Failed on path '{path}'. Image: {imgFile}")
        raise


# Starts application.
root = tk.Tk()
root.geometry('+100+100')
root.resizable(width=False, height=False)
app = Application(master=root)
app.mainloop()
javabridge.kill_vm()
