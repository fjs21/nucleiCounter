import sys
import tkinter as tk

# tkmacosx attempts to fix problems with unresponsive UI on macOS
import platform

if platform.system() == "Darwin":  # Check if the system is macOS
    print('Importing tkmacosx')
    import tkmacosx

from tkinter import ttk
import tkinter.font as tkFont
import tkinter.filedialog as fileDialog
from tkinter.ttk import Progressbar

from matplotlib.backends.backend_pdf import PdfPages

from settings import Settings
from commonFunctions import *
from singleCompositeImage import singleCompositeImage

# start JVM for compatibility with VSI files
import javabridge
import bioformats

print('Python %s on %s' % (sys.version, sys.platform))
print('JAVA_HOME =', os.environ['JAVA_HOME'])

# initiate javabridge
javabridge.start_vm(class_path=bioformats.JARS)


def init_logger(jb):
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    From https://github.com/pskeshu/microscoper/blob/master/microscoper/io.py
    """
    rootLoggerName = jb.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jb.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jb.get_static_field("ch/qos/logback/classic/Level",
                                   "WARN",
                                   "Lch/qos/logback/classic/Level;")

    jb.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)


init_logger(javabridge)

settings = Settings()


class Application(ttk.Frame):
    def __init__(self, master=None):
        """Setup tk application."""
        super().__init__(master)
        self.name = None
        self.progress = None
        self.console = None
        self.console_scrollbar = None
        self.folder_root = None
        self.master = master
        self.master.title("Sim Lab: In vitro analysis")
        self.pack()
        self.main_container = ttk.Frame(master)
        self.main_container.pack(side="top", fill="both", expand=True)
        self.top_frame = ttk.Frame(self.main_container)
        self.bottom_frame = ttk.Frame(self.main_container)
        self.top_frame.pack(side="top", fill="x", expand=True)
        self.bottom_frame.pack(side="bottom", fill="both", expand=True)
        self.create_widgets()

    def create_widgets(self):
        """Creates widgets on initial window."""
        n1 = ttk.Label(self.top_frame,
                       text="""Experiment Name""",
                       font=tkFont.Font(size=14))
        n1.focus_set()

        l1 = ttk.Label(self.top_frame,
                       text="""1. Select folder to process:""",
                       font=tkFont.Font(size=12))

        l2 = ttk.Label(self.top_frame,
                       text="""2. Define pattern for image files:""",
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(size=12))

        l3 = ttk.Label(self.top_frame,
                       text="""Choose slice for each antigen label, use -1 to indicate antigen not present.""",
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(size=12))

        l4 = ttk.Label(self.top_frame,
                       text="""3. Which image slice contains the DAPI image?""",
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(size=12))

        l5 = ttk.Label(self.top_frame,
                       text="""4. DAPI gamma = """,
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(size=12))

        dapi_blocksize_label = ttk.Label(self.top_frame,
                                         text="""DAPI blocksize: """,
                                         justify=tk.LEFT,
                                         anchor='w',
                                         font=tkFont.Font(size=12))

        dapi_C_label = ttk.Label(self.top_frame,
                                 text="""DAPI C: """,
                                 justify=tk.LEFT,
                                 anchor='w',
                                 font=tkFont.Font(size=12))

        l6 = ttk.Label(self.top_frame,
                       text="""5. Which image slice contains the O4 image?""",
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(size=12))

        l7 = ttk.Label(self.top_frame,
                       text="""6. O4 gamma = """,
                       justify=tk.LEFT,
                       anchor='w',
                       font=tkFont.Font(size=12))

        l_EdUch = ttk.Label(self.top_frame,
                            text="""7. Which image slice contains the EdU image?""",
                            justify=tk.LEFT,
                            anchor='w',
                            font=tkFont.Font(size=12))

        l_EdUg = ttk.Label(self.top_frame,
                           text="""8. EdU gamma = """,
                           justify=tk.LEFT,
                           anchor='w',
                           font=tkFont.Font(size=12))

        l_Olig2Ch = ttk.Label(self.top_frame,
                              text="""Which image slice contains the Olig2 image?""",
                              justify=tk.LEFT,
                              anchor='w',
                              font=tkFont.Font(size=12))

        l_Olig2g = ttk.Label(self.top_frame,
                             text="""Olig2 gamma = """,
                             justify=tk.LEFT,
                             anchor='w',
                             font=tkFont.Font(size=12))

        l_mCherryCh = ttk.Label(self.top_frame,
                                text="""Which image slice contains the NLS mCherry image?""",
                                justify=tk.LEFT,
                                anchor='w',
                                font=tkFont.Font(size=12))

        l_mCherryg = ttk.Label(self.top_frame,
                               text="""NLS mCherry gamma = """,
                               justify=tk.LEFT,
                               anchor='w',
                               font=tkFont.Font(size=12))

        l10 = ttk.Label(self.top_frame,
                        text="""9. Which image slice contains the Gfap image?""",
                        justify=tk.LEFT,
                        anchor='w',
                        font=tkFont.Font(size=12))

        l11 = ttk.Label(self.top_frame,
                        text="""10. Gfap threshold = """,
                        justify=tk.LEFT,
                        anchor='w',
                        font=tkFont.Font(size=12))

        l12 = ttk.Label(self.top_frame,
                        text="""11. Scalefactor (1.0 is equivalent to 1.5385 pixels per 1.0 μm) = """,
                        justify=tk.LEFT,
                        anchor='w',
                        font=tkFont.Font(size=12))

        l13 = ttk.Label(self.top_frame,
                        text="""Enable debug?""",
                        anchor='e',
                        font=tkFont.Font(size=12))

        prediction_cutoff_label = ttk.Label(self.top_frame,
                                            text="""O4 Keras Model Prediction cutoff:""",
                                            anchor='e',
                                            font=tkFont.Font(size=12))

        n1.grid(row=0, column=0, sticky='w', pady=2)
        l1.grid(row=1, column=0, sticky='w', pady=2)
        l2.grid(row=2, column=0, sticky='w', pady=2)

        l3.grid(row=3, column=0, columnspan=4, sticky='ew', pady=2)
        # DAPI
        l4.grid(row=4, column=0, sticky='w', pady=2)
        l5.grid(row=4, column=2, sticky='w', pady=2)
        dapi_blocksize_label.grid(row=5, column=2, sticky='e', pady=2)
        dapi_C_label.grid(row=6, column=2, sticky='e', pady=2)
        # O4
        l6.grid(row=7, column=0, sticky='w', pady=2)
        l7.grid(row=7, column=2, sticky='w', pady=2)
        # EdU
        l_EdUch.grid(row=8, column=0, sticky='w', pady=2)
        l_EdUg.grid(row=8, column=2, sticky='w', pady=2)
        # Olig2
        l_Olig2Ch.grid(row=9, column=0, sticky='w', pady=2)
        l_Olig2g.grid(row=9, column=2, sticky='w', pady=2)
        # NLS mCherry
        l_mCherryCh.grid(row=10, column=0, sticky='w', pady=2)
        l_mCherryg.grid(row=10, column=2, sticky='w', pady=2)

        l10.grid(row=11, column=0, sticky='w', pady=2)
        l11.grid(row=11, column=2, sticky='w', pady=2)
        l12.grid(row=12, column=0, sticky='w', pady=2)
        l13.grid(row=12, column=2, sticky='w', pady=2)
        prediction_cutoff_label.grid(row=13, column=0, sticky='w', pady=2)

        """Now for the text entry and other boxes"""

        def selection_changed(event=None):
            # print("Selection changed")
            selected_value = combo.get()

            self.name.set(selected_value)
            # retrieve values from other experiment

            self.folder_root.set(settings.experiments[selected_value]['root'])
            pattern.set(settings.experiments[selected_value]['pattern'])
            # DAPI
            dapi_ch.set(settings.experiments[selected_value].get('dapi_ch', 0))
            dapi_gamma.set(settings.experiments[selected_value].get('dapi_gamma', 1.0))
            # O4
            o4_ch.set(settings.experiments[selected_value].get('o4_ch', -1))
            o4_gamma.set(settings.experiments[selected_value].get('o4_gamma', 1.0))
            # EdU
            edu_ch.set(settings.experiments[selected_value].get('edu_ch', -1))
            edu_gamma.set(settings.experiments[selected_value].get('edu_gamma', 1.0))
            # Olig2
            olig2_ch.set(settings.experiments[selected_value].get('olig2_ch', -1))
            olig2_gamma.set(settings.experiments[selected_value].get('olig2_gamma', 1.0))
            # mCherry
            mCherry_ch.set(settings.experiments[selected_value].get('mCherry_ch', -1))
            mCherry_gamma.set(settings.experiments[selected_value].get('mCherry_gamma', 1.0))
            # Gfap
            gfap_ch.set(settings.experiments[selected_value].get('gfap_ch', -1))
            gfap_th.set(settings.experiments[selected_value].get('gfap_th', 1000))

            scalefactor.set(settings.experiments[selected_value].get('scalefactor', 1))
            prediction_cutoff.set(settings.experiments[selected_value].get('prediction_cutoff', 0.5))

            debug.set(settings.experiments[selected_value].get('debug', 0))
            # print("Selected:", selected_value)

        self.name = tk.StringVar()
        experiments = list(settings.experiments)
        # print(experiments)
        self.name.set(settings.defaults["name"])
        combo = ttk.Combobox(self.top_frame, values=experiments,
                             width=80, textvariable=self.name,
                             font=tkFont.Font(size=14))
        combo.bind("<<ComboboxSelected>>", selection_changed)

        e1 = ttk.Frame(self.top_frame)
        ttk.Button(e1,
                   text="Browse",
                   command=lambda: self.select_folder()).pack(side=tk.RIGHT)

        self.folder_root = tk.StringVar()
        self.folder_root.set(settings.defaults["root"])
        ttk.Entry(e1, width=80, textvariable=self.folder_root,
                  font=tkFont.Font(size=12)).pack(side=tk.LEFT, fill="x", expand=True)

        # set file pattern
        pattern = tk.StringVar()
        pattern.set(settings.defaults["pattern"])
        e2 = ttk.Entry(self.top_frame, width=20, textvariable=pattern,
                       font=tkFont.Font(size=12))

        # set DAPI channel
        dapi_ch = tk.IntVar()
        dapi_ch.set(settings.defaults["dapi_ch"])
        e3 = ttk.Entry(self.top_frame, width=20, textvariable=dapi_ch,
                       font=tkFont.Font(size=12))

        # set DAPI gamma
        dapi_gamma = tk.DoubleVar()
        dapi_gamma.set(settings.defaults["dapi_gamma"])
        e4 = ttk.Entry(self.top_frame, width=20, textvariable=dapi_gamma,
                       font=tkFont.Font(size=12))

        # set DAPI blocksize
        dapi_blocksize = tk.IntVar()
        dapi_blocksize.set(11)  # need to be incorporated into settings
        dapi_blocksize_entry = ttk.Entry(self.top_frame, width=20, textvariable=dapi_blocksize,
                                         font=tkFont.Font(size=12))
        # set DAPI C
        dapi_C = tk.IntVar()
        dapi_C.set(2)
        dapi_C_entry = ttk.Entry(self.top_frame, width=20, textvariable=dapi_C,
                                 font=tkFont.Font(size=12))

        # set O4 channel
        o4_ch = tk.IntVar()
        o4_ch.set(settings.defaults["o4_ch"])
        e5 = ttk.Entry(self.top_frame, width=20, textvariable=o4_ch,
                       font=tkFont.Font(size=12))

        # set O4 gamma
        o4_gamma = tk.DoubleVar()
        o4_gamma.set(settings.defaults["o4_gamma"])
        e6 = ttk.Entry(self.top_frame, width=20, textvariable=o4_gamma,
                       font=tkFont.Font(size=12))

        # set EdU channel
        edu_ch = tk.IntVar()
        edu_ch.set(settings.defaults["edu_ch"])
        e_EdUCh = ttk.Entry(self.top_frame, width=20, textvariable=edu_ch,
                            font=tkFont.Font(size=12))

        # set EdU gamma
        edu_gamma = tk.DoubleVar()
        edu_gamma.set(settings.defaults["edu_gamma"])
        e_EdUg = ttk.Entry(self.top_frame, width=20, textvariable=edu_gamma,
                           font=tkFont.Font(size=12))

        # set Olig2 channel
        olig2_ch = tk.IntVar()
        if "olig2_ch" in settings.defaults:
            olig2_ch.set(settings.defaults["olig2_ch"])
        else:
            olig2_ch.set(-1)
        e_Olig2Ch = ttk.Entry(self.top_frame, width=20, textvariable=olig2_ch,
                              font=tkFont.Font(size=12))

        # set Olig2 gamma
        olig2_gamma = tk.DoubleVar()
        if "olig2_gamma" in settings.defaults:
            olig2_gamma.set(settings.defaults["olig2_gamma"])
        e_Olig2g = ttk.Entry(self.top_frame, width=20, textvariable=olig2_gamma,
                             font=tkFont.Font(size=12))

        # set mCherry channel
        mCherry_ch = tk.IntVar()
        if "mCherry_ch" in settings.defaults:
            mCherry_ch.set(settings.defaults["mCherry_ch"])
        else:
            mCherry_ch.set(-1)
        e_mCherryCh = ttk.Entry(self.top_frame, width=20, textvariable=mCherry_ch,
                                font=tkFont.Font(size=12))

        # set EdU gamma
        mCherry_gamma = tk.DoubleVar()
        if "mCherry_gamma" in settings.defaults:
            mCherry_gamma.set(settings.defaults["mCherry_gamma"])
        e_mCherryg = ttk.Entry(self.top_frame, width=20, textvariable=mCherry_gamma,
                               font=tkFont.Font(size=12))

        # set Gfap channel
        gfap_ch = tk.IntVar()
        gfap_ch.set(settings.defaults["gfap_ch"])
        e9 = ttk.Entry(self.top_frame, width=20, textvariable=gfap_ch,
                       font=tkFont.Font(size=12))

        # set Gfap threshold
        gfap_th = tk.IntVar()
        gfap_th.set(settings.defaults["gfap_th"])
        e10 = ttk.Entry(self.top_frame, width=20, textvariable=gfap_th,
                        font=tkFont.Font(size=12))

        # set scalefactor
        scalefactor = tk.DoubleVar()
        scalefactor.set(settings.defaults["scalefactor"])
        e11 = ttk.Entry(self.top_frame, width=20, textvariable=scalefactor,
                        font=tkFont.Font(size=12))

        # debug mode?
        debug = tk.BooleanVar()
        debug.set(settings.defaults["debug"])
        e12 = ttk.Checkbutton(self.top_frame, text='', variable=debug,
                              onvalue=True, offvalue=False)

        # prediction_cutoff
        prediction_cutoff = tk.DoubleVar()
        if 'prediction_cutoff' in settings.defaults:
            prediction_cutoff.set(settings.defaults['prediction_cutoff'])
        else:
            prediction_cutoff.set(0.5)
        prediction_cutoff_entry = ttk.Entry(self.top_frame, width=20,
                                            textvariable=prediction_cutoff,
                                            font=tkFont.Font(size=12))

        # Experiment
        combo.grid(row=0, column=1, columnspan=3, sticky='ew', pady=2)
        # Folder
        e1.grid(row=1, column=1, columnspan=3, sticky='ew', pady=2)
        # File type
        e2.grid(row=2, column=1, sticky='w', pady=2)
        # DAPI
        e3.grid(row=4, column=1, sticky='w', pady=2)
        e4.grid(row=4, column=3, sticky='w', pady=2)
        dapi_blocksize_entry.grid(row=5, column=3, sticky='w', pady=2)
        dapi_C_entry.grid(row=6, column=3, sticky='w', pady=2)
        # O4
        e5.grid(row=7, column=1, sticky='w', pady=2)
        e6.grid(row=7, column=3, sticky='w', pady=2)
        # EdU
        e_EdUCh.grid(row=8, column=1, sticky='w', pady=2)
        e_EdUg.grid(row=8, column=3, sticky='w', pady=2)
        # Olig2
        e_Olig2Ch.grid(row=9, column=1, sticky='w', pady=2)
        e_Olig2g.grid(row=9, column=3, sticky='w', pady=2)
        # mCherry
        e_mCherryCh.grid(row=10, column=1, sticky='w', pady=2)
        e_mCherryg.grid(row=10, column=3, sticky='w', pady=2)

        e9.grid(row=11, column=1, sticky='w', pady=2)
        e10.grid(row=11, column=3, sticky='w', pady=2)
        e11.grid(row=12, column=1, sticky='w', pady=2)
        e12.grid(row=12, column=3, sticky='w', pady=2)
        prediction_cutoff_entry.grid(row=13, column=1, columnspan=1, sticky='w', pady=2)

        # start button
        button2 = ttk.Button(self.bottom_frame,
                             text="Start",
                             command=lambda: self.start_analysis(name=self.name.get(),
                                                                 folder_root=self.folder_root.get(),
                                                                 pattern=pattern.get(),
                                                                 dapi_ch=dapi_ch.get(),
                                                                 dapi_gamma=dapi_gamma.get(),
                                                                 dapi_blocksize=dapi_blocksize.get(),
                                                                 dapi_C=dapi_C.get(),
                                                                 o4_ch=o4_ch.get(),
                                                                 o4_gamma=o4_gamma.get(),
                                                                 edu_ch=edu_ch.get(),
                                                                 edu_gamma=edu_gamma.get(),
                                                                 olig2_ch=olig2_ch.get(),
                                                                 olig2_gamma=olig2_gamma.get(),
                                                                 mCherry_ch=mCherry_ch.get(),
                                                                 mCherry_gamma=mCherry_gamma.get(),
                                                                 gfap_ch=gfap_ch.get(),
                                                                 gfap_th=gfap_th.get(),
                                                                 scalefactor=scalefactor.get(),
                                                                 prediction_cutoff=prediction_cutoff.get(),
                                                                 debug=debug.get()))
        button2.pack(side="top")

        # define output console area
        ttk.Label(self.bottom_frame,
                  text="""Output""",
                  justify=tk.LEFT,
                  anchor='n',
                  font=tkFont.Font(size=12)).pack(side='left', fill='y')
        # add scroll bar
        self.console_scrollbar = ttk.Scrollbar(self.bottom_frame, orient=tk.VERTICAL)
        self.console_scrollbar.pack(side="right", fill='y')
        # add text console widget
        self.console = tk.Text(self.bottom_frame,
                               yscrollcommand=self.console_scrollbar,
                               font=tkFont.Font(size=12))
        self.console.pack(side="top", fill="both", expand=True)
        self.console.bind("<Key>", lambda e: "break")
        # assign scroll bar to console yview
        self.console_scrollbar.config(command=self.console.yview)
        # add progress bar
        self.progress = Progressbar(self.bottom_frame,
                                    length=200, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side="bottom")

    def select_folder(self):
        import os
        directory_path = fileDialog.askdirectory(title='Select source folder containing image files',
                                                 initialdir=self.folder_root.get())
        if directory_path != "":
            self.folder_root.set(os.path.abspath(directory_path))

    def start_analysis(
            self,
            name: str,
            folder_root: str,
            pattern: str,
            dapi_ch: int,
            dapi_gamma: float = 1.0,
            dapi_blocksize: int = 11,
            dapi_C: int = 2,
            o4_ch: int = -1,
            o4_gamma: float = 1.0,
            edu_ch: int = -1,
            edu_gamma: float = 1.0,
            olig2_ch: int = -1,
            olig2_gamma: float = 1.0,
            mCherry_ch: int = -1,
            mCherry_gamma: float = 1.0,
            gfap_ch: int = -1,
            gfap_th: int = 1000,
            scalefactor: float = 1.0,
            prediction_cutoff: float = 0.5,
            debug: bool = False):

        # clear console
        self.console.delete(1.0, tk.END)

        # save settings
        settings.updateDefaults(
            name,
            folder_root,
            pattern,
            dapi_ch,
            o4_ch,
            edu_ch,
            olig2_ch,
            mCherry_ch,
            gfap_ch,
            dapi_gamma,
            o4_gamma,
            edu_gamma,
            olig2_gamma,
            mCherry_gamma,
            gfap_th,
            scalefactor,
            prediction_cutoff,
            debug)

        if name == "temp":
            self.console.insert("end", "\nTemporary analysis. experiments.json will not be updated")
        else:
            settings.saveExperimentalParameters()
            self.console.insert("end", "\nUpdating list of experiment parameters")

        # set o4_ch and edu_ch to none if -1
        if o4_ch == -1:
            o4_ch = None
            self.console.insert("end", "\nSkipping O4 channel & analysis")

        if edu_ch == -1:
            edu_ch = None
            self.console.insert("end", "\nSkipping EdU channel & analysis")

        if olig2_ch == -1:
            olig2_ch = None
            self.console.insert("end", "\nSkipping Olig2 channel & analysis")

        if mCherry_ch == -1:
            mCherry_ch = None
            self.console.insert("end", "\nSkipping NLS-mCherry channel & analysis")

        if gfap_ch == -1:
            gfap_ch = None
            self.console.insert("end", "\nSkipping Gfap channel & analysis")

        # start analysis
        files = find(pattern, folder_root, excluded_subfolder='keras')

        if len(files) == 0:
            self.console.insert("end", f"No files found in '{folder_root}'. Check the input.")
            return
        else:
            self.console.insert("end", f"\nFound {len(files)} matching '{pattern}' in '{folder_root}'")
        self.console.insert("end", "\n***************************")
        self.console.insert("end", "\nStarting to analyze images")
        self.console.update()

        # select file sample
        if debug:
            # select five files at random
            # files = list(files[i] for i in random.sample(list(range(len(files))), 5))

            # select first two files to do manual count comparisons
            self.console.insert("end", "\ndebug: selecting first two files to do manual count comparisons")
            files = list(files[i] for i in range(0, 2))

            # self.console.insert("end", "\ndebug: specific file hard encoded...")
            # debug_path = '/Users/frasersim/Library/CloudStorage/Box-Box/NewLabData/People/Greg/Nog Expan/Plate1(2.4.24)/Folder_20240204/EdUOlig2'
            # files = [{"path": debug_path, "name": "NogExpan_EduOlig2_Control_C5_ImageID-26729.vsi"}]

        results = []

        model = None
        if o4_ch is not None:
            if os.path.exists(os.path.join(folder_root, settings.kerasModel)):
                self.console.insert("end", "\nUsing experiment specific model.")
                model = loadKerasModel(os.path.join(folder_root, settings.kerasModel))
            else:
                self.console.insert("end", "\nUsing old general model.")
                model = loadKerasModel('o4counter_wAug_5.1.h5')

        # rest progress bar
        self.progress["value"] = 0
        self.progress.update()
        # calculate increment for progress bar
        fileNumber = len(files)
        currentFileNumber = 0
        i = 100 / fileNumber

        with PdfPages(fullPath(folder_root, 'results_nucleiCounter.pdf')) as export_pdf:

            for file in files:

                # increment currentFileNumber
                currentFileNumber += 1

                # increment progress bar
                self.progress["value"] += i
                self.progress.update()

                path = file['path']
                imgFile = file['name']

                # parse file names
                well = parseFileName(imgFile)
                if debug:
                    print(f"Well: {well}")

                try:
                    sCI = singleCompositeImage(
                        path=path,
                        imgFile=imgFile,
                        dapi_ch=dapi_ch,
                        dapi_gamma=dapi_gamma,
                        o4_ch=o4_ch,
                        o4_gamma=o4_gamma,
                        EdU_ch=edu_ch,
                        EdU_gamma=edu_gamma,
                        Olig2_ch=olig2_ch,
                        Olig2_gamma=olig2_gamma,
                        mCherry_ch=mCherry_ch,
                        mCherry_gamma=mCherry_gamma,
                        gfap_ch=gfap_ch,
                        gfap_th=gfap_th,
                        scalefactor=scalefactor,
                        debug=debug)
                    sCI.processDAPI(threshold_method='th2', blocksize=dapi_blocksize,
                                    C=dapi_C, debug=debug)  # based on manual counts (see OneNote)

                    if o4_ch is not None:
                        sCI.processCells()
                        sCI.getPredictions(model)
                        sCI.processPredictions(export_pdf, prediction_cutoff=prediction_cutoff, debug=debug)

                    if edu_ch is not None:
                        sCI.countEdUchannel(export_pdf)

                    if olig2_ch is not None:
                        sCI.countOlig2channel(export_pdf)

                    if mCherry_ch is not None:
                        sCI.countmCherrychannel(export_pdf)

                    if gfap_ch is not None:
                        sCI.countGfapchannel(export_pdf)

                    if olig2_ch is not None and edu_ch is not None:
                        if debug:
                            print('Counting Olig2 and EdU double positive cells...')
                        Olig2EdUpos_count, Olig2EdU_mask, Olig2EdU_watershed, \
                            Olig2EdU_centroid_x, \
                            Olig2EdU_centroid_y = sCI.countNuclearMarker(export_pdf,
                                                                         name1='EdU',
                                                                         channel1=sCI.EdU_ch,
                                                                         gamma1=sCI.EdU_gamma,
                                                                         name2='Olig2',
                                                                         nucleiMask=sCI.Olig2_mask,
                                                                         nucleiCount=sCI.Olig2pos_count,
                                                                         nucleiWatershed=sCI.Olig2_watershed,
                                                                         centroid_x=sCI.Olig2_centroid_x,
                                                                         centroid_y=sCI.Olig2_centroid_y)
                        if debug:
                            print(f"Olig2 & EdU double positive count: {Olig2EdUpos_count}")

                    if mCherry_ch is not None and edu_ch is not None:
                        if debug:
                            print('Counting mCherry and EdU double positive cells...')
                        mCherryEdUpos_count, mCherryEdU_mask, mCherryEdU_watershed, \
                            mCherryEdU_centroid_x, \
                            mCherryEdU_centroid_y = sCI.countNuclearMarker(export_pdf,
                                                                           name1='EdU',
                                                                           channel1=sCI.EdU_ch,
                                                                           gamma1=sCI.EdU_gamma,
                                                                           name2='mCherry',
                                                                           nucleiMask=sCI.mCherry_mask,
                                                                           nucleiCount=sCI.mCherrypos_count,
                                                                           nucleiWatershed=sCI.mCherry_watershed,
                                                                           centroid_x=sCI.mCherry_centroid_x,
                                                                           centroid_y=sCI.mCherry_centroid_y)
                        if debug:
                            print(f"mCherry & EdU double positive count: {mCherryEdUpos_count}")

                    if olig2_ch is not None and mCherry_ch is not None:
                        if debug:
                            print('Counting Olig2 and mCherry double positive cells...')
                        Olig2mCherrypos_count, Olig2mCherry_mask, Olig2mCherry_watershed, \
                            Olig2mCherry_centroid_x, \
                            Olig2mCherry_centroid_y = sCI.countNuclearMarker(export_pdf,
                                                                             name1='Olig2',
                                                                             channel1=sCI.Olig2_ch,
                                                                             gamma1=sCI.Olig2_gamma,
                                                                             name2='mCherry',
                                                                             nucleiMask=sCI.mCherry_mask,
                                                                             nucleiCount=sCI.mCherrypos_count,
                                                                             nucleiWatershed=sCI.mCherry_watershed,
                                                                             centroid_x=sCI.mCherry_centroid_x,
                                                                             centroid_y=sCI.mCherry_centroid_y)
                        if debug:
                            print(f"mCherry & Olig2 double positive count: {Olig2mCherrypos_count}")

                    if edu_ch is not None and olig2_ch is not None and mCherry_ch is not None:
                        if debug:
                            print('Counting EdU, mCherry and Olig2 triple positive cells...')
                        EdUmCherryOlig2pos_count, EdUCherryOlig2_mask, EdUmCherryOlig2_watershed, \
                            EdUmCherryOlig2_centroid_x, \
                            EdUmCherryOlig2_centroid_y \
                            = sCI.countNuclearMarker(export_pdf,
                                                     name1='EdU',
                                                     channel1=sCI.EdU_ch,
                                                     gamma1=sCI.EdU_gamma,
                                                     name2='mCherry+Olig2',
                                                     nucleiMask=Olig2mCherry_mask,
                                                     nucleiCount=Olig2mCherrypos_count,
                                                     nucleiWatershed=Olig2mCherry_watershed,
                                                     centroid_x=Olig2mCherry_centroid_x,
                                                     centroid_y=Olig2mCherry_centroid_y
                                                     )
                        if debug:
                            print(f"EdU, mCherry & Olig2 triple positive count: {EdUmCherryOlig2pos_count}")

                    if debug:
                        sCI.reportResults()
                        self.console.insert('end', f"\nimgFile: {sCI.imgFile} found {sCI.nucleiCount} DAPI+ nuclei.")

                        if o4_ch is not None:
                            self.console.insert('end', f" O4+: {sCI.o4pos_count}.")

                        if edu_ch is not None:
                            self.console.insert('end', f" EdU+: {sCI.EdUpos_count}.")

                        if olig2_ch is not None:
                            self.console.insert('end', f" Olig2+: {sCI.Olig2pos_count}.")

                        if mCherry_ch is not None:
                            self.console.insert('end', f" mCherry+: {sCI.mCherrypos_count}.")

                        if gfap_ch is not None:
                            self.console.insert('end', f" Gfap+: {sCI.gfappos_count}.")

                        if edu_ch is not None and olig2_ch is not None:
                            self.console.insert('end', f" EdU+Olig2+: {Olig2EdUpos_count}.")

                        if edu_ch is not None and mCherry_ch is not None:
                            self.console.insert('end', f" EdU+mCherry+: {mCherryEdUpos_count}.")

                        if olig2_ch is not None and mCherry_ch is not None:
                            self.console.insert('end', f" Olig2+mCherry+: {Olig2EdUpos_count}.")

                        if edu_ch is not None and olig2_ch is not None and mCherry_ch is not None:
                            self.console.insert('end', f" EdU+Olig2+mCherry+: {EdUmCherryOlig2pos_count}.")

                        self.console.update()

                        # report result of nuclei count
                    result = {
                        'path': sCI.path,
                        'imgFile': sCI.imgFile,
                        'nucleiCount': sCI.nucleiCount}

                    # add well details parsed from fileName
                    if well is not None:
                        result['well'] = well

                    # add O4 counts
                    if o4_ch is not None:
                        if (sCI.o4pos_count + sCI.o4neg_count) > 0:
                            o4_percentage = sCI.o4pos_count / (sCI.o4pos_count + sCI.o4neg_count)
                        else:
                            o4_percentage = 0
                            self.console.insert('end', f"Error calculating O4% in {sCI.imgFile}.")

                        result['o4pos_count'] = sCI.o4pos_count
                        result['o4neg_count'] = sCI.o4neg_count
                        result['o4%'] = "{:.2%}".format(o4_percentage)

                    # add EdU counts
                    if edu_ch is not None:
                        result['edupos_count'] = sCI.EdUpos_count

                    # add Olig2 counts
                    if olig2_ch is not None:
                        result['olig2pos_count'] = sCI.Olig2pos_count

                    # add mCherry counts
                    if mCherry_ch is not None:
                        result['mCherrypos_count'] = sCI.mCherrypos_count

                    # add Gfap counts
                    if gfap_ch is not None:
                        result['gfappos_count'] = sCI.gfappos_count

                    if olig2_ch is not None and edu_ch is not None:
                        result['Olig2EdUpos_count'] = Olig2EdUpos_count

                    if mCherry_ch is not None and edu_ch is not None:
                        result['mCherryEdUpos_count'] = mCherryEdUpos_count

                    if mCherry_ch is not None and olig2_ch is not None:
                        result['Olig2mCherrypos_count'] = Olig2mCherrypos_count

                    if edu_ch is not None and olig2_ch is not None and mCherry_ch is not None:
                        result['EdUmCherryOlig2pos_count'] = EdUmCherryOlig2pos_count

                    results.append(result)

                    self.console.insert("end", f"\nCompleted '{imgFile}'. {currentFileNumber} of {fileNumber} files.")
                except Exception as e:
                    self.console.insert("end", f"\nFailed on path '{path}'. Image: {imgFile}")
                    self.console.insert("end", f"\n{str(e)} on {imgFile}")

                self.console.update()
                self.console.yview("end")

            # Done with File loop
            self.console.insert("end", f"\nFinished analysis, exporting PDF file...")
            self.console.update()
            self.console.yview("end")

        # output results as csv
        import csv
        filename = fullPath(folder_root, 'results_nucleiCounter.csv')
        with open(filename, 'w', newline='') as f:
            w = csv.DictWriter(f, results[0].keys())
            w.writeheader()
            w.writerows(results)

        self.console.insert("end", f'\nResults saved to {filename}.')
        self.console.insert("end", '\nAll Done')


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    window.geometry(f"{width}x{height}+{x}+{y}")


# Starts application.
root = tk.Tk()

width = 1200
height = 900

root.minsize(width=width, height=height)
style = ttk.Style()
style.theme_use("aqua")
app = Application(master=root)
center_window(root, width, height)

app.mainloop()
javabridge.kill_vm()
