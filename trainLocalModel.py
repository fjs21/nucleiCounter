"""
TEST: This file will use the annotations in an annotations.json file in
an experimental folder to train a keras model using the local computer
"""
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter.ttk import Progressbar

import json
import shutil

from settings import Settings
from commonFunctions import *

settings = Settings()

# setup image folder & settings for dataset sizes
# defaults
base_dir = 'o4modelimages'
val_size_default = 500
test_size_default = 200
batch_size = 200
epochs = 100


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

        # initialize directory variables
        self.train_dir = None
        self.validation_dir = None
        self.test_dir = None
        self.train_o4neg_dir = None
        self.train_o4pos_dir = None
        self.validation_o4neg_dir = None
        self.validation_o4pos_dir = None
        self.test_o4neg_dir = None
        self.test_o4pos_dir = None

        # initialize for widgets
        self.experiment = None
        self.marker = None
        self.debug = None
        self.progress = None
        self.console = None
        self.console_scrollbar = None

        self.master = master
        self.master.title("Train a local Keras model")
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

        experiment_label = tk.Label(self.top_frame,
                      text="""Experiment Name""",
                      font=tkFont.Font(family="Calibri", size=14))

        marker_label = tk.Label(self.top_frame,
                      text="""Choose whether to use XML classifications or manual annotations.""",
                      justify=tk.LEFT,
                      anchor='w',
                      font=tkFont.Font(family="Calibri", size=12))

        validation_label = tk.Label(self.top_frame,
                                text="""Minimum number of images for validation data set:""",
                                justify=tk.LEFT,
                                anchor='w',
                                font=tkFont.Font(family="Calibri", size=12))

        test_label = tk.Label(self.top_frame,
                                text="""Minimum number of images for test data set:""",
                                justify=tk.LEFT,
                                anchor='w',
                                font=tkFont.Font(family="Calibri", size=12))

        debug_label = tk.Label(self.top_frame,
                       text="""Enable debug?""",
                       anchor='e',
                       font=tkFont.Font(family="Calibri", size=12))

        experiment_label.grid(row=0, column=0, sticky='w', pady=2)
        marker_label.grid(row=1, column=0, sticky='w', pady=2)
        validation_label.grid(row=2, column=0, sticky='w', pady=2)
        test_label.grid(row=3, column=0, sticky='w', pady=2)
        debug_label.grid(row=4, column=0, sticky='w', pady=2)

        self.experiment = tk.StringVar()
        experiments = list(settings.experiments)
        print(experiments)
        self.experiment.set(settings.defaults["name"])
        experiments_combo = ttk.Combobox(self.top_frame, values=experiments,
                             width=80, textvariable=self.experiment,
                             font=tkFont.Font(family="Calibri", size=14))

        self.marker = tk.StringVar()
        self.marker.set('classification')
        marker_combo = ttk.Combobox(self.top_frame, values=['classification', 'annotation'],
                                    width=20, textvariable=self.marker,
                                    font=tkFont.Font(family='Calibri', size=12))

        self.val_size = tk.StringVar()
        self.val_size.set(str(val_size_default))
        val_size_entry = tk.Entry(self.top_frame, width =20,
                                  textvariable=self.val_size,
                                  font=tkFont.Font(family='Calibri', size=12))

        self.test_size = tk.StringVar()
        self.test_size.set(str(test_size_default))
        test_size_entry = tk.Entry(self.top_frame, width =20,
                                  textvariable=self.test_size,
                                  font=tkFont.Font(family='Calibri', size=12))

        # debug mode?
        self.debug = tk.BooleanVar()
        self.debug.set(settings.defaults["debug"])
        debug_check = tk.Checkbutton(self.top_frame, text='', variable=self.debug,
                             onvalue=True, offvalue=False,
                             anchor='w')

        experiments_combo.grid(row=0, column=1, sticky='w', pady=2)
        marker_combo.grid(row=1, column=1, sticky='w', pady=2)
        val_size_entry.grid(row=2, column=1, sticky='w', pady=2)
        test_size_entry.grid(row=3, column=1, sticky='w', pady=2)
        debug_check.grid(row=4, column=1, sticky='w', pady=2)

        # setup files button
        button2 = tk.Button(self.bottom_frame,
                            text="Consolidate files to local directory",
                            command=lambda: self.start_analysis(),
                            font=tkFont.Font(family="Calibri", size=12))

        button2.pack(side="top")

        # train model button
        button3 = tk.Button(self.bottom_frame,
                            text="Train model using local images",
                            command=lambda: self.train_model(),
                            font=tkFont.Font(family="Calibri", size=12))

        button3.pack(side="top")

        # define output console area
        tk.Label(self.bottom_frame,
                 text="""Output""",
                 justify=tk.LEFT,
                 anchor='n',
                 font=tkFont.Font(family="Calibri", size=12)).pack(side='left', fill='y')

        # add text console widget
        self.console = tk.Text(self.bottom_frame,
                               yscrollcommand=self.console_scrollbar,
                               font=tkFont.Font(family="", size=12))
        self.console.pack(side="top", fill="both", expand=True)
        self.console.bind("<Key>", lambda e: "break")
        # add scroll bar
        self.console_scrollbar = tk.Scrollbar(self.bottom_frame, orient=tk.VERTICAL, command=self.console.yview)
        self.console_scrollbar.pack(side="right", fill='y')
        self.console.config(yscrollcommand=self.console_scrollbar.set)

        # add progress bar
        self.progress = Progressbar(self.bottom_frame,
                                    length=200, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side="bottom", fill="y", expand=True)

        text_redirector = TextRedirector(self.console)
        sys.stdout = text_redirector

    def setupLocalFolders(self, base_dir, empty: bool = False):

        # define subdirectories for training, validation and test
        self.train_dir = os.path.join(base_dir, 'train')
        self.validation_dir = os.path.join(base_dir, 'validation')
        self.test_dir = os.path.join(base_dir, 'test')
        self.train_o4neg_dir = os.path.join(self.train_dir, 'o4neg')
        self.train_o4pos_dir = os.path.join(self.train_dir, 'o4pos')
        self.validation_o4neg_dir = os.path.join(self.validation_dir, 'o4neg')
        self.validation_o4pos_dir = os.path.join(self.validation_dir, 'o4pos')
        self.test_o4neg_dir = os.path.join(self.test_dir, 'o4neg')
        self.test_o4pos_dir = os.path.join(self.test_dir, 'o4pos')

        if empty:
            # delete existing and create new
            create_empty_folder(base_dir)

            os.mkdir(self.train_dir)
            os.mkdir(self.validation_dir)
            os.mkdir(self.test_dir)

            os.mkdir(self.train_o4neg_dir)
            os.mkdir(self.train_o4pos_dir)
            os.mkdir(self.validation_o4neg_dir)
            os.mkdir(self.validation_o4pos_dir)
            os.mkdir(self.test_o4neg_dir)
            os.mkdir(self.test_o4pos_dir)

    def start_analysis(self, export: bool = True, debug: bool = False):
        """ Run analysis - process folder for presence of count markers"""

        print(f"Starting local consolidation of images for model training & validation.")

        self.setupLocalFolders(base_dir, empty=True)

        experiment = self.experiment.get()

        folder_root = settings.experiments[experiment]['root']

        # load annotations
        filename = fullPath(folder_root, 'annotations.json')
        with open(filename, 'r') as f:
            annotations = json.load(f)

        train_o4pos_i = train_o4neg_i = 0
        val_o4pos_i = val_o4neg_i = 0
        test_o4pos_i = test_o4neg_i = 0

        self.progress["value"] = 0
        self.progress.update()
        # calculate increment for progress bar
        fileNumber = len(annotations)
        progress_increment = 100 / fileNumber

        print(f"Found {len(annotations)} cells in this experiment.")
        print("Starting copy operation...")

        for cell in annotations:

            # increment progress bar
            self.progress["value"] += progress_increment
            self.progress.update()

            if not self.marker.get() in cell:
                # not a valid cell
                print(f"No marker for cell {cell['cell']}")
                continue

            src = fullPath(os.path.join(settings.experiments[experiment]['root'], 'keras'), cell['cell'])
            dst = None

            if not os.path.exists(src):
                # image file not found, most likely renamed during manual annotation
                print(f"No image for cell {cell['cell']}")
                continue

            if cell[self.marker.get()] == 1:
                """ O4+ cell """
                #  order changed to maximize training data set available
                if val_o4pos_i < int(self.val_size.get()):
                    dst = fullPath(self.validation_o4pos_dir, cell['cell'])
                    val_o4pos_i += 1
                elif test_o4pos_i < int(self.test_size.get()):
                    dst = fullPath(self.test_o4pos_dir, cell['cell'])
                    test_o4pos_i += 1
                else:
                    dst = fullPath(self.train_o4pos_dir, cell['cell'])
                    train_o4pos_i += 1

            elif cell[self.marker.get()] == 0:
                """ O4- cell """
                #  order changed to maximize training data set available
                if val_o4neg_i < int(self.val_size.get()):
                    dst = fullPath(self.validation_o4neg_dir, cell['cell'])
                    val_o4neg_i += 1
                elif test_o4neg_i < int(self.test_size.get()):
                    dst = fullPath(self.test_o4neg_dir, cell['cell'])
                    test_o4neg_i += 1
                else:
                    dst = fullPath(self.train_o4neg_dir, cell['cell'])
                    train_o4neg_i += 1

            else:
                """ Cell annotation not set or unknown '-1'. """
                continue

            # copy image file to local folder for model building
            shutil.copyfile(src, dst)

        print(f"total training o4- images: {len(os.listdir(self.train_o4neg_dir))}")
        print(f"total training o4+ images: {len(os.listdir(self.train_o4pos_dir))}")
        print(f"total validation o4- images: {len(os.listdir(self.validation_o4neg_dir))}")
        print(f"total validation o4+ images: {len(os.listdir(self.validation_o4pos_dir))}")
        print(f"total test o4- images: {len(os.listdir(self.test_o4neg_dir))}")
        print(f"total test o4+ images: {len(os.listdir(self.test_o4pos_dir))}")

        print("All Done. Ready for model fitting.")
        print("**********************************")

    def train_model(self):

        self.setupLocalFolders(base_dir, empty=False)
        root_folder = settings.experiments[self.experiment.get()]['root']

        from keras import models
        from keras import layers
        from keras import optimizers
        from keras.preprocessing.image import ImageDataGenerator
        from keras.callbacks import EarlyStopping

        # Keras model
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        print(model.summary())

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=90,
            horizontal_flip=True,
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(
            rescale=1. / 255)

        print("Setup training dataset folder.")
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary')

        print("Setup validation dataset folder.")
        validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(128, 128),
            batch_size=batch_size,
            class_mode='binary')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        print("****************")
        print("Fitting model...")
        history = model.fit(
            train_generator,
            # steps_per_epoch=int(5 * train_size / batch_size),  # oversample 5x
            # steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[early_stopping])
        print("Done.")
        print("****************")

        model.save(os.path.join(root_folder, settings.kerasModel))

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # output results as csv
        import csv
        filename = os.path.join(root_folder, 'KerasModelFit_results_' + settings.kerasModel + '.csv')
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['acc', 'val_acc', 'loss', 'val_loss'])
            for i in range(len(history.history['acc'])):
                w.writerow([acc[i], val_acc[i], loss[i], val_loss[i]])

        # Evaluate model
        print("****************")
        print("Evaluating model...")
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(128, 128),
            batch_size=20,
            class_mode='binary')
        print("Done.")

        test_loss, test_acc = model.evaluate(test_generator, steps=50)
        print(f"Test accuracy (%): {test_acc}")


# Starts application.
root = tk.Tk()
root.geometry('+100+100')
root.resizable(width=False, height=False)
app = Application(master=root)
app.mainloop()
