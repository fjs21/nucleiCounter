"""
TEST: This file will use the annotations in an annotations.json file in
an experimental folder to train a keras model using the local computer
"""
"""
GUI-based tool to consolidate image data and train a Keras CNN model
to classify O4+ versus O4− cells using annotations.

Features:
- Loads cell annotations from annotations.json
- Organizes image data into train/validation/test splits
- Trains CNN model with augmentation, class balancing, and evaluation
"""
import sys
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter.ttk import Progressbar

import json
import shutil
from PIL import Image

from settings import Settings
from commonFunctions import *

settings = Settings()

# setup image folder & settings for dataset sizes
# defaults
base_dir = 'o4modelimages'
val_size_default = 500
test_size_default = 200
batch_size = 64 # 200
epochs = 100 # 100


class TextRedirector:
    """ This class allows capturing of stdout into console in GUI """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.bind("<Control-c>", lambda e: self.copy_selection(e))
        self.text_widget.bind("<Command-c>", lambda e: self.copy_selection(e))

    def write(self, text):
        self.text_widget.insert('end', text)
        self.text_widget.see('end')
        self.text_widget.update()

    def flush(self):
        pass

    def copy_selection(self, event):
        try:
            self.text_widget.clipboard_clear()
            selected_text = self.text_widget.get("sel.first", "sel.last")
            self.text_widget.clipboard_append(selected_text)
        except tk.TclError:
            pass


# Utility function for printing to the console widget
def console_print(console, text):
    console.insert("end", text + "\n")
    console.see("end")
    console.update()


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

        # New model parameter variables
        self.filters = tk.StringVar(value='32,64,128,128')
        self.dropout = tk.StringVar(value='0.5')

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
        self.trained_model = None

    def create_widgets(self):
        """Creates widgets on initial window."""
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

        # New model parameter input rows
        filters_label = tk.Label(self.top_frame, text="Conv layer filters (comma-separated):", anchor='w',
                                 font=tkFont.Font(family="Calibri", size=12))
        filters_entry = tk.Entry(self.top_frame, width=20, textvariable=self.filters,
                                 font=tkFont.Font(family='Calibri', size=12))

        dropout_label = tk.Label(self.top_frame, text="Dropout rate (0.0 - 1.0):", anchor='w',
                                 font=tkFont.Font(family="Calibri", size=12))
        dropout_entry = tk.Entry(self.top_frame, width=20, textvariable=self.dropout,
                                 font=tkFont.Font(family='Calibri', size=12))

        # Help label for model parameters
        help_label = tk.Label(self.top_frame,
            text="""Adjust model parameters only if you want to experiment with the CNN architecture.\n
- Conv filters: affects depth of feature maps in successive layers (default: 32,64,128,128)\n
- Dropout: controls regularization to reduce overfitting (default: 0.5)\n
For typical use, leave these settings unchanged.""",
            justify=tk.LEFT,
            anchor='w',
            fg='gray',
            font=tkFont.Font(family="Calibri", size=10))
        help_label.grid(row=7, column=0, columnspan=2, sticky='w', pady=(4, 10))

        experiment_label.grid(row=0, column=0, sticky='w', pady=2)
        marker_label.grid(row=1, column=0, sticky='w', pady=2)
        validation_label.grid(row=2, column=0, sticky='w', pady=2)
        test_label.grid(row=3, column=0, sticky='w', pady=2)
        debug_label.grid(row=4, column=0, sticky='w', pady=2)
        filters_label.grid(row=5, column=0, sticky='w', pady=2)
        filters_entry.grid(row=5, column=1, sticky='w', pady=2)
        dropout_label.grid(row=6, column=0, sticky='w', pady=2)
        dropout_entry.grid(row=6, column=1, sticky='w', pady=2)

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
        self.experiment_path_label = tk.Label(self.top_frame, text="📁", fg="blue", font=tkFont.Font(family="Calibri", size=14))
        self.experiment_path_label.config(cursor="hand2")
        self.experiment_path_label.grid(row=0, column=2, sticky='w', padx=(10, 0))

        self.experiment_tooltip = tk.StringVar()
        tooltip_label = tk.Label(self.top_frame, text="", font=tkFont.Font(family="Calibri", size=10), fg="gray")

        def update_experiment_path(*args):
            experiment = self.experiment.get()
            if experiment in settings.experiments:
                path = settings.experiments[experiment]['root']
                self.experiment_tooltip.set(path)
            else:
                self.experiment_tooltip.set("")

        def open_experiment_folder(event=None):
            experiment = self.experiment.get()
            if experiment in settings.experiments:
                folder_path = settings.experiments[experiment]['root']
                import subprocess
                subprocess.run(["open", folder_path])

        def show_tooltip(event=None):
            tooltip_label.config(text=self.experiment_tooltip.get())
            # Place tooltip slightly below and to the right of the icon, relative to top_frame
            tooltip_label.place(x=event.x_root - self.top_frame.winfo_rootx(), y=event.y_root - self.top_frame.winfo_rooty() + 20)

        def hide_tooltip(event=None):
            tooltip_label.place_forget()

        self.experiment.trace_add('write', lambda *args: update_experiment_path())
        update_experiment_path()

        self.experiment_path_label.bind("<Button-1>", open_experiment_folder)
        self.experiment_path_label.bind("<Enter>", show_tooltip)
        self.experiment_path_label.bind("<Leave>", hide_tooltip)
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

        # convert TIF to PNG button
        button4 = tk.Button(self.bottom_frame,
                            text="Convert TIF to PNG in o4modelimages",
                            command=lambda: self.convert_tif_to_png(),
                            font=tkFont.Font(family="Calibri", size=12))
        button4.pack(side="top")

        # train model button
        button3 = tk.Button(self.bottom_frame,
                            text="Train model using local images",
                            command=lambda: self.train_model(),
                            font=tkFont.Font(family="Calibri", size=12))
        button3.pack(side="top")

        # Analyze misclassified images button
        button_misclassify = tk.Button(self.bottom_frame,
                            text="Analyze misclassified test images",
                            command=lambda: self.analyze_misclassifications(),
                            font=tkFont.Font(family="Calibri", size=12))
        button_misclassify.pack(side="top")

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

        # temporarily disable redirector
        #text_redirector = TextRedirector(self.console)
        #sys.stdout = text_redirector

    def convert_tif_to_png(self):
        """
        Recursively convert all .tif images in o4modelimages/ to .png format.
        Retains original .tif images as backup.
        Only converts .tif files if the corresponding .png does not already exist.
        """
        import glob
        import os

        # Initialize and reset the progress bar
        self.progress["value"] = 0
        self.progress.update()

        tif_files = glob.glob(os.path.join(base_dir, '**', '*.tif'), recursive=True)
        console_print(self.console, f"\nFound {len(tif_files)} .tif files in {base_dir}")

        progress_increment = 100 / len(tif_files) if tif_files else 0

        for tif_path in tif_files:
            png_path = tif_path.rsplit('.', 1)[0] + '.png'
            if os.path.exists(png_path):
                console_print(self.console, f"\nSkipping {tif_path}, PNG already exists.")
                # Update progress bar even if skipping
                self.progress["value"] += progress_increment
                self.progress.update()
                continue
            try:
                with Image.open(tif_path) as im:
                    im.convert('RGB').save(png_path, format='PNG')
                console_print(self.console, f"\nConverted {tif_path} -> {png_path}")
            except Exception as e:
                console_print(self.console, f"\nFailed to convert {tif_path}: {e}")
            # Update progress bar after each file
            self.progress["value"] += progress_increment
            self.progress.update()

        console_print(self.console, f"\nConversion complete.")

    def setupLocalFolders(self, base_dir, empty: bool = False):
        """
        Creates directory structure for train/validation/test datasets with O4+ and O4− subfolders.
        If 'empty' is True, existing folders are deleted and recreated.
        """
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
        """
        Parses annotations.json for the selected experiment and distributes cell images
        into training, validation, and test folders based on their O4 annotation status.
        If the input image is a .tif, it is converted to .png using PIL.
        """
        console_print(self.console, f"\nStarting local consolidation of images for model training & validation.")

        self.setupLocalFolders(base_dir, empty=True)

        experiment = self.experiment.get()

        folder_root = settings.experiments[experiment]['root']

        # load annotations with error handling
        try:
            filename = fullPath(folder_root, 'annotations.json')
            with open(filename, 'r') as f:
                annotations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            console_print(self.console, f"\nError loading annotations.json: {e}")
            return

        train_o4pos_i = train_o4neg_i = 0
        val_o4pos_i = val_o4neg_i = 0
        test_o4pos_i = test_o4neg_i = 0

        self.progress["value"] = 0
        self.progress.update()
        # calculate increment for progress bar
        fileNumber = len(annotations)
        progress_increment = 100 / fileNumber

        console_print(self.console, f"\nFound {len(annotations)} cells in this experiment.")
        console_print(self.console, f"Starting copy operation...")

        for cell in annotations:

            # increment progress bar
            self.progress["value"] += progress_increment
            self.progress.update()

            if not self.marker.get() in cell:
                # not a valid cell
                console_print(self.console, f"\nNo marker for cell {cell['cell']}")
                continue

            src = fullPath(os.path.join(settings.experiments[experiment]['root'], 'keras'), cell['cell'])
            dst = None

            if not os.path.exists(src):
                # image file not found, most likely renamed during manual annotation
                console_print(self.console, f"\nNo image for cell {cell['cell']}")
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

            # Convert .tif to .png and update dst, or skip/warn for non-.tif images
            if cell['cell'].lower().endswith('.tif'):
                dst = dst.rsplit('.', 1)[0] + '.png'
                try:
                    im = Image.open(src)
                    im.convert('RGB').save(dst, format='PNG')
                except Exception as e:
                    console_print(self.console, f"\nError converting {src} to PNG: {e}")
                    continue
            else:
                console_print(self.console, f"\nSkipping non-TIF image: {src}")
                continue

        console_print(self.console, f"\ntotal training o4- images: {len(os.listdir(self.train_o4neg_dir))}")
        console_print(self.console, f"\ntotal training o4+ images: {len(os.listdir(self.train_o4pos_dir))}")
        console_print(self.console, f"\ntotal validation o4- images: {len(os.listdir(self.validation_o4neg_dir))}")
        console_print(self.console, f"\ntotal validation o4+ images: {len(os.listdir(self.validation_o4pos_dir))}")
        console_print(self.console, f"\ntotal test o4- images: {len(os.listdir(self.test_o4neg_dir))}")
        console_print(self.console, f"\ntotal test o4+ images: {len(os.listdir(self.test_o4pos_dir))}")

        console_print(self.console, f"\nAll Done. Ready for model fitting.")
        console_print(self.console, f"\n**********************************")

    def train_model(self):
        """
        Builds and trains a convolutional neural network (CNN) using Keras to classify
        whether individual cell-centered images are O4+ or O4−.

        Steps:
        - Builds model with batch normalization and dropout
        - Loads training and validation data with augmentation
        - Computes class weights to handle imbalance
        - Trains model and evaluates on test data with metrics
        - Saves model and training results
        """

        self.setupLocalFolders(base_dir, empty=False)
        root_folder = settings.experiments[self.experiment.get()]['root']

        from keras import models
        from keras import layers
        from tensorflow.keras.optimizers import Adam
        from keras.callbacks import EarlyStopping
        from keras.metrics import AUC, Precision, Recall
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Parse model parameters from GUI
        filter_list = [int(f.strip()) for f in self.filters.get().split(',')]
        dropout_rate = float(self.dropout.get())

        # Define CNN model architecture with dynamic filters and dropout
        model = models.Sequential()
        model.add(layers.Input(shape=(128, 128, 3)))
        for f in filter_list:
            model.add(layers.Conv2D(f, (3, 3)))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), AUC()])
        model.summary(print_fn=lambda x: console_print(self.console, x))

        # Set up data generators with augmentation for training and rescaling for validation
        import tensorflow as tf
        from tensorflow.keras.utils import image_dataset_from_directory

        AUTOTUNE = tf.data.AUTOTUNE

        console_print(self.console, f"\nSetting up training dataset using image_dataset_from_directory...")
        train_generator = image_dataset_from_directory(
            self.train_dir,
            image_size=(128, 128),
            batch_size=batch_size,
            label_mode='binary',
            shuffle=True
        ).prefetch(buffer_size=AUTOTUNE)
        train_count = sum(1 for _ in train_generator)
        console_print(self.console, f"Found {train_count * batch_size} files belonging to 2 classes.")

        console_print(self.console, f"\nSetting up validation dataset using image_dataset_from_directory...")
        validation_generator = image_dataset_from_directory(
            self.validation_dir,
            image_size=(128, 128),
            batch_size=batch_size,
            label_mode='binary',
            shuffle=False
        ).prefetch(buffer_size=AUTOTUNE)
        val_count = sum(1 for _ in validation_generator)
        console_print(self.console, f"Found {val_count * batch_size} files belonging to 2 classes.")

        # Compute class weights to address O4+ underrepresentation
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        # Collect true labels from train_generator (tf.data.Dataset)
        train_labels = np.concatenate([np.array(y) for _, y in train_generator], axis=0).astype(int).flatten()
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels)
        class_weight_dict = dict(enumerate(class_weights))
        console_print(self.console, f"\nClass weights: {class_weight_dict}")

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # --- Custom console logger callback for Keras training ---
        class ConsoleLogger(tf.keras.callbacks.Callback):
            def __init__(self, console):
                super().__init__()
                self.console = console

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                msg = f"Epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
                console_print(self.console, msg)
        # --------------------------------------------------------

        console_print(self.console, f"\n****************")
        console_print(self.console, f"\nFitting model...")
        # Train the model using class weights and early stopping and console logger
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            class_weight=class_weight_dict,
            callbacks=[early_stopping, ConsoleLogger(self.console)])
        console_print(self.console, f"\nDone.")
        console_print(self.console, f"\n****************")

        # Save the trained model
        model.save(os.path.join(root_folder, settings.kerasModel))
        self.trained_model = model

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Write training history to CSV file
        import csv
        filename = os.path.join(root_folder, 'KerasModelFit_results_' + settings.kerasModel + '.csv')
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['accuracy', 'val_accuracy', 'loss', 'val_loss'])
            for i in range(len(history.history['accuracy'])):
                w.writerow([acc[i], val_acc[i], loss[i], val_loss[i]])

        # Prepare test data generator
        console_print(self.console, f"\n****************")
        console_print(self.console, f"\nEvaluating model...")
        test_generator = image_dataset_from_directory(
            self.test_dir,
            image_size=(128, 128),
            batch_size=batch_size,
            label_mode='binary',
            shuffle=False
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
        console_print(self.console, f"\nDone.")

        # Evaluate model on test data and output performance
        test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_generator)
        console_print(self.console, f"\nTest accuracy: {test_acc}")
        console_print(self.console, f"\nTest precision: {test_precision}")
        console_print(self.console, f"\nTest recall: {test_recall}")
        console_print(self.console, f"\nTest AUC: {test_auc}")
        # Advanced evaluation: confusion matrix and ROC-AUC
        from sklearn.metrics import confusion_matrix, roc_auc_score

        y_pred_probs = model.predict(test_generator, steps=50)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        # --- Identify and save misclassified image filenames ---
        # Get filenames from test_generator
        file_paths = []
        for batch in test_generator:
            images, labels = batch
            if hasattr(images, 'file_paths'):  # Not available in tf.data Dataset, so this is just a placeholder
                file_paths.extend(images.file_paths)

        # Workaround: manually collect file paths using glob
        import glob
        test_image_paths = sorted(glob.glob(os.path.join(self.test_dir, '*', '*.png')))

        misclassified_dir = os.path.join(root_folder, "misclassified")
        os.makedirs(misclassified_dir, exist_ok=True)

        # Get true labels
        y_true = [y.numpy() for _, y in test_generator]
        y_true = tf.concat(y_true, axis=0)[:len(y_pred)]

        import os
        import shutil
        for i, (yp, yt) in enumerate(zip(y_pred, y_true)):
            if yp != yt:
                src_path = test_image_paths[i]
                fname = os.path.basename(src_path)
                pred_label = "o4pos" if yp == 1 else "o4neg"
                true_label = "o4pos" if yt == 1 else "o4neg"
                dst_path = os.path.join(misclassified_dir, f"{true_label}_as_{pred_label}_{fname}")
                shutil.copy2(src_path, dst_path)

        console_print(self.console, f"\nSaved misclassified images to {misclassified_dir}")
        # --- End misclassified block ---

        console_print(self.console, f"\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        console_print(self.console, f"{cm}")
        console_print(self.console, f"\nROC-AUC: {roc_auc_score(y_true, y_pred_probs[:len(y_true)])}")

        # Save confusion matrix to CSV
        import pandas as pd
        cm_df = pd.DataFrame(cm, index=['O4-', 'O4+'], columns=['Predicted O4-', 'Predicted O4+'])
        cm_df.to_csv(os.path.join(root_folder, 'confusion_matrix.csv'))

        # Save confusion matrix as image
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(6,4))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(root_folder, 'confusion_matrix.png'))
        plt.close()

    def analyze_misclassifications(self):
        import os  # Ensure os is imported before use
        # Ensure test_dir is initialized before use
        self.setupLocalFolders(base_dir, empty=False)
        if self.trained_model is None:
            # Load model if not already loaded
            import keras
            root_folder = settings.experiments[self.experiment.get()]['root']
            model_path = os.path.join(root_folder, settings.kerasModel)
            self.trained_model = keras.models.load_model(model_path)

        import tensorflow as tf
        import shutil
        import numpy as np
        import glob

        from sklearn.metrics import confusion_matrix, roc_auc_score

        test_generator = tf.keras.utils.image_dataset_from_directory(
            self.test_dir,
            image_size=(128, 128),
            batch_size=64,
            label_mode='binary',
            shuffle=False
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

        y_pred_probs = self.trained_model.predict(test_generator)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = [y.numpy() for _, y in test_generator]
        y_true = tf.concat(y_true, axis=0)[:len(y_pred)]

        test_image_paths = sorted(glob.glob(os.path.join(self.test_dir, '*', '*.png')))
        root_folder = settings.experiments[self.experiment.get()]['root']
        misclassified_dir = os.path.join(root_folder, "misclassified")
        os.makedirs(misclassified_dir, exist_ok=True)

        for i, (yp, yt) in enumerate(zip(y_pred, y_true)):
            if yp != yt:
                src_path = test_image_paths[i]
                fname = os.path.basename(src_path)
                pred_label = "o4pos" if yp == 1 else "o4neg"
                true_label = "o4pos" if yt == 1 else "o4neg"
                dst_path = os.path.join(misclassified_dir, f"{true_label}_as_{pred_label}_{fname}")
                shutil.copy2(src_path, dst_path)

        console_print(self.console, f"\nSaved misclassified images to {misclassified_dir}")
        cm = confusion_matrix(y_true, y_pred)
        console_print(self.console, f"\nConfusion Matrix:\n{cm}")
        console_print(self.console, f"\nROC-AUC: {roc_auc_score(y_true, y_pred_probs[:len(y_true)])}")
# Starts application.
root = tk.Tk()
root.geometry('+100+100')
root.resizable(width=False, height=False)

# Bring window to front
root.lift()
root.attributes('-topmost', True)
root.focus_force()
root.after_idle(root.attributes, '-topmost', False)

app = Application(master=root)
app.mainloop()

