# Nuclei Counter: Automated Analysis of In Vitro Images

This project provides tools to analyze microscopy images of in vitro experiments, including detection and quantification of various markers (e.g., DAPI, O4, EdU, Olig2, mCherry, Gfap) and model training using Keras.

---

## 📁 Project Structure

### `nucleiCounter.py`
- **GUI-based image processing tool**
- Allows batch processing of image folders
- Detects nuclei, counts marker-positive cells
- Produces summary CSV and annotated PDF output
- Loads experiment parameters from `settings.json`
- Includes debug mode for visualization and development

Usage:
```bash
python nucleiCounter.py
```

### `manualAnnotation.py`

Manual image annotation tool for O4 cell classification.

This script launches a Tkinter-based GUI to manually review and correct automated classifications of cell images (O4+, O4–, or unclassified). It operates on images in the `keras/` subdirectory and updates an `annotations.json` file associated with the selected experiment.

#### Features
- Interactive GUI to select the experiment and cell types (O4+, O4–, unclassified)
- Annotates each image via keyboard input:
  - **Y** → mark as O4-positive
  - **N** → mark as O4-negative
  - **Q** → quit annotation session
- Image filenames are updated accordingly (e.g., renames `unknown` to `o4pos`)
- Resized, gamma-corrected image display with cell of interest marked

#### Usage
```bash
python manualAnnotation.py
```
### `processFolderForModelTraining.py`

GUI-based script for preprocessing microscope images and exporting single-cell crops and metadata for model training.

This tool reads experimental image folders, processes DAPI and O4 channels, applies cell segmentation, and associates optional marker annotations if present. It ultimately exports a training-ready set of cell images and a corresponding `annotations.json` metadata file.

#### Features
- Graphical user interface to:
  - Select experiment configuration
  - Toggle debug mode for testing on small image subsets
- Image processing pipeline includes:
  - DAPI-based nucleus detection
  - O4 channel processing
  - Marker detection if XML annotation files are available
  - Cell-wise classification as O4+, O4–, or unknown
- Supports export of:
  - Individual cell images into `keras/` directory
  - Metadata in `annotations.json` for use in training
- Plots distribution of marker-to-cell distances if marker data are available

#### Output Format
Each cell is exported as a `.tif` file named with its predicted label:
- `o4pos.N.tif` for O4-positive cells
- `o4neg.N.tif` for O4-negative cells
- `unknown.N.tif` for unclassified cells

The `annotations.json` contains structured metadata for each cell:
```json
{
  "cell": "o4pos.0.tif",
  "path": "...",                  // original folder
  "imgFile": "...",              // original composite image
  "markerFile": "...",           // XML marker file if used
  "cellIndex": 4,                // index of cell in parent image
  "centroid": [x, y],            // XY coordinates
  "classification": 1           // 0 = O4-, 1 = O4+, -1 = unknown
}
```

### `trainLocalModel.py`

GUI-based script to train a Keras convolutional neural network (CNN) for classifying O4+ versus O4− cells based on labeled cell images.

#### Features
- Graphical user interface (GUI) for:
  - Selecting experiment and annotation type
  - Setting validation/test image count
  - Configuring CNN hyperparameters (filters, dropout)
  - Viewing output and training progress
- Converts `.tif` images to `.png` as needed
- Loads annotations from `annotations.json`
- Splits data into training, validation, and test sets
- Dynamically builds CNN based on GUI parameters
- Applies data augmentation and class balancing
- Evaluates model using confusion matrix and ROC-AUC
- Saves results as CSV, PNG, and trained model files

#### GUI Parameters
- **Experiment Name**: selects the experiment folder from `settings.py`
- **Annotation Type**: choose `classification` or `annotation` from JSON
- **Validation/Test Size**: minimum number of images per class
- **Conv Layer Filters**: comma-separated values (e.g., `32,64,128,128`)
- **Dropout Rate**: float between 0 and 1 (e.g., `0.5`)

#### Outputs
- Model saved to experiment folder (HDF5 format)
- Confusion matrix:
  - As CSV: `confusion_matrix.csv`
  - As PNG image: `confusion_matrix.png`
- Training history: `KerasModelFit_results_<modelname>.csv`

#### Usage
Run from command line:
```bash
python trainLocalModel.py
```
