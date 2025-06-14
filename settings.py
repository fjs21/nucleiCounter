import json
import numpy as np


class Settings:
    """A class to store all settings."""
    def __init__(self):
        """Initialize static settings."""

        # Define size of individual cell images
        self.parameters = None
        self.width = 128
        self.height = 128

        self.dotColors = ['orange', 'fuchsia']

        # Define distance between centroid and marker cutoff over which cell will not be classified
        self.fD_cutoff = 50

        # Keras Model
        self.kerasFolder = 'kerasFolder'
        self.kerasModel = 'o4counter_model.h5'

        # Settings for thresholdSegmentation (used for EdU, DAPI is hard coded for now)
        self.opening_kernel = np.ones((3, 3), np.uint8)
        self.opening_iterations = 3
        self.background_kernel = np.ones((3, 3), np.uint8)
        self.background_iterations = 3

        # Defaults
        filename = "defaults.json"
        try:
            with open(filename) as f:
                self.defaults = json.load(f)
            # print(self.defaults)
            if 'name' not in self.defaults:
                self.defaults['name'] = 'default'

        except FileNotFoundError:
            print("Defaults not found")
            self.defaults = {
                "name": "temp",
                "root": "/Users/frasersim/Desktop/2022-03-04 (Evans OKN & Noggin plate)",
                "pattern": "*.vsi",
                "dapi_ch": 0,
                "o4_ch": -1,
                "edu_ch": -1,
                "gfap_ch": -1,
                "dapi_gamma": 1.0,
                "o4_gamma": 1.0,
                "edu_gamma": 1.0,
                "gfap_th": 1000,
                "scalefactor": 1.0,
                "prediction_cutoff": 0.5,
                "debug": True}

        # Load experiments
        filename = "experiments.json"
        # load existing
        try:
            with open(filename, "r") as f:
                self.experiments = json.load(f)
        except FileNotFoundError:
            print(f"The file '{filename}' was not found.")
            self.experiments = {}

    def updateDefaults(self,
                       name: str,
                       root: str,
                       pattern: str,
                       dapi_ch: int,
                       o4_ch: int = -1,
                       edu_ch: int = -1,
                       olig2_ch: int = -1,
                       mCherry_ch: int = -1,
                       gfap_ch: int = -1,
                       dapi_gamma: float = 1.0,
                       o4_gamma: float = 1.0,
                       edu_gamma: float = 1.0,
                       olig2_gamma: float = 1.0,
                       mCherry_gamma: float = 1.0,
                       gfap_th: int = 1000,
                       scalefactor: float = 1.0,
                       prediction_cutoff: float = 0.5,
                       debug: bool = False):
        """ Save defaults to temporary file. """

        self.parameters = {
            "name": name,
            "root": root,
            "pattern": pattern,
            "dapi_ch": dapi_ch,
            "o4_ch": o4_ch,
            "edu_ch": edu_ch,
            "olig2_ch": olig2_ch,
            "mCherry_ch": mCherry_ch,
            "gfap_ch": gfap_ch,
            "dapi_gamma": dapi_gamma,
            "o4_gamma": o4_gamma,
            "edu_gamma": edu_gamma,
            "olig2_gamma": olig2_gamma,
            "mCherry_gamma": mCherry_gamma,
            "gfap_th": gfap_th,
            "scalefactor": scalefactor,
            "prediction_cutoff": prediction_cutoff,
            "debug": debug
        }

        filename = "defaults.json"
        with open(filename, "w") as f:
            json.dump(self.parameters, f)
        print("New defaults saved")

    def saveExperimentalParameters(self):
        """ This saves the experimental parameters for future use."""
        # replace existing or add new
        if self.parameters['name'] in self.experiments:
            print("Experiment already exists, updating...")

        else:
            print("Experiment does not yet exist, adding...")

        self.experiments[self.parameters['name']] = self.parameters
        # export
        filename = "experiments.json"
        with open(filename, "w") as f:
            json.dump(self.experiments, f)
