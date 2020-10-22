class Settings:
    """A class to store all settings."""

    def __init__(self):
        """Initialize static settings."""
        
        # Define size of individual cell images
        self.width = 128
        self.height = 128

        # Define folders for input of data
        self.folder_dicts = [
            {
                # name: foldername        
                'name': 'Test',
                # root: path to folder
                'root': 'C:\\Users\\fjs21\\OneDrive\\Research&Lab\\Data\\Transwell Experiments 2020',
                # pattern: define file
                'pattern': '*.tif',
                # which channel is DAPI?
                'dapi_ch': 0,
                # which channel is O4?
                'o4_ch': None,
                # which channel is EdU?
                'EdU_ch': None,                
                # which marker counts the O4 counts
                'marker_index' : None
            },
            {
                # name: foldername        
                'name': 'Roopa September 2020 BK experiment',
                # root: path to folder
                'root': 'C:\\scratch\\sep 2020 sample',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 1,
                # which channel is O4?
                'o4_ch': 2,
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None
            },        
            {
                # name: foldername        
                'name': 'Jackie Transwell and IGFBP2 experiment',
                # root: path to folder
                'root': 'C:\\scratch\\Transwell + IGFBP2 Ab',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                # which channel is O4?
                'o4_ch': None,
                # which channel is EdU?
                'EdU_ch': 1,
                # which marker counts the O4 counts
                'marker_index' : None
            },
            {
                # name: foldername        
                'name': 'Roopa Feb 2020 BK experiment',
                # root: path to folder
                'root': 'Y:\\People\\Roopa\\dk si alpha expts\\feb 2020 sample',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 1,
                # which channel is O4?
                'o4_ch': 2,
                # which channel is EdU?
                'EdU_ch': None,                
                # which marker counts the O4 counts
                'marker_index' : None
            },  
            {
                # name: foldername        
                'name': 'Roopa July 31 2020 BK experiment',
                # root: path to folder
                'root': 'Y:\\People\\Roopa\\dk si alpha expts\\july 2020',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 1,
                # which channel is O4?
                'o4_ch': 2,
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None
            },          ]             

        # Define distance between centroid and marker cutoff over which cell will not be classified
        self.fD_cutoff = 50

        # Keras Model
        self.kerasModel = 'o4counter_wAug_5.1.h5'