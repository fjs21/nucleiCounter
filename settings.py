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
                # name: folder 0       
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
                'marker_index' : None,
            },
            {
                # name: folder 1        
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
                'marker_index' : None,
                # set gamma and treshold method
                'gamma': 0.5,
                'thres': 'th2',            },        
            {
                # name: folder 2        
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
                'marker_index' : None,
                # set gamma and treshold method
                'gamma': 1.0,
                'thres': 'th2',            },
            {
                # name: folder 3        
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
                'marker_index' : None,
                # set gamma and treshold method
                'gamma': 0.5,
                'thres': 'th2',            },  
            {
                # name: folder 4        
                'name': 'Roopa July 2020 BK experiment (Discard)',
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
                'marker_index' : None,
                # set gamma
                'gamma': 0.5,
                'thres': 'th2',
            },
            {
                # name: folder 5        
                'name': "Ahmed's transwell experiment",
                # root: path to folder
                'root': 'Y:\\People\\Ahmed\\Transwell Experiments\\Exp 3\\Proliferation',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                # which channel is O4?
                'o4_ch': None,
                # which channel is EdU?
                'EdU_ch': 1,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set gamma
                'gamma': 3.5,
                'thres': 'th2',
            },
            {
                # name: folder 6        
                'name': "Roopa July 31 2019 BK experiment",
                # root: path to folder
                'root': 'Y:\\People\\Roopa\\dk si alpha expts\\siBK 0731 sample',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 1,
                'dapi_gamma': 0.5,
                # which channel is O4?
                'o4_ch': 2,
                'o4_gamma': 0.2,
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold                
                'thres': 'th2',
            },
            {
                # name: folder 7        
                'name': "Test of DAPI/GFP transwell images",
                # root: path to folder
                # 'root': 'C:\\scratch\\Migration Assays',
                'root': 'C:\\Users\fjs21\\Box\\NewLabData\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(1) no A594',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                # which channel is O4?
                'o4_ch': None,
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set gamma
                'gamma': 1,
                'thres': 'th2',
            },
            {
                # name: folder 8       
                'name': "Migration Assay (Experiment 1) - DAPI/GFP/A594 transwell images",
                # root: path to folder
                # 'root': 'Y:\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(1)',
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(1)',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 1.0,                
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,                   
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },
            {
                # name: folder 9       
                'name': "Migration Assay (Experiment 2) - DAPI/GFP/A594 transwell images",
                # root: path to folder
                # 'root': 'Y:\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(2)',
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(2)',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 1.0,
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,                
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },                     
            {
                # name: folder 10
                'name': "Migration Assay (Experiment 3) - DAPI/GFP/A594 transwell images",
                # root: path to folder
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(3)',
                # 'root': 'Y:\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(3)',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 1.0,
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },
            {
                # name: folder 11
                'name': "Migration Assay (Experiment 4) - DAPI/GFP/A594 transwell images",
                # root: path to folder
                # 'root': 'Y:\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(4)',
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(4)',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 1.0,
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },
           {
                # name: folder 12
                'name': "Migration Assay (Experiment 5) - DAPI/GFP/A594 transwell images",
                # root: path to folder
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(5)',
                # 'root': 'Y:\\People\\Jackie\\Migration Assays (SULF2 paper)\\Migration Assay(5)',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 1.0,
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,
                # which channel is EdU?
                'EdU_ch': None,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },            
           {
                # name: folder 13
                'name': "Transwell experiment Jan 2021",
                # root: path to folder
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\Transwell Exps_2021\\Expt 1',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 1.0,
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,
                # which channel is EdU?
                'EdU_ch': 1,
                'EdU_gamma': 1.0,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },           
           {
                # name: folder 14
                'name': "Transwell experiment Feb 2021",
                # root: path to folder
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\Transwell Exps_2021\\Expt 2',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 1.0,
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,
                # which channel is EdU?
                'EdU_ch': 1,
                'EdU_gamma': 1.0,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },
           {
                # name: folder 14
                'name': "IGFBP2 antibody experiment",
                # root: path to folder
                'root': 'C:\\Users\\fjs21\\Box\\NewLabData\\People\\Jackie\\IGFBP2',
                # pattern: define file
                'pattern': '*.vsi',
                # which channel is DAPI?
                'dapi_ch': 0,
                'dapi_gamma': 0.8,
                # which channel is O4?
                'o4_ch': None,
                'o4_gamma': 1,
                # which channel is EdU?
                'EdU_ch': 1,
                'EdU_gamma': 1.0,
                # which marker counts the O4 counts
                'marker_index' : None,
                # set threshold method
                'thres': 'th2',
                # settings for autoFL channel
                'autoFL_dilate': False,
                'autoFL_gamma': 1,
            },
        ]
        # Define distance between centroid and marker cutoff over which cell will not be classified
        self.fD_cutoff = 50

        # Keras Model
        self.kerasModel = 'o4counter_wAug_5.1.h5'