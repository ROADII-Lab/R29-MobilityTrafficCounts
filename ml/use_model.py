# Mobility Counts Prediction
# 
import pandas as pd

# Import custom modules
import module_ai
import module_data
import setup_funcs
import pickle
import os

# init ai module
ai = module_ai.ai()
ai.model_filename_root = 'C:/Users/William.Chupp/OneDrive - DOT OST/Documents/ROADII-DATAPROGRAM/R29-MobilityCounts/'

# init data module
source_data = module_data.data()
# Normalization Functions - DO NOT CHANGE FOR FUTURE USE
source_data.norm_functions = ['tmc_norm', 'tstamp_norm', 'startdate_norm', 'density_norm', 'time_before_after']

# setup data sources
cols1 = ['tmc_code',
         'measurement_tstamp',
        'average_speed_All',
        'speed_All',
        'data_density_All',
        'data_density_Pass',
        'data_density_Truck', 
        'travel_time_seconds_All',
        'start_latitude',
        'start_longitude',
        'end_latitude',
        'end_longitude',
        'miles',
        'aadt',
        'urban_code',
        'thrulanes_unidir',
        'f_system',
        'route_sign',
        'thrulanes',
        'zip',
        'Population_2022'
        ]

source_data.OUTPUT_FILE_PATH = r'C:/Users/William.Chupp/OneDrive - DOT OST/Documents/ROADII-DATAPROGRAM/R29-MobilityCounts/JOINED_FILES/NPMRDS_TMC_TMAS_NE_C.csv'

use_custom_cols = True # CHANGE TO TRUE IF USING COLUMNS ABOVE, OTHERWISE USING COLUMNS FROM module_data
use_pkl = True

# IF USING UPDATED DATASET (any new columns calculated or selected)
# make sure to delete existing norm_data.pkl otherwise old data will be loaded
if use_pkl and os.path.isfile("../norm_data.pkl"):
    print("Loaded .pkl file")
    source_data.dataset = pickle.load(open("../norm_data.pkl", "rb"))
else:
    source_data.read()
    import pickle
    pickle.dump(source_data.dataset, open("../norm_data.pkl", "wb"))

normalized_df = source_data.normalized()
# If using custom columns, add the calculated columns to the custom cols1
# Otherwise, set columns equal to features training data set that already contains calculated_columns
if (use_custom_cols):
    cols1.extend(source_data.calculated_columns)
else :
    cols1 = source_data.features_training_set
print(normalized_df)

print(source_data.calculated_columns)
print(cols1)

result = setup_funcs.train_model(ai, normalized_df, cols1, 'VOL')

# Test the final model
print("Testing final model...")
print(setup_funcs.test_model(ai, normalized_df, cols1, 'VOL'))

# TODO: calculate average percent diff between predictions and y_test 