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

# init data module
source_data = module_data.data()
# Normalization Functions - DO NOT CHANGE FOR FUTURE USE
source_data.norm_functions = ['tmc_norm', 'tstamp_norm', 'startdate_norm', 'density_norm', 'time_before_after']

# setup data sources
cols1 = ['measurement_tstamp',
        'average_speed_All',
        # 'average_speed_All_before',
        # 'average_speed_All_after',
        'speed_All',
        'speed_All_before',
        'speed_All_after',
        'travel_time_seconds_All',
        # 'travel_time_seconds_All_before',
        # 'travel_time_seconds_All_after',
        'start_latitude',
        'start_longitude',
        'end_latitude',
        'end_longitude',
        'miles',
        # 'miles_before',
        # 'miles_after',
        'aadt',
        'thrulanes_unidir',
        'f_system',
        'route_sign',
        'thrulanes',
        'zip',
        'Population_2022'
        ]

source_data.OUTPUT_FILE_PATH = r'../data/NPMRDS_TMC_TMAS_NE_C.csv'

use_custom_cols = False # CHANGE TO TRUE IF USING COLUMNS ABOVE, OTHERWISE USING COLUMNS FROM module_data


# IF USING UPDATED DATASET (any new columns calculated or selected)
# make sure to delete existing norm_data.pkl otherwise old data will be loaded
if os.path.isfile("../data/norm_data.pkl"):
    print("Loaded .pkl file")
    normalized_df = pickle.load(open("../data/norm_data.pkl", "rb"))
else:
    result_df = source_data.read()
    normalized_df = source_data.normalized()
    import pickle
    pickle.dump(normalized_df, open("../data/norm_data.pkl", "wb"))

# If using custom columns, add the calculated columns to the custom cols1
# Otherwise, set columns equal to features training data set that already contains calculated_columns
if (use_custom_cols):
    cols1.extend(source_data.calculated_columns)
else :
    cols1 = source_data.features_training_set
print(normalized_df)
result = setup_funcs.train_model(ai, normalized_df, cols1, 'VOL')

# Test the final model
print("Testing final model...")
print(setup_funcs.test_model(ai, normalized_df, cols1, 'VOL'))

# TODO: calculate average percent diff between predictions and y_test 