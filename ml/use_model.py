# Mobility Counts Prediction
# 
import pandas as pd

# Import custom modules
import module_ai
import module_data
import setup_funcs
import pickle
import os

def check_valid_file_path_env_var(var_name):
    # Get the value of the environment variable
    path = os.getenv(var_name)

    # Check if the variable is set and has data
    if path:
        # Check if the path is a valid file
        if os.path.isfile(path):
            return True, path
        else:
            print(f"The path {path} does not point to a valid file.")
            return False, None
    else:
        print(f"{var_name} is not set or is empty.")
        return False, None

# Don't run without knowing where the data is located! Set environment variable using "setx" in Windows, or "export" in linux.
data_path_is_set, data_path = check_valid_file_path_env_var("TRIMS_DATA_SOURCE")
if data_path_is_set:

    # init ai module
    ai = module_ai.ai()

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

    source_data.OUTPUT_FILE_PATH = str(data_path)

    use_custom_cols = True # CHANGE TO TRUE IF USING COLUMNS ABOVE, OTHERWISE USING COLUMNS FROM module_data
    #use_pkl = True
	use_pt = True

	'''
    # IF USING UPDATED DATASET (any new columns calculated or selected)
    # make sure to delete existing norm_data.pkl otherwise old data will be loaded
    if use_pkl and os.path.isfile("../norm_data.pkl"):
        print("Loaded .pkl file")
        source_data.dataset = pickle.load(open("../data/norm_data.pkl", "rb"))
    else:
        source_data.read()
        import pickle
        pickle.dump(source_data.dataset, open("../data/norm_data.pkl", "wb"))
	'''

	# IF USING UPDATED DATASET (any new columns calculated or selected)
    # make sure to delete existing norm_data.pkl otherwise old data will be loaded
    if use_pt and os.path.isfile("../norm_data.pt"):
        print("Loaded .pt file")
        source_data.dataset = pickle.load(open("../data/norm_data.pt", "rb"))
    else:
        source_data.read()
        import pickle
        pickle.dump(source_data.dataset, open("../data/norm_data.pt", "wb"))

    normalized_df = source_data.normalized()
    # If using custom columns, add the calculated columns to the custom cols1
    # Otherwise, set columns equal to features training data set that already contains calculated_columns
    if (use_custom_cols):
        cols1.extend(source_data.calculated_columns)
    else :
        cols1 = source_data.features_training_set
    print(normalized_df)

    # print(source_data.calculated_columns)
    # print(cols1)

    result = setup_funcs.train_model(ai, normalized_df, cols1, 'VOL')

    # Test the final model
    print("Testing final model...")
    print(setup_funcs.test_model(ai, normalized_df, cols1, 'VOL'))

    # TODO: calculate average percent diff between predictions and y_test 
