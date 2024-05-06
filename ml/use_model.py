# Mobility Counts Prediction

import pandas as pd
import os
import sys
import pickle

# Import custom modules
import module_ai
import module_data
import setup_funcs

def main(data_path):
    # Validate the provided data path
    if not os.path.isfile(data_path):
        print(f"The path {data_path} does not point to a valid file.")
        return
    
    # Initialize AI and data modules
    ai = module_ai.ai()
    source_data = module_data.data()

    # Normalization Functions - Set and do not change unless necessary
    norm_functions = ['tmc_norm', 'tstamp_norm', 'startdate_norm', 'density_norm', 'time_before_after']
    source_data.norm_functions = norm_functions

    # Columns to use
    # columns = [
    #     'tmc_code', 'measurement_tstamp', 'average_speed_All', 'speed_All', 'data_density_All',
    #     'data_density_Pass', 'data_density_Truck', 'travel_time_seconds_All', 'start_latitude',
    #     'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'aadt', 'urban_code',
    #     'thrulanes_unidir', 'f_system', 'route_sign', 'thrulanes', 'zip', 'Population_2022'
    # ]
    columns = [
        'tmc_code', 'measurement_tstamp', 'speed_All', 'data_density_All',
        'data_density_Pass', 'data_density_Truck', 'travel_time_seconds_All', 'start_latitude',
        'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'aadt', 'urban_code',
        'thrulanes_unidir', 'f_system', 'route_sign', 'thrulanes', 'zip'
    ]

    # Setup data source path
    source_data.OUTPUT_FILE_PATH = data_path

    use_custom_columns = True  # CHANGE TO FALSE IF NOT USING COLUMNS ABOVE

    # Check and handle previously normalized data
    norm_data_path = "../data/norm_data.pkl"
    if os.path.isfile(norm_data_path):
        print("Loading from cache (.pkl file)...")
        source_data.dataset = pickle.load(open(norm_data_path, "rb"))
    else:
        source_data.read()
        with open(norm_data_path, "wb") as file:
            pickle.dump(source_data.dataset, file)

    # Normalize the dataset
    normalized_df = source_data.normalized()

    # Use custom columns or the default set
    if use_custom_columns:
        columns.extend(source_data.calculated_columns)
    else:
        columns = source_data.features_training_set

    print(normalized_df)

    # Train and test the model
    result = setup_funcs.train_model(ai, normalized_df, columns, 'VOL')
    print("Testing final model...")
    print(setup_funcs.test_model(ai, normalized_df, columns, 'VOL'))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <data_file_path>")
    else:
        main(sys.argv[1])