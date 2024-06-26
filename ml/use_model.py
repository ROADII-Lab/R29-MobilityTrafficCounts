# Mobility Counts Prediction

import pandas as pd
import os
import sys
import pickle
import argparse

# Import custom modules
import module_ai
import module_data
import setup_funcs

def main(data_directory, train, model):
    # Validate the provided data directory
    if not os.path.isdir(data_directory):
        print(f"The path {data_directory} does not point to a valid directory.")
        return
    
    # Initialize AI and data modules
    ai = module_ai.ai()
    source_data = module_data.data()

    # Normalization Functions - Set and do not change unless necessary
    norm_functions = ['tmc_norm', 'tstamp_norm', 'density_norm', 'time_before_after']
    source_data.norm_functions = norm_functions

    # Columns to use
    columns = [
        'tmc_code', 'measurement_tstamp', 'speed_All', 'data_density_All',
        'data_density_Pass', 'data_density_Truck', 'travel_time_seconds_All', 'start_latitude',
        'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'aadt', 'urban_code',
        'thrulanes_unidir', 'f_system', 'route_sign', 'thrulanes', 'zip', 'DIR'
    ]

    # Load all pickle files in the directory
    print("Loading data...")
    dataframes = []
    for file_name in os.listdir(data_directory):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(data_directory, file_name)
            with open(file_path, "rb") as file:
                data = pickle.load(file)
                dataframes.append(data)
    
    if not dataframes:
        print("No pickle files found in the provided directory.")
        return

    # Combine all dataframes into one
    source_data.dataset = pd.concat(dataframes, ignore_index=True)

    # Normalize the dataset
    normalized_df = source_data.normalized()

    # Use custom columns or the default set
    use_custom_columns = True  # CHANGE TO FALSE IF NOT USING COLUMNS ABOVE
    if use_custom_columns:
        columns.extend(source_data.calculated_columns)
    else:
        columns = source_data.features_training_set

    print(normalized_df)

    if train:
        # Train and test the model
        result = setup_funcs.train_model(ai, normalized_df, columns, 'VOL')
        print("Testing final model...")
        print(setup_funcs.test_model(ai, normalized_df, columns, 'VOL'))
    else:
        # Run inference using the data and model provided
        if model:
            print(setup_funcs.use_model(ai, model, normalized_df, columns, 'VOL'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mobility Counts Prediction Script')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--data', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model', type=str, required=False, help='Path to the model file')

    args = parser.parse_args()

    main(args.data, args.train, args.model)
