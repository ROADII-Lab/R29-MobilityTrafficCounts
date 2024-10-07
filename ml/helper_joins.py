import pandas as pd
import pickle
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import for time manipulation
from matplotlib.ticker import ScalarFormatter
import os
import module_data
import re

def run_joins():
    # init data module
    source_data = module_data.data()
    source_data.join_and_save()

def TMAS_to_pkl():
    # File path
    TMAS_path = r"C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\TMAS_Class_Clean_2022.csv"
    dtype_dict = {
                'STATE': int,
                'STATION_ID': str,
                'DIR': float,  
                'DATE': str,  
                'YEAR': int,
                'MONTH': int,
                'DAY': int,  
                'HOUR': int,
                'DAY_TYPE': str,
                'PEAKING': str,
                'VOL': float,
                'F_SYSTEM': float,
                'URB_RURAL': str,
                'COUNTY': int,
                'REPCTY': str,
                'ROUTE_SIGN': float,
                'ROUTE_NUMBER': str,
                'LAT': float,
                'LONG': float,
                'STATE_NAME': str,
                'COUNTY_NAME': str,
                'HPMS_TYPE10': float,
                'HPMS_TYPE25': float,
                'HPMS_TYPE40': float,
                'HPMS_TYPE50': float,
                'HPMS_TYPE60': float,
                'HPMS_ALL': float,
                'NOISE_AUTO': float,
                'NOISE_MED_TRUCK': float,
                'NOISE_HVY_TRUCK': float,
                'NOISE_BUS': float,
                'NOISE_MC': float,
                'NOISE_ALL': float
            }
    TMAS_DATA = pd.read_csv(TMAS_path, dtype= dtype_dict)
    print("csv read")
    numstations = TMAS_DATA['STATION_ID'].nunique(dropna=True)
    print(numstations)

    pickle.dump(TMAS_DATA, open(r"C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\TMAS_Class_Clean_2022.pkl", "wb"))
    print("dumped to pkl")

def plotting_dataset():
    print('loading data')
    DATASET_PATH =  r'C:\Users\Michael.Barzach\OneDrive - DOT OST\R29-MobilityCounts\JOINED_FILES/NPMRDS_TMC_TMAS_US_SUBSET_1000_22.pkl'

    if DATASET_PATH.endswith('.pkl'):
        output = pickle.load(open(DATASET_PATH, "rb"))
    else:
        output = pd.read_csv(DATASET_PATH, low_memory=False)
    print("data loaded")

    # Selecting a subset of the data if required
    # Change value in ADDITIONAL_SUBSET and STATES to reflect subset you want
    # If ADDITIONAL_SUBSET is empty, no subset is taken
    ADDITIONAL_SUBSET = ''
    STATES = ['RHODE ISLAND', 'VERMONT', 'NEW HAMPSHIRE', 'CONNECTICUT', 'MAINE', 'MASSACHUSETTS', 'NEW YORK', 'DELAWARE', 'PENNSYLVANIA', 'NEW JERSEY']
    if ADDITIONAL_SUBSET != '':
        output = output[output['State'].isin(STATES)]
        SUBSET_TEXT = f'Selected Subset: {ADDITIONAL_SUBSET}, '
    else:
        SUBSET_TEXT = ''

    # Ensure measurement_tstamp is a datetime object
    output['measurement_tstamp'] = pd.to_datetime(output['measurement_tstamp'])

    # Create a figure for the plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Histogram of all volume counts for the year
    axes[0].hist(output['VOL'], bins=30, alpha=0.7)
    axes[0].set_yscale('log')
    scalar_formatter = ScalarFormatter()
    scalar_formatter.set_scientific(False)
    axes[0].yaxis.set_major_formatter(scalar_formatter)
    axes[0].set_title('Annual Volume Histogram')
    axes[0].set_xlabel('Volume (VOL)')
    axes[0].set_ylabel('Count')

    # Plot 2: Hourly averages for each day
    for day, group_data in output.groupby(output['measurement_tstamp'].dt.dayofweek):
        hourly_avg = group_data.groupby(group_data['measurement_tstamp'].dt.hour)['VOL'].mean()
        axes[1].plot(hourly_avg.index, hourly_avg.values, marker='o', linestyle='-', label=f'Day {day+1}')
    axes[1].set_xlabel('Hour of the Day')
    axes[1].set_ylabel('Average Volume')
    axes[1].set_title('Avg Hourly Volume by Day')
    axes[1].set_xticks(range(0, 24, 4))
    axes[1].legend(loc='upper left')

    # Plot 3: Max volume per station
    max_volume_per_station = output.groupby('STATION_ID')['VOL'].max()
    axes[2].hist(max_volume_per_station, bins=30, alpha=0.7)
    axes[2].set_xlabel('Max Volume')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Max Volume per Station')
    axes[2].set_yscale('log')
    axes[2].yaxis.set_major_formatter(scalar_formatter)

    plt.tight_layout()
    plt.show()
    '''
    max_volume_per_station.rename(columns={'VOL': 'Max_VOL'}, inplace=True)  # Correctly renaming the column to 'Max_VOL'
    # Extract rows for each station ID where the volume equals the maximum volume found
    top_stations = pd.merge(output, max_volume_per_station, how='inner', left_on=['STATION_ID', 'VOL'], right_on=['STATION_ID', 'Max_VOL'])
    top_stations = top_stations[['STATION_ID', 'VOL', 'County', 'State', 'road', 'intersection', 'measurement_tstamp']].sort_values(by='VOL', ascending=False).head(10)

    # Save to CSV
    csv_path = r'../data/US_Subset/Top_Stations_Max_Volume_Details.csv'
    top_stations.to_csv(csv_path, index=False)
    print(f"Top stations data saved to '{csv_path}'")
    '''
def QAQC_Joins():

    print('loading data')
    DATASET_PATH =  r'C:\Users\Michael.Barzach\OneDrive - DOT OST\R29-MobilityCounts\JOINED_FILES/NPMRDS_TMC_TMAS_US_SUBSET_1000_22.pkl'
    # Load the data
    #handle input data file differently if .pkl or .csv
    if (DATASET_PATH.endswith('.pkl')):
        output = pickle.load(open(DATASET_PATH, "rb"))
    else:
        output = pd.read_csv(DATASET_PATH, low_memory=False)
    print("data loaded")
    
    prejoin = pickle.load(open(r'../data/prejoin.pkl', "rb"))
    #tmas = pickle.load(open(r'C:\Users\Michael.Barzach\Documents\ROADII\TMAS_Class_Clean_2021.pkl', "rb"))
    print('loaded test pkl files')
    goutput = output.groupby(['tmc_code'])
    gprejoin = prejoin.groupby(['tmc_code'])
    print('Number of unique TMCs before Join: ' + str(len(gprejoin)))
    print('Number of unique TMCs after join: ' + str(len(goutput)))

    
    goutput = output.groupby(['tmc_code', 'measurement_tstamp'])
    gprejoin = prejoin.groupby(['tmc_code', 'measurement_tstamp'])

    print('Number of unique TMCs/times before Join: ' + str(len(gprejoin)))
    print('Number of unique TMCs/times after join: ' + str(len(goutput)))
    

    prejoin_station_ids = set(prejoin['STATION_ID'])
    output_station_ids = set(output['STATION_ID'])

    missing_stations = prejoin_station_ids - output_station_ids

    print(f"{len(missing_stations)} stations missing in the output dataframe: " + str(missing_stations))

def generate_available_stations():

    # Convert TMAS to pkl
    TMAS_to_pkl()
    # Define the path to your pickle file
    pickle_filepath = r"C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\TMAS_Class_Clean_2022.pkl"

    # Define the desired output CSV filepath (replace with your chosen path)
    output_filepath = r"C:\Users\Michael.Barzach\Documents\ROADII\TMAS Data\tmas22_unique.csv"

    # Load data from the pickle file
    with open(pickle_filepath, 'rb') as f:
        data = pickle.load(f)

    # check if 'STATION_ID' and 'STATE_NAME' exist
    if not all(col in data.columns for col in ['STATION_ID', 'STATE_NAME']):
        raise ValueError("Missing required columns: 'STATION_ID' and 'STATE_NAME'")

    # Ensure 'STATION_ID' is a string type
    data['STATION_ID'] = data['STATION_ID'].astype(str)

    # Group by station ID and state name, get size (count) and reset index
    unique_combos = data.groupby(['STATION_ID', 'STATE_NAME']).size().to_frame(name='count').reset_index()

    # Select only 'STATION_ID' and 'STATE_NAME' columns
    selected_columns = ['STATION_ID', 'STATE_NAME']
    unique_combos = unique_combos[selected_columns]  # Select desired columns

    # Save unique combinations to CSV
    unique_combos.to_csv(output_filepath, index=False)

    print(f"Successfully saved unique combinations of 'STATION_ID' and 'STATE_NAME' to {output_filepath}")

def load_pkl():

    # Load the data from the pickle file
    data = pickle.load(open(r'C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\NPMRDS_TMC_TMAS_US_SUBSET_20_22.pkl', "rb"))
    data2 = pickle.load(open(r'C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\NPMRDS_TMC_TMAS_US_SUBSET_HOALL_22.pkl', "rb"))

    breakpoint()

    # Create a regex pattern to match '104' followed by any character, and '04680'
    pattern = re.compile(r'(104.04680)')

    # Apply the filter using the regex pattern
    filtered_data = data[data['tmc_code'].apply(lambda x: bool(pattern.search(str(x))))]

    # Ensure that filtered_data is not empty
    if not filtered_data.empty:
        # Replace '04680' with '04679', keeping the character intact
        filtered_data['tmc_code'] = filtered_data['tmc_code'].apply(lambda x: re.sub(r'04680', '04679', str(x)))

        # Replace all values in the 'DIR' column with 3
        filtered_data['DIR'] = 3

        # Save the modified data back to a pickle file
        output_path = r'C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\modified_data.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(filtered_data, f)

        print("Data saved to", output_path)
    else:
        print("No data matched the filtering criteria.")

def generate_metrics_for_predictions_pkl():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import os

    # Prompt user for the path to save the output text file
    output_file_path = input("Please enter the path where you want to save the output text file (e.g., C:\\output.txt): ")

    # Open the output file
    with open(output_file_path, 'w') as output_file:
        # Redirect print statements to both console and file
        def print_and_write(message):
            print(message)
            output_file.write(message + '\n')

        # Load data
        pkl_file_path = r'C:\Users\Michael.Barzach\Documents\GitHub\R29-MobilityTrafficCounts\data\NPMRDS_TMC_TMAS_US_SUBSET_HOALL_22_predictions.pkl'
        data = pickle.load(open(pkl_file_path, "rb"))

        # Ensure data is a DataFrame
        df = pd.DataFrame(data)

        # Ensure 'measurement_tstamp' is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['measurement_tstamp']):
            df['measurement_tstamp'] = pd.to_datetime(df['measurement_tstamp'])

        # Check if required columns exist
        required_columns = ['VOL', 'Predicted_VOL', 'measurement_tstamp', 'tmc_code']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in DataFrame: {missing_columns}")

        # Convert 'VOL' and 'Predicted_VOL' to numeric
        df['VOL'] = pd.to_numeric(df['VOL'], errors='coerce')
        df['Predicted_VOL'] = pd.to_numeric(df['Predicted_VOL'], errors='coerce')

        # Count zeros in 'VOL' and 'Predicted_VOL'
        zeros_in_vol = (df['VOL'] == 0).sum()
        zeros_in_pred_vol = (df['Predicted_VOL'] == 0).sum()
        print_and_write(f"Number of zeros in 'VOL': {zeros_in_vol}")
        print_and_write(f"Number of zeros in 'Predicted_VOL': {zeros_in_pred_vol}")

        # Rows where either 'VOL' or 'Predicted_VOL' is zero
        zeros_in_either = df[(df['VOL'] == 0) | (df['Predicted_VOL'] == 0)].copy()
        num_rows_with_zeros = len(zeros_in_either)
        print_and_write(f"Number of rows with zero in either 'VOL' or 'Predicted_VOL': {num_rows_with_zeros}")

        # Compute average and median absolute difference for rows with zero in either column
        zeros_in_either['Absolute_Difference'] = (zeros_in_either['Predicted_VOL'] - zeros_in_either['VOL']).abs()
        avg_absolute_difference = zeros_in_either['Absolute_Difference'].mean()
        median_absolute_difference = zeros_in_either['Absolute_Difference'].median()
        print_and_write(f"Average absolute difference for rows with zero in 'VOL' or 'Predicted_VOL': {avg_absolute_difference:.2f}")
        print_and_write(f"Median absolute difference for rows with zero in 'VOL' or 'Predicted_VOL': {median_absolute_difference:.2f}")

        # Analysis for zeros in 'VOL'
        zeros_vol_df = df[df['VOL'] == 0]
        tmc_codes_with_zeros_vol = zeros_vol_df['tmc_code'].unique()
        num_tmc_codes_with_zeros_vol = len(tmc_codes_with_zeros_vol)
        print_and_write(f"Number of tmc_codes with zeros in 'VOL': {num_tmc_codes_with_zeros_vol}")

        # Calculate percentage of zero counts per tmc_code in 'VOL'
        percent_zero_counts_vol = []
        tmc_codes_vol = []
        for tmc in tmc_codes_with_zeros_vol:
            tmc_df = df[df['tmc_code'] == tmc]
            zero_count = (tmc_df['VOL'] == 0).sum()
            total_count = len(tmc_df)
            percent_zero = (zero_count / total_count) * 100
            percent_zero_counts_vol.append(percent_zero)
            tmc_codes_vol.append(tmc)

        avg_percent_zero_vol = np.mean(percent_zero_counts_vol)
        min_percent_zero_vol = np.min(percent_zero_counts_vol)
        max_percent_zero_vol = np.max(percent_zero_counts_vol)
        max_percent_zero_vol_tmc = tmc_codes_vol[np.argmax(percent_zero_counts_vol)]

        print_and_write("\n'VOL' Zero Count Analysis:")
        print_and_write(f"Average percentage of zero counts across tmc_codes with zeros in 'VOL': {avg_percent_zero_vol:.2f}%")
        print_and_write(f"Minimum percentage of zero counts across tmc_codes with zeros in 'VOL': {min_percent_zero_vol:.2f}%")
        print_and_write(f"Maximum percentage of zero counts across tmc_codes with zeros in 'VOL': {max_percent_zero_vol:.2f}%")
        print_and_write(f"tmc_code with maximum percentage of zero counts in 'VOL': {max_percent_zero_vol_tmc}")

        # Analysis for zeros in 'Predicted_VOL'
        zeros_pred_vol_df = df[df['Predicted_VOL'] == 0]
        tmc_codes_with_zeros_pred_vol = zeros_pred_vol_df['tmc_code'].unique()
        num_tmc_codes_with_zeros_pred_vol = len(tmc_codes_with_zeros_pred_vol)
        print_and_write(f"\nNumber of tmc_codes with zeros in 'Predicted_VOL': {num_tmc_codes_with_zeros_pred_vol}")

        # Handle case when there are zeros in 'Predicted_VOL'
        if num_tmc_codes_with_zeros_pred_vol > 0:
            percent_zero_counts_pred_vol = []
            tmc_codes_pred_vol = []
            for tmc in tmc_codes_with_zeros_pred_vol:
                tmc_df = df[df['tmc_code'] == tmc]
                zero_count = (tmc_df['Predicted_VOL'] == 0).sum()
                total_count = len(tmc_df)
                percent_zero = (zero_count / total_count) * 100
                percent_zero_counts_pred_vol.append(percent_zero)
                tmc_codes_pred_vol.append(tmc)

            avg_percent_zero_pred_vol = np.mean(percent_zero_counts_pred_vol)
            min_percent_zero_pred_vol = np.min(percent_zero_counts_pred_vol)
            max_percent_zero_pred_vol = np.max(percent_zero_counts_pred_vol)
            max_percent_zero_pred_vol_tmc = tmc_codes_pred_vol[np.argmax(percent_zero_counts_pred_vol)]

            print_and_write("\n'Predicted_VOL' Zero Count Analysis:")
            print_and_write(f"Average percentage of zero counts across tmc_codes with zeros in 'Predicted_VOL': {avg_percent_zero_pred_vol:.2f}%")
            print_and_write(f"Minimum percentage of zero counts across tmc_codes with zeros in 'Predicted_VOL': {min_percent_zero_pred_vol:.2f}%")
            print_and_write(f"Maximum percentage of zero counts across tmc_codes with zeros in 'Predicted_VOL': {max_percent_zero_pred_vol:.2f}%")
            print_and_write(f"tmc_code with maximum percentage of zero counts in 'Predicted_VOL': {max_percent_zero_pred_vol_tmc}")
        else:
            print_and_write("\nNo zeros found in 'Predicted_VOL'. Skipping zero count analysis for 'Predicted_VOL'.")

        # Exclude rows where 'VOL' or 'Predicted_VOL' is zero for percent difference calculations
        df_non_zero = df[(df['VOL'] != 0) & (df['Predicted_VOL'] != 0)].copy()
        total_non_zero_rows = len(df_non_zero)
        print_and_write(f"\nTotal number of rows used for percent difference calculations: {total_non_zero_rows}")

        # Function to calculate performance metrics
        def calculate_performance_metrics(df_input, description):
            """
            Calculate performance metrics comparing 'VOL' and 'Predicted_VOL' in the dataframe.
            """
            # Calculate absolute percent difference
            df_input['Percent_Difference'] = ((df_input['Predicted_VOL'] - df_input['VOL']).abs() / df_input['VOL']) * 100

            # Histogram with bins of size 10 from 0 to 300
            bins = np.arange(0, 310, 10)  # From 0 to 300 in steps of 10
            plt.figure(figsize=(10, 6))
            plt.hist(df_input['Percent_Difference'], bins=bins, log=True, edgecolor='black')
            plt.xlabel('Percent Error')
            plt.ylabel('Distribution (Log Scale)')
            plt.title('Histogram of Percent Differences for Holdout Data')
            plt.grid(axis='y', alpha=0.75)
            plt.show()

            # Define daytime and nighttime hours
            day_hours = list(range(7, 19))  # 7 AM to 7 PM
            night_hours = list(range(0, 7)) + list(range(19, 24))  # 7 PM to 7 AM

            # Calculate percent difference for daytime
            day_df = df_input[df_input['measurement_tstamp'].dt.hour.isin(day_hours)]
            day_diff = day_df['Percent_Difference'].mean()

            # Calculate percent difference for nighttime
            night_df = df_input[df_input['measurement_tstamp'].dt.hour.isin(night_hours)]
            night_diff = night_df['Percent_Difference'].mean()

            # Calculate percentage within various thresholds for overall
            thresholds = list(range(5, 51, 5))  # 5%, 10%, 15%, ... 50%
            overall_within_percentages = []
            for threshold in thresholds:
                within_threshold = (df_input['Percent_Difference'] <= threshold).mean() * 100
                overall_within_percentages.append(within_threshold)

            # Compile results into a dictionary
            results = {
                'Description': description,
                'Total Rows': len(df_input),
                'Overall Percent Difference': df_input['Percent_Difference'].mean(),
                'Daytime Percent Difference': day_diff,
                'Nighttime Percent Difference': night_diff,
                'Thresholds': thresholds,
                'Overall Percentage Within': overall_within_percentages,
            }

            return results

        # Calculate metrics excluding rows with zero in 'VOL' or 'Predicted_VOL'
        results_non_zero = calculate_performance_metrics(df_non_zero, 'Excluding Rows with Zero in VOL or Predicted_VOL')

        # Print the results
        def print_results(results):
            print_and_write(f"\nMetrics ({results['Description']}):")
            print_and_write(f"Total number of rows: {results['Total Rows']}")
            print_and_write(f"Overall Percent Difference: {results['Overall Percent Difference']:.2f}%")
            print_and_write(f"Daytime Percent Difference: {results['Daytime Percent Difference']:.2f}%")
            print_and_write(f"Nighttime Percent Difference: {results['Nighttime Percent Difference']:.2f}%")
            print_and_write("Percentage Within Thresholds:")
            for threshold, percentage in zip(results['Thresholds'], results['Overall Percentage Within']):
                print_and_write(f"  Within {threshold}%: {percentage:.2f}%")

        print_results(results_non_zero)

        # Plot percentage within thresholds for the non-zero data
        plt.figure(figsize=(10, 6))
        plt.plot(results_non_zero['Thresholds'], results_non_zero['Overall Percentage Within'], marker='o')
        plt.xlabel('Error Threshold (%)')
        plt.ylabel('Percentage of Data Points Within Threshold (%)')
        plt.title('Percentage of Data Points Within Error Thresholds (Excluding Zeros)')
        plt.grid(True)
        plt.xticks(results_non_zero['Thresholds'])
        plt.show()

        # Create a histogram of traffic counts ('VOL') for the station with max zero counts
        tmc_with_max_zeros_df = df[df['tmc_code'] == max_percent_zero_vol_tmc]

        # Plot histogram of 'VOL' for this tmc_code
        plt.figure(figsize=(10, 6))
        plt.hist(tmc_with_max_zeros_df['VOL'], bins=30, edgecolor='black')
        plt.xlabel('Traffic Counts (VOL)')
        plt.ylabel('Frequency')
        plt.title(f"Histogram of Traffic Counts for TMC Code: {max_percent_zero_vol_tmc}")
        plt.grid(axis='y', alpha=0.75)
        plt.show()


    # End of with block; output file is closed







    
def main():
    #TMAS_to_pkl()
    #QAQC_Joins()
    #plotting_dataset()
    #run_joins()
    #generate_available_stations()
    generate_metrics_for_predictions_pkl()




if __name__ == "__main__":
    main()