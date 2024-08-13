import pandas as pd
import os
import random
import module_data

# Take in both folders as input that contains the existing batch of data used to train the model
folder1_path = r'C:\Users\Michael.Barzach\OneDrive - DOT OST\R29-MobilityCounts\JOINED_FILES\NPMRDS_TMC_TMAS_US_SUBSET_500_2021_v2'
folder2_path = r'C:\Users\Michael.Barzach\OneDrive - DOT OST\R29-MobilityCounts\JOINED_FILES\NPMRDS_TMC_TMAS_US_SUBSET_1000_22'

# Function to extract unique station IDs from file names
def extract_unique_stations(folder_path):
    station_ids = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl"):  # Assuming the files are pickle files
            parts = filename.split('_')  # Extract parts assuming format stationid_state_year.pkl
            if len(parts) == 3:
                station_id = parts[0]  # Extract station ID
                station_ids.add(station_id)
    return station_ids

# Extract unique station IDs from both folders
uniqueStations_training = extract_unique_stations(folder1_path) | extract_unique_stations(folder2_path)

# Remove all rows containing stations from uniqueStations_training from the CSV
geojoin_csv_path = r'C:\Users\Michael.Barzach\Documents\ROADII\Output\USSubset_ALL_22\TMC_2022Random_US_Subset_ALL_2022.csv'
geojoin_df = pd.read_csv(geojoin_csv_path)

# Filter out rows with STATION_ID in uniqueStations_training and keep track of removed stations
removed_stations = geojoin_df[geojoin_df['STATION_ID'].astype(str).isin(uniqueStations_training)]['STATION_ID'].unique()
filtered_df = geojoin_df[~geojoin_df['STATION_ID'].astype(str).isin(uniqueStations_training)]

# Randomly select 20 station IDs to keep
unique_stations_remaining = filtered_df['STATION_ID'].unique()
stations_to_keep = random.sample(list(unique_stations_remaining), min(20, len(unique_stations_remaining)))

# Filter to keep only the selected 20 stations
final_filtered_df = filtered_df[filtered_df['STATION_ID'].isin(stations_to_keep)]

# Save the final filtered CSV at the same file path as above but add '_2' to the file name
output_csv_path = geojoin_csv_path.replace('.csv', '_2.csv')
final_filtered_df.to_csv(output_csv_path, index=False)

# Print list of stations removed and stations kept
print("Stations successfully removed from the CSV:")
for station in removed_stations:
    print(station)

print("\nStations selected to keep:")
for station in stations_to_keep:
    print(station)

print(f"\nFiltered CSV saved to: {output_csv_path}")

# Create Data Class and Run Joins with new input files
source_data = module_data.data()

source_data.tmas = source_data.tmas_data()
if not os.path.isfile(source_data.tmas.TMAS_PKL_FILE):
# tmas.read() opens the csv and saves as pkl for performance during later joins
# if the pkl file already exists, this doesn't need to be run
    source_data.tmas.read()
source_data.npmrds = source_data.npmrds_data()
source_data.tmc = source_data.tmc_data()

# Set output file paths
source_data.OUTPUT_FILE_PATH = r'../data/NPMRDS_TMC_TMAS_US_SUBSET_20_22.pkl'

# Set TMAS file paths
source_data.tmas.TMAS_DATA_FILE = r'../data/TMAS_Class_Clean_2022.csv'
source_data.tmas.TMAS_PKL_FILE = r'../data/TMAS_Class_Clean_2022.pkl'

# Set NPMRDS data locations
source_data.npmrds.NPMRDS_ALL_FILE = r'..\data\US_ALL_22\all2022_NPMRDS_ALL.csv'
source_data.npmrds.NPMRDS_PASS_FILE = r'..\data\US_ALL_22\all2022_NPMRDS_PASS.csv'
source_data.npmrds.NPMRDS_TRUCK_FILE = r'..\data\US_ALL_22\all2022_NPMRDS_TRUCK.csv'


# Set TMC data locations (from geojoins)
source_data.tmc.TMC_STATION_FILE = r'..\data\US_ALL_22\TMC_2022Random_US_Subset_ALL_2022_2.csv'
source_data.tmc.TMC_ID_FILE = r'..\data\US_ALL_22\TMC_Identification.csv'

source_data.dataset_year = '2022'

source_data.join_and_save()

