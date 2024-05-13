import pandas as pd
import pickle
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import for time manipulation
from matplotlib.ticker import ScalarFormatter
import os


def TMAS_to_pkl():
    # File path
    TMAS_path = r'C:\Users\Michael.Barzach\OneDrive - DOT OST\R29-MobilityCounts\TMAS Data\TMAS 2021\TMAS_Class_Clean_2021.csv'
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
    breakpoint()

    pickle.dump(TMAS_DATA, open(r'C:\Users\Michael.Barzach\Documents\ROADII\TMAS_Class_Clean_2021.pkl', "wb"))
    print("dumped to pkl")

def plotting_dataset():
    print('loading data')
    DATASET_PATH = r'../data/NPMRDS_TMC_TMAS_US_SUBSET.pkl'
    
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
    max_volume_per_station = output.groupby('STATION_ID')['VOL'].max().reset_index()
    axes[2].hist(max_volume_per_station, bins=30, alpha=0.7)
    axes[2].set_xlabel('Max Volume')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Max Volume per Station')
    axes[2].set_yscale('log')
    axes[2].yaxis.set_major_formatter(scalar_formatter)

    plt.tight_layout()
    plt.show()

    max_volume_per_station.rename(columns={'VOL': 'Max_VOL'}, inplace=True)  # Correctly renaming the column to 'Max_VOL'
    # Extract rows for each station ID where the volume equals the maximum volume found
    top_stations = pd.merge(output, max_volume_per_station, how='inner', left_on=['STATION_ID', 'VOL'], right_on=['STATION_ID', 'Max_VOL'])
    top_stations = top_stations[['STATION_ID', 'VOL', 'County', 'State', 'road', 'intersection', 'measurement_tstamp']].sort_values(by='VOL', ascending=False).head(10)

    # Save to CSV
    csv_path = r'../data/US_Subset/Top_Stations_Max_Volume_Details.csv'
    top_stations.to_csv(csv_path, index=False)
    print(f"Top stations data saved to '{csv_path}'")

def QAQC_Joins():

    print('loading data')
    DATASET_PATH =  r'../data/NPMRDS_TMC_TMAS_US_SUBSET.pkl'
    # Load the data
    #handle input data file differently if .pkl or .csv
    if (DATASET_PATH.endswith('.pkl')):
        output = pickle.load(open(DATASET_PATH, "rb"))
    else:
        output = pd.read_csv(DATASET_PATH, low_memory=False)
    print("data loaded")
    
    breakpoint()

    prejoin = pickle.load(open(r'../data/prejoin.pkl', "rb"))
    tmas = pickle.load(open(r'C:\Users\Michael.Barzach\Documents\ROADII\TMAS_Class_Clean_2021.pkl', "rb"))
    print('loaded test pkl files')
    breakpoint()
    goutput = output.groupby(['tmc_code'])
    gprejoin = prejoin.groupby(['tmc_code'])
    print('Number of unique TMCs before Join: ' + str(len(gprejoin)))
    print('Number of unique TMCs after join: ' + str(len(goutput)))

    
    goutput = output.groupby(['tmc_code', 'measurement_tstamp'])
    gprejoin = prejoin.groupby(['tmc_code', 'measurement_tstamp'])

    print('Number of unique TMCs/times before Join: ' + str(len(gprejoin)))
    print('Number of unique TMCs/times after join: ' + str(len(goutput)))
    
    # check for null values from TMC station

    # Assuming your dataframes are named 'df_prejoin' and 'df_output'
    prejoin_station_ids = set(prejoin['STATION_ID'])
    output_station_ids = set(output['STATION_ID'])

    missing_stations = prejoin_station_ids - output_station_ids

    print("Stations missing in the output dataframe: " + str(missing_stations))


#TMAS_to_pkl()
#QAQC_Joins()
plotting_dataset()

