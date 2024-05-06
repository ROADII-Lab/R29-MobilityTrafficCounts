import pandas as pd
import pickle
import csv


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


def QAQC_Joins():


    output = pickle.load(open(r'../data/NPMRDS_TMC_TMAS_US_SUBSET.pkl', "rb"))
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


TMAS_to_pkl()
#QAQC_Joins()

