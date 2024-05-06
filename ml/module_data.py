import dask.dataframe as dd
import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from math import radians, sin, cos, acos
import pickle

# common functions
def merge(dataframe1, dataframe2, key):
    # Merge the datasets based on a common key (e.g., TMC Code)
    return pd.merge(dataframe1, dataframe2, on=key)

def csv_read(data_file_path):
    # generic function to read in a csv from a file path / S3 bucket and return it as a pandas dataframe
    try:
        df = pd.read_csv(data_file_path)
    except OSError as err:
        print("File open error: ", err)
        df = None
    return df
    
def norm_multiple_choice(list1):
    # Stack Overflow
    # Python - Is there a function to normalize strings and convert them to integers/floats?
    # https://stackoverflow.com/questions/63580106/is-there-a-function-to-normalize-strings-and-convert-them-to-integers-floats
    # d = {x: i for i, x in enumerate(sorted(set(list1)))} # sorting causes an error, cannot quantitativetly compare a string to a blank
    d = {x: i for i, x in enumerate(set(list1))}
    return [d[s] for s in list1]      

class data(object):
    # class object that handles pre-processing of various data sources

    def __init__(self) -> None:

        # Placeholder for the complete dataset (all data conflated)
        self.dataset = None
        self.normalized_dataset = None
        self.prepared_dataset = None
        self.calculated_columns = []
        self.norm_functions = ['tmc_norm', 'tstamp_norm', 'startdate_norm', 'density_norm', 'time_before_after']


        # setup data sources - Uncomment if running joins for creating dataset
        #self.tmas = self.tmas_data()
        #self.tmas.read()
        #self.npmrds = self.npmrds_data()
        #self.tmc = self.tmc_data()
        
        # output
        self.always_cache_data = True
        self.OUTPUT_FILE_PATH = r'../data/NPMRDS_TMC_TMAS_US_SUBSET.pkl'
        self.prejoin = r'../data/prejoin.pkl'
        # pre-defined features for input into the AI model
        self.features_column_names = ['tmc_code', # traffic monitoring station id, needed for groupby() operations                          
                                'measurement_tstamp', # already normalized (yyyy-mm-dd hh:mm:ss)
                                'active_start_date', ## text field to normalize (yyyy-mm-dd hh:mm:ss +- time zone)
                                # 'average_speed_All', # (int)
                                'speed_All', # (int)
                                'travel_time_seconds_All', # (float)
                                'data_density_All', ## text field to normalize
                                'data_density_Pass', ## text field to normalize
                                'data_density_Truck', ## text field to normalize
                                'start_latitude', # (float)
                                'start_longitude', # (float)
                                'end_latitude', # (float)
                                'end_longitude', # (float)
                                'miles', # (float)
                                'f_system', ## numerical field to incorporate (int)
                                'urban_code', ## numerical field to incorporate (int)
                                'aadt', # (int)
                                'thrulanes_unidir', # (int)
                                'route_sign',
                                'thrulanes',
                                'zip',
                                # 'MONTH', # (int)
                                # 'DAY', # (int)
                                # 'HOUR', # (int)
                                # 'DAY_TYPE', ## text field to normalize
                                #'PEAKING', ## text field to normalize
                                #'URB_RURAL', ## text field to normalize
                                'VOL', # (int)
                                #'F_SYSTEM', ## numerical field to incorporate (int)
                                # 'HPMS_ALL', # (int)
                                #'NOISE_ALL', # (int)
                                # 'Population_2022' # (int) population by county
                                ]
        
        self.features_training_set = ['tmc_code', # traffic monitoring station id, needed for groupby() operations                           
                                'measurement_tstamp', # already normalized (yyyy-mm-dd hh:mm:ss)
                                'active_start_date', ## text field to normalize (yyyy-mm-dd hh:mm:ss +- time zone)
                                # 'average_speed_All', # (int)
                                'speed_All', # (int)
                                'travel_time_seconds_All', # (float)
                                'data_density_All', ## text field to normalize
                                'data_density_Pass', ## text field to normalize
                                'data_density_Truck', ## text field to normalize
                                'start_latitude', # (float)
                                'start_longitude', # (float)
                                'end_latitude', # (float)
                                'end_longitude', # (float)
                                'miles', # (float)
                                'f_system', ## numerical field to incorporate (int)
                                'urban_code', ## numerical field to incorporate (int)
                                'aadt', # (int)
                                'thrulanes_unidir', # (int)
                                'route_sign',
                                'thrulanes',
                                'zip',
                                # 'MONTH', # (int)
                                # 'DAY', # (int)
                                # 'HOUR', # (int)
                                # 'DAY_TYPE', ## text field to normalize
                                #'PEAKING', ## text field to normalize
                                #'URB_RURAL', ## text field to normalize
                                #'VOL', # (int)
                                #'F_SYSTEM', ## numerical field to incorporate (int)
                                # 'HPMS_ALL', # (int)
                                #'NOISE_ALL' # (int)
                                # 'Population_2022' # (int) population by county
                                ]

        self.features_target = "VOL"

    def join_and_save(self):

        # Check if the file exists at self.prejoin
        if os.path.exists(self.prejoin):
            NPMRDS_TMC = pickle.load(open(self.prejoin, "rb"))
            print("NPMRDS_TMC pkl read")
        else:
            # If file doesn't exist, create file with joins
            NPMRDS_Join = self.npmrds.CombineNPMRDS()
            print("NPMRDS Joined")
            NPMRDS_TMC = self.TMC_Join(NPMRDS_Join)
            print("TMC AND NPMRDS Joined")

        final_output = self.TMAS_Join(NPMRDS_TMC)
        print("Joined with TMAS, now outputting to pkl")
        pickle.dump(final_output, open(self.OUTPUT_FILE_PATH, "wb"))
        print("Data output to pkl")

        self.dataset = final_output

        return final_output

    def read(self):
        # loads up all the required datasets and runs the joins only if needed
        if self.always_cache_data == False:
            return self.join_and_save()
        else:
            try:
                # If handle input data file differently if .pkl or .csv
                if (self.OUTPUT_FILE_PATH.endswith('.pkl')):
                    final_output = pickle.load(open(self.OUTPUT_FILE_PATH, "rb"))
                else:
                    final_output = pd.read_csv(self.OUTPUT_FILE_PATH, dtype={'tmc_code': 'string'}, low_memory=False)
                print("Loading cached data...")
                self.dataset = final_output
            except Exception as err:
                print(str(err))
                final_output = self.join_and_save()

        return final_output

    def TMAS_Join(self, NPMRDS_TMC):
        """
        Take in NPMRDS_TMC DataFrame and TMAS_DATA_FILE path as input parameters,
        join on station ID, and return the joined DataFrame.

        Performs the join in a chunkwise manner on NPMRDS_TMC for memory efficiency.
        Tracks progress based on the number of chunks processed.

        """
        # Set chunksize for memory efficient processing of NPMRDS_TMC
        chunksize = 5000000   # Adjust this value based on your memory constraints and file size

        # Read TMAS data .pkl file into a DataFrame
        TMAS_Data = pickle.load(open(self.tmas.TMAS_PKL_FILE, "rb"))
        NPMRDS_TMC['STATION_ID'] = NPMRDS_TMC['STATION_ID'].astype(str)
        TMAS_Data['STATION_ID'] = TMAS_Data['STATION_ID'].astype(str)

        print("TMAS pkl Read")
       # '''
        # Create a pandas Series for the datetime values
        datetime_series = pd.to_datetime(
        {'year': 2000 + TMAS_Data['YEAR'],
        'month': TMAS_Data['MONTH'],
        'day': TMAS_Data['DAY'],
        'hour': TMAS_Data['HOUR']})

        TMAS_Data['measurement_tstamp'] = datetime_series
        print("merging")

        # Split NPMRDS_TMC into chunks
        chunks = []
        for i in range(0, len(NPMRDS_TMC), chunksize):
            # Get the end index for the current chunk
            end_index = min(i + chunksize, len(NPMRDS_TMC))  # Limit to DataFrame length
            chunks.append(NPMRDS_TMC[i:end_index])
        print("Split into Chunks")
        total_chunks = len(chunks)  # Get total number of chunks
        # Track processed chunks to monitor progress
        processed_chunks = 0
        breakpoint()
        # Process NPMRDS_TMC chunks
        joined_data = []
        for chunk in chunks:
            # Perform the join on the current chunk with the entire TMAS_Data
            chunk_joined = pd.merge(chunk, TMAS_Data, on=['STATION_ID', 'measurement_tstamp'], how='inner')
            joined_data.append(chunk_joined)

            # Update processed chunks counter and calculate percentage complete
            processed_chunks += 1
            percentage_complete = (processed_chunks / total_chunks) * 100

            # Print progress message
            print(f"Processing complete: {percentage_complete:.2f}%")

        # Combine all joined chunks into a single DataFrame
        NPMRDS_TMC_TMAS = pd.concat(joined_data)
        
        return NPMRDS_TMC_TMAS

    def TMC_Join(self, NPMRDS_Join):
        ''' Read in NPMRDS Data frame that has All, Pass, and Truck and join with TMC_ID and TMC_Station'''
        dtype_dict = {
            'STATION_ID': str,
        }
        # Read in TMC Station file as a csv, rename Tmc column to tmc_code
        TMC_Station = pd.read_csv(self.tmc.TMC_STATION_FILE, dtype= dtype_dict)
        TMC_Station = TMC_Station.rename(columns = {'Tmc': 'tmc_code'})
        
        # Read in TMC ID file as a csv, rename tmc column to tmc_code
        TMC_ID = pd.read_csv(self.tmc.TMC_ID_FILE)
        TMC_ID = TMC_ID.rename(columns = {'tmc': 'tmc_code'})

        #Join NPMRDS Data frame with TMC Station and TMC ID df on tmc_code
        NPMRDS_TMC = pd.merge(NPMRDS_Join, TMC_Station,on=['tmc_code'], how = 'inner') 
        NPMRDS_TMC = pd.merge(NPMRDS_TMC, TMC_ID,on=['tmc_code'], how = 'inner')
        # UNCOMMENT LINE BELOW IF DOING QAQC CHECKS
        #pickle.dump(NPMRDS_TMC, open(self.prejoin, "wb"))
        return NPMRDS_TMC

    # AI-centric functions ----------------------------------------------------------------------------------------------------------------------

    # Check if tmc code contains P, N, +, - and return 0 - 3 accordingly or -1 for none
    def tmc_value(self, tmc_code):
        tmc_code = str(tmc_code).lower()
        if ('p' in tmc_code):
            return 0
        elif ('n' in tmc_code):
            return 1
        elif ('-' in tmc_code):
            return 2
        elif ('+' in tmc_code):
            return 3
        else:
            return -1
    
    # Insert column based on code found in tmc_code, remove characters from tmc_code, convert to int
    def tmc_norm(self):
        self.prepared_dataset.insert(1, 'TMC_Value'
                                       , self.prepared_dataset.apply(lambda row: self.tmc_value(row['tmc_code']), axis=1))
        # Replace all non numerical characters in tmc_code, then convert column to int
        self.prepared_dataset['tmc_code'] = self.prepared_dataset['tmc_code'].str.lower().str.replace('p', '').str.replace('n', '').str.replace('+', '').str.replace('-', '')
        self.prepared_dataset['tmc_code'] = self.prepared_dataset['tmc_code'].astype(int)
        # Add this value to calculated column name
        self.calculated_columns.append('TMC_Value')

    # Function to normalize measurement_tstamp
    def tstamp_norm(self):
        # sort the data only by timestamp
        self.prepared_dataset.sort_values('measurement_tstamp', inplace=True)
        # format the timestamps
        # convert 'measurement_tstamp' from (yyyy-mm-dd hh:mm:ss) to integer seconds
        self.prepared_dataset['measurement_tstamp'] = pd.to_datetime(self.prepared_dataset['measurement_tstamp'], errors='coerce')
        self.prepared_dataset['measurement_tstamp'] = self.prepared_dataset['measurement_tstamp'].interpolate(method='linear')
        self.prepared_dataset['measurement_tstamp'] = pd.to_datetime(self.prepared_dataset['measurement_tstamp']).astype('int64') // 10**9

    # Function to normalize active_start_date
    def startdate_norm(self):
        # convert 'active_start_date' from (yyyy-mm-dd hh:mm:ss +- time zone) to integer seconds UTC
        self.prepared_dataset['active_start_date'] = pd.to_datetime(self.prepared_dataset['active_start_date'], errors='coerce')
        self.prepared_dataset['active_start_date'] = self.prepared_dataset['active_start_date'].dt.tz_convert(None)
        self.prepared_dataset['active_start_date'] = pd.to_datetime(self.prepared_dataset['active_start_date']).astype('int64') // 10**9


    # Function to normalize data_density_XXXX
    def density_norm(self):
        # convert multiple-choice (i.e., equally weighted) string data fields into integer fields using Python enumerate
        self.prepared_dataset['data_density_All'] = norm_multiple_choice(self.prepared_dataset['data_density_All'])
        self.prepared_dataset['data_density_Pass'] = norm_multiple_choice(self.prepared_dataset['data_density_Pass'])
        self.prepared_dataset['data_density_Truck'] = norm_multiple_choice(self.prepared_dataset['data_density_Truck'])

    # Creates time before and after datasets in the prepared_dataset
    # This should always be last normalization called since it creates columns based on other calculated columns
    def time_before_after(self):
        self.prepared_dataset = self.prepared_dataset.sort_values(by=['tmc_code','TMC_Value','measurement_tstamp'],ascending=[True,True,True])
        # create a "before" and an "after" dataframe representing shift by -/+ one time increment
        df_before = self.prepared_dataset.groupby(by=['tmc_code','TMC_Value']).shift(periods=-1)
        #df_before = self.prepared_dataset.groupby(by=['tmc_code','measurement_tstamp']).shift(periods=-1)
        #the above line commented out is a way to display in the debug window that the groupby() worked as intended, making each group smaller/showing that the shifting took place on the correct indices
        df_after = self.prepared_dataset.groupby(by=['tmc_code','TMC_Value']).shift(periods=1)
        #df_after = self.prepared_dataset.groupby(by=['tmc_code','measurement_tstamp']).shift(periods=1)
        #the above line commented out is a way to display in the debug window that the groupby() worked as intended, making each group smaller/showing that the shifting took place on the correct indices
        #Columns to exclude from time before/after - mostly columns that won't change for a given time period shift
        excluded_columns = ['tmc_code', 'aadt', 'TMC_Value', 'start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'f_system', 'urban_code', 'thrulanes_unidir', 'route_sign', 'thrulanes', 'zip', 'Population_2022']
        # Loop through all columns not uppercase (uppercase not part of training) and not in excluded_columns and create a before/after column
        for col in self.prepared_dataset.columns:
            if (not col.isupper() and col not in excluded_columns):
                col_name_before = col+"_before"
                col_name_after = col+"_after"
                self.prepared_dataset.insert(len(self.prepared_dataset.columns), col_name_before, df_before[col])
                self.prepared_dataset.insert(len(self.prepared_dataset.columns), col_name_after, df_after[col])
                self.calculated_columns.append(col_name_before)
                self.calculated_columns.append(col_name_after)

    # Apply all normalizations to dataset here by looping through self.norm_functions and calling all (ORDER MATTERS)
    def apply_normalization(self):
        # sort the data i) by road link TMC id; ii) then by timestamp
        self.prepared_dataset = self.dataset[self.features_column_names].copy()
        self.prepared_dataset = self.prepared_dataset.sort_values(by=['tmc_code','measurement_tstamp'],ascending=[True,True])
        for function_name in self.norm_functions:
            method = getattr(self, function_name)
            if callable(method):
                method()

        # Update column names and training set to include all calculated columns, then reset calculated columns
        self.features_column_names.extend(self.calculated_columns)
        self.features_training_set.extend(self.calculated_columns)  

    def normalized(self):
        # Call apply_normalization to run all normalization functions (that modify the prepared dataset)
        self.apply_normalization()

        # Create normalized dataset from prepared dataset
        self.normalized_dataset = self.prepared_dataset[self.features_column_names].copy()

        # kill any rows that contain null values TODO: Should modify this to replace values instead depending on what the value is...
        self.normalized_dataset = self.normalized_dataset.dropna()
        
        # Sort normalized data set by tmc_code, TMC_Value, measurement_tstamp
        self.normalized_dataset = self.normalized_dataset.sort_values(by=['tmc_code','TMC_Value','measurement_tstamp'],ascending=[True,True,True])
        return self.normalized_dataset
    
    
    def generate_feature(self, obj_tmas, obj_npmrds, obj_census):
        # accepts data objects and returns a conflated feature for a specified time/GIS slice NOTE: this might not be needed anymore
        pass

    # Data Sources ------------------------------------------------------------------------------------------------------------------------------

    class tmas_data:
        def __init__(self) -> None:
            
            # setup default data location
            self.TMAS_DATA_FILE = r'../data/TMAS_Class_Clean_2021.csv'
            self.TMAS_PKL_FILE = r'../data/TMAS_Class_Clean_2021.pkl'
            self.df = None

        def read(self):
            # read in raw TMAS data
            try:
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
                TMAS_DATA = pd.read_csv(self.TMAS_DATA_FILE, dtype= dtype_dict)
                print("TMAS csv read")
                pickle.dump(TMAS_DATA, open(self.TMAS_PKL_FILE, "wb"))
                print("dumped to pkl")
            except OSError as err:
                print("TMAS OS file read error:", err)
                self.df = None
            
    class npmrds_data(object):
        # class object for NPMRDS data source

        def __init__(self) -> None:
            
            # setup default data locations
            self.NPMRDS_ALL_FILE = r'../data/US_Subset/US_500_ALL.csv'
            self.NPMRDS_PASS_FILE = r'../data/US_Subset/US_500_PASS.csv'
            self.NPMRDS_TRUCK_FILE = r'../data/US_Subset/US_500_TRUCK.csv'
        
        def CombineNPMRDS(self):
            '''Read in NPMRDS Files as input parameters, join them on tmc_code and measurement_tstamp'''
            
            # Read in NPMRDS Files
            NPMRDS_All = pd.read_csv(self.NPMRDS_ALL_FILE)
            NPMRDS_Pass = pd.read_csv(self.NPMRDS_PASS_FILE)
            NPMRDS_Truck = pd.read_csv(self.NPMRDS_TRUCK_FILE)
            
            # Convert 'measurement_tstamp' column to date time object for all NPMRDS Files    
            NPMRDS_All['measurement_tstamp'] = pd.to_datetime(NPMRDS_All['measurement_tstamp'])
            NPMRDS_Pass['measurement_tstamp'] = pd.to_datetime(NPMRDS_Pass['measurement_tstamp'])
            NPMRDS_Truck['measurement_tstamp'] = pd.to_datetime(NPMRDS_Truck['measurement_tstamp'])
            
            # Add Suffix to Truck data, not including join columns. Once joined all data will have a suffix
            NPMRDS_All = NPMRDS_All.rename(columns = {c: c + '_All' for c in NPMRDS_All.columns if c not in ['tmc_code', 'measurement_tstamp']})
            NPMRDS_Pass = NPMRDS_Pass.rename(columns = {c: c + '_Pass' for c in NPMRDS_Pass.columns if c not in ['tmc_code', 'measurement_tstamp']})
            NPMRDS_Truck = NPMRDS_Truck.rename(columns = {c: c + '_Truck' for c in NPMRDS_Truck.columns if c not in ['tmc_code', 'measurement_tstamp']})

            # Combine all TMAS .csv
            NPMRDS_Join = pd.merge(NPMRDS_All, NPMRDS_Pass, on=['tmc_code','measurement_tstamp'], how = 'inner')
            NPMRDS_Join = pd.merge(NPMRDS_Join, NPMRDS_Truck,on=['tmc_code','measurement_tstamp'], how = 'inner')
            return NPMRDS_Join

    class tmc_data(object):
        # class object for TMS data source
        
        def __init__(self) -> None:

            # setup default data locations
            self.TMC_STATION_FILE = r'../data/US_Subset/TMC_2021Random_US_Subset_500.csv'
            self.TMC_ID_FILE = r'../data/US_Subset/TMC_Identification.csv'