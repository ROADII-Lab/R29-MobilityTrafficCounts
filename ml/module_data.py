import dask.dataframe as dd
import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from math import radians, sin, cos, acos

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

# HELPER FUNCTIONS TO CALCULATED nearest_Vol
def distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371  # Earth radius in kilometers (you can adjust this value)
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) * sin(dlat / 2) + cos(lat1) * cos(lat2) * sin(dlon / 2) * sin(dlon / 2)
    c = 2 * acos(min(1, a))  # Limit acos value to be between -1 and 1
    distance = earth_radius * c

    return distance 

def calculate_vol(df_group):
  # Iterate through each row and find the minimum distance
  for idx, row in df_group.iterrows():
    min_distance = float('inf') #starting at very large number
    nearest_vol = None
    found_match = False  # Flag to track if a match is found
    for inner_idx, inner_row in df_group.iterrows():
      if idx != inner_idx: #excluding considering itself as a match
        dist = distance(row['start_latitude'], row['start_longitude'], 
                                         inner_row['start_latitude'], inner_row['start_longitude'])
        if dist < min_distance:
          min_distance = dist
          nearest_vol = inner_row['VOL']
          found_match = True  # Set flag if a match is found
    if not found_match:  # Check if no match was found
      nearest_vol = -1  # Assign -1 if no match
    row['nearest_VOL'] = nearest_vol
  return df_group       

class data(object):
    # class object that handles pre-processing of various data sources

    def __init__(self) -> None:

        # Placeholder for the complete dataset (all data conflated)
        self.dataset = None
        self.normalized_dataset = None
        self.prepared_dataset = None
        self.calculated_columns = []
        self.norm_functions = ['tmc_norm', 'tstamp_norm', 'startdate_norm', 'density_norm', 'time_before_after']


        # setup data sources
        # self.tmas = self.tmas_data()
        # self.tmas.read()
        
        # self.npmrds = self.npmrds_data()
        # self.tmc = self.tmc_data()
        
        # output
        self.always_cache_data = True
        self.OUTPUT_FILE_PATH = r'../data/NPMRDS_TMC_TMAS_NE_C.csv'
        # pre-defined features for input into the AI model
        self.features_column_names = ['tmc_code', # traffic monitoring station id, needed for groupby() operations                          
                                'measurement_tstamp', # already normalized (yyyy-mm-dd hh:mm:ss)
                                'active_start_date', ## text field to normalize (yyyy-mm-dd hh:mm:ss +- time zone)
                                'average_speed_All', # (int)
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
                                'Population_2022' # (int) population by county
                                ]
        
        self.features_training_set = ['tmc_code', # traffic monitoring station id, needed for groupby() operations                           
                                'measurement_tstamp', # already normalized (yyyy-mm-dd hh:mm:ss)
                                'active_start_date', ## text field to normalize (yyyy-mm-dd hh:mm:ss +- time zone)
                                'average_speed_All', # (int)
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
                                'Population_2022' # (int) population by county
                                ]

        self.features_target = "VOL"

    def join_and_save(self):
        # run the joins and then save the results to storage
        NPMRDS_Join = self.npmrds.CombineNPMRDS()
        NPMRDS_TMC = self.TMC_Join(NPMRDS_Join)
        final_output = self.TMAS_Join(NPMRDS_TMC)
        final_output.to_csv(self.OUTPUT_FILE_PATH, index = True)

        self.dataset = final_output

        return final_output

    def read(self):
        # loads up all the required datasets and runs the joins only if needed
        if self.always_cache_data == False:
            return self.join_and_save()
        else:
            try:
                print(self.OUTPUT_FILE_PATH)
                final_output = pd.read_csv(self.OUTPUT_FILE_PATH, dtype={'tmc_code': 'string'})
                print("Loading cached data...")
                self.dataset = final_output
            except Exception as err:
                print(str(err))
                final_output = self.join_and_save()

        return final_output

    def TMAS_Join(self, NPMRDS_TMC):
        ''' Take in NPMRDS_TMC DF and TMAS_DATA_FILE path as input paramter, join on station ID, return Joined DF'''
        # Read in TMAS_Data file as a csv
        TMAS_Data = dd.read_csv(self.tmas.TMAS_DATA_FILE, dtype = {'STATION_ID': 'object'}, low_memory = False)
        
        # Filter TMAS_Data to only include Middlesex County in MA and save computed dataframe
        TMAS_Data_Filtered = TMAS_Data[(TMAS_Data['STATE_NAME'] == 'MA') & (TMAS_Data['COUNTY_NAME'] == 'Middlesex County')]
        TMAS_Data_Filtered = TMAS_Data_Filtered.compute()
        
        # Add a column to the dataframe called 'measurement_tstamp', the name of the datetime column in NPMRDS_TMC
        # that creates a datetime column given data in the existing dataframe
        TMAS_Data_Filtered['measurement_tstamp'] = TMAS_Data_Filtered.apply(lambda row: datetime.datetime(2000 + int(row['YEAR']),int(row['MONTH']), int(row['DAY']), int(row['HOUR'])), axis = 1)
        
        # Join NPMRDS_TMC with TMAS_Data
        NPMRDS_TMC_TMAS = pd.merge(NPMRDS_TMC, TMAS_Data_Filtered,on=['STATION_ID', 'measurement_tstamp'], how = 'inner')    
        
        return NPMRDS_TMC_TMAS

    def TMC_Join(self, NPMRDS_Join):
        ''' Read in NPMRDS Data frame that has All, Pass, and Truck and join with TMC_ID and TMC_Station'''
   
        # Read in TMC Station file as a csv, rename Tmc column to tmc_code
        TMC_Station = pd.read_csv(self.tmc.TMC_STATION_FILE)
        TMC_Station = TMC_Station.rename(columns = {'Tmc': 'tmc_code'})
        
        # Read in TMC ID file as a csv, rename tmc column to tmc_code
        TMC_ID = pd.read_csv(self.tmc.TMC_ID_FILE)
        TMC_ID = TMC_ID.rename(columns = {'tmc': 'tmc_code'})

        #Join NPMRDS Data frame with TMC Station and TMC ID df on tmc_code
        NPMRDS_TMC = pd.merge(NPMRDS_Join, TMC_Station,on=['tmc_code'], how = 'inner') 
        NPMRDS_TMC = pd.merge(NPMRDS_TMC, TMC_ID,on=['tmc_code'], how = 'inner')
            
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
        self.prepared_dataset['measurement_tstamp'] = pd.to_datetime(self.prepared_dataset['measurement_tstamp']).view('int64') // 10**9

    # Function to normalize active_start_date
    def startdate_norm(self):
        # convert 'active_start_date' from (yyyy-mm-dd hh:mm:ss +- time zone) to integer seconds UTC
        self.prepared_dataset['active_start_date'] = pd.to_datetime(self.prepared_dataset['active_start_date'], errors='coerce')
        self.prepared_dataset['active_start_date'] = self.prepared_dataset['active_start_date'].dt.tz_convert(None)
        self.prepared_dataset['active_start_date'] = pd.to_datetime(self.prepared_dataset['active_start_date']).view('int64') // 10**9


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
        
        """
        This function modifies the class variable 'self.prepared_dataset' by adding 
        a new column 'nearest_VOL' containing the VOL value of the nearest point 
        based on measurement_tstamp, TMC_Value, f_system, start_latitude and 
        start_longitude.
        """
    def calculate_nearest_vol(self):
        # Group by the three primary columns
        grouped_df = self.prepared_dataset.groupby(['measurement_tstamp', 'TMC_Value', 'f_system'])
        # Apply the function to each group and update the dataframe in-place
        self.prepared_dataset = grouped_df.apply(calculate_vol).reset_index()
        self.calculated_columns.append('nearest_VOL')

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
        self.calculated_columns = []

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
            self.TMAS_DATA_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/TMAS_Class_Clean_2021.csv'
            self.df = None

        def read(self):
            # read in raw TMAS data
            try:
                self.df = dd.read_csv(self.TMAS_DATA_FILE, dtype = {'STATION_ID': 'object'}, low_memory = False)
            except OSError as err:
                print("OS file read error:", err)
                self.df = None
            
    class npmrds_data(object):
        # class object for NPMRDS data source

        def __init__(self) -> None:
            
            # setup default data locations
            self.NPMRDS_ALL_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/Middlesex_MA_2021_TMAS_Matches_ALL.csv'
            self.NPMRDS_PASS_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/Middlesex_MA_2021_TMAS_Matches_PASSENGER.csv'
            self.NPMRDS_TRUCK_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/Middlesex_MA_2021_TMAS_Matches_TRUCKS.csv'
        
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
            NPMRDS_Truck = NPMRDS_Truck.rename(columns = {c: c + '_Truck' for c in NPMRDS_Truck.columns if c not in ['tmc_code', 'measurement_tstamp']})

            # Combine all TMAS .csv
            NPMRDS_Join = pd.merge(NPMRDS_All, NPMRDS_Pass, on=['tmc_code','measurement_tstamp'], how = 'inner', suffixes = ('_All', '_Pass'))
            NPMRDS_Join = pd.merge(NPMRDS_Join, NPMRDS_Truck,on=['tmc_code','measurement_tstamp'], how = 'inner')

            return NPMRDS_Join

    class tmc_data(object):
        # class object for TMS data source
        
        def __init__(self) -> None:

            # setup default data locations
            self.TMC_STATION_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/TMC_2021.csv'
            self.TMC_ID_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/TMC_Identification.csv'