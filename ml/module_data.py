import dask.dataframe as dd
import datetime
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

        '''
        # setup data sources
        self.tmas = self.tmas_data()
        self.tmas.read()
        
        self.npmrds = self.npmrds_data()
        self.tmc = self.tmc_data()
    '''
        
        # output
        self.always_cache_data = True
        self.OUTPUT_FILE_PATH = r'C:\Users\Michael.Barzach\OneDrive - DOT OST\R29-MobilityCounts\JOINED_FILES\NPMRDS_TMC_TMAS_NE_C.csv'
        # pre-defined features for input into the AI model
        self.features_column_names = ['tmc_code', # traffic monitoring station id, needed for groupby() operations                          
                                'measurement_tstamp', # already normalized (yyyy-mm-dd hh:mm:ss)
                                'active_start_date', ## text field to normalize (yyyy-mm-dd hh:mm:ss +- time zone)
                                'average_speed_All', # (int)
                                'speed_All', # (int)
                                'travel_time_seconds_All', # (float)
                                #'data_density_All', ## text field to normalize
                                #'data_density_Pass', ## text field to normalize
                                #'data_density_Truck', ## text field to normalize
                                'start_latitude', # (float)
                                'start_longitude', # (float)
                                'end_latitude', # (float)
                                'end_longitude', # (float)
                                'miles', # (float)
                                'f_system', ## numerical field to incorporate (int)
                                #'urban_code', ## numerical field to incorporate (int)
                                'aadt', # (int)
                                'thrulanes_unidir', # (int)
                                'route_sign',
                                'thrulanes',
                                'zip',
                                'MONTH', # (int)
                                'DAY', # (int)
                                'HOUR', # (int)
                                'DAY_TYPE', ## text field to normalize
                                'PEAKING', ## text field to normalize
                                'URB_RURAL', ## text field to normalize
                                'VOL', # (int)
                                #'F_SYSTEM', ## numerical field to incorporate (int)
                                'HPMS_ALL', # (int)
                                'NOISE_ALL', # (int)
                                'Population_2022' # (int) population by county
                                ]
        
        self.features_training_set = ['tmc_code', # traffic monitoring station id, needed for groupby() operations                           
                                'measurement_tstamp', # already normalized (yyyy-mm-dd hh:mm:ss)
                                'active_start_date', ## text field to normalize (yyyy-mm-dd hh:mm:ss +- time zone)
                                'average_speed_All', # (int)
                                'speed_All', # (int)
                                'travel_time_seconds_All', # (float)
                                #'data_density_All', ## text field to normalize
                                #'data_density_Pass', ## text field to normalize
                                #'data_density_Truck', ## text field to normalize
                                'start_latitude', # (float)
                                'start_longitude', # (float)
                                'end_latitude', # (float)
                                'end_longitude', # (float)
                                'miles', # (float)
                                'f_system', ## numerical field to incorporate (int)
                                #'urban_code', ## numerical field to incorporate (int)
                                'aadt', # (int)
                                'thrulanes_unidir', # (int)
                                'route_sign',
                                'thrulanes',
                                'zip',
                                #'MONTH', # (int)
                                #'DAY', # (int)
                                #'HOUR', # (int)
                                #'DAY_TYPE', ## text field to normalize
                                #'PEAKING', ## text field to normalize
                                #'URB_RURAL', ## text field to normalize
                                #'VOL', # (int)
                                #'F_SYSTEM', ## numerical field to incorporate (int)
                                #'HPMS_ALL', # (int)
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
                final_output = pd.read_csv(self.OUTPUT_FILE_PATH, dtype={'tmc_code': 'string'})
                print("Loading cached data...")
                self.dataset = final_output
            except error as err:
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
        self.normalized_dataset.insert(1, 'TMC_Value'
                                       , self.normalized_dataset.apply(lambda row: self.tmc_value(row['tmc_code']), axis=1))
        # Replace all non numerical characters in tmc_code, then convert column to int
        self.normalized_dataset['tmc_code'] = self.normalized_dataset['tmc_code'].str.lower().str.replace('p', '').str.replace('n', '').str.replace('+', '').str.replace('-', '')
        self.normalized_dataset['tmc_code'] = self.normalized_dataset['tmc_code'].astype(int)
        # Add this value to calculated column name
        self.calculated_columns.append('TMC_Value')

    # Higher order function to add normalization to dataset through passed function
    def add_normalization(self, func):
        # If normalized dataset hasn't been created, create copy of dataset with feature columns
        if (self.normalized_dataset is None):
            self.normalized_dataset = self.dataset[self.features_column_names].copy()
        # Run passed function
        func()

    # Apply all normalizations to dataset here by calling add_normalization for all normalization functions
    def apply_normalization(self):
        self.add_normalization(self.tmc_norm)

    def normalized(self):
        
        
        # modify data types to be normalized by the AI training data pre-processing steps
        self.normalized_dataset = self.dataset[self.features_column_names].copy()
        self.apply_normalization()
        # format the timestamps
        # convert 'measurement_tstamp' from (yyyy-mm-dd hh:mm:ss) to integer seconds
        
        # sort the data only by timestamp
        self.normalized_dataset.sort_values('measurement_tstamp', inplace=True)
        self.normalized_dataset['measurement_tstamp'] = pd.to_datetime(self.normalized_dataset['measurement_tstamp'], errors='coerce')
        self.normalized_dataset['measurement_tstamp'] = self.normalized_dataset['measurement_tstamp'].interpolate(method='linear')
        self.normalized_dataset['measurement_tstamp'] = pd.to_datetime(self.normalized_dataset['measurement_tstamp']).view('int64') // 10**9
        
        # convert 'active_start_date' from (yyyy-mm-dd hh:mm:ss +- time zone) to integer seconds UTC
        self.normalized_dataset['active_start_date'] = pd.to_datetime(self.normalized_dataset['active_start_date'], errors='coerce')
        self.normalized_dataset['active_start_date'] = self.normalized_dataset['active_start_date'].dt.tz_convert(None)
        self.normalized_dataset['active_start_date'] = pd.to_datetime(self.normalized_dataset['active_start_date']).view('int64') // 10**9
        
        '''
        # TODO: add vectorization for string data
        # convert multiple-choice (i.e., equally weighted) string data fields into integer fields using Python enumerate
        #self.normalized_dataset['data_density_All'] = norm_multiple_choice(self.normalized_dataset['data_density_All'])
        #self.normalized_dataset['data_density_Pass'] = norm_multiple_choice(self.normalized_dataset['data_density_Pass'])
        #self.normalized_dataset['data_density_Truck'] = norm_multiple_choice(self.normalized_dataset['data_density_Truck'])
        self.normalized_dataset['DAY_TYPE'] = norm_multiple_choice(self.normalized_dataset['DAY_TYPE'])
        self.normalized_dataset['PEAKING'] = norm_multiple_choice(self.normalized_dataset['PEAKING'])
        self.normalized_dataset['URB_RURAL'] = norm_multiple_choice(self.normalized_dataset['URB_RURAL'])
        '''
        # kill any rows that contain null values TODO: Should modify this to replace values instead depending on what the value is...
        self.normalized_dataset = self.normalized_dataset.dropna()

        # Add all calculated column to features training set and column names so they are included moving forward
        self.features_column_names.extend(self.calculated_columns)
        self.features_training_set.extend(self.calculated_columns)

        return self.normalized_dataset
    
    def prepared(self):
        # modify data types - extend each data entry to +/- one time increment
        #   time increment = 1 hour for data within each traffic station
        # this is a pre-processing step before AI training
        self.prepared_dataset = self.normalized_dataset[self.features_column_names].copy()
        
        # sort the data i) by traffic station id; ii) then by timestamp
        #   (placeholder for more sorting criteria...)
        self.prepared_dataset = self.prepared_dataset.sort_values(by=['tmc_code','measurement_tstamp'],ascending=[True,True])
        
        # create a "before" and an "after" dataframe representing shift by -/+ one time increment
        df_before = self.prepared_dataset.groupby(by=['tmc_code']).shift(periods=-1)
        #df_before = self.prepared_dataset.groupby(by=['tmc_code','measurement_tstamp']).shift(periods=-1)
        #the above line commented out is a way to display in the debug window that the groupby() worked as intended, making each group smaller/showing that the shifting took place on the correct indices
        df_after = self.prepared_dataset.groupby(by=['tmc_code']).shift(periods=1)
        #df_after = self.prepared_dataset.groupby(by=['tmc_code','measurement_tstamp']).shift(periods=1)
        #the above line commented out is a way to display in the debug window that the groupby() worked as intended, making each group smaller/showing that the shifting took place on the correct indices
        
        # ------------------------------------------------------
        # insert the time-shifted columns
        # measurement_tstamp - plus/minus one time increment
        self.prepared_dataset.insert(1,"measurement_tstamp_before",df_before['measurement_tstamp'])
        self.prepared_dataset.insert(3,"measurement_tstamp_after",df_after['measurement_tstamp'])
        
        # active_start_date - plus/minus one time increment
        self.prepared_dataset.insert(4,"active_start_date_before",df_before['active_start_date'])
        self.prepared_dataset.insert(6,"active_start_date_after",df_after['active_start_date'])
        
        ### average_speed_All - varies with time
        self.prepared_dataset.insert(7,"average_speed_All_before",df_before['average_speed_All'])
        self.prepared_dataset.insert(9,"average_speed_All_after",df_after['average_speed_All'])
        
        ### speed_All - varies with time
        self.prepared_dataset.insert(10,"speed_All_before",df_before['speed_All'])
        self.prepared_dataset.insert(12,"speed_All_after",df_after['speed_All'])
        
        ### travel_time_seconds_All - varies with time
        self.prepared_dataset.insert(13,"travel_time_seconds_All_before",df_before['travel_time_seconds_All'])
        self.prepared_dataset.insert(15,"travel_time_seconds_All_after",df_after['travel_time_seconds_All'])
        
        ### start_latitude - varies with time
        self.prepared_dataset.insert(16,"start_latitude_before",df_before['start_latitude'])
        self.prepared_dataset.insert(18,"start_latitude_after",df_after['start_latitude'])
        
        ### start_longitude - varies with time
        self.prepared_dataset.insert(19,"start_longitude_before",df_before['start_longitude'])
        self.prepared_dataset.insert(21,"start_longitude_after",df_after['start_longitude'])
        
        ### end_latitude - varies with time
        self.prepared_dataset.insert(22,"end_latitude_before",df_before['end_latitude'])
        self.prepared_dataset.insert(24,"end_latitude_after",df_after['end_latitude'])
        
        ### end_longitude - varies with time
        self.prepared_dataset.insert(25,"end_longitude_before",df_before['end_longitude'])
        self.prepared_dataset.insert(27,"end_longitude_after",df_after['end_longitude'])
        
        ### miles - varies with time
        self.prepared_dataset.insert(28,"miles_before",df_before['miles'])
        self.prepared_dataset.insert(30,"miles_after",df_after['miles'])
        
        # f_system - road type is constant
        self.prepared_dataset.insert(31,"f_system_before",df_before['f_system'])
        self.prepared_dataset.insert(33,"f_system_after",df_after['f_system'])
        
        # urban_code - urban code is constant
        self.prepared_dataset.insert(34,"urban_code_before",df_before['urban_code'])
        self.prepared_dataset.insert(36,"urban_code_after",df_after['urban_code'])
        
        ### aadt - varies with time
        self.prepared_dataset.insert(37,"aadt_before",df_before['aadt'])
        self.prepared_dataset.insert(39,"aadt_after",df_after['aadt'])
        
        # thrulanes_unidir - number of lanes is constant
        self.prepared_dataset.insert(40,"thrulanes_unidir_before",df_before['thrulanes_unidir'])
        self.prepared_dataset.insert(42,"thrulanes_unidir_after",df_after['thrulanes_unidir'])
        
        # route_sign - route signing type is constant
        self.prepared_dataset.insert(43,"route_sign_before",df_before['route_sign'])
        self.prepared_dataset.insert(45,"route_sign_after",df_after['route_sign'])
        
        # thrulanes - number of lanes is constant
        self.prepared_dataset.insert(46,"thrulanes_before",df_before['thrulanes'])
        self.prepared_dataset.insert(48,"thrulanes_after",df_after['thrulanes'])
        
        # zip - zip code is constant
        self.prepared_dataset.insert(49,"zip_before",df_before['zip'])
        self.prepared_dataset.insert(51,"zip_after",df_after['zip'])
        
        # DAY_TYPE - day type is constant (each piece of input data is one hour)
        self.prepared_dataset.insert(52,"DAY_TYPE_before",df_before['DAY_TYPE'])
        self.prepared_dataset.insert(54,"DAY_TYPE_after",df_after['DAY_TYPE'])
        
        # PEAKING - peaking direction is constant
        self.prepared_dataset.insert(55,"PEAKING_before",df_before['PEAKING'])
        self.prepared_dataset.insert(57,"PEAKING_after",df_after['PEAKING'])
        
        # URB_RURAL - urban rural classification is constant
        self.prepared_dataset.insert(58,"URB_RURAL_before",df_before['URB_RURAL'])
        self.prepared_dataset.insert(60,"URB_RURAL_after",df_after['URB_RURAL'])
        
        breakpoint()
        
        return self.prepared_dataset
    
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