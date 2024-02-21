# -*- coding: utf-8 -*-
"""
CombineNPMRDS, TMC_Join, and TMAS_Join functions and main() test
"""



import pandas as pd
import dask.dataframe as dd
import datetime


def CombineNPMRDS (NPMRDS_ALL_FILE, NPMRDS_PASS_FILE,NPMRDS_TRUCK_FILE ):
    '''Read in NPMRDS Files as input parameters, join them on tmc_code and measurement_tstamp'''
    
    # Read in NPMRDS Files
    NPMRDS_All = pd.read_csv(NPMRDS_ALL_FILE)
    NPMRDS_Pass = pd.read_csv(NPMRDS_PASS_FILE)
    NPMRDS_Truck = pd.read_csv(NPMRDS_TRUCK_FILE)
    
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

def TMC_Join(NPMRDS_Join, TMC_STATION_FILE, TMC_ID_FILE):
    ''' Read in NPMRDS Data frame that has All, Pass, and Truck and join with TMC_ID and TMC_Station'''
   
    # Read in TMC Station file as a csv, rename Tmc column to tmc_code
    TMC_Station = pd.read_csv(TMC_STATION_FILE)
    TMC_Station = TMC_Station.rename(columns = {'Tmc': 'tmc_code'})
    
    # Read in TMC ID file as a csv, rename tmc column to tmc_code
    TMC_ID = pd.read_csv(TMC_ID_FILE)
    TMC_ID = TMC_ID.rename(columns = {'tmc': 'tmc_code'})

    #Join NPMRDS Data frame with TMC Station and TMC ID df on tmc_code
    NPMRDS_TMC = pd.merge(NPMRDS_Join, TMC_Station,on=['tmc_code'], how = 'inner') 
    NPMRDS_TMC = pd.merge(NPMRDS_TMC, TMC_ID,on=['tmc_code'], how = 'inner')    
    return NPMRDS_TMC

def TMAS_Join(NPMRDS_TMC, TMAS_DATA_FILE):
    ''' Take in NPMRDS_TMC DF and TMAS_DATA_FILE path as input paramter, join on station ID, return Joined DF'''
    # Read in TMAS_Data file as a csv
    TMAS_Data = dd.read_csv(TMAS_DATA_FILE, dtype = {'STATION_ID': 'object'}, low_memory = False)
    
    # Filter TMAS_Data to only include Middlesex County in MA and save computed dataframe
    TMAS_Data_Filtered = TMAS_Data[(TMAS_Data['STATE_NAME'] == 'MA') & (TMAS_Data['COUNTY_NAME'] == 'Middlesex County')]
    TMAS_Data_Filtered = TMAS_Data_Filtered.compute()
    
    # Add a column to the dataframe called 'measurement_tstamp', the name of the datetime column in NPMRDS_TMC
    # that creates a datetime column given data in the existing dataframe
    TMAS_Data_Filtered['measurement_tstamp'] = TMAS_Data_Filtered.apply(lambda row: datetime.datetime(2000 + int(row['YEAR']),int(row['MONTH']), int(row['DAY']), int(row['HOUR'])), axis = 1)
    
    # Join NPMRDS_TMC with TMAS_Data
    NPMRDS_TMC_TMAS = pd.merge(NPMRDS_TMC, TMAS_Data_Filtered,on=['STATION_ID', 'measurement_tstamp'], how = 'inner')    
    
    return NPMRDS_TMC_TMAS

    
def main():
    
    
    # File Paths on S3 Bucket
    NPMRDS_ALL_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/Middlesex_MA_2021_TMAS_Matches_ALL.csv'
    NPMRDS_PASS_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/Middlesex_MA_2021_TMAS_Matches_PASSENGER.csv'
    NPMRDS_TRUCK_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/Middlesex_MA_2021_TMAS_Matches_TRUCKS.csv'
    TMC_STATION_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/TMC_2021.csv'
    TMC_ID_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/TMC_Identification.csv'
    TMAS_DATA_FILE = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/TMAS_Class_Clean_2021.csv'
    OUTPUT_FILE_PATH = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/NPMRDS_TMC_TMAS.csv'
    
    #Calling Join Functions
    NPMRDS_Join = CombineNPMRDS(NPMRDS_ALL_FILE, NPMRDS_PASS_FILE, NPMRDS_TRUCK_FILE)
    NPMRDS_TMC = TMC_Join(NPMRDS_Join,TMC_STATION_FILE, TMC_ID_FILE)
    NPMRDS_TMC_TMAS = TMAS_Join(NPMRDS_TMC, TMAS_DATA_FILE)

    # Output Final Join to CSV
    NPMRDS_TMC_TMAS.to_csv(OUTPUT_FILE_PATH, index = False)
    
if __name__ == '__main__':   
     main()