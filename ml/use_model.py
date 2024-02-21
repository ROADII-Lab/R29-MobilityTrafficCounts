# Mobility Counts Prediction
# 
import pandas as pd

# Import custom modules
import module_ai
import module_data
import module_census
import setup_funcs
import pickle
import os



# init ai module
ai = module_ai.ai()

# init data module
source_data = module_data.data()

# init census data (using saved data for now)
# census_dataobj = module_census.census_data()
# census_data = census_dataobj.get_population_data_by_city(25)

# setup data sources
cols1 = ['measurement_tstamp',
        'average_speed_All',
        'speed_All',
        'travel_time_seconds_All',
        'start_latitude',
        'start_longitude',
        'end_latitude',
        'end_longitude',
        'miles',
        'aadt',
        'thrulanes_unidir',
        'f_system',
        'route_sign',
        'thrulanes',
        'zip',
        ]

source_data.OUTPUT_FILE_PATH = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/NPMRDS_TMC_TMAS_NE.csv'

census_df = pd.DataFrame() #pd.DataFrame(census_data)
if os.path.isfile("norm_data.pkl"):
    normalized_df = pickle.load(open("norm_data.pkl", "rb"))
else:
    result_df = source_data.read()
    normalized_df = source_data.normalized()
    import pickle
    pickle.dump(normalized_df, open("norm_data.pkl", "wb"))

result = setup_funcs.train_model(ai, normalized_df, cols1, 'VOL')

# TODO: calculate average percent diff between predictions and y_test 