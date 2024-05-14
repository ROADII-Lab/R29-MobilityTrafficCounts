# Mobility Counts Prediction
# 
import pandas as pd

# Import custom modules
import module_ai
import module_data
import module_census

# helper functions / hooks for object calls
def train_model(ai, normalized_df, in_cols, target_col):
    # setup training data
    ai.features = in_cols
    ai.target = target_col
    x_train, y_train, x_test, y_test = ai.format_training_data(normalized_df)

    # init the model
    ai.model_init(x_train)

    # train the model
    ai.train(ai.model, x_train, y_train, x_test, y_test)

    return 

def test_model(ai, normalized_df, in_cols, target_col):
    # setup training / test data
    ai.features = in_cols
    ai.target = target_col
    # x_train, y_train, x_test, y_test = ai.format_training_data(normalized_df)

    # load the model or use a model that's already loaded
    if ai.model != None or ai.test_loader == None:
        predictions, y_test, test_loss, accuracy = ai.test(ai.model, ai.test_loader)
            
    else:
        print("No model or data loaded!")
        return 0
    
    return predictions, y_test, test_loss, accuracy

def setup():
    # init ai module
    ai = module_ai.ai()

    # init data module
    source_data = module_data.data()

    # init census data (using saved data for now)
    # census_dataobj = module_census.census_data()
    # census_data = census_dataobj.get_population_data_by_city(25)

    # setup data sources
    census_df = pd.DataFrame() #pd.DataFrame(census_data)
    result_df = source_data.read()
    normalized_df = source_data.normalized()

    return census_df, result_df, normalized_df, ai, source_data