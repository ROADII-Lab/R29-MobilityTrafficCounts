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
    '''
    This line not neccesary anymore, since we are passing
    x_test and y_test as input parameter. 
    This ensures we aren't testing model on data it was trained on.
    '''
    #x_train, y_train, x_test, y_test = ai.format_training_data(normalized_df)

    # load the model or use a model that's already loaded
    if ai.model != None:
        predictions, ai.y_test, test_loss, accuracy = ai.test(ai.model, ai.x_test, ai.y_test)
    else:
        if(ai.model_load(ai.x_test)):
            predictions, ai.y_test, test_loss, accuracy = ai.test(ai.model, ai.x_test, ai.y_test)
            
        else:
            print("No model loaded!")
            return 0
    
    return predictions, ai.y_test, test_loss, accuracy

def setup():
    # init ai module
    ai = module_ai.ai()

    # init data module
    source_data = module_data.data()

    # setup data sources
    census_df = pd.DataFrame() #pd.DataFrame(census_data)
    result_df = source_data.read()
    normalized_df = source_data.normalized()

    return census_df, result_df, normalized_df, ai, source_data