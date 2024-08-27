# Mobility Counts Prediction
# 
import pandas as pd

# Import custom modules
import module_ai
import module_data

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
    if ai.model != None or ai.test_loader != None:
        predictions, y_test, test_loss, R2, Within15 = ai.test(ai.model, ai.test_loader)
            
    else:
        print("No model or data loaded!")
        return 0
    
    return predictions, y_test, test_loss, (R2, Within15)

def use_model(ai, model_filename, normalized_df, in_cols, output_column_name = "VOL"):
    def remove_column_if_exists(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        # Remove a column from a DataFrame if it exists.
        if column_name in df.columns:
            df = df.drop(columns=[column_name])
        return df
    # make sure the output column does not already exist
    normalized_df = remove_column_if_exists(normalized_df, output_column_name)

    # setup training / test data
    ai.features = in_cols
    ai.target = output_column_name

    # load the model or use a model that's already loaded
    if ai.model == None:
        ai.model_load(normalized_df, model_filename)

    if ai.model != None:
        predictions = ai.model_inference(normalized_df)
    else:
        print("No model loaded, check file path!")
        return None
    column_name = 'Predicted_' + output_column_name
    normalized_df.reset_index(drop=True, inplace=True)
    predictions.reset_index(drop=True, inplace=True)
    normalized_df[column_name]= predictions['predicted']
    return normalized_df

def setup():
    # init ai module
    ai = module_ai.ai()

    # init data module
    source_data = module_data.data()

    # setup data sources
    result_df = source_data.read()
    normalized_df = source_data.normalized()

    return result_df, normalized_df, ai, source_data