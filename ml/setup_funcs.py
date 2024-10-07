# Mobility Counts Prediction
# 
import pandas as pd
import numpy as np

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

def calculate_performance_metrics(answer_df_merged):
    """
    Calculate performance metrics comparing 'VOL' and 'Predicted_VOL' in the dataframe.
    """

    # Ensure the 'measurement_tstamp' is in datetime format
    answer_df_merged['measurement_tstamp'] = pd.to_datetime(answer_df_merged['measurement_tstamp'])

    # Count zeros in 'VOL' and 'Predicted_VOL' Columns
    zeros_in_vol = (answer_df_merged['VOL'] == 0).sum()
    zeros_in_pred_vol = (answer_df_merged['Predicted_VOL'] == 0).sum()

    # Rows where either 'VOL' or 'Predicted_VOL' is zero
    zeros_in_either = answer_df_merged[(answer_df_merged['VOL'] == 0) | (answer_df_merged['Predicted_VOL'] == 0)].copy()
    num_rows_with_zeros = len(zeros_in_either)

    # Compute average and median absolute difference for rows with zero in either column
    zeros_in_either['Absolute_Difference'] = (zeros_in_either['Predicted_VOL'] - zeros_in_either['VOL']).abs()
    avg_absolute_difference = zeros_in_either['Absolute_Difference'].mean()
    median_absolute_difference = zeros_in_either['Absolute_Difference'].median()

    # Exclude rows where 'VOL' or 'Predicted_VOL' is zero for percent difference calculations
    df_non_zero = answer_df_merged[(answer_df_merged['VOL'] != 0) & (answer_df_merged['Predicted_VOL'] != 0)].copy()
    total_non_zero_rows = len(df_non_zero)

    # Calculate absolute percent difference overall
    df_non_zero['Percent_Difference'] = ((df_non_zero['Predicted_VOL'] - df_non_zero['VOL']).abs() / df_non_zero['VOL']) * 100

    # Define daytime and nighttime hours
    day_hours = list(range(7, 19))  # 7 AM to 7 PM
    night_hours = list(range(0, 7)) + list(range(19, 24))  # 7 PM to 7 AM

    # Calculate percent difference for daytime
    day_df = df_non_zero[df_non_zero['measurement_tstamp'].dt.hour.isin(day_hours)]
    day_diff = day_df['Percent_Difference'].mean()

    # Calculate percent difference for nighttime
    night_df = df_non_zero[df_non_zero['measurement_tstamp'].dt.hour.isin(night_hours)]
    night_diff = night_df['Percent_Difference'].mean()

    # Calculate percentage within various thresholds for overall
    thresholds = list(range(5, 76, 5))  # 5%, 10%, ..., 75%
    overall_within_percentages = []
    for threshold in thresholds:
        within_threshold = (df_non_zero['Percent_Difference'] <= threshold).mean() * 100
        overall_within_percentages.append(within_threshold)

    # Compile results into a dictionary
    results = {
        'Overall Percent Difference': df_non_zero['Percent_Difference'].mean(),
        'Daytime Percent Difference': day_diff,
        'Nighttime Percent Difference': night_diff,
        'Thresholds': thresholds,
        'Overall Percentage Within': overall_within_percentages,
        'Zeros in VOL': zeros_in_vol,
        'Zeros in Predicted_VOL': zeros_in_pred_vol,
        'Rows with zeros': num_rows_with_zeros,
        'Average Absolute Difference (zeros)': avg_absolute_difference,
        'Median Absolute Difference (zeros)': median_absolute_difference
    }

    return results

# Calculate data metrics when 'VOL' column is not present
def calculate_data_metrics(df):
    max_volume_time = df.loc[df['Predicted_VOL'].idxmax(), 'measurement_tstamp']
    morning_peak = df[df['measurement_tstamp'].dt.hour.between(6, 9)]['Predicted_VOL']
    evening_peak = df[df['measurement_tstamp'].dt.hour.between(16, 19)]['Predicted_VOL']
    metrics = {
        'Max Volume Time': max_volume_time,
        'Average Morning Peak': morning_peak.mean(),
        'Median Morning Peak': morning_peak.median(),
        'Std Morning Peak': morning_peak.std(),
        'Average Evening Peak': evening_peak.mean(),
        'Median Evening Peak': evening_peak.median(),
        'Std Evening Peak': evening_peak.std(),
        'max_volume' : df['Predicted_VOL'].idxmax()
    }
    return metrics


