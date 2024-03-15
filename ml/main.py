# Mobility Counts Prediction
# 
import streamlit as st
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
    x_train, y_train, x_test, y_test = ai.format_training_data(normalized_df)

    # load the model or use a model that's already loaded
    if ai.model != None:
        ai.test(ai.model, x_test, y_test)
    else:
        if(ai.model_load(x_test)):
            predictions, y_test, test_loss, test_accuracy = ai.test(ai.model, x_test, y_test)
            return test_accuracy
        else:
            print("No model loaded!")
            return 0

    return test_accuracy

def find_string_index(alist, search_string):
    alist = list(alist)
    try:
        return alist.index(search_string)
    except ValueError:
        return False

@st.cache_data
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

# run setup
census_df, result_df, normalized_df, ai, source_data = setup()

# UI title
st.title("Mobility Counts Prediction")

# setup a display of the raw input data
st.header("Raw Data")
st.dataframe(result_df[0:50])

st.header("Normalized Training Data")
st.dataframe(normalized_df[0:50])

#choose input features
cols1 = source_data.features_training_set
in_cols = st.multiselect(label = "Choost input columns", options=cols1, default=source_data.features_training_set)

# Target Col Dropdown
cols2 = normalized_df.columns
target_col = st.selectbox(label = 'Choose a target column', options= cols2, index=find_string_index(cols2, source_data.features_target))
# UI buttons
col1, col2 = st.columns(2)
with col1:
    if st.button('Train Model'):
        st.write('Model training started...')
        result = train_model(ai, normalized_df, in_cols, target_col)
        st.write(result)

with col2:
    list_of_model_files = ai.get_model_list('.')
    ai.model_filename = st.selectbox(label = 'Choose a model file', options=list_of_model_files)
    if st.button('Test Model'):
        st.write('Model testing started...')
        test_accuracy = test_model(ai, normalized_df, in_cols, target_col)
        st.write('Model Accuracy = ', test_accuracy)