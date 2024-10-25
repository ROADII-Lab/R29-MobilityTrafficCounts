# Mobility Traffic Counts AI Prediction
# ROADII TRIMS development team

# Import standard libraries 
import base64
from colour import Color
import datetime as dt
import folium
from folium import IFrame
import geopandas as gpd
from geopandas import GeoDataFrame
import logging
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import pytz
import seaborn as sb
from shapely.geometry import Point
from shapely import wkt
from statistics import mean
from string import Template
import streamlit as st
from streamlit_folium import st_folium
import time
import warnings
import wx
import csv 

# Import ROADII team's modules
import module_ai
import module_data
import setup_funcs
from load_shapes import load_shape_csv  # Ensure this function is correctly defined in load_shapes.py

# Enable Wide Mode
st.set_page_config(layout="wide")
# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific Streamlit warnings by setting the logging level to ERROR
logging.getLogger('streamlit').setLevel(logging.ERROR)

# GUI home page
st.title('Mobility Traffic Counts AI Prediction')

# Display current/system datetime
now = dt.datetime.now(pytz.timezone('UTC'))
date_time = now.strftime('%m/%d/%Y, %H:%M:%S')
st.write('Current Datetime is ', date_time, ' UTC')

# Display software build/version - placeholder
st.write('Current Build is v1.0')

# Function to find string index in a list
def find_string_index(alist, search_string):
    alist = list(alist)
    try:
        return alist.index(search_string)
    except ValueError:
        return False

# Measure runtime of train_model or test_model
def run_timer(text, now):
    print('%s%.3f' % (text, time.time()-now))
    return time.time()

# Rotate among commonly-used Streamlit colors
def icon_color_modulo(indx):
    if indx % 8 == 1:
        return 'blue'
    elif indx % 8 == 2:
        return 'green'
    elif indx % 8 == 3:
        return 'orange'
    elif indx % 8 == 4:
        return 'red'
    elif indx % 8 == 5:
        return 'purple'
    elif indx % 8 == 6:
        return 'pink'
    elif indx % 8 == 7:
        return 'beige'
    else:
        return 'gray'

# Color a Streamlit icon with respect to traffic density
def icon_color_quintile(vol, all_vol, pred):
    if pred:  # Predicted data with red gradation
        if vol < np.percentile(all_vol, 20):
            return '#ede8e8'
        elif vol < np.percentile(all_vol, 40):
            return '#d6adad'
        elif vol < np.percentile(all_vol, 60):
            return '#d15f5f'
        elif vol < np.percentile(all_vol, 80):
            return '#c51919'
        else:
            return 'darkred'
    else:  # Input data with blue gradation
        if vol < np.percentile(all_vol, 20):
            return 'lightblue'
        elif vol < np.percentile(all_vol, 40):
            return '#77b5e1'
        elif vol < np.percentile(all_vol, 60):
            return '#3c81e2'
        elif vol < np.percentile(all_vol, 80):
            return '#1242d1'
        else:
            return '#0412a4'

# Filter dataframe by U.S. state
def get_tmcs_state(in_df, state):
    out_df = in_df[in_df['State'] == state]
    out_df = out_df.drop_duplicates(subset='tmc_code', keep='first')
    return out_df

# Filter dataframe by U.S. county
def get_tmcs_county(in_df, county):
    out_df = in_df[in_df['County'] == county]
    out_df = out_df.drop_duplicates(subset='tmc_code', keep='first')
    return out_df

# Filter dataframe by tmc_code
def get_tmcs_tmccode(in_df, tmc_code):
    out_df = in_df[in_df['tmc_code'] == tmc_code]
    return out_df

# Filter dataframe by datetime range
def filter_datetime(in_df, start_date, end_date):
    out_df = in_df[(in_df['measurement_tstamp'] >= start_date) & (in_df['measurement_tstamp'] < end_date)]
    out_df = out_df.drop_duplicates(subset='tmc_code', keep='first')
    return out_df

# Run setup and cache data
@st.cache_data
def setup(filePath):
    ai = module_ai.ai()
    source_data = module_data.data()
    source_data.OUTPUT_FILE_PATH = filePath

    norm_functions = ['tmc_norm', 'tstamp_norm', 'density_norm', 'time_before_after']
    source_data.norm_functions = norm_functions

    result_df = source_data.read()
    normalized_df = source_data.normalized()

    unique_result_df = result_df.drop_duplicates(subset='tmc_code', keep='first')

    st.session_state['result_df'] = result_df
    st.session_state['normalized_df'] = normalized_df
    st.session_state['ai'] = ai
    st.session_state['source_data'] = source_data

    return result_df, normalized_df, ai, source_data

# Function to delete rows where 'State' is empty
def delete_with_minus(df):
    empty_state_indices = df[df['State'].isin(['', ' '])].index
    df.drop(empty_state_indices, inplace=True)
    return df

# Timer function
def lapTimer(text, now):
    print('%s%.3f' % (text, time.time() - now))
    return time.time()

# Function to perform geojoin and generate CSV
def run_step1_geojoin(PATH_TMAS_STATION, OUTPUT_CSVFile, sample_size, PATH_tmc_shp_folder):
    warnings.simplefilter(action='ignore', category=Warning)
    now = time.time()

    crs = {'init': 'epsg:4326'}
    shp = load_shape_csv(PATH_tmc_shp_folder, crs)

    if shp.empty:
        raise ValueError("The loaded shapefile is empty. Please check the directory path and contents.")

    if 'geometry' not in shp.columns:
        possible_geom_cols = ['geom', 'the_geom', 'geometry']
        geom_col = None
        for col in possible_geom_cols:
            if col in shp.columns:
                geom_col = col
                break
        if geom_col:
            shp = shp.set_geometry(geom_col)
        else:
            raise ValueError("No geometry column found in the shapefile CSVs.")

    now = lapTimer('  took: ', now)

    tmas_station = pd.read_csv(PATH_TMAS_STATION, dtype={'STATION_ID': str})
    sample_size = int(sample_size)
    tmas_station = tmas_station.drop_duplicates(subset=['STATION_ID', 'STATE_NAME'])
    if sample_size < len(tmas_station):
        tmas_station_sampled = tmas_station.sample(n=sample_size, random_state=42)
    else:
        tmas_station_sampled = tmas_station
    tmas_station = tmas_station_sampled

    tmas_station['geometry'] = tmas_station.apply(lambda row: Point(row["LONG"], row["LAT"]), axis=1)
    tmas_station.reset_index(drop=True, inplace=True)
    geo_tmas = GeoDataFrame(tmas_station.copy(), crs=crs, geometry='geometry')

    now = lapTimer('  took: ', now)

    # Map DIR strings to numerical values
    direction_mapping = {
        'NORTHBOUND': 1,
        'EASTBOUND': 3,
        'SOUTHBOUND': 5,
        'WESTBOUND': 7
    }
    shp['dir_num'] = shp['DIR'].map(direction_mapping)

    geo_tmas['geometry'] = geo_tmas['geometry'].buffer(0.001)
    intersect = gpd.sjoin(shp, geo_tmas, how='inner', predicate='intersects')

    intersect_dir = intersect[intersect['dir_num'] == intersect['DIR']]
    intersect_dir = intersect_dir[intersect_dir['F_System'] == intersect_dir['F_SYSTEM']]

    tier1 = intersect_dir.loc[:, ['Tmc', 'STATION_ID', 'DIR', 'geometry', 'County', 'State', 'LAT', 'LONG']]
    tier1['tier'] = 1
    tier1 = delete_with_minus(tier1)
    now = lapTimer('  took: ', now)

    if not os.path.exists(os.path.dirname(OUTPUT_CSVFile)):
        os.makedirs(os.path.dirname(OUTPUT_CSVFile))
    tier1.to_csv(OUTPUT_CSVFile, index=False)

    pure_csv_filename = os.path.splitext(os.path.basename(OUTPUT_CSVFile))[0] + '_pure.csv'
    pure_csv_path = os.path.join(os.path.dirname(OUTPUT_CSVFile), pure_csv_filename)
    tier1['Tmc'].to_csv(pure_csv_path, index=False, header=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    tier1.plot(ax=ax, column='tier', cmap='viridis', legend=False)
    ax.set_title('Display of Stations in dataset')
    plt.close(fig)
    return fig

# Function to choose a file using wxPython dialog
def file_picker(label, key, style=wx.FD_OPEN, button_key=None, wildcard="All files (*.*)|*.*"):
    if st.button(f'{label}', key=button_key):
        app = wx.App(False)
        dialog = wx.FileDialog(None, f'Select the {label}:', wildcard=wildcard, style=style)
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
            st.session_state[key] = file_path
            st.write(f"Selected {label}: {file_path}")
        else:
            st.write(f"No {label} selected.")
        dialog.Destroy()
    elif key in st.session_state:
        st.write(f"Selected {label}: {st.session_state[key]}")

def folder_picker(label, key, style=wx.DD_DEFAULT_STYLE, button_key=None):
    if st.button(f'Choose {label}', key=button_key):
        app = wx.App(False)
        dialog = wx.DirDialog(None, f'Select the {label}:', style=style)
        if dialog.ShowModal() == wx.ID_OK:
            dir_path = dialog.GetPath()
            st.session_state[key] = dir_path
            st.write(f"Selected {label}: {dir_path}")
        else:
            st.write(f"No {label} selected.")
        dialog.Destroy()
    elif key in st.session_state:
        st.write(f"Selected {label}: {st.session_state[key]}")

# Function to create and run data joins
def run_joins(mode):
    source_data = module_data.data()

    source_data.tmas = source_data.tmas_data()
    source_data.npmrds = source_data.npmrds_data()
    source_data.tmc = source_data.tmc_data()

    output_folder = st.session_state.get('output_folder_tab2', r'../data')
    dataset_title = st.session_state.get('dataset_title_tab2', 'NPMRDS_TMC_TMAS_US_SUBSET_20_22')
    output_file_path = os.path.join(output_folder, f"{dataset_title}.pkl")
    source_data.OUTPUT_FILE_PATH = output_file_path

    if mode == "Training/Testing Dataset":
        tmas_file_path = st.session_state.get('tmc_station_file_tab1', 'nofile')
        source_data.tmas.TMAS_DATA_FILE = tmas_file_path

        if tmas_file_path.endswith('.pkl') or tmas_file_path == 'nofile':
            source_data.tmas.TMAS_PKL_FILE = tmas_file_path
        else:
            source_data.tmas.TMAS_PKL_FILE = os.path.join(output_folder, f"{dataset_title}_TMAS_Class_Clean.pkl")
            source_data.tmas.TMAS_CSV_FILE = tmas_file_path
            if not os.path.isfile(source_data.tmas.TMAS_PKL_FILE):
                source_data.tmas.read()
    else:
        # Inference Dataset Mode: Set TMAS files to 'nofile'
        source_data.tmas.TMAS_DATA_FILE = 'nofile'
        source_data.tmas.TMAS_PKL_FILE = 'nofile'

    source_data.npmrds.NPMRDS_ALL_FILE = st.session_state.get('npmrds_all_file_tab2', r'..\data\US_ALL_22\all2022_NPMRDS_ALL.csv')
    source_data.npmrds.NPMRDS_PASS_FILE = st.session_state.get('npmrds_pass_file_tab2', r'..\data\US_ALL_22\all2022_NPMRDS_PASS.csv')
    source_data.npmrds.NPMRDS_TRUCK_FILE = st.session_state.get('npmrds_truck_file_tab2', r'..\data\US_ALL_22\all2022_NPMRDS_TRUCK.csv')

    source_data.tmc.TMC_ID_FILE = st.session_state.get('tmc_id_file_tab2', r'..\data\US_ALL_22\TMC_Identification.csv')

    if mode == "Training/Testing Dataset":
        source_data.tmc.TMC_STATION_FILE = st.session_state.get('tmc_station_file_tab1', 'nofile')
    else:
        source_data.tmc.TMC_STATION_FILE = 'nofile'

    source_data.join_and_save()
    st.write("Data joined and saved successfully.")

    # Display success message with output location
    st.success(f"Successfully output to location: **{output_folder}**")

# Function to merge normalized and raw data
def merge_normalized_and_raw_data(raw_df, normalized_df):
    def normalize_tmc_code(df):
        def tmc_value(tmc_code):
            tmc_code = str(tmc_code).lower()
            if 'p' in tmc_code:
                return 0
            elif 'n' in tmc_code:
                return 1
            elif '-' in tmc_code:
                return 2
            elif '+' in tmc_code:
                return 3
            else:
                return -1

        df['TMC_Value'] = df['tmc_code'].apply(tmc_value)
        df['tmc_code_raw'] = df['tmc_code']
        df['tmc_code'] = df['tmc_code'].astype(str).str.lower().str.replace('p', '').str.replace('n', '').str.replace('+', '').str.replace('-', '')
        df['tmc_code'] = df['tmc_code'].astype(int)
        return df

    raw_df = normalize_tmc_code(raw_df)

    normalized_df['measurement_tstamp'] = pd.to_datetime(
        normalized_df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H'
    )

    merged_df = pd.merge(
        raw_df,
        normalized_df,
        on=['tmc_code', 'measurement_tstamp', 'DIR'],
        how='inner',
        suffixes=('_raw', '_norm')
    )

    return merged_df

# Define Tabs in the desired order with numbering
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '0. 0 - Introduction',
    '1. 1 - Generate Dataset',
    '2. 2 - Use a Traffic Counts Model',
    '3. 3 - Results',
    '4. 4 - Train Model',
    '5. 5 - About'
])

# GUI tab #0: Introduction
with tab0:
    st.header('Introduction')
    st.write("""
    **Welcome to the Mobility Traffic Counts AI Prediction Tool!**

    ***Overview***

    The Mobility Traffic Counts code base geographically matches traffic counting station data with probe-collected speed data on the U.S. National Highway System (NHS), to produce training datasets for roadway traffic volume prediction across the entire road system. The code provides a Graphical User Interface (GUI) to easily load input data, select input and target columns, and train a model using basic AI neural network methods.
    
    This code base will make it easier and more approachable for transportation agencies to develop a neural network model to output estimates of historical traffic count data on NHS roadway links for which real world measured counts are not available. This is the case for most NHS roadway links. The intended user base includes state and local agencies looking to produce and use more complete traffic speed and traffic volume datasets. Applications of these resulting datasets and the code in this repository include highway planning projects and highway management projects, as well as future forecasting efforts.

    This application is designed to assist in predicting traffic counts using advanced AI models. Below is an overview of the workflow:

    
    ***Definitions***

    **Highway Performance Monitoring System (HPMS):** 
    A national level highway information system that includes information on the extent, condition, performance, use, and operating characteristics of the nation’s highways. Notably, HPMS includes traffic data reporting, AADT estimates at a segment level, vehicle miles traveled, road type, and urban and rural distinctions.
 
    **Traffic Monitoring Analysis System (TMAS):** 
    An internal FHWA data program that assists in the collection of data on traffic volumes, vehicle classification, and truck weights for traffic statistics and analysis. Reports generated from the data include average daily vehicle traffic patterns for each hour, day, and month. These data are collected at XYZ counting stations....
 
    **National Performance Management Research Data Set (NPMRDS):** 
    A vehicle probe-based travel time data set acquired by the FHWA. The dataset is an archived speed, travel time, and location referencing dataset. It includes hourly speeds and travel times at 5-minute intervals for passenger vehicles, trucks, and a combination of the two. NPMRDS also includes traffic counts as an average for any day of the week for the year.

    ***Workflow***
    1. **Generate Dataset:**
       - Generate a dataset to predict traffic volumes on roads with no existing traffic counting stations (TMAS).
       - Generate a dataset to predict traffic volumes on roads with existing traffic counting stations (TMAS). This is used for testing the performance of the model or training a new model.


    2. **Use a Traffic Counts Model:**
       - Upload or select a pre-trained AI model.
       - Apply the model to the generated dataset to obtain traffic volume predictions.

    3. **Results:**
       - View performance metrics comparing actual and predicted traffic volumes or view predictions for roads with no measured traffic volumes.
       - Explore station locations on an interactive map.

    4. **Train Model:**
       - Train a new AI model using your dataset.
       - Select input features and target variables for model training.

    5. **About:**
       - Access additional resources, documentation, and contact information.

    **Get Started:**
    Navigate through the tabs to generate your dataset, apply AI models, view results, and train new models. Ensure that all required files are available and correctly formatted before proceeding to each step.

    **Need Assistance?**
    Refer to the [GitHub Repository](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/tree/main) for more information or contact our team at William.Chupp@dot.gov or Eric.Englin@dot.gov.
    """)

# GUI tab #1: Generate Dataset
with tab1:
    st.header('Dataset Creation')

    st.write("""
    **Depending on your needs, you can generate an Inference Dataset or a Training/Testing Dataset.**
    
    - **Inference Dataset:** This will create a dataset that can generate predictions for any TMC road segments. This Dataset does not require the use of TMAS Counting Station data that provides existing traffic counts in select locations throughout the USA.
    - **Training/Testing Dataset:** This will create a dataset that only contains road segment data that overlaps with TMAS Counting Station data. This dataset is used for training a new model or testing the performance of an existing model.
    
    Ensure that the year(s) of the TMAS data matches the year(s) of NPMRDS data if generating a Training/Testing Dataset.
    
    **To download the Shapefile for the USA, go to this [website](https://npmrds.ritis.org/analytics/shapefiles) (requires access) and scroll to the bottom of the page to National Shape Files and select the most recent shape file for the United States.**
    """)

    st.markdown('---') 

    # Toggle switch for dataset type
    dataset_mode = st.radio(
        "Select Dataset Type:",
        ("Inference Dataset", "Training/Testing Dataset"),
        index=0
    )

    if dataset_mode == "Training/Testing Dataset":
        st.subheader("Step 1: Geo Join TMC Road Links with TMAS Stations")
        st.write("This step performs a geospatial join between TMC shapefiles and TMAS stations.")

        file_picker('TMAS Station data file', 'PATH_TMAS_STATION', button_key='tmas_station_file_picker_tab1')
        folder_picker('Directory of Shapefiles converted to CSV', 'PATH_tmc_shp', button_key='tmc_shp_folder_picker_tab1')
        sample_size = st.number_input(
            'Sample Size (number of stations to sample, set to 10000 for all stations)',
            min_value=1,
            value=10000,
            step=1
        )

        title_input = st.text_input("Enter a title for the output CSV:", value="default_title")

        if st.button('Run Step 1: Geo Join', key='run_step1_button_tab1'):
            if 'PATH_TMAS_STATION' in st.session_state and 'PATH_tmc_shp' in st.session_state:
                try:
                    PATH_TMAS_STATION = st.session_state['PATH_TMAS_STATION']
                    PATH_tmc_shp_folder = st.session_state['PATH_tmc_shp']

                    processed_title = "TMC_Station_" + title_input.replace(" ", "_")
                    if not processed_title.endswith('.csv'):
                        processed_title += '.csv'

                    OUTPUT_CSVFile = os.path.join(os.getcwd(), 'data', processed_title)
                    if not os.path.exists(os.path.dirname(OUTPUT_CSVFile)):
                        os.makedirs(os.path.dirname(OUTPUT_CSVFile))

                    fig = run_step1_geojoin(
                        PATH_TMAS_STATION=PATH_TMAS_STATION,
                        OUTPUT_CSVFile=OUTPUT_CSVFile,
                        sample_size=sample_size,
                        PATH_tmc_shp_folder=PATH_tmc_shp_folder
                    )

                    st.success(f"`{processed_title}` saved at {OUTPUT_CSVFile}")

                    st.pyplot(fig)

                    tier1 = pd.read_csv(OUTPUT_CSVFile)
                    if 'Tmc' not in tier1.columns:
                        st.error("The 'Tmc' column is missing from the Geo Joined CSV.")
                    else:
                        Tmc_ids = tier1['Tmc'].tolist()

                        pure_csv_filename = os.path.splitext(os.path.basename(OUTPUT_CSVFile))[0] + '_pure.csv'
                        pure_csv_path = os.path.join(os.path.dirname(OUTPUT_CSVFile), pure_csv_filename)
                        tier1['Tmc'].to_csv(pure_csv_path, index=False, header=False)

                        st.success(f"`{pure_csv_filename}` saved at {pure_csv_path}")

                        Tmc_ids_str = ','.join(map(str, Tmc_ids))
                        
                        st.subheader("TMC IDs CSV Output")
                        st.text_area("TMC IDs CSV", value=Tmc_ids_str, height=100, disabled=True)
                        st.write("You can select and copy the text above.")

                        # Automatically set the TMC Station file path for Training/Testing Dataset
                        st.session_state['tmc_station_file_tab1'] = OUTPUT_CSVFile

                        st.success(f"TMC Station file path set to `{OUTPUT_CSVFile}`")
                except Exception as e:
                    st.error(f"An error occurred during the Geo Join process: {e}")
            else:
                st.error("Please choose the TMAS Station data file and the directory of Shapefile CSVs.")



    # Step 2: Download and Extract NPMRDS Data
    st.markdown('---')  

    if dataset_mode == "Inference Dataset":
        st.subheader("Step 1: Download and Extract NPMRDS Data for Inference")
        st.write("""
        **Download and extract NPMRDS data from this [website](https://npmrds.ritis.org/analytics/download/) (requires access).** 

        - Additional info on the use of the NPMRDS RITIS site can be found in the README.

        **The NPMRDS RITIS Massive Data Downloader will give you the NPMRDS All Data, NPMRDS Passenger Data, and NPMRDS Truck Data files as well as the TMC ID file. These will come in 3 separate zip files where the TMC Identification file will be the same in all, but the data files will be different and will need to be identified by opening the readme in each zip. These data files should then be saved and named accordingly so they are not mixed up. These files are selected in Step 3 below.**

        **Tips on Using NPMRDS RITIS Massive Data Downloader:**
        1. **Select Segment Type:** Choose the type of segments you want to download.
        2. **Select Year:** Select the year for your data.
        3. **Select Roads:** Use the built-in tools on the website to select a region or set of roads.
        4. **Select One or More Date Ranges:** Choose the date range of the data you would like to download.
        5. **Select Days of Week:** Select every day of the week (default).
        6. **Select One or More Times of Day:** Select 12:00 AM to 11:59 PM (default).
        7. **Select Data Sets and Measures:** Check the boxes for Passenger, Trucks, and their sub-measures.
        8. **Select Units for Travel Time:** Choose "seconds".
        9. **Volume Data:** Leave unchecked.
        10. **Null Record Handling:** Leave unchecked.
        11. **Select Averaging:** Choose "1 hour".
        12. **Provide Title:** Enter a relevant title for your download.

        **Note:** This mode does not utilize any TMAS data.
        """)
    else:
        st.subheader("Step 2: Download and Extract NPMRDS Data for Training/Testing")
        st.write("""
        Download and extract NPMRDS data from this [website](https://npmrds.ritis.org/analytics/download/) (requires access). 

        - Additional info on the use of the NPMRDS RITIS site can be found in the README.

        **The NPMRDS RITIS Massive Data Downloader will give you the NPMRDS All Data, NPMRDS Passenger Data, and NPMRDS Truck Data files as well as the TMC ID file. These will come in 3 separate zip files where the TMC Identification file will be the same in all, but the data files will be different and will need to be identified by opening the readme in each zip. These data files should then be saved and named accordingly so they are not mixed up. These files are selected in Step 3 below.**

        **Tips on Using NPMRDS RITIS Massive Data Downloader:**
        1. **Select Segment Type:** Choose the type of segments you want to download.
        2. **Select Year:** Select the year of data that aligns with your TMAS data.
        3. **Select Roads:** Paste in comma-separated TMC codes into the "Segment Codes" tab and press "Add Segments".
        4. **Select One or More Date Ranges:** Choose the date range of the data you would like to download.
        5. **Select Days of Week:** Select every day of the week (default).
        6. **Select One or More Times of Day:** Select 12:00 AM to 11:59 PM (default).
        7. **Select Data Sets and Measures:** Check the boxes for Passenger, Trucks, and their sub-measures.
        8. **Select Units for Travel Time:** Choose "seconds".
        9. **Volume Data:** Leave unchecked.
        10. **Null Record Handling:** Leave unchecked.
        11. **Select Averaging:** Choose "1 hour".
        12. **Provide Title:** Enter a relevant title for your download.
    """)

    # Step 3: Select Files to Run Joins Between NPMRDS, TMC, and TMAS Data
    st.markdown('---')

    if dataset_mode == "Inference Dataset":
        st.subheader("Step 2: Select Files to Run Joins Between NPMRDS, TMC, and TMAS Data")
        # Only show necessary file pickers for Inference
        file_picker('NPMRDS All Data file', 'npmrds_all_file_tab2', button_key='npmrds_all_file_picker_tab1')
        file_picker('NPMRDS Passenger Data file', 'npmrds_pass_file_tab2', button_key='npmrds_pass_file_picker_tab1')
        file_picker('NPMRDS Truck Data file', 'npmrds_truck_file_tab2', button_key='npmrds_truck_file_picker_tab1')
        file_picker('TMC ID file', 'tmc_id_file_tab2', button_key='tmc_id_file_picker_tab1')
    else:
        st.subheader("Step 3: Select Files to Run Joins Between NPMRDS, TMC, and TMAS Data")
        # Show all file pickers for Training/Testing
        # TMCS Station file is already set after Step 1
        st.write("**TMC Station File:**")
        if 'tmc_station_file_tab1' in st.session_state:
            st.write(f"Selected TMC Station File: {st.session_state['tmc_station_file_tab1']}")
        else:
            st.write("TMC Station file will be set after running Step 1.")

        file_picker('NPMRDS All Data file', 'npmrds_all_file_tab2', button_key='npmrds_all_file_picker_tab1')
        file_picker('NPMRDS Passenger Data file', 'npmrds_pass_file_tab2', button_key='npmrds_pass_file_picker_tab1')
        file_picker('NPMRDS Truck Data file', 'npmrds_truck_file_tab2', button_key='npmrds_truck_file_picker_tab1')
        file_picker('TMC ID file', 'tmc_id_file_tab2', button_key='tmc_id_file_picker_tab1')

    # Step 4: Choose the Output File Path
    st.markdown('---') 

    st.subheader("Final Step: Choose the Output File Path and Press **Join and Save Data** button.")
    
    if 'output_folder_tab2' not in st.session_state:
        st.session_state['output_folder_tab2'] = ''
    if 'dataset_title_tab2' not in st.session_state:
        st.session_state['dataset_title_tab2'] = ''

    col1, col2 = st.columns([2, 3])

    with col1:
        folder_picker('Choose Output Folder', 'output_folder_tab2', button_key='output_folder_picker_tab1')

    with col2:
        st.text_input("Enter a title for the output dataset:", value="NPMRDS_TMC_TMAS_US_SUBSET_20_22", key='dataset_title_tab2')

    if st.button('**Join and Save Data**', key='join_and_save_button_tab1'):
        if dataset_mode == "Inference Dataset":
            required_keys = [
                'npmrds_all_file_tab2', 
                'npmrds_pass_file_tab2', 
                'npmrds_truck_file_tab2', 
                'tmc_id_file_tab2', 
                'output_folder_tab2', 
                'dataset_title_tab2'
            ]
        else:
            required_keys = [
                'tmc_station_file_tab1',
                'npmrds_all_file_tab2', 
                'npmrds_pass_file_tab2', 
                'npmrds_truck_file_tab2', 
                'tmc_id_file_tab2', 
                'output_folder_tab2', 
                'dataset_title_tab2'
            ]
        
        # Check if all required keys are present and not empty
        missing_keys = [key for key in required_keys if key not in st.session_state or not st.session_state[key]]
        if not missing_keys:
            try:
                output_folder = st.session_state['output_folder_tab2']
                dataset_title = st.session_state['dataset_title_tab2']
                if not output_folder:
                    st.error("Please select an output folder.")
                elif not dataset_title:
                    st.error("Please enter a title for the dataset.")
                else:
                    run_joins(dataset_mode)
                    # Success message is handled inside run_joins
            except Exception as e:
                st.error(f"An error occurred during the join process: {e}")
        else:
            if dataset_mode == "Inference Dataset":
                st.error("Please ensure all required NPMRDS files, output folder, and dataset title are provided before joining.")
            else:
                st.error("Please ensure all required NPMRDS files, TMC Station file, output folder, and dataset title are provided before joining.")

# GUI tab #2: Use a Traffic Counts Model
with tab2:
    st.header('Use a Traffic Counts Model')
    
# Input column(s) and target column GUI buttons (previous single window)
    folder_picker('Choose Model Storage Location', 'model_storage_path', button_key='model_storage_path_picker')
    ai = module_ai.ai()
    if 'model_storage_path' in st.session_state:
        list_of_model_files = ai.get_model_list(st.session_state['model_storage_path'])
    else: list_of_model_files = []
    model_filename = st.selectbox(label='Choose a model file', options=list_of_model_files)

    # File picker for generated dataset path
    file_picker('Choose Generated Dataset (.pkl)', 'raw_dataset_path', button_key='raw_dataset_picker_tab2', wildcard="Pickle files (*.pkl)|*.pkl")

    if 'raw_dataset_path' in st.session_state:
        raw_dataset_path = st.session_state['raw_dataset_path']

        try:
            t_result_df, t_normalized_df, t_ai, t_source_data = setup(raw_dataset_path)
            st.success("Dataset loaded successfully.")
        except Exception as e:
            st.error(f"Error loading the dataset: {e}")
            t_result_df, t_normalized_df, t_ai, t_source_data = pd.DataFrame(), pd.DataFrame(), module_ai.ai(), module_data.data()

        if not t_result_df.empty and not t_normalized_df.empty and model_filename:
            if st.button('Use Model', key='use_model_button_tab2'):
                st.write('Running loaded model on test dataset...')
                try:
                    columns = [
                        'tmc_code', 'measurement_tstamp', 'speed_All', 'data_density_All',
                        'data_density_Pass', 'data_density_Truck', 'travel_time_seconds_All', 'start_latitude',
                        'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'aadt', 'urban_code',
                        'thrulanes_unidir', 'f_system', 'route_sign', 'thrulanes', 'zip', 'DIR'
                    ]
                    columns.extend(t_source_data.calculated_columns)

                    # Convert columns to numeric and handle errors if present
                    for col in t_normalized_df.columns:
                        try:
                            t_normalized_df[col] = pd.to_numeric(t_normalized_df[col], errors='coerce')
                        except Exception as e:
                            print(f"Error converting column {col}: {e}")
                    

                    answer_df = setup_funcs.use_model(t_ai, model_filename, t_normalized_df, columns, 'VOL')
                    if answer_df.empty:
                        st.write("Model or data was not properly loaded. Please check your inputs.")
                    else:
                        answer_df_merged = merge_normalized_and_raw_data(t_result_df, answer_df)
                        output_file_path = os.path.join('data', f"{os.path.splitext(os.path.basename(raw_dataset_path))[0]}_predictions.pkl")
                        answer_df_merged.to_pickle(output_file_path)
                        st.session_state['answer_df_merged'] = answer_df_merged
                        st.success(f"DataFrame successfully output to location: **{output_file_path}** -- View Results in the **Results** Tab")
                except Exception as e:
                    st.error(f"An error occurred while using the model: {e}")
        else:
            st.write("Please ensure both the model file and the generated dataset are selected.")

# Cache data preparation for the map
@st.cache_data
def prepare_map_data(df):
    if 'start_latitude_norm' in df.columns and 'start_longitude_norm' in df.columns:
        lat_col, lon_col = 'start_latitude_norm', 'start_longitude_norm'
    elif 'start_latitude' in df.columns and 'start_longitude' in df.columns:
        lat_col, lon_col = 'start_latitude', 'start_longitude'
    else:
        return None, None, None

    return df, lat_col, lon_col

# Generate and cache the folium map
@st.cache_resource
def create_folium_station_map(df):
    # Use a reduced dataset or sampling to minimize map complexity
    sample_df = df[['start_latitude_norm', 'start_longitude_norm', 'tmc_code']].dropna().sample(500)  # Limit to 500 points

    # Create a folium map centered on the USA
    map_center = [39.8283, -98.5795]  # Approximate center of the USA
    folium_map = folium.Map(location=map_center, zoom_start=4)

    # Add points to the map
    for _, row in sample_df.iterrows():
        folium.Marker(
            location=[row['start_latitude_norm'], row['start_longitude_norm']],
            popup=row['tmc_code'],
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(folium_map)

    return folium_map

# Extract unique TMC codes and dates for dropdowns
@st.cache_data
def get_unique_tmc_codes_and_dates(df):
    tmc_codes = df['tmc_code'].unique()
    dates = df['measurement_tstamp'].dt.date.unique()
    return sorted(tmc_codes), sorted(dates)

# Cache the filtered DataFrame
@st.cache_data
def filter_data_by_tmc_and_date(df, selected_tmc, selected_date):
    filtered_df = df[(df['tmc_code'] == selected_tmc) & (df['measurement_tstamp'].dt.date == selected_date)]
    return filtered_df

# GUI tab #3: Results
with tab3:
    st.header('Traffic Counts Prediction Results')
    
    st.subheader("Upload a Predictions File")
    if st.button('Choose Predictions File', key='choose_predictions_file_tab3'):
        app = wx.App(False)
        dialog = wx.FileDialog(None, 'Select a _predictions.pkl File:', wildcard="Pickle files (*.pkl)|*.pkl", style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            uploaded_file_path = dialog.GetPath()
            st.session_state['uploaded_predictions_path'] = uploaded_file_path
            st.write(f"Selected Predictions File: {uploaded_file_path}\nAttempting to Load...")
        else:
            st.write("No Predictions File selected.")
        dialog.Destroy()
    
    if 'uploaded_predictions_path' in st.session_state:
        uploaded_file_path = st.session_state['uploaded_predictions_path']
        try:
            uploaded_df = pd.read_pickle(uploaded_file_path)
            st.session_state['uploaded_df'] = uploaded_df
            st.success("Uploaded predictions file loaded successfully.")
        except Exception as e:
            st.error(f"Error loading the uploaded file: {e}")

    df_to_display = None

    if 'answer_df_merged' in st.session_state:
        df_to_display = st.session_state['answer_df_merged']

    if 'uploaded_df' in st.session_state:
        df_to_display = st.session_state['uploaded_df']

    if df_to_display is not None and not df_to_display.empty:
        # Calculate overall performance metrics for the entire dataset
        if 'VOL' in df_to_display.columns:
            overall_performance_metrics = setup_funcs.calculate_performance_metrics(df_to_display)
        else:
            overall_performance_metrics = setup_funcs.calculate_data_metrics(df_to_display)

        # Display overall performance metrics
        st.header('Overall Performance Metrics')
        if 'VOL' in df_to_display.columns:
            st.write(f"**Overall Percent Difference:** {overall_performance_metrics['Overall Percent Difference']:.2f}%")
            st.write(f"**Daytime Percent Difference (7 AM - 7 PM):** {overall_performance_metrics['Daytime Percent Difference']:.2f}%")
            st.write(f"**Nighttime Percent Difference (7 PM - 7 AM):** {overall_performance_metrics['Nighttime Percent Difference']:.2f}%")
            st.write(f"**Number of zeros in 'VOL':** {overall_performance_metrics['Zeros in VOL']}")
            st.write(f"**Number of zeros in 'Predicted_VOL':** {overall_performance_metrics['Zeros in Predicted_VOL']}")
            st.write(f"**Number of rows with zero in either 'VOL' or 'Predicted_VOL':** {overall_performance_metrics['Rows with zeros']}")
            st.write(f"**Average absolute difference for zeros:** {overall_performance_metrics['Average Absolute Difference (zeros)']:.2f}")
            st.write(f"**Median absolute difference for zeros:** {overall_performance_metrics['Median Absolute Difference (zeros)']:.2f}")
            
            # Time of maximum difference and TMC code
            max_diff_index = (df_to_display['VOL'] - df_to_display['Predicted_VOL']).abs().idxmax()
            max_diff_row = df_to_display.loc[max_diff_index]
            st.write(f"**Time of Maximum Difference:** {max_diff_row['measurement_tstamp']}")
            st.write(f"**TMC Code of Maximum Difference:** {max_diff_row['tmc_code']}")
            
            # Percentage within Error Thresholds plot
            if 'Thresholds' in overall_performance_metrics and 'Overall Percentage Within' in overall_performance_metrics:
                st.header('Percentage Within Error Thresholds (Excluding Zeros)')
                fig_overall = px.line(
                    x=overall_performance_metrics['Thresholds'], 
                    y=overall_performance_metrics['Overall Percentage Within'],
                    labels={'x': 'Error Threshold (%)', 'y': 'Percentage of Data Points Within Threshold (%)'},
                    title='Percentage of Data Points Within Error Thresholds (Excluding Zeros)',
                    markers=True
                )
                # Use container width
                st.plotly_chart(fig_overall, use_container_width=True)
        else:
            st.write(f"**Average Morning (6 AM - 9 AM):** {overall_performance_metrics['Average Morning Peak']:.2f}")
            st.write(f"**Average Evening  (4 PM - 7 PM):** {overall_performance_metrics['Average Evening Peak']:.2f}")
            # Include any other overall metrics you wish to display

        st.header('Visualize Predictions')

        # Extract unique TMC codes for dropdown
        @st.cache_data
        def get_unique_tmc_codes(df):
            tmc_codes = df['tmc_code'].unique()
            return sorted(tmc_codes)

        # Cache function to get dates for a given TMC code
        @st.cache_data
        def get_dates_for_tmc(df, selected_tmc):
            dates = df[df['tmc_code'] == selected_tmc]['measurement_tstamp'].dt.date.unique()
            return sorted(dates)

        # Create dropdown for TMC codes
        tmc_codes = get_unique_tmc_codes(df_to_display)
        selected_tmc = st.selectbox('Select TMC Code', tmc_codes)

        # After TMC code is selected, get available dates
        if selected_tmc:
            dates = get_dates_for_tmc(df_to_display, selected_tmc)
            if dates:
                selected_date = st.selectbox('Select Date', dates)
            else:
                st.write("No dates available for the selected TMC code.")
                selected_date = None
        else:
            selected_date = None

        # Proceed only if a date is selected
        if selected_date is not None:
            # Filter the data for the selected TMC code and date
            filtered_df = filter_data_by_tmc_and_date(df_to_display, selected_tmc, selected_date)

            if filtered_df.empty:
                st.write("No data available for the selected TMC code and date.")
            else:
                # Display the direction (assumed to be unique for each TMC code)
                direction = filtered_df['DIR'].iloc[0]
                st.write(f"Direction: {direction}")
                
                # Prepare the data for visualization
                filtered_df['hour'] = filtered_df['measurement_tstamp'].dt.hour

                if 'VOL' in df_to_display.columns:
                    # Case: 'VOL' column is present
                    st.write("Comparing 'VOL' and 'Predicted_VOL'")
                    avg_df = filtered_df.groupby('hour', as_index=False).agg({'VOL': 'mean', 'Predicted_VOL': 'mean'})
                    fig = px.line(
                        avg_df, 
                        x='hour', 
                        y=['VOL', 'Predicted_VOL'],
                        labels={'value': 'Traffic Counts', 'variable': 'Legend', 'hour': 'Hour of Day'},
                        title=f'Average Traffic Counts for TMC Code {selected_tmc} on {selected_date}',
                        markers=True
                    )
                    # Use container width
                    st.plotly_chart(fig, use_container_width=True)

                    # Performance metrics for the selected day
                    performance_metrics = setup_funcs.calculate_performance_metrics(filtered_df)

                    if performance_metrics:
                        st.header(f'Performance Metrics for {selected_date}')
                        st.write(f"**Overall Percent Difference:** {performance_metrics['Overall Percent Difference']:.2f}%")
                        st.write(f"**Daytime Percent Difference (7 AM - 7 PM):** {performance_metrics['Daytime Percent Difference']:.2f}%")
                        st.write(f"**Nighttime Percent Difference (7 PM - 7 AM):** {performance_metrics['Nighttime Percent Difference']:.2f}%")
                else:
                    # Case: No 'VOL' column present
                    st.write("Displaying Predicted Traffic Counts")
                    avg_df = filtered_df.groupby('hour', as_index=False).agg({'Predicted_VOL': 'mean'})
                    fig = px.line(
                        avg_df, 
                        x='hour', 
                        y='Predicted_VOL',
                        labels={'Predicted_VOL': 'Predicted Traffic Counts', 'hour': 'Hour of Day'},
                        title=f'Average Predicted Traffic Counts for TMC Code {selected_tmc} on {selected_date}',
                        markers=True
                    )
                    # Use container width
                    st.plotly_chart(fig, use_container_width=True)

                    # Data metrics for the selected day
                    data_metrics = setup_funcs.calculate_data_metrics(filtered_df)
                    if data_metrics:
                        st.header(f'Metrics for {selected_date}')
                        st.write(f"**Average Morning (6 AM - 9 AM):** {data_metrics['Average Morning Peak']:.2f}")
                        st.write(f"**Average Evening (4 PM - 7 PM):** {data_metrics['Average Evening Peak']:.2f}")

        else:
            st.write("Please select a date to proceed.")

        # Generate and display the folium map
        if df_to_display is not None and not df_to_display.empty:
            if 'station_map' not in st.session_state:
                st.session_state['station_map'] = create_folium_station_map(df_to_display)

            st.header('Map of Stations in dataset')
            # Adjust the width to use the full container width
            st_folium(st.session_state['station_map'], width='100%', height=500)
        else:
            st.write("No data available to display the map.")

# GUI tab #4: Train Model
with tab4:
    st.header('Train a Traffic Counts Model')
    
    if st.button('Choose Source Data File', key='train_model_source_file_tab4'):
        app = wx.App(False)
        dialog = wx.FileDialog(None, 'Select a File:', style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
            try:
                result_df, normalized_df, ai, source_data = setup(file_path)
                st.session_state['train_model_source_file_tab4'] = file_path
                st.success("Source data file loaded successfully.")
            except Exception as e:
                st.error(f"Error loading the source data file: {e}")
                result_df, normalized_df, ai, source_data = pd.DataFrame(), pd.DataFrame(), module_ai.ai(), module_data.data()
        else:
            st.write("No source data file selected.")
            result_df, normalized_df, ai, source_data = pd.DataFrame(), pd.DataFrame(), module_ai.ai(), module_data.data()
        dialog.Destroy()
    elif 'train_model_source_file_tab4' not in st.session_state:
        result_df = pd.DataFrame()
        normalized_df = pd.DataFrame()
        ai = module_ai.ai()
        source_data = module_data.data()
    else:
        file_path = st.session_state.get('train_model_source_file_tab4', None)
        if file_path:
            try:
                result_df, normalized_df, ai, source_data = setup(file_path)
                st.success("Source data file loaded successfully.")
            except Exception as e:
                st.error(f"Error loading the source data file: {e}")
                result_df, normalized_df, ai, source_data = pd.DataFrame(), pd.DataFrame(), module_ai.ai(), module_data.data()
        else:
            result_df = pd.DataFrame()
            normalized_df = pd.DataFrame()
            ai = module_ai.ai()
            source_data = module_data.data()

    if 'result_df' in st.session_state and not st.session_state['result_df'].empty:
        st.header('Input Data')
        st.dataframe(st.session_state['result_df'].head(50))
    else:
        st.header('Input Data')
        st.write("No data loaded.")

    if 'normalized_df' in st.session_state and not st.session_state['normalized_df'].empty:
        st.header('Input Data Normalized for AI Training')
        st.dataframe(st.session_state['normalized_df'].head(50))
    else:
        st.header('Input Data Normalized for AI Training')
        st.write("No normalized data available.")

    if 'source_data' in st.session_state:
        cols1 = st.session_state['source_data'].features_training_set
        in_cols = st.multiselect(
            label='Choose input column(s) (select one or more):', 
            options=cols1, 
            default=st.session_state['source_data'].features_training_set
        )
    else:
        in_cols = []

    if 'normalized_df' in st.session_state and not st.session_state['normalized_df'].empty:
        cols2 = st.session_state['normalized_df'].columns
        default_target = st.session_state['source_data'].features_target if 'source_data' in st.session_state else (cols2[0] if len(cols2) > 0 else None)
        target_col_index = find_string_index(cols2, default_target) if default_target else 0
        target_col = st.selectbox(
            label='Choose target column (select only one):', 
            options=cols2, 
            index=target_col_index if target_col_index is not False else 0
        )
    else:
        target_col = None

    if st.button('Train Model', key='train_model_button_tab4'):
        if 'ai' in st.session_state and 'normalized_df' in st.session_state and in_cols and target_col:
            st.write('Model training started...')
            try:
                result = setup_funcs.train_model(
                    st.session_state['ai'], 
                    st.session_state['normalized_df'], 
                    in_cols, 
                    target_col
                )
                st.write(result)
                st.success("Model training completed successfully.")
            except Exception as e:
                st.error(f"An error occurred during model training: {e}")
        else:
            st.error("Please load the data and select input columns and target column before training the model.")

# GUI tab #5: About
with tab5:
    st.header('About')
    
    st.write('Visit our [GitHub](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/tree/main) for more information.')
    st.write('Download our [Readme](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/blob/main/resources/readme.pdf).')
    st.write('Questions? Contact William.Chupp@dot.gov or Eric.Englin@dot.gov.')
