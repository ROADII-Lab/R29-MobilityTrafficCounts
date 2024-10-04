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
import csv  # Added for CSV writing

# Import ROADII team's modules
import module_ai
import module_data
import setup_funcs
from load_shapes import load_shape_csv  # Ensure this function is correctly defined in load_shapes.py

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific Streamlit warnings by setting the logging level to ERROR
logging.getLogger('streamlit').setLevel(logging.ERROR)

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
    unique_result_df['geometry'] = unique_result_df['geometry'].apply(wkt.loads)
    geo_df = gpd.GeoDataFrame(unique_result_df, geometry='geometry')

    st.session_state['result_df'] = result_df
    st.session_state['normalized_df'] = normalized_df
    st.session_state['ai'] = ai
    st.session_state['source_data'] = source_data
    st.session_state['geo_df'] = geo_df

    return result_df, normalized_df, ai, source_data, geo_df

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

    shp['dir_num'] = np.nan
    shp.loc[shp['Direction'] == 'N', 'dir_num'] = 1
    shp.loc[shp['Direction'] == 'E', 'dir_num'] = 3
    shp.loc[shp['Direction'] == 'S', 'dir_num'] = 5
    shp.loc[shp['Direction'] == 'W', 'dir_num'] = 7

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
def file_picker(label, key, style=wx.FD_OPEN, button_key=None):
    if st.button(f'Choose {label}', key=button_key):
        app = wx.App(False)
        dialog = wx.FileDialog(None, f'Select the {label}:', style=style)
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
def run_joins():
    source_data = module_data.data()

    source_data.tmas = source_data.tmas_data()
    source_data.npmrds = source_data.npmrds_data()
    source_data.tmc = source_data.tmc_data()

    output_folder = st.session_state.get('output_folder_tab4', r'../data')
    dataset_title = st.session_state.get('dataset_title_tab4', 'NPMRDS_TMC_TMAS_US_SUBSET_20_22')
    output_file_path = os.path.join(output_folder, f"{dataset_title}.pkl")
    source_data.OUTPUT_FILE_PATH = output_file_path

    tmas_file_path = st.session_state.get('PATH_TMAS_STATION', r'nofile')
    source_data.tmas.TMAS_DATA_FILE = tmas_file_path

    if tmas_file_path.endswith('.pkl') or tmas_file_path == 'nofile':
        source_data.tmas.TMAS_PKL_FILE = tmas_file_path
    else:
        source_data.tmas.TMAS_PKL_FILE = os.path.join(output_folder, f"{dataset_title}_TMAS_Class_Clean.pkl")
        source_data.tmas.TMAS_CSV_FILE = tmas_file_path
        if not os.path.isfile(source_data.tmas.TMAS_PKL_FILE):
            source_data.tmas.read()

    source_data.npmrds.NPMRDS_ALL_FILE = st.session_state.get('npmrds_all_file', r'..\data\US_ALL_22\all2022_NPMRDS_ALL.csv')
    source_data.npmrds.NPMRDS_PASS_FILE = st.session_state.get('npmrds_pass_file', r'..\data\US_ALL_22\all2022_NPMRDS_PASS.csv')
    source_data.npmrds.NPMRDS_TRUCK_FILE = st.session_state.get('npmrds_truck_file', r'..\data\US_ALL_22\all2022_NPMRDS_TRUCK.csv')

    source_data.tmc.TMC_STATION_FILE = st.session_state.get('tmc_station_file', r'nofile')
    source_data.tmc.TMC_ID_FILE = st.session_state.get('tmc_id_file', r'..\data\US_ALL_22\TMC_Identification.csv')
    source_data.join_and_save()
    st.write("Data joined and saved successfully.")

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

# Define Tabs in the desired order
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '1. Use a Traffic Counts Model',
    '2. Results',
    '3. Generate Dataset',
    '4. Train Model',
    '5. About'
])

# GUI tab #1: Use a Traffic Counts Model
with tab1:
    st.header('Use a Traffic Counts Model')
    
    folder_picker('Choose Model Storage Location', 'model_storage_path', button_key='model_storage_path_picker')

    if 'model_storage_path' in st.session_state:
        ai = module_ai.ai()
        list_of_model_files = ai.get_model_list(st.session_state['model_storage_path'])
    else:
        list_of_model_files = []
    if list_of_model_files:
        model_filename = st.selectbox(label='Choose a model file', options=list_of_model_files)
    else:
        model_filename = None
        st.write("No model files found in the selected directory.")

    file_picker('Created dataset - NPMRDS + TMC + TMAS (.pkl file)', 'raw_dataset_path', button_key='raw_dataset_picker')

    if 'raw_dataset_path' in st.session_state:
        raw_dataset_path = st.session_state['raw_dataset_path']

        t_result_df, t_normalized_df, t_ai, t_source_data, t_geo_df = setup(raw_dataset_path)

        if 'VOL' in t_result_df.columns:
            st.write("The 'VOL' column is present in the dataset. Proceeding with model usage.")
            
            columns = [
                'tmc_code', 'measurement_tstamp', 'speed_All', 'data_density_All',
                'data_density_Pass', 'data_density_Truck', 'travel_time_seconds_All', 'start_latitude',
                'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'aadt', 'urban_code',
                'thrulanes_unidir', 'f_system', 'route_sign', 'thrulanes', 'zip', 'DIR'
            ]
            
            columns.extend(t_source_data.calculated_columns)

            base_name = os.path.basename(raw_dataset_path)
            name, ext = os.path.splitext(base_name)

            output_file_path = os.path.join('data', f"{name}_predictions{ext}")

            if model_filename:
                if st.button('Use Model'):
                    st.write('Running loaded model on test dataset...')
                    answer_df = setup_funcs.use_model(t_ai, model_filename, t_normalized_df, columns, 'VOL')
                    if answer_df.empty:
                        st.write("Model or data was not properly loaded. Please check your inputs.")
                    else:
                        answer_df_merged = merge_normalized_and_raw_data(t_result_df, answer_df)
                        answer_df_merged.to_pickle(output_file_path)
                        st.session_state['answer_df_merged'] = answer_df_merged
                        st.success(f"DataFrame saved to {output_file_path} -- View Results in the **Results** Tab")
            else:
                st.error("Please select a valid model file before using the model.")

        else:
            st.write("The 'VOL' column is not present in the dataset. Generating predictions using the selected model.")
            
            columns = [
                'tmc_code', 'measurement_tstamp', 'speed_All', 'data_density_All',
                'data_density_Pass', 'data_density_Truck', 'travel_time_seconds_All', 'start_latitude',
                'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'aadt', 'urban_code',
                'thrulanes_unidir', 'f_system', 'route_sign', 'thrulanes', 'zip', 'DIR'
            ]
            columns.extend(t_source_data.calculated_columns)

            if model_filename:
                if st.button('Generate Predictions'):
                    st.write('Generating predictions using the selected model...')
                    predictions = setup_funcs.use_model(t_ai, model_filename, t_normalized_df, columns, None)
                    if predictions.empty:
                        st.write("Failed to generate predictions. Please check your inputs.")
                    else:
                        predictions_merged = merge_normalized_and_raw_data(t_result_df, predictions)
                        output_file_path = os.path.join('data', f"{name}_predictions{ext}")
                        predictions_merged.to_pickle(output_file_path)
                        st.session_state['predictions_merged'] = predictions_merged
                        st.success(f"Predictions saved to {output_file_path}")
            else:
                st.error("Please select a valid model file before generating predictions.")

# GUI tab #2: Results
with tab2:
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
            st.success("Uploaded predictions file loaded successfully. Generating Performance Metrics and Visualizations.")
        except Exception as e:
            st.error(f"Error loading the uploaded file: {e}")

    df_to_display = None
    performance_metrics = None

    if 'answer_df_merged' in st.session_state:
        df_to_display = st.session_state['answer_df_merged']
        performance_metrics = setup_funcs.calculate_performance_metrics(df_to_display)

    if 'uploaded_df' in st.session_state:
        df_to_display = st.session_state['uploaded_df']
        if 'VOL' in df_to_display.columns:
            performance_metrics = setup_funcs.calculate_performance_metrics(df_to_display)
        else:
            performance_metrics = setup_funcs.calculate_performance_metrics_no_vol(df_to_display)

    if df_to_display is not None:
        if 'VOL' in df_to_display.columns:
            st.write("Comparing 'VOL' and 'Predicted_VOL'")
        else:
            st.write("Displaying 'Predicted_VOL' data")
        
        if performance_metrics is not None:
            st.header('Performance Metrics')
            if 'Overall Percent Difference' in performance_metrics:
                st.write(f"**Overall Percent Difference:** {performance_metrics['Overall Percent Difference']:.2f}%")
            if 'Daytime Percent Difference' in performance_metrics:
                st.write(f"**Daytime Percent Difference:** {performance_metrics['Daytime Percent Difference']:.2f}%")
            if 'Nighttime Percent Difference' in performance_metrics:
                st.write(f"**Nighttime Percent Difference:** {performance_metrics['Nighttime Percent Difference']:.2f}%")
            if 'Zeros in VOL' in performance_metrics:
                st.write(f"**Number of zeros in 'VOL':** {performance_metrics['Zeros in VOL']}")
            if 'Zeros in Predicted_VOL' in performance_metrics:
                st.write(f"**Number of zeros in 'Predicted_VOL':** {performance_metrics['Zeros in Predicted_VOL']}")
            if 'Rows with zeros' in performance_metrics:
                st.write(f"**Number of rows with zero in either 'VOL' or 'Predicted_VOL':** {performance_metrics['Rows with zeros']}")
            if 'Average Absolute Difference (zeros)' in performance_metrics:
                st.write(f"**Average absolute difference for zeros:** {performance_metrics['Average Absolute Difference (zeros)']:.2f}")
            if 'Median Absolute Difference (zeros)' in performance_metrics:
                st.write(f"**Median absolute difference for zeros:** {performance_metrics['Median Absolute Difference (zeros)']:.2f}")
            
            if 'Thresholds' in performance_metrics and 'Overall Percentage Within' in performance_metrics:
                st.header('Percentage Within Error Thresholds (Excluding Zeros)')
                # Use Plotly Express to create the line chart with axis titles
                fig = px.line(x=performance_metrics['Thresholds'], y=performance_metrics['Overall Percentage Within'],
                              labels={'x': 'Error Threshold (%)', 'y': 'Percentage of Data Points Within Threshold (%)'},
                              title='Percentage of Data Points Within Error Thresholds (Excluding Zeros)',
                              markers=True)
                st.plotly_chart(fig)
    
        st.header('Visualize Predictions vs Measured Values')
        if 'tmc_code_raw' in df_to_display.columns:
            tmc_codes = df_to_display['tmc_code_raw'].unique()
            tmc_code = st.selectbox('Select TMC Code', tmc_codes)
        else:
            # Create 'tmc_code_raw' if it doesn't exist
            if 'tmc_code' in df_to_display.columns:
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

                df_to_display['TMC_Value'] = df_to_display['tmc_code'].apply(tmc_value)
                df_to_display['tmc_code_raw'] = df_to_display['tmc_code']
                df_to_display['tmc_code'] = df_to_display['tmc_code'].astype(str).str.lower().str.replace('p', '').str.replace('n', '').str.replace('+', '').str.replace('-', '')
                df_to_display['tmc_code'] = df_to_display['tmc_code'].astype(int)
                st.session_state['uploaded_df'] = df_to_display  # Update the session state
                tmc_codes = df_to_display['tmc_code_raw'].unique()
                tmc_code = st.selectbox('Select TMC Code', tmc_codes)
            else:
                st.write("No TMC codes available in the data.")
                tmc_code = None

        if tmc_code:
            available_directions = df_to_display[df_to_display['tmc_code_raw'] == tmc_code]['DIR'].unique()
            direction = st.selectbox('Select Direction', available_directions)

            date1 = st.date_input('Select Date:', value=dt.datetime.today())

            filtered_df = df_to_display[
                (df_to_display['tmc_code_raw'] == tmc_code) & 
                (df_to_display['DIR'] == direction) & 
                (df_to_display['measurement_tstamp'].dt.date == date1)
            ]
            if filtered_df.empty:
                st.write("No data available for the selected TMC code, direction, and date.")
            else:
                filtered_df['hour'] = filtered_df['measurement_tstamp'].dt.hour
                if 'VOL' in df_to_display.columns:
                    avg_df = filtered_df.groupby('hour', as_index=False).agg({'VOL': 'mean', 'Predicted_VOL': 'mean'})
                else:
                    avg_df = filtered_df.groupby('hour', as_index=False).agg({'Predicted_VOL': 'mean'})

                if 'VOL' in df_to_display.columns:
                    fig2 = px.line(avg_df, x='hour', y=['VOL', 'Predicted_VOL'],
                                labels={'value': 'Traffic Counts', 'variable': 'Legend', 'hour': 'Hour of Day'},
                                title=f'Average Traffic Counts for TMC Code {tmc_code}, Direction {direction} on {date1}')
                else:
                    fig2 = px.line(avg_df, x='hour', y='Predicted_VOL',
                                labels={'Predicted_VOL': 'Predicted Traffic Counts', 'hour': 'Hour of Day'},
                                title=f'Average Predicted Traffic Counts for TMC Code {tmc_code}, Direction {direction} on {date1}')
                st.plotly_chart(fig2)

    st.header('Map of Stations in dataset')

    if 'geo_df' not in st.session_state or st.session_state['geo_df'].empty:
        st.write("No geospatial data available to display the map.")
    elif df_to_display is not None and not df_to_display.empty:
        @st.cache_resource
        def create_station_map(df):
            if 'start_latitude_norm' in df.columns and 'start_longitude_norm' in df.columns:
                lat_col = 'start_latitude_norm'
                lon_col = 'start_longitude_norm'
            else:
                lat_col = 'start_latitude'
                lon_col = 'start_longitude'
            map_fig = px.scatter_geo(
                df,
                lat=lat_col,
                lon=lon_col,
                hover_name='tmc_code',  # Use 'tmc_code' instead of 'tmc_code_raw'
                projection='albers usa',
                title='Display of Stations in dataset'
            )
            map_fig.update_geos(
                visible=True,
                resolution=50,
                showcountries=True, countrycolor="Black",
                showcoastlines=True, coastlinecolor="Black",
                showland=True, landcolor="white",
                showocean=True, oceancolor="lightblue",
            )
            return map_fig

        map_fig = create_station_map(df_to_display)
        st.plotly_chart(map_fig)
    else:
        st.write("No data available to display the map.")

# GUI tab #3: Generate Dataset
with tab3:
    st.header('Dataset Creation')
    
    st.write("""
    **If generating a training or testing dataset, upload a TMAS file from [TMAS Source](https://www.fhwa.dot.gov/environment/air_quality/methodologies/dana/). If generating predictions for a dataset without TMAS data, leave the TMAS file blank.**
    
    Ensure that the year(s) of the TMAS data matches the year(s) of NPMRDS data.
    
    **To download the Shapefile for the USA, go to this [website](https://npmrds.ritis.org/analytics/shapefiles) (requires access) and scroll to the bottom of the page to National Shape Files and select the most recent shape file for the United States.**
    """)
    
    st.markdown('---')  # Visual separator

    st.subheader("Step 1: Geo Join TMC Road Links with TMAS Stations")
    st.write("This step performs a geospatial join between TMC shapefiles and TMAS stations.")

    file_picker('TMAS Station data file', 'PATH_TMAS_STATION', button_key='tmas_station_file_picker')
    folder_picker('Directory of Shapefile CSVs', 'PATH_tmc_shp', button_key='tmc_shp_folder_picker')
    sample_size = st.number_input('Sample Size (number of stations to sample, set to 10000 for all stations)', min_value=1, value=10000, step=1)

    title_input = st.text_input("Enter a title for the output CSV:", value="default_title")

    if st.button('Run Step 1: Geo Join', key='run_step1_button_tab4'):
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
                    st.success(f"`{pure_csv_filename}` saved at {pure_csv_path}")
                    tier1['Tmc'].to_csv(pure_csv_path, index=False, header=False)

                    Tmc_ids_str = ','.join(map(str, Tmc_ids))
                    
                    st.subheader("TMC IDs CSV Output")
                    st.text_area("TMC IDs CSV", value=Tmc_ids_str, height=100, disabled=True)
                    st.write("You can select and copy the text above.")

            except Exception as e:
                st.error(f"An error occurred during the Geo Join process: {e}")
        else:
            st.error("Please choose the TMAS Station data file and the directory of Shapefile CSVs.")

    st.markdown('---')  # Visual separator

    st.subheader("Step 2: Download and Extract NPMRDS Data")
    st.write("""
    Download and extract NPMRDS data from this [website](https://npmrds.ritis.org/analytics/download/) (requires access). 

    - If testing against TMAS data, this can be done by pasting in the comma-separated TMC values generated in Step 1 (from the text box above or from `tmc_pure.csv`).
    - If using the model for inference, this can be done simply by selecting the region on the NPMRDS RITIS website to download road links for your area of choice.
    - Additional info on the use of the NPMRDS RITIS site can be found in the README.

    **The NPMRDS RITIS Massive Data Downloader will give you the NPMRDS All Data, NPMRDS Passenger Data, and NPMRDS Truck Data files as well as the TMC ID file. These will come in 3 separate zip files where the TMC Identification file will be the same in all, but the data files will be different and will need to be identified by opening the readme in each zip. These data files should then be saved and named accordingly so they are not mixed up. These files are selected in Step 3 below.**

    **Tips on Using NPMRDS RITIS Massive Data Downloader:**
    1. **Select Segment Type:** Choose the type of segments you want to download.
    2. **Select Year:** Select the year of data that aligns with your TMAS data (if testing).
    3. **Select Roads:** 
        - If testing, paste in comma-separated TMC codes into the "Segment Codes" tab and press "Add Segments".
        - Otherwise, use the built-in tools on the website to select a region or set of roads.
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

    st.markdown('---')  # Visual separator

    st.subheader("Step 3: Select Files to Run Joins Between NPMRDS, TMC, and TMAS Data")
    file_picker('NPMRDS All Data file', 'npmrds_all_file', button_key='npmrds_all_file_picker_tab4')
    file_picker('NPMRDS Passenger Data file', 'npmrds_pass_file', button_key='npmrds_pass_file_picker_tab4')
    file_picker('NPMRDS Truck Data file', 'npmrds_truck_file', button_key='npmrds_truck_file_picker_tab4')
    file_picker('TMC Station file', 'tmc_station_file', button_key='tmc_station_file_picker_tab4')
    file_picker('TMC ID file', 'tmc_id_file', button_key='tmc_id_file_picker_tab4')

    st.markdown('---')  # Visual separator

    st.subheader("Step 4: Choose the Output File Path")
    
    if 'output_folder_tab4' not in st.session_state:
        st.session_state['output_folder_tab4'] = ''
    if 'dataset_title_tab4' not in st.session_state:
        st.session_state['dataset_title_tab4'] = ''

    col1, col2 = st.columns([2, 3])

    with col1:
        folder_picker('Choose Output Folder', 'output_folder_tab4', button_key='output_folder_picker_tab4')

    with col2:
        st.text_input("Enter a title for the output dataset:", value="NPMRDS_TMC_TMAS_US_SUBSET_20_22", key='dataset_title_tab4')

    if st.button('Join and Save Data', key='join_and_save_button_tab4'):
        required_keys = [
            'npmrds_all_file', 
            'npmrds_pass_file', 
            'npmrds_truck_file', 
            'tmc_station_file', 
            'tmc_id_file', 
            'output_folder_tab4', 
            'dataset_title_tab4'
        ]
        if all(key in st.session_state for key in required_keys):
            try:
                output_folder = st.session_state['output_folder_tab4']
                dataset_title = st.session_state['dataset_title_tab4']
                if not output_folder:
                    st.error("Please select an output folder.")
                elif not dataset_title:
                    st.error("Please enter a title for the dataset.")
                else:
                    run_joins()
                    st.success("Data joined and saved successfully.")
            except Exception as e:
                st.error(f"An error occurred during the join process: {e}")
        else:
            st.error("Please ensure all required files, output folder, and dataset title are provided before joining.")

# GUI tab #4: Train Model
with tab4:
    st.header('Train a Traffic Counts Model')
    
    if st.button('Choose source data file', key='train_model_source_file'):
        app = wx.App(False)
        dialog = wx.FileDialog(None, 'Select a File:', style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()
            result_df, normalized_df, ai, source_data, geo_df = setup(file_path)
            st.session_state['train_model_source_file'] = file_path
        else:
            result_df = pd.DataFrame()
            normalized_df = pd.DataFrame()
            ai = module_ai.ai()
            source_data = module_data.data()
            geo_df = pd.DataFrame()
        dialog.Destroy()
    elif 'train_model_source_file' not in st.session_state:
        result_df = pd.DataFrame()
        normalized_df = pd.DataFrame()
        ai = module_ai.ai()
        source_data = module_data.data()
        geo_df = pd.DataFrame()
    else:
        file_path = st.session_state.get('train_model_source_file', None)
        if file_path:
            result_df, normalized_df, ai, source_data, geo_df = setup(file_path)
        else:
            result_df = pd.DataFrame()
            normalized_df = pd.DataFrame()
            ai = module_ai.ai()
            source_data = module_data.data()
            geo_df = pd.DataFrame()

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
        default_target = st.session_state['source_data'].features_target if 'source_data' in st.session_state else cols2[0] if len(cols2) > 0 else None
        target_col_index = find_string_index(cols2, default_target) if default_target else 0
        target_col = st.selectbox(
            label='Choose target column (select only one):', 
            options=cols2, 
            index=target_col_index if target_col_index is not False else 0
        )
    else:
        target_col = None

    if st.button('Train Model', key='train_model_button'):
        if 'ai' in st.session_state and 'normalized_df' in st.session_state and in_cols and target_col:
            st.write('Model training started...')
            result = setup_funcs.train_model(
                st.session_state['ai'], 
                st.session_state['normalized_df'], 
                in_cols, 
                target_col
            )
            st.write(result)
        else:
            st.error("Please load the data and select input columns and target column before training the model.")

# GUI tab #5: About
with tab5:
    st.header('About')
    
    st.write('Visit our [GitHub](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/tree/main) for more information.')
    st.write('Download our [Readme](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/blob/main/resources/readme.pdf).')
    st.write('Questions? Contact William.Chupp@dot.gov or Eric.Englin@dot.gov.')
