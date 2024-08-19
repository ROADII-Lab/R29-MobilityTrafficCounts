# Mobility Traffic Counts AI Prediction
# ROADII TRIMS development team

# import standard libraries 
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

# import ROADII team's modules
import module_ai
import module_data

# suppress all warnings
warnings.filterwarnings("ignore")

# suppress specific Streamlit warnings by setting the logging level to ERROR
logging.getLogger('streamlit').setLevel(logging.ERROR)


import setup_funcs

### -------------------------------
### Streamlit GUI - visual elements and support subfunctions
# create dropdown menu selector
def find_string_index(alist, search_string):
	alist = list(alist)
	try:
		return alist.index(search_string)
	except ValueError:
		return False

# measure runtime of train_model or test_model
def run_timer(text, now):
    print('%s%.3f' %(text,time.time()-now))
    return time.time()

# rotate among commonly-used streamlit colors
def icon_color_modulo(indx):
	# common streamlit colors include
	# ['red', 'blue', 'green', 'purple', 'orange',
	#  'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
	#  'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue',
	#  'lightgreen', 'gray', 'black', 'lightgray']
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

# color a Streamlit icon with respect to traffic density
def icon_color_quintile(vol, all_vol, pred):
	# color the icons by which quintile the traffic volume is in
	# quintile is easy to code using five common streamlit colors in gradation (e.g., blue series below)
	
	# alternate way is to use Python library colour v0.1.5; https://pypi.org/project/colour/
	# to create a list of hex colors that are a gradation from start color to end color
	# the function currently uses quintiles but can be changed to deciles or any gradation
	if pred: # plot predicted data with red gradation
		#white1 = Color('white')
		#colors1 = list(white1.range_to(Color('darkred'),10)) --> produces list of hex colors with 10 increments
		# 'white'
		# '#ede8e8'
		# '#dfcdcd'
		# '#d6adad'
		# '#d18888'
		# '#d15f5f'
		# '#d63131'
		# '#c51919'
		# '#aa0a0a'
		# 'darkred'
		if vol<np.percentile(all_vol,20):
			return '#ede8e8'
		elif vol>=np.percentile(all_vol,20) and vol<np.percentile(all_vol,40):
			return '#d6adad'
		elif vol>=np.percentile(all_vol,40) and vol<np.percentile(all_vol,60):
			return '#d15f5f'
		elif vol>=np.percentile(all_vol,60) and vol<np.percentile(all_vol,80):
			return '#c51919'
		else:
			return 'darkred'
	else: # plot input data with blue gradation
		#lightblue1 = Color('lightblue')
		#colors2 = list(lightblue1.range_to(Color('darkblue'),10)) -->
		# 'lightblue'
		# '#93c8e3'
		# '#77b5e1'
		# '#5a9ee1'
		# '#3c81e2'
		# '#1c5fe5'
		# '#1242d1'
		# '#0a28bb'
		# '#0412a4'
		# 'darkblue'
		if vol<np.percentile(all_vol,20):
			return 'lightblue'
		elif vol>=np.percentile(all_vol,20) and vol<np.percentile(all_vol,40):
			return '#77b5e1'
		elif vol>=np.percentile(all_vol,40) and vol<np.percentile(all_vol,60):
			return '#3c81e2'
		elif vol>=np.percentile(all_vol,60) and vol<np.percentile(all_vol,80):
			return '#1242d1'
		else:
			return '#0412a4'

# filter dataframe by U.S. state
def get_tmcs_state(in_df, state):
	# This function will return a filtered dataframe for a given Tmc value
	#	given a larger dataframe, and a tmc value to use as a filter
	out_df = in_df[in_df['State'] == state]
	out_df.drop_duplicates(subset='tmc_code', keep='first')
	# old code
	#resulting_df = tier1[(tier1['County'] == 'MIDDLESEX') & (tier1['State'] == 'MASSACHUSETTS') & (tier1['STATION_ID'] == STATION_ID) ]
	#resulting_df = resulting_df[["Tmc", "STATION_ID", "DIR"]]
	return out_df

# filter dataframe by U.S. county
def get_tmcs_county(in_df, county):
	out_df = in_df[in_df['County'] == county]
	out_df.drop_duplicates(subset='tmc_code', keep='first')
	return out_df

def get_tmcs_tmccode(in_df, tmc_code):
	out_df = in_df[in_df['tmc_code'] == tmc_code]
	return out_df

# filter dataframe by datetime range
def filter_datetime(in_df, start_date, end_date):
	out_df = in_df[in_df['measurement_tstamp'] >= start_date & in_df['measurement_tstamp'] < end_date]
	out_df.drop_duplicates(subset='tmc_code', keep='first')
	return out_df

### -------------------
### Streamlit GUI setup
# run setup
@st.cache_data
def setup(filePath):
    # init ai module
    ai = module_ai.ai()
    
    # init data module
    source_data = module_data.data()
    source_data.OUTPUT_FILE_PATH = filePath

	# Normalization Functions - Set and do not change unless necessary
    norm_functions = ['tmc_norm', 'tstamp_norm', 'density_norm', 'time_before_after']
    source_data.norm_functions = norm_functions
    
    # setup data sources
    result_df = source_data.read()
    normalized_df = source_data.normalized()

    # Create geo_df here with only unique tmc_code values (used for plotting geometries)
    unique_result_df = result_df.drop_duplicates(subset='tmc_code', keep='first')
    unique_result_df['geometry'] = unique_result_df['geometry'].apply(wkt.loads)
    geo_df = gpd.GeoDataFrame(unique_result_df, geometry='geometry')
    return result_df, normalized_df, ai, source_data, geo_df


# GUI home page
st.title('Mobility Traffic Counts AI Prediction')

# Display current/system datetime
now = dt.datetime.now(pytz.timezone('UTC'))
date_time = now.strftime('%m/%d/%Y, %H:%M:%S')
st.write('Current Datetime is ',date_time,' UTC')

# Display software build/version - placeholder
st.write('Current Build is v0.12345')


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

# Function to create and run data joins
def run_joins():
    source_data = module_data.data()

    source_data.tmas = source_data.tmas_data()
    source_data.npmrds = source_data.npmrds_data()
    source_data.tmc = source_data.tmc_data()

    # Set output file paths
    source_data.OUTPUT_FILE_PATH = st.session_state.get('output_file_path', r'../data/NPMRDS_TMC_TMAS_US_SUBSET_20_22.pkl')

    # Set TMAS file paths on the instance
    tmas_file_path = st.session_state.get('tmas_file', r'../data/TMAS_Class_Clean_2022.csv')
    source_data.tmas.TMAS_DATA_FILE = tmas_file_path

    # If the TMAS data file is a .pkl file, set the TMAS_PKL_FILE
    if tmas_file_path.endswith('.pkl'):
        source_data.tmas.TMAS_PKL_FILE = tmas_file_path
    else:
        source_data.tmas.TMAS_PKL_FILE = r'../data/TMAS_Class_Clean_2022.pkl'
        if not os.path.isfile(source_data.tmas.TMAS_PKL_FILE):
            source_data.tmas.read()

    # Set NPMRDS data locations on the instance
    source_data.npmrds.NPMRDS_ALL_FILE = st.session_state.get('npmrds_all_file', r'..\data\US_ALL_22\all2022_NPMRDS_ALL.csv')
    source_data.npmrds.NPMRDS_PASS_FILE = st.session_state.get('npmrds_pass_file', r'..\data\US_ALL_22\all2022_NPMRDS_PASS.csv')
    source_data.npmrds.NPMRDS_TRUCK_FILE = st.session_state.get('npmrds_truck_file', r'..\data\US_ALL_22\all2022_NPMRDS_TRUCK.csv')

    # Set TMC data locations on the instance (from geojoins)
    source_data.tmc.TMC_STATION_FILE = st.session_state.get('tmc_station_file', r'..\data\US_ALL_22\TMC_2022Random_US_Subset_ALL_2022_2.csv')
    source_data.tmc.TMC_ID_FILE = st.session_state.get('tmc_id_file', r'..\data\US_ALL_22\TMC_Identification.csv')

    source_data.dataset_year = '2022'

    source_data.join_and_save()
    st.write("Data joined and saved successfully.")

# Take in the output of use_model and the raw_df and merge them to incldue all columns including vol, station_id, and geometries
def merge_normalized_and_raw_data(raw_df, normalized_df):
    # Function to normalize tmc_code
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
        df['tmc_code'] = df['tmc_code'].str.lower().str.replace('p', '').str.replace('n', '').str.replace('+', '').str.replace('-', '')
        df['tmc_code'] = df['tmc_code'].astype(int)
        return df

    # Normalize tmc_code in the raw dataframe
    raw_df = normalize_tmc_code(raw_df)

    # Convert timestamp in normalized dataframe to match the raw dataframe format
    normalized_df['measurement_tstamp'] = pd.to_datetime(
        normalized_df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H'
    )

    # Merge the dataframes
    merged_df = pd.merge(
        raw_df,
        normalized_df,
        on=['tmc_code', 'measurement_tstamp', 'DIR'],
        how='inner',
        suffixes=('_raw', '_norm')
    )

    return merged_df
# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(['1. Train Model', '2. Test Model', '3. Results', '4. Generate Training/Testing Dataset', '5. About'])

# GUI tab #1: Train Model
with tab1:
    st.header('Train a Traffic Counts Model')
    
    # FilePicker - Choose source data file
    if st.button('Choose source data file', key='train_model_source_file'):
        ap = wx.App()
        ap.MainLoop()
        dialog = wx.FileDialog(None, 'Select a File:', style=wx.FD_DEFAULT_STYLE)
        st.session_state.dialog = dialog
        if dialog.ShowModal() == wx.ID_OK:
            file_path = dialog.GetPath()  # file_path will contain the path of the folder you have selected as string
            result_df, normalized_df, ai, source_data, geo_df = setup(file_path)
            # breakpoint() # to fill [normalized_df] and work with it
        else:
            result_df = pd.DataFrame()
            normalized_df = pd.DataFrame()
            ai = module_ai.ai()
            source_data = module_data.data()
            geo_df = pd.DataFrame()
    elif 'dialog' not in st.session_state:
        result_df = pd.DataFrame()
        normalized_df = pd.DataFrame()
        ai = module_ai.ai()
        source_data = module_data.data()
        geo_df = pd.DataFrame()
    else:
        file_path = st.session_state.dialog.GetPath()  # file_path will contain the path of the file you have selected as string
        result_df, normalized_df, ai, source_data, geo_df = setup(file_path)
        
    # Raw data pane
    st.header('Input Data')
    st.dataframe(result_df[0:50])
    
    # Normalized data pane
    st.header('Input Data Normalized for AI Training')
    st.dataframe(normalized_df[0:50])
    
    # Choose input column(s) drop-down menu
    cols1 = source_data.features_training_set
    in_cols = st.multiselect(label='Choose input column(s) (select one or more):', options=cols1, default=source_data.features_training_set)
    
    # Choose target column
    cols2 = normalized_df.columns
    target_col = st.selectbox(label='Choose target column (select only one):', options=cols2, index=find_string_index(cols2, source_data.features_target))
    
    if st.button('Train Model', key='train_model_button'):
        st.write('Model training started...')
        result = setup_funcs.train_model(ai, normalized_df, in_cols, target_col)
        st.write(result)



# GUI tab #2: Use a Traffic Counts Model
with tab2:
    st.header('Use a Traffic Counts Model')
    
    # Input column(s) and target column GUI buttons (previous single window)
    list_of_model_files = ai.get_model_list('..\models')
    model_filename = st.selectbox(label='Choose a model file', options=list_of_model_files)

    # File picker for raw dataset path
    file_picker('created dataset - NPMRDS + TMC + TMAS (.pkl file)', 'raw_dataset_path', button_key='raw_dataset_picker')

    if 'raw_dataset_path' in st.session_state:
        raw_dataset_path = st.session_state['raw_dataset_path']
        
        # Columns to use
        columns = [
            'tmc_code', 'measurement_tstamp', 'speed_All', 'data_density_All',
            'data_density_Pass', 'data_density_Truck', 'travel_time_seconds_All', 'start_latitude',
            'start_longitude', 'end_latitude', 'end_longitude', 'miles', 'aadt', 'urban_code',
            'thrulanes_unidir', 'f_system', 'route_sign', 'thrulanes', 'zip', 'DIR'
        ]
        
        t_result_df, t_normalized_df, t_ai, t_source_data, t_geo_df = setup(raw_dataset_path)
        columns.extend(t_source_data.calculated_columns)

        if st.button('Use Model'):
            st.write('Running Loaded model on test dataset...')
            answer_df = setup_funcs.use_model(t_ai, model_filename, t_normalized_df, columns, 'VOL')
            if answer_df.empty:
                st.write("Model or data was not properly loaded. Please check your inputs.")
            else:
                answer_df_merged = merge_normalized_and_raw_data(t_result_df, answer_df)
                # Save the results
                base_name = os.path.basename(raw_dataset_path)
                name, ext = os.path.splitext(base_name)
                output_file_path = os.path.join('..', 'data', f"{name}_predictions{ext}")
                answer_df_merged.to_pickle(output_file_path)
                st.session_state['answer_df_merged'] = answer_df_merged
                st.write(f"DataFrame saved to {output_file_path}")

    # Check if answer_df_merged is available in session state
    if 'answer_df_merged' in st.session_state:
        answer_df_merged = st.session_state['answer_df_merged']

        # Create interactive visualization
        st.header('Visualize Predictions vs Measured Values')
        tmc_code = st.selectbox('Select TMC Code', answer_df_merged['tmc_code'].unique())

        # Filter the direction options based on the selected TMC Code
        available_directions = answer_df_merged[answer_df_merged['tmc_code'] == tmc_code]['DIR'].unique()
        direction = st.selectbox('Select Direction', available_directions)

        day_of_week = st.selectbox('Select Day of the Week', answer_df_merged['measurement_tstamp'].dt.day_name().unique())

        filtered_df = answer_df_merged[(answer_df_merged['tmc_code'] == tmc_code) & 
                                       (answer_df_merged['DIR'] == direction) & 
                                       (answer_df_merged['measurement_tstamp'].dt.day_name() == day_of_week)]

        # Aggregate data to get the average traffic volume for each hour
        filtered_df['hour'] = filtered_df['measurement_tstamp'].dt.hour
        avg_df = filtered_df.groupby('hour').agg({'VOL': 'mean', 'Predicted_VOL': 'mean'}).reset_index()

        fig = px.line(avg_df, x='hour', y=['VOL', 'Predicted_VOL'],
                      labels={'value': 'Traffic Counts', 'variable': 'Legend'},
                      title=f'Average Traffic Counts for TMC Code {tmc_code}, Direction {direction} on {day_of_week}')
        st.plotly_chart(fig)
    else:
        st.write('Please select a raw dataset and run the model to proceed.')

# GUI tab #3: Results
with tab3:
    st.header('Traffic Counts Prediction Results')
    
    # If no data has been loaded yet, then this will not display anything on this tab 
    # or let the user create a map which would cause an error
    if geo_df.empty:
        st.write("No data available to display results. Please load the data first.")
    else:
        # Subheader for date/time selection
        st.subheader('Select Date of Analysis (1-day span)')

        date1 = st.date_input('Select Date to continue:', value=None)
        while date1 is None:
            st.stop()

        st.write('Date =', date1)

        print('Filtering data for chosen date')
        # Filter the data for the selected date
        filtered_df = result_df[
            (result_df['measurement_tstamp'].dt.year == date1.year) &
            (result_df['measurement_tstamp'].dt.month == date1.month) &
            (result_df['measurement_tstamp'].dt.day == date1.day)
        ]
        print('Data filtered for chosen date\nGenerating plots and folium map')

        if filtered_df.empty:
            st.write("No data available for the selected date.")
        else:
            # Create list of unique TMC stations
            unique_tmcs = filtered_df.drop_duplicates(subset='tmc_code', keep='first')

            # Subheader for Streamlit-folium map #1, display input data
            st.subheader('Display of Input Data (filtered by date selection)')

            # Create a Folium map centered around automatic or specified point
            map_center_in = [37.5, -85.0]
            map_in = folium.Map(location=map_center_in, zoom_start=5)

            # Create the datatips folder if it doesn't exist
            datatips_folder = '../datatips'
            if not os.path.exists(datatips_folder):
                os.makedirs(datatips_folder)

            # Group the filtered data by TMC station
            grouped_df = filtered_df.groupby('tmc_code')

            # Iterate through unique TMC stations, plot the geometry, add basic datatip at that location
            for tmc_code, group in grouped_df:
                pred = False  # different color scheme is used for input vs. predicted data

                # Extract the hour from the measurement_tstamp
                group = group.copy()
                group['hour'] = group['measurement_tstamp'].dt.hour

                # Create the figure object
                fig, ax = plt.subplots()
                
                # Plot the data
                ax.plot(group['hour'], group['VOL'])
                
                # Set axis labels and title
                ax.set_xlabel('Hour')
                ax.set_ylabel('Volume')
                ax.set_title('Hourly Volume')
                fig.tight_layout()
                path1 = os.path.join(datatips_folder, str(tmc_code).replace('.0', '') + '.jpg')
                fig.savefig(path1, bbox_inches='tight', pad_inches=0.9)
                plt.close(fig)

                with open(path1, 'rb') as image_file:
                    encode1 = base64.b64encode(image_file.read()).decode('UTF-8')
                
                html1 = f'''
                    <html>
                        <body style="margin: 0;">
                            tmc_code={str(tmc_code).replace(".0", "")}<br>
                            max_traffic={group["VOL"].max()}<br>
                            <img src="data:image/jpg;base64,{encode1}" style="width: 100%;">
                        </body>
                    </html>
                '''
                iframe1 = IFrame(html1, width=325, height=375)  # Adjust height based on content
                popup1 = folium.Popup(iframe1, max_width=325)
                color1 = icon_color_quintile(group['VOL'].iloc[0], filtered_df['VOL'], pred)  # Keep the original marker color
                icon1 = folium.plugins.BeautifyIcon(icon='car', icon_shape='marker', border_width=1, border_color='black', background_color=color1)
                tooltip1 = str(tmc_code)

                # Get geometry from geo_df
                geometry = geo_df.loc[geo_df['tmc_code'] == tmc_code, 'geometry'].values[0]
                loc1 = [group['start_latitude'].mean(), group['start_longitude'].mean()]

                folium.Marker(location=loc1, popup=popup1, icon=icon1, tooltip=tooltip1).add_to(map_in)
                folium.GeoJson(geometry, style_function=lambda x: {'color': '#000000', 'weight': 3}).add_to(map_in)  # Set to black

            # Save .html file and render in Streamlit GUI
            map_in.save('map_in.html')
            map_in_html = open('map_in.html', 'r', encoding='utf-8').read()
            st.components.v1.html(map_in_html, height=600, width=800)

            # -------------------------------------------------------------
            # Subheader for Streamlit-folium map #2, display predicted data
            st.subheader('Display of Predicted Data (filtered by date selection)')

            # Create a Folium map centered around automatic or specified point
            map_center_pr = [37.5, -85.0]
            map_pr = folium.Map(location=map_center_pr, zoom_start=5)

            for tmc_code, group in grouped_df:
                pred = False  # different color scheme is used for input vs. predicted data

                popup1 = folium.Popup(f"TMC_code: {str(tmc_code)}, Initial_vol: {group['VOL'].iloc[0]}", parse_html=True)
                color1 = icon_color_quintile(group['VOL'].iloc[0], filtered_df['VOL'], pred)  # Keep the original marker color
                icon1 = folium.plugins.BeautifyIcon(icon='car', icon_shape='marker', border_width=1, border_color='black', background_color=color1)
                tooltip1 = str(tmc_code)
                
                # Get geometry from geo_df
                geometry = geo_df.loc[geo_df['tmc_code'] == tmc_code, 'geometry'].values[0]
                loc1 = [group['start_latitude'].mean(), group['start_longitude'].mean()]

                folium.Marker(location=loc1, popup=popup1, icon=icon1, tooltip=tooltip1).add_to(map_pr)
                folium.GeoJson(geometry, style_function=lambda x: {'color': '#000000', 'weight': 3}).add_to(map_pr)  # Set to black

            for tmc_code, group in grouped_df:
                pred = True  # different color scheme is used for input vs. predicted data

                popup2 = folium.Popup(f"TMC_code: {str(tmc_code)}, Initial_vol: {group['VOL'].iloc[0]}", parse_html=True)
                color2 = icon_color_quintile(group['VOL'].iloc[0], filtered_df['VOL'], pred)  # Keep the original marker color
                icon2 = folium.plugins.BeautifyIcon(icon='car', icon_shape='marker', border_width=1, border_color='black', background_color=color2)
                tooltip2 = str(tmc_code)
                
                # Get geometry from geo_df
                geometry = geo_df.loc[geo_df['tmc_code'] == tmc_code, 'geometry'].values[0]
                loc1 = [group['start_latitude'].mean(), group['start_longitude'].mean()]

                folium.Marker(location=loc1, popup=popup2, icon=icon2, tooltip=tooltip2).add_to(map_pr)
                folium.GeoJson(geometry, style_function=lambda x: {'color': '#000000', 'weight': 3}).add_to(map_pr)  # Set to black

            # Save .html file and render in Streamlit GUI
            map_pr.save('map_pr.html')
            map_pr_html = open('map_pr.html', 'r', encoding='utf-8').read()
            st.components.v1.html(map_pr_html, height=600, width=800)

# GUI tab #4: Generate Training/Testing Dataset
with tab4:
    st.header('Dataset Creation')
    
    if result_df.empty:
        st.write('If generating a training or testing dataset, upload a TMAS file from [TMAS Source](https://www.fhwa.dot.gov/environment/air_quality/methodologies/dana/). If generating predictions for a dataset without TMAS data, leave the TMAS file blank.')
        st.write('Ensure that the year(s) of the TMAS data matches the year(s) of NPMRDS data.')

        st.write("Step 1: Geo Join TMC road links with TMAS stations")
        st.write("Step 2: Download and extract NPMRDS data from this [website](https://npmrds.ritis.org/analytics/download/) (requires access) by pasting in the comma separated TMC values")

        st.write("Step 3: Select files to run joins between NPMRDS, TMC, and TMAS data")
        
        file_picker('TMAS data file', 'tmas_file', button_key='tmas_file_picker')
        file_picker('NPMRDS All Data file', 'npmrds_all_file', button_key='npmrds_all_file_picker')
        file_picker('NPMRDS Passenger Data file', 'npmrds_pass_file', button_key='npmrds_pass_file_picker')
        file_picker('NPMRDS Truck Data file', 'npmrds_truck_file', button_key='npmrds_truck_file_picker')
        file_picker('TMC Station file', 'tmc_station_file', button_key='tmc_station_file_picker')
        file_picker('TMC ID file', 'tmc_id_file', button_key='tmc_id_file_picker')

        st.write("Step 4: Choose the output file path")
        file_picker('output file path', 'output_file_path', style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, button_key='output_path_picker_tab4')

        if st.button('Join and Save Data', key='join_and_save_button'):
            run_joins()

# GUI tab #5: About
with tab5:
	st.header('About')
	
	# GitHub link
	st.write('Visit our [GitHub](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/tree/main) for more information.')
	
	# Readme link
	st.write('Download our [Readme](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/blob/main/resources/readme.pdf).')
	
	# POCs
	st.write('Questions? Contact William.Chupp@dot.gov or Eric.Englin@dot.gov.')
	

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------