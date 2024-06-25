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

# Define tabs
tab1, tab2, tab3, tab4 = st.tabs(['1. Train Model','2. Test Model','3. Results','4. About'])

# GUI tab #1: Train Model
with tab1:
	st.header('Train a Traffic Counts Model')
	
	# FilePicker - Choose source data file
	if st.button('Choose source data file'):
		ap = wx.App()
		ap.MainLoop()
		dialog = wx.FileDialog(None,'Select a folder:', style=wx.FD_DEFAULT_STYLE)
		st.session_state.dialog = dialog
		if dialog.ShowModal() == wx.ID_OK:
			folder_path = dialog.GetPath() # folder_path will contain the path of the folder you have selected as string
			result_df, normalized_df, ai, source_data, geo_df = setup(folder_path)
			#breakpoint() # to fill [normalized_df] and work with it
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
		folder_path = st.session_state.dialog.GetPath() # folder_path will contain the path of the folder you have selected as string
		result_df, normalized_df, ai, source_data, geo_df = setup(folder_path)
		
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
	
	if st.button('Train Model'):
		st.write('Model training started...')
		result = setup_funcs.train_model(ai, normalized_df, in_cols, target_col)
		st.write(result)
	
	
# GUI tab #2: Test/Use Model
with tab2:
	st.header('Test a Traffic Counts Model')
	
	# Input column(s) and target column GUI buttons (previous single window)
	list_of_model_files = ai.get_model_list('..\models')
	ai.model_filename = st.selectbox(label = 'Choose a model file', options=list_of_model_files)
	if st.button('Test Model'):
		st.write('Model testing started...')
		test_accuracy = setup_funcs.test_model(ai, normalized_df, in_cols, target_col)
		st.write('Model Accuracy = ', test_accuracy)

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

	
# GUI tab #4: About
with tab4:
	st.header('About')
	
	# GitHub link
	st.write('Visit our [GitHub](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/tree/main) for more information.')
	
	# Readme link
	st.write('Download our [Readme](https://github.com/ITSJPO-TRIMS/R29-MobilityTrafficCounts/blob/main/resources/readme.pdf).')
	
	# POCs
	st.write('Questions? Contact William.Chupp@dot.gov or Eric.Englin@dot.gov.')
	

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------