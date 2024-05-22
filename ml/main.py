# Mobility Counts Prediction
# ROADII TRIMS development team

# import standard libraries 
import datetime as dt
import folium
import geopandas as gpd
from geopandas import GeoDataFrame
import pandas as pd
import pytz
from shapely.geometry import Point
from statistics import mean
import streamlit as st
from streamlit_folium import st_folium
import time
import wx

# import ROADII team's modules
import module_ai
import module_census
import module_data

# datetime slider
# https://docs.kanaries.net/topics/Python/streamlit-datetime-slider
# general slider
# https://www.youtube.com/watch?list=TLGGTGC9_GhF4fYyMTA1MjAyNA&v=sCvdt79asrE

### -------------------------------------------------------
### Streamlit GUI - train model and test model capabilities
# train_model function call and button
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

# test_model function call and button
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


### -------------------------------
### Streamlit GUI - visual elements and support subfunctions
# dropdown menu selector function
def find_string_index(alist, search_string):
	alist = list(alist)
	try:
		return alist.index(search_string)
	except ValueError:
		return False

# rotate among commonly-used streamlit colors
def heat_color_modulo(indx):
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

# measuring runtime of train_model or test_model
def run_timer(text, now):
    print('%s%.3f' %(text,time.time()-now))
    return time.time()

# dataframe filtering function
def get_tmcs_state(in_df, state):
	# This function will return a filtered dataframe for a given Tmc value
	#	given a larger dataframe, and a tmc value to use as a filter
	out_df = in_df[in_df['State'] == state]
	out_df.drop_duplicates(subset='tmc_code', keep='first')
	# old code
	#resulting_df = tier1[(tier1['County'] == 'MIDDLESEX') & (tier1['State'] == 'MASSACHUSETTS') & (tier1['STATION_ID'] == STATION_ID) ]
	#resulting_df = resulting_df[["Tmc", "STATION_ID", "DIR"]]
	return out_df

# dataframe filtering function
def get_tmcs_county(in_df, county):
	out_df = in_df[in_df['County'] == county]
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
	
	# init census data (using saved data for now)
	# census_dataobj = module_census.census_data()
	# census_data = census_dataobj.get_population_data_by_city(25)
	
    # setup data sources
	census_df = pd.DataFrame() #pd.DataFrame(census_data)
	result_df = source_data.read()
	normalized_df = source_data.normalized()
	return census_df, result_df, normalized_df, ai, source_data


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
			census_df, result_df, normalized_df, ai, source_data = setup(folder_path)
			#breakpoint() # to fill [normalized_df] and work with it
		else:
			census_df = pd.DataFrame()
			result_df = pd.DataFrame()
			normalized_df = pd.DataFrame()
			ai = module_ai.ai()
			source_data = module_data.data()
	elif 'dialog' not in st.session_state:
			census_df = pd.DataFrame()
			result_df = pd.DataFrame()
			normalized_df = pd.DataFrame()
			ai = module_ai.ai()
			source_data = module_data.data()
	else:
		folder_path = st.session_state.dialog.GetPath() # folder_path will contain the path of the folder you have selected as string
		census_df, result_df, normalized_df, ai, source_data = setup(folder_path)
		
	# Create list of unique TMC stations
	unique_tmcs_raw = result_df.drop_duplicates(subset='tmc_code', keep='first')
	unique_tmcs = normalized_df.drop_duplicates(subset='tmc_code', keep='first')
	
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
		result = train_model(ai, normalized_df, in_cols, target_col)
		st.write(result)
	
	
# GUI tab #2: Test/Use Model
with tab2:
	st.header('Test a Traffic Counts Model')
	
	# Input column(s) and target column GUI buttons (previous single window)
	list_of_model_files = ai.get_model_list('..\models')
	ai.model_filename = st.selectbox(label = 'Choose a model file', options=list_of_model_files)
	if st.button('Test Model'):
		st.write('Model testing started...')
		test_accuracy = test_model(ai, normalized_df, in_cols, target_col)
		st.write('Model Accuracy = ', test_accuracy)


# GUI tab #3: Show results
with tab3:
	st.header('Traffic Counts Prediction Results')
	
	# Create list of unique TMC stations
	unique_tmcs_raw = result_df.drop_duplicates(subset='tmc_code', keep='first')
	unique_tmcs = normalized_df.drop_duplicates(subset='tmc_code', keep='first')
	#unique_tmcs.insert(2,'tmc_code_string',unique_tmcs_raw['tmc_code'])
	#potentially to do: insert original tmc_code string to show in map displays
	
	# -------------------------------------
	# different way to show a Streamlit map
	#my_map1 = pd.DataFrame({'lat': (unique_tmcs['start_latitude']+unique_tmcs['end_latitude'])/2,'lon': (unique_tmcs['start_longitude']+unique_tmcs['end_longitude'])/2})
	#st.map(my_map1)
	#st.map(data=my_map_df,latitude=42.5,longitude=-71.0)
	
	# old versions mockup of maps
	#st.subheader('Display of Input Data')
	#st.image('input_data.jpg', width=600)
	#st.subheader('Display of Predicted Data')
	#st.image('predicted_data.jpg', width=600)	
	#st.subheader('End of Results tab')
	
	# ---------------------------------------------------------
	# Subheader for Streamlit-folium map #1, display input data
	st.subheader('Display of Input Data')
	
	# Create a Folium map centered around automatic or specified point
	# Find where the st_folium map should be centered
	#map_center = [unique_tmcs.unary_union.centroid.y, unique_tmcs.unary_union.centroid.x]
	map_center_in = [37.5,-85.0]
	map_in = folium.Map(location=map_center_in, zoom_start=5)
	
	# Iterate through unique TMC stations, plot the centroid of start/end lat/lon, add basic datatip at that location
	for indx, row in unique_tmcs.iterrows():
		#loc1 = [mean([row['start_latitude'],row['end_latitude']]), mean([row['start_longitude'],row['end_longitude']])]
		loc1 = [row['start_latitude'],row['start_longitude']]
		popup1 = folium.Popup(f"TMC_code: {str(row['tmc_code'])}, Initial_vol: {row['VOL']}", parse_html=True)
		icon1 = folium.Icon(color=heat_color_modulo(indx))
		tooltip1 = str(row['tmc_code'])
		folium.Marker(location=loc1, popup=popup1, icon=icon1, tooltip=tooltip1).add_to(map_in)
		points1 = [ [row['start_latitude'], row['start_longitude']], [row['end_latitude'], row['end_longitude']] ]
		folium.PolyLine(locations=points1, color=heat_color_modulo(indx), weight=3).add_to(map_in)
	
	# save .html file and render in Streamlit GUI
	map_in.save('map_in.html')
	map_in_html = open('map_in.html','r',encoding='utf-8').read()
	st.components.v1.html(map_in_html,height=600,width=800)
	
	
	# -------------------------------------------------------------
	# Subheader for Streamlit-folium map #2, display predicted data
	# Display i) input data road segments and datatips ...
	# 	ii) new road segments predicted by AI model, and datatips
	st.subheader('Display of Predicted Data')
	
	# Create a Folium map centered around automatic or specified point
	map_center_pr = [37.5,-85.0]
	map_pr = folium.Map(location=map_center_pr, zoom_start=5)
	
	# Iterate through unique TMC stations, plot the centroid of start/end lat/lon, add basic datatip at that location
	for indx, row in unique_tmcs.iterrows():
		#loc1 = [mean([row['start_latitude'],row['end_latitude']]), mean([row['start_longitude'],row['end_longitude']])]
		loc1 = [row['start_latitude'],row['start_longitude']]
		popup1 = folium.Popup(f"TMC_code: {str(row['tmc_code'])}, Initial_vol: {row['VOL']}", parse_html=True)
		icon1 = folium.Icon(color=heat_color_modulo(indx))
		tooltip1 = str(row['tmc_code'])
		folium.Marker(location=loc1, popup=popup1, icon=icon1, tooltip=tooltip1).add_to(map_pr)
		points1 = [ [row['start_latitude'], row['start_longitude']], [row['end_latitude'], row['end_longitude']] ]
		folium.PolyLine(locations=points1, color=heat_color_modulo(indx), weight=3).add_to(map_pr)
	
	# save .html file and render in Streamlit GUI
	map_pr.save('map_pr.html')
	map_pr_html = open('map_pr.html','r',encoding='utf-8').read()
	st.components.v1.html(map_pr_html,height=600,width=800)
	
	
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