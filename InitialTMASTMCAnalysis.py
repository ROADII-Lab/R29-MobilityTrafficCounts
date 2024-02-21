# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:41:59 2023

@author: william.chupp
"""
import pandas as pd
import numpy as np
import datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
import time
import geopandas as gpd
import sys
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pathlib
from load_shapes import *
import pkg_resources
from tqdm.tk import tqdm
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import psutil

def lapTimer(text,now):
    print('%s%.3f' %(text,time.time()-now))
    return time.time()

now=time.time()

crs = {'init' :'epsg:4326'}
#b1. Preparing TMC data for geoprocessing
#Read the TMC shapefile
print('Geoprocessing TMC data')

PATH_tmc_shp = r'..\USAShape_2019'
PATH_TMAS_STATION = r'..\TMAS_Station_2019.csv'


shp = load_shape(PATH_tmc_shp)

# In[]

now=lapTimer('  took: ',now)


#a. Read TMAS Station and Classification

tmas_station = pd.read_csv(PATH_TMAS_STATION, dtype={'STATION_ID':str})
    
#a2. Preparing TMAS station data for geoprocessing
#Create a new attribute as points from lat and long
print('Geoprocessing TMAS station data')
tmas_station['geometry']=tmas_station.apply(lambda row: Point(row["LONG"], row["LAT"]), axis=1)
tmas_station.reset_index(drop=True, inplace=True)    #Start the dataframe index from 0
#Create a geodataframe to test geopandas capabilities
geo_tmas = GeoDataFrame(tmas_station.copy(), crs=crs, geometry='geometry')
    
now=lapTimer('  took: ',now)
#b Read TMC Indentification data
'''
print('Reading TMC Identification data')
tmc = pd.read_csv(tmc_identification)
dir_dic = {'EASTBOUND':'EB', 'NORTHBOUND':'NB', 'WESTBOUND':'WB', 'SOUTHBOUND':'SB'}
tmc['direction'].replace(dir_dic, inplace=True)
tmc['direction']=tmc['direction'].str.extract('(EB|NB|SB|WB)')
'''
shp['dir_num']=np.nan
shp['dir_num'].loc[shp['Direction']=='N']=1
shp['dir_num'].loc[shp['Direction']=='E']=3
shp['dir_num'].loc[shp['Direction']=='S']=5
shp['dir_num'].loc[shp['Direction']=='W']=7

##################################################
#c. Tier 1: space join
#c1. Merging TMC Link and TMAS Station data using a 0.2-mile buffer
print('Merging TMAS and TMC data')
geo_tmas['geometry'] = geo_tmas['geometry'].buffer(0.001)
intersect = gpd.sjoin(shp, geo_tmas, op='intersects')
#c2. Selecting only the data that matches on direction
intersect_dir = intersect[intersect['dir_num']==intersect['DIR']]
intersect_dir = intersect_dir[intersect_dir['F_System']==intersect_dir['F_SYSTEM']]
#c3. Assigning interected data as tier 1
tier1 = intersect_dir.loc[:,['Tmc','STATION_ID', 'DIR']]
tier1['tier']=1
now=lapTimer('  took: ', now)

