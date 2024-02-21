"""
Create visualization of output from Join_TMAS_NPMRDS.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Set CSV file path and read into variable
    NPMRDS_TMC_TMAS_PATH = r's3://prod.sdc.dot.gov.team.roadii/UseCaseR29-MobilityCounts/NPMRDS_TMC_TMAS_Join/NPMRDS_TMC_TMAS.csv'
    NPMRDS_TMC_TMAS = pd.read_csv(NPMRDS_TMC_TMAS_PATH, low_memory = False)
    
    # Plot that shows pure timestamp counts for each TMAS station
    station_counts = NPMRDS_TMC_TMAS.groupby('STATION_ID')['measurement_tstamp'].count()
    plt.figure(figsize=(12,6))
    plt.bar(station_counts.index, station_counts.values)
    plt.xlabel('TMAS Station')
    plt.ylabel('Count of one hour timestamps')
    plt.title('Count of timestamps for each TMAS Station')
    plt.xticks(rotation = 45)
    plt.show()
    
    # Heatmap that shows availablity by month
    heatmap_data_month = NPMRDS_TMC_TMAS.pivot_table(index = 'STATION_ID', columns = ['YEAR', 'MONTH'], values = 'measurement_tstamp', aggfunc = 'count', fill_value = 0 )
    plt.figure(figsize = (15,10))
    sns.heatmap(heatmap_data_month, cmap = 'Blues', linewidths = 0.5, cbar = True)
    plt.xlabel('Month')
    plt.ylabel('STATION_ID')
    plt.title('Data Availability by Month for Each Station')
    plt.show()
    
    # Heatmap that shows availability by hour
    heatmap_data_hour = NPMRDS_TMC_TMAS.pivot_table(index = 'STATION_ID', columns = 'HOUR', values = 'measurement_tstamp', aggfunc = 'count', fill_value = 0 )
    plt.figure(figsize = (15,10))
    sns.heatmap(heatmap_data_hour, cmap = 'Blues', linewidths = 0.5, cbar = True)
    plt.xlabel('Hour of the Day')
    plt.ylabel('STATION_ID')
    plt.title('Data Availability by Hour for Each Station')

    
    
if __name__ == "__main__":
    main()