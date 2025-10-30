import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from RelativeHumidity.rh_into_df import process_rh_file

excel_files_directory = '/Users/briannamarcosano/Documents/Thesis/Data/Temp Data/Good Data'
# excel_files_directory = 'CNN_LSTM_Thesis/LSTM/Data/Raw_and_avg/Raw Tabular'
def saturation_vapor_pressure(temp):
    """ Calculate saturation vapor pressure using the Magnus-Tetens approximation """
    return 0.6108 * np.exp(17.27 * temp / (temp + 237.3))

def set_weekly_temporal_avgs():
    desired_columns = [
        'STATION', 'NAME', 'County', 'LATITUDE', 'LONGITUDE',
        'ELEVATION', 'DATE', 'AWND', 'PRCP', 'SNOW', 'SNWD',
        'TAVG', 'TMAX', 'TMIN', 'vpdmin (hPa)', 'vpdmax (hPa)'
    ]

    rh_dataframes = []
    weather_dataframes = []
    for filename in os.listdir(excel_files_directory):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):
            file_path = os.path.join(excel_files_directory, filename)
            try:
                df = pd.read_excel(file_path, engine='openpyxl', usecols=desired_columns)
                if not df.empty:
                    weather_dataframes.append(df)
                else:
                    print(f"Skipping empty file: {file_path}")
            except pd.errors.EmptyDataError:
                print(f"Skipping empty file: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")


    # Concatenate RH dataframes and weather dataframes separately
    rh_df = process_rh_file() 
    # rh_df = pd.concat(merrimack_rh_df, ignore_index=True)
    weather_df = pd.concat(weather_dataframes, ignore_index=True)

    rh_daily_avg = rh_df.groupby('DATE')['RH'].mean().reset_index()
    rh_daily_avg['DATE'] = pd.to_datetime(rh_daily_avg['DATE'])
    # Ensure 'DATE' is datetime
    # rh_df['DATE'] = pd.to_datetime(rh_df['DATE'])
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

    # Merge RH data with weather data based on DATE and County
    final_df = weather_df.merge(rh_daily_avg[['DATE', 'RH']], on='DATE', how='left')

    # Set DATE as index
    final_df.set_index('DATE', inplace=True)


    columns_to_average = ['AWND', 'PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX', 'TMIN', 'vpdmin (hPa)', 'vpdmax (hPa)', 'RH']
    columns_to_sum = ['PRCP', 'SNOW', 'SNWD']
    columns_to_avg_neg = ['AWND', 'vpdmin (hPa)', 'vpdmax (hPa)','TAVG', 'TMAX', 'TMIN', 'RH']
    #cols_to_max = ['TAVG', 'TMAX', 'TMIN']

    # Create an aggregation dictionary
    agg_dict = {col: 'sum' for col in columns_to_sum}
    agg_dict.update({col: 'mean' for col in columns_to_avg_neg})
    # agg_dict.update({col: 'max' for col in cols_to_max})
    groupby_cols = ['STATION', 'LATITUDE', 'LONGITUDE', 'County', 'ELEVATION']
    # monthly_sums = final_df.groupby(['STATION', 'LATITUDE', 'LONGITUDE', 'County', 'ELEVATION'])[columns_to_average].resample('M').sum().reset_index()

    weekly_averages = final_df.groupby(groupby_cols).resample('W')[columns_to_average].mean().reset_index()

    weekly_summary_sum_avg = final_df.groupby(groupby_cols).resample('W').agg({**{col: 'sum' for col in columns_to_sum}, **{col: 'mean' for col in columns_to_avg_neg}}).reset_index()
    weekly_summary_max_sum_avg = final_df.groupby(groupby_cols).resample('W').agg(agg_dict).reset_index()

    weekly_averages.to_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_weekly.csv', index=True)
    weekly_averages = weekly_averages.to_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_weekly.csv', index=True)
    weekly_summary_sum_avg.to_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_SUM_and_AVG.csv', index=True)

    weekly_summary_max_sum_avg.to_csv('CNN_LSTM_Thesis/LSTM/Data/weekly_temporal_data_MAX_SUM_and_AVG.csv', index=True)
#     return weekly_sum_to_csv
    
set_weekly_temporal_avgs()