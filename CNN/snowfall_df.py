from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# source MODIS_env/bin/activate

pio.templates.default = "plotly_white"


def preprocess_data(start_date):
    #all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_weekly.csv') # weekly averaged features
    #all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/monthly_max.csv') ##### not good results, do not continue
    #all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/monthly_sums.csv')
    #all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_SUM_and_AVG.csv')
    #all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_by_day.csv')
    #all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/bi_weekly_temporal_data_MAX_SUM_and_AVG.csv') # sum of snow BI WEEKLY
    
    # all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/weekly_temporal_data_MAX_SUM_and_AVG.csv') # sum of snow

    # all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_SUM_and_AVG.csv') # sum of snow
    all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/weekly_processed_data.csv') # sum of snow

    numeric_df = all_df.loc[:, 'County':]
    
    # Date processing
    numeric_df['DATE'] = pd.to_datetime(numeric_df['DATE'], format='%Y-%m-%d')
    
    # Enhanced temporal features
    numeric_df['year'] = numeric_df['DATE'].dt.year
    numeric_df['month'] = numeric_df['DATE'].dt.month
    numeric_df['day'] = numeric_df['DATE'].dt.day
    # numeric_df['dayofweek'] = numeric_df['DATE'].dt.dayofweek
    # numeric_df['season'] = numeric_df['month'].map(lambda x: 
    #     'Winter' if x in [12, 1, 2] else
    #     'Spring' if x in [3, 4, 5] else
    #     'Summer' if x in [6, 7, 8] else 'Fall'
    # )
    
    # Create season dummy variables
    # season_dummies = pd.get_dummies(numeric_df['season'], prefix='season')
    # numeric_df = pd.concat([numeric_df, season_dummies], axis=1)
    
    # Indexing and cleaning
    numeric_df.index = numeric_df['DATE']
    numeric_df = numeric_df.dropna(subset=["SNOWsum"])
    temp_cols = ['TAVGmean', 'TMAXmax', 'TMINmin', 'TMINmean', 'TMAXmean']
    vpd_cols = ['vpdmin (hPa)mean', 'vpdmax (hPa)mean']
    numeric_df[vpd_cols] = numeric_df[vpd_cols].fillna(method='ffill')
    numeric_df[temp_cols] = numeric_df[temp_cols].fillna(method='ffill')
    numeric_df = numeric_df.fillna(0)
    
    # Handle missing values more carefully
    # for column in numeric_df.columns:
    #     if numeric_df[column].dtype in ['float64', 'int64']:
    #         # Fill missing values with column median instead of 0
    #         median_val = numeric_df[column].median()
    #         numeric_df[column] = numeric_df[column].fillna(median_val)
    
    # Create target variable
    target = "Target_Snowfall"
    numeric_df[target] = numeric_df["SNOWsum"]
    numeric_df = numeric_df.drop(['SNOWsum'], axis=1)
    
    # Filter date range
    nearest_start_date = min(numeric_df.index, key=lambda x: abs(x - pd.to_datetime(start_date)))
    test_start = pd.to_datetime(nearest_start_date)
    end_date = test_start + pd.DateOffset(years=23) + pd.DateOffset(months=2)
    
    # Filter for snow season (October to April)
    df_temp = numeric_df[(numeric_df.index >= test_start) & (numeric_df.index <= end_date) # & 
        # (numeric_df.index.month.isin([10, 11, 12, 1, 2, 3, 4]))
    ]
    
    # Filter for Coos county
    df_temp = df_temp[df_temp['County'] == 'Coos']
    
    # Drop unnecessary columns
    columns_to_drop = ['County', 'DATE', 'ELEVATION', 'freezing_daysum', 'AWNDmean']
    df_temp = df_temp.drop(columns_to_drop, axis=1)
    
    return df_temp

def transform_data(X_train, X_val, X_test, y_train, y_val, y_test):
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaled_dict = {}

    X_scaler = scaler.fit(X_train)

    scaled_dict['X_train_trns'] = X_scaler.transform(X_train)
    scaled_dict['X_val_trns'] = X_scaler.transform(X_val)
    scaled_dict['X_test_trns'] = X_scaler.transform(X_test)

    y_scaler = scaler2.fit(y_train)

    scaled_dict['y_train_trns'] = y_scaler.transform(y_train)
    scaled_dict['y_val_trns'] = y_scaler.transform(y_val)
    scaled_dict['y_test_trns'] = y_scaler.transform(y_test)

    return scaled_dict, X_scaler, y_scaler

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)

    X = df.drop(columns=[target_col])
    y = df[[target_col]]
    # y = df[target_col]
    #X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test



def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


def inverse_transform_df_out(df, scaler):
    # mm_scaler_pred = MinMaxScaler()
    # mm_scaler_pred.min_, mm_scaler_pred.scale_ = scaler.min_[3], scaler.scale_[3]
    mm_scaler_pred = scaler
    df_copy = df.copy()
    
    for c in df:
        c = [int(i) for i in c]
        # reshaped = df_copy[c].values.reshape(-1, 1)
        reshaped = df_copy[c].reshape(-1, 1)
        # inv_trns = mm_scaler_pred.inverse_transform(reshaped)
        inv_trns = scaler.inverse_transform(reshaped)
        df_copy[c] = inv_trns.flatten()
    
    return df_copy


    
def inverse_transform_df_out_scalers(df, scalers):
    # mm_scaler_pred = MinMaxScaler()
    # mm_scaler_pred.min_, mm_scaler_pred.scale_ = scaler.min_[3], scaler.scale_[3]
    df_copy = df.copy()
    for c in scalers:
        if c in df_copy.columns:
            # reshaped = df_copy[c].values.reshape(-1, 1)
            reshaped = df_copy[c].reshape(-1, 1)
            inv_trns = scalers[c].inverse_transform(reshaped)
            df_copy[c] = inv_trns.flatten()
    
    return df_copy


def create_sequences(data, seq_length):
    '''
        Currently unused
        Sequences now created in custom dataset
    '''
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)



