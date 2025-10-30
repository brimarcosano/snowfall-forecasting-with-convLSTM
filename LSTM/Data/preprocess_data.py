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
    all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/temporal_data_SUM_and_AVG.csv') # sum of snow   

    numeric_df = all_df.loc[:, 'County':]
    numeric_df['DATE'] = pd.to_datetime(numeric_df['DATE'], format='%Y-%m-%d')
    numeric_df['year'] = numeric_df['DATE'].dt.year
    numeric_df['month'] = numeric_df['DATE'].dt.month
    numeric_df['day'] = numeric_df['DATE'].dt.day

    numeric_df.index = numeric_df['DATE']
    # numeric_df = numeric_df.drop(['DATE'], axis=1)

    # drop other unneeded columns and rows with missing SNOW data. Fill NaN with 0
    numeric_df = numeric_df.dropna(subset=["SNOW"])
    num_filled = numeric_df.isna().sum().sum()

    numeric_df = numeric_df.fillna(0)

    print("num filled", num_filled)

    target = "Target_Snowfall"
    numeric_df[target] = numeric_df["SNOW"]
    ######### DROP SNOW ##########
    numeric_df = numeric_df.drop(['SNOW'], axis=1)

    df = pd.DataFrame(numeric_df, columns=numeric_df.columns, index=numeric_df.index)

    nearest_start_date = min(df.index, key=lambda x: abs(x - pd.to_datetime(start_date)))
    test_start = pd.to_datetime(nearest_start_date)

    end_date = test_start + pd.DateOffset(years=27) + pd.DateOffset(months=0)

    df_temp = df[(df.index >= test_start) & (df.index <= end_date)] # ((df.index.month >= 9) | (df.index.month <= 3))]

    df_temp = df_temp[df_temp['County'] == 'Coos']
    df_temp = df_temp.drop(['County'], axis=1)
    df_temp = df_temp.drop(['AWND','DATE', 'ELEVATION'], axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_temp, target, 0.1)

    return X_train, X_val, X_test, y_train, y_val, y_test

def transform_data(X_train, X_val, X_test, y_train, y_val, y_test):
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()
    # scaled_dict = {'train': {}, 'val': {}, 'test': {}}
    scaled_dict = {}

    X_scaler = scaler.fit(X_train)

    # scaled_dict['train']['X_train_trns'] = X_scaler.transform(X_train)
    # scaled_dict['val']['X_val_trns'] = X_scaler.transform(X_val)
    # scaled_dict['test']['X_test_trns'] = X_scaler.transform(X_test)

    scaled_dict['X_train_trns'] = X_scaler.transform(X_train)
    scaled_dict['X_val_trns'] = X_scaler.transform(X_val)
    scaled_dict['X_test_trns'] = X_scaler.transform(X_test)

    y_scaler = scaler2.fit(y_train)

    # scaled_dict['train']['y_train_trns'] = y_scaler.transform(y_train)
    # scaled_dict['val']['y_val_trns'] = y_scaler.transform(y_val)
    # scaled_dict['test']['y_test_trns'] = y_scaler.transform(y_test)
    scaled_dict['y_train_trns'] = y_scaler.transform(y_train)
    scaled_dict['y_val_trns'] = y_scaler.transform(y_val)
    scaled_dict['y_test_trns'] = y_scaler.transform(y_test)

    return scaled_dict, X_scaler, y_scaler

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)

    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def drop_consecutive_zeros(group):
    return not ((group['SNOW'] == 0).all() and (len(group) > 1))


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

def create_lagged_target(target_sensor, df):
    '''
        Currently unused
        Create a lagged target column for the target sensor
    '''
    forecast_lead = 0
    target = f"{target_sensor}_lead{forecast_lead}"
    target = "SNOW_lead7"
    df[target] = df[target_sensor].shift(-forecast_lead)
    df['target'] = df.loc[df['County'] == 'Coos', 'SNOW'].shift(-forecast_lead)

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
