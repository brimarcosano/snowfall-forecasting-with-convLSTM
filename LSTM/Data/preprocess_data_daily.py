import plotly.io as pio
from sklearn.model_selection import train_test_split
import pandas as pd

pio.templates.default = "plotly_white"

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    # X, y = feature_label_split(df, target_col)
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data_daily(start_date):

    all_df = pd.read_csv('CNN_LSTM_Thesis/LSTM/Data/Raw_and_avg/temporal_data_by_day.csv', index_col='DATE')

    meta = []
    meta = pd.DataFrame.from_records(
        meta, columns=['LATITUDE','LONGITUDE','County','ELEVATION']
    )
    meta.to_csv('CNN_LSTM_Thesis/LSTM/Data/Unused/tabular_metadata.csv', index=False)

    # use columns from "county" forward
    numeric_df = all_df.loc[:, 'County':]

    #numeric_df.index = pd.to_datetime(numeric_df.index, format="%m/%d/%y")
    numeric_df.index = pd.to_datetime(numeric_df.index)
    #numeric_df = numeric_df.interpolate(method='linear')


    # Setting target and creating new column with values in SNOW column
    # target = "Target_Snowfall"
    # numeric_df[target] = numeric_df["SNOW"]

    # drop other unneeded columns and rows with no SNOW data. Fill NaN with 0
    numeric_df = numeric_df.drop(['ELEVATION'], axis=1)
    numeric_df = numeric_df.dropna(subset=["SNOW"])
    numeric_df = numeric_df.fillna(0)

    target_sensor = "SNOW"
    forecast_lead = 20
    # 1 month into future
    target = f"{target_sensor}_lead{forecast_lead}"
    target = "SNOW_lead7"
    numeric_df[target] = numeric_df[target_sensor].shift(-forecast_lead)

    subset = ["AWND", "PRCP","SNOW","SNWD", "TAVG", "TMIN", "vpdmin (hPa)", "vpdmax (hPa)", target]

    df = pd.DataFrame(numeric_df, columns=numeric_df.columns, index=numeric_df.index)

    # FOR DAILY DATA TEST START:
    test_start = pd.to_datetime(start_date)
    nearest_end_date = min(df.index, key=lambda x: abs(x - (test_start + pd.DateOffset())))

    end_date = test_start + pd.DateOffset(years=22) + pd.DateOffset(months=3)

    df_temp = df[(df.index >= test_start) & (df.index <= end_date)] # & ((df.index.month >= 10) | (df.index.month <= 6))]

    df_temp = df_temp[df_temp['County'] == 'Carroll']
    df_temp = df_temp.drop(['County'], axis=1)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_temp, target, 0.15)

    return X_train, X_val, X_test, y_train, y_val, y_test

def transform_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler):
    X_train_trns= scaler.fit_transform(X_train)
    X_val_trns = scaler.transform(X_val)
    X_test_trns = scaler.transform(X_test)
    y_train_trns = scaler.fit_transform(y_train)
    y_test_trns = scaler.transform(y_test)
    y_val_trns = scaler.transform(y_val)

    return X_train_trns, X_test_trns, X_val_trns, y_train_trns, y_test_trns, y_val_trns

    
def inverse_transform_df_out(df, scaler):
    # mm_scaler_pred = MinMaxScaler()
    # mm_scaler_pred.min_, mm_scaler_pred.scale_ = scaler.min_[3], scaler.scale_[3]
    mm_scaler_pred = scaler

    for c in df:
        reshaped = df[c].values.reshape(-1, 1)
        # inv_trns = mm_scaler_pred.inverse_transform(reshaped)
        inv_trns = scaler.inverse_transform(reshaped)
        df[c] = inv_trns.flatten()
    
    return df



# def format_predictions(predictions, vals, df_test, scaler):
#     values = np.concatenate(vals, axis=0).ravel()
#     preds = np.concatenate(predictions, axis=0).ravel()
#     df_result = pd.DataFrame(data={"actuals": values, "prediction": preds}, index=df_test.head(len(preds)).index)
#     df_result = df_result.sort_index()
#     df_result = inverse_transform(scaler, df_result, ["actuals","prediction"])
#     return df_result

