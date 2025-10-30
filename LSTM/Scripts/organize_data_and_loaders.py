from Data.preprocess_data import preprocess_data, inverse_transform_df_out, transform_data
from Data.LSTM_CustomDataset import CustomDataset
import pandas as pd

def get_normalized_dfs(vals, df):
    # final_scaled_dict = {}
    # for k, v in scaled_dict.items():
    #     # if k.split('_')[1] == 'train':
    #     if k.split('_')[0] == 'X':
    #         final_scaled_dict[k] = pd.DataFrame(v, columns=X.columns)
        # else:
        #     final_scaled_dict[k] = pd.DataFrame(v, columns=y.columns, index=X.index)
    normalized_df = pd.DataFrame(vals, columns=df.columns, index=df.index)
    # normalized_df = normalized_df.sort_values('date', inplace=True)
    return normalized_df
    # return final_scaled_dict['X_train_trns'], final_scaled_dict['X_val_trns'], final_scaled_dict['X_test_trns']
    # , scaled_dict[y_train_trns], scaled_dict[y_val_trns], scaled_dict[y_test_trns]

def get_custom_datasets(scaled_dict):
    datasets = {}

    train_ds = CustomDataset(scaled_dict['X_train_trns'],scaled_dict['y_train_trns'])
    val_ds = CustomDataset(scaled_dict['X_val_trns'],scaled_dict['y_val_trns'])
    test_ds = CustomDataset(scaled_dict['X_test_trns'],scaled_dict['y_test_trns'])

    return train_ds, val_ds, test_ds