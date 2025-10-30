

import torch
import os
import numpy as np
import torch
import os
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y, seq_len=7):
        self.seq_len = seq_len
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequences = self.create_inout_sequences()
    

    def create_inout_sequences(self):
        # inout_seq = []
        # L = len(self.X)

        # for i in range(L - self.seq_len):
        #     train_seq = self.X[i:i + self.seq_len]
        #     train_label = self.y[i + self.seq_len]
        #     inout_seq.append((train_seq ,train_label))
        
        # return inout_seq
        inout_seq = []
        L = len(self.X)
        for i in range(L):
            if i < self.seq_len - 1:
                padding = self.X[0].repeat(self.seq_len - i - 1, 1)
                train_seq = torch.cat((padding, self.X[0:(i + 1), :]), 0)
            else:
                train_seq = self.X[(i - self.seq_len + 1):(i + 1), :]
            train_label = self.y[i]
            inout_seq.append((train_seq, train_label))
        return inout_seq
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, i):
        return self.sequences[i]

       # return self.X.shape[0]
    # def __len__(self):
    #     return len(self.X)
    #    # return self.X.shape[0]

    # def __getitem__(self, i):
    #     # i += 7
    #     if i >= self.seq_len - 1:
    #         i_start = i - self.seq_len + 1
    #         x = self.X[i_start:(i + 1), :]
    #     else:
    #         padding = self.X[0].repeat(self.seq_len - i - 1, 1)
    #         x = self.X[0:(i + 1), :]
    #         x = torch.cat((padding, x), 0)

    #     if i + 1 < len(self.y):
    #         y = self.y[i + 1]
    #     else:
    #         y = self.y[-1]  # or any other default value

    #     return x, y

        # return x, self.y[i]
    
        # sample = self.X[i]
        # return torch.Tensor(sample['sequence'].astype('float32')), torch.Tensor(sample['target'].astype('float32'))

def create_inout_sequences(df, seq_len, pw, target_cols, drop_targets=False):
    data = dict() # Store results into a dictionary
    L = len(df)
    for i in range(L-seq_len):

        # Get current sequence  
        sequence = df[i:i+seq_len].values
        # Get values right after the current sequence
        target = df[i+seq_len:i+seq_len+pw][target_cols].values
        data[i] = {'sequence': sequence, 'target': target}
    
    return data
#https://towardsdatascience.com/time-series-forecasting-with-deep-learning-in-pytorch-lstm-rnn-1ba339885f0c

    # if i >= seq_len - 1:
    #     i_start = i - seq_len + 1
    #     x = X[i_start:(i + 1), :]
    # else:
    #     padding = X[0].repeat(seq_len - i - 1, 1)
    #     x = X[0:(i + 1), :]
    #     x = torch.cat((padding, x), 0)

    # # Get the county for the current sample
    # county_index = X.columns.get_loc('Coos')
    # county = x[-1, county_index]

    # # Select the y value for the correct county
    # y = y[y['Coos'] == county]

    # return x, y[i]


# sequence stuff about words:# https://towardsdatascience.com/dataloader-for-sequential-data-using-pytorch-deep-learning-framework-part-2-ed3ad5f6ad82


        # if i >= self.seq_len - 1:
        #     i_start = i - self.seq_len + 1
        #     x = self.X[i_start:(i + 1), :]
        # else:
        #     padding = self.X[0].repeat(self.seq_len - i - 1, 1)
        #     x = self.X[0:(i + 1), :]
        #     x = torch.cat((padding, x), 0)

        # return x, self.y[i]



        # input_data, label = self.data[i]
        # return input_data, label
    


# class CustomDataset(Dataset):
#     def __init__(self, data_frame, img_dir, scaler=None): #, labels, tabular_transform, img_transform):
#         self.data_frame = data_frame
#         self.img_dir = img_dir
#         self.scaler = scaler

#     def get_subset(self, indices, start, end):
#         return indices[start : start + end]

#     def __len__(self):
#         return len(self.data_frame)
#         # return len(self.data_frame) - self.window

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         print(idx)

#         start_idx = idx
#         end_idx = idx + self.window

#         dates = sorted(self.data_frame.index)
#         # tabular_by_idx = dates[idx]
#         # tabular = self.data_frame.loc[tabular_by_idx, :]
#         # tabular = self.data_frame[idx:idx+self.window]
#         tabular_by_idx = dates[start_idx:end_idx]
#         tabular = self.data_frame.loc[tabular_by_idx]

#         for date in tabular.index:
#             tabular_date = int(date.strftime('%Y%m%d'))

#         # tabular = tabular[["ELEVATION", "AWND", "PRCP", "SNOW", "SNWD", "TAVG", "TMIN", "vpdmin (hPa)", "vpdmax (hPa)"]]
#         tabular = tabular[["AWND", "PRCP", "SNWD", "TAVG", "TMIN", "vpdmin (hPa)", "vpdmax (hPa)"]]
#         # tabular = tabular.tolist()
#         tabular = torch.FloatTensor(tabular.values)
#         y = torch.FloatTensor(y)
                    
#         return images, tabular, y
    
