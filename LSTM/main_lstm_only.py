import torch
import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim

from Models.LSTM import LSTM, custom_collate
from Scripts.organize_data_and_loaders import get_normalized_dfs, get_custom_datasets
from Scripts.early_stopping import EarlyStopping
from Scripts.evaluate import calculate_metrics

from Scripts.train_lstm_only import train, test_model, predict, validate
from Data.preprocess_data import preprocess_data
from Scripts.plot import plot_losses
from sklearn.preprocessing import MinMaxScaler, RobustScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# export PYTHONPATH=/Users/briannamarcosano/Documents/SF_Testing_code:$PYTHONPATH
# source MODIS_env/bin/activate


start_date = '1995-01-01'
#start_date = '2001-01-01'
target = "Target_Snowfall"

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(start_date)
# scaled_dict, X_scaler, y_scaler = transform_data(X_train, X_val, X_test, y_train, y_val, y_test)
cols = X_train.columns

# X_scaler = MinMaxScaler()
# y_scaler = MinMaxScaler()
X_scaler = RobustScaler()
y_scaler = RobustScaler()

scaled_dict = {
    'X_train_trns': X_scaler.fit_transform(X_train).astype(np.float32),
    'X_val_trns': X_scaler.transform(X_val).astype(np.float32),
    'X_test_trns': X_scaler.transform(X_test).astype(np.float32)
}
scaled_dict['y_train_trns'] = y_train['Target_Snowfall'].astype(np.float32).to_numpy().reshape(-1, 1)
scaled_dict['y_val_trns'] = y_val['Target_Snowfall'].astype(np.float32).to_numpy().reshape(-1, 1)
scaled_dict['y_test_trns'] = y_test['Target_Snowfall'].astype(np.float32).to_numpy().reshape(-1, 1)

# creating normalized DataFrames for later use
train_scaled_df, val_scaled_df, test_scaled_df = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
train_scaled_df = get_normalized_dfs(scaled_dict['X_train_trns'], X_train)
val_scaled_df = get_normalized_dfs(scaled_dict['X_val_trns'], X_val)
test_scaled_df = get_normalized_dfs(scaled_dict['X_test_trns'], X_test)

train_scaled_df[target] = y_train
val_scaled_df[target] = y_val
test_scaled_df[target] = y_test

# creating custom datasetsand loaders
train_ds, val_ds, test_ds = get_custom_datasets(scaled_dict)
print(len(train_ds), len(val_ds), len(test_ds))

batch_size =15 # 16 with min max scaler
lr = .001
num_epochs = 400
# 370, or 320 without relu or 200 with relu?
# convergence happens much faster with relu

params = {'batch_size': batch_size,
          'collate_fn':custom_collate, 
          'shuffle': False }

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, collate_fn=custom_collate, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, **params)
test_loader = torch.utils.data.DataLoader(test_ds, **params)


'''
Model and train loop
'''
model = LSTM()
#model.load_state_dict(torch.load('saved_models/best_lstm_model.pt'))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=8,verbose=True)

train_losses, test_losses, val_losses = np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs)
best_val_loss = float('inf')  # Set to positive infinity initially
early_stopping = EarlyStopping(patience=8, min_delta=0.01)

for epoch in range(num_epochs): 
    train_loss = train(model = model, criterion = criterion, optimizer = optimizer, train_loader = train_loader)
    val_loss = validate(val_loader, model, criterion)
    # test_loss = test_model(test_loader, model, criterion)

    train_losses[epoch] = train_loss
    val_losses[epoch] = val_loss

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Train Loss: {train_loss:.3}, " 
             f"Val Loss: {val_loss:.3} \n---------")  #f"Test Loss: {test_loss:.3},
        
    # Save the model if val loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_dir = 'saved_models'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'lsm_only_best_model.pt')
        torch.save(model.state_dict(), save_path)

    if early_stopping.early_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break


'''
Predict and evaluate
'''
model.load_state_dict(torch.load(save_path))
predicted_col, actuals_col = "Model forecast", "Target_Snowfall"

Xy_train_df, Xy_val_df, Xy_test_df  = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
y_cols = ['Target_Snowfall', predicted_col]

train_predictions, y_list = predict(train_loader, model)
train_scaled_df[predicted_col] = train_predictions
cols_to_inverse_train = cols
train_scaled_df[cols_to_inverse_train] = X_scaler.inverse_transform(train_scaled_df[cols_to_inverse_train])
#train_scaled_df[y_cols] = y_scaler.inverse_transform(train_scaled_df[y_cols])

predictions, y_list = predict(val_loader, model)
val_scaled_df[predicted_col] = predictions
val_og = val_scaled_df.copy()
cols_to_inverse_val = cols
val_scaled_df[cols_to_inverse_val] = X_scaler.inverse_transform(val_scaled_df[cols_to_inverse_val]).round(2)
#val_scaled_df[y_cols] = y_scaler.inverse_transform(val_scaled_df[y_cols])
mae, mse, rmse, r2 = calculate_metrics(val_scaled_df[target], val_scaled_df[predicted_col])
print(f"Val: mae: {mae:.3}, mse: {mse:.3}, rmse: {rmse:.3}, r2: {r2:.3}\n")

test_predictions, y_list = predict(test_loader, model)
test_scaled_df[predicted_col] = test_predictions
cols_to_inverse_test = cols
test_scaled_df[cols_to_inverse_test] = X_scaler.inverse_transform(test_scaled_df[cols_to_inverse_test])
#test_scaled_df[y_cols] = y_scaler.inverse_transform(test_scaled_df[y_cols])
te_mae, te_mse, te_rmse, te_r2 = calculate_metrics(test_scaled_df[target], test_scaled_df[predicted_col])
print(f"Test: mae: {te_mae:.3}, mse: {te_mse:.3}, rmse: {te_rmse:.3}, r2: {te_r2:.3}\n")

print(test_scaled_df[[target, predicted_col]].tail(15))

print(f'{num_epochs}, {batch_size}, {lr}')

Xy_train_df, Xy_val_df, Xy_test_df = train_scaled_df.copy(), val_scaled_df.copy(), test_scaled_df.copy()
dfs = {'train_preds': Xy_train_df, 'val_preds': Xy_val_df, 'test_preds': Xy_test_df}

df_out = pd.concat([Xy_train_df, Xy_val_df, Xy_test_df])
df_out = df_out.sort_index()

plot_losses(train_losses, val_losses)
# fig = plot_snowfall_predictions(df_out, train_end=Xy_train_df.index[-1], val_end=Xy_val_df.index[-1], test_end=Xy_test_df.index[-1])
# fig.show()
#plot_tricolor(predicts_df, Xy_train_df, Xy_val_df, Xy_test_df, dfs, actuals_col)

