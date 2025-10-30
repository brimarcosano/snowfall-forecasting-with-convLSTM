import torch
import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim

from Models.LSTM import LSTM, custom_collate

from Scripts.train_lstm_only import train, test_model, predict, validate
from Data.preprocess_data_daily import preprocess_data_daily, inverse_transform_df_out, transform_data
from Data.LSTM_CustomDataset import CustomDataset
from Scripts.plot import plot_bicolor, plot_tricolor, plot_losses

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Params and data
'''
start_date = '2000-01-01'
batch_size = 18 #changing batch size doesn't seem to do much?
lr = .00001  #.0001 best so far  #val loss plateus more with .001?
num_epochs = 60

scaler = MinMaxScaler()

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data_daily(start_date)
X_train_trns, X_test_trns, X_val_trns, y_train_trns, y_test_trns, y_val_trns = transform_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler)

train_normal_df = pd.DataFrame(X_train_trns, columns=X_train.columns, index=X_train.index)
test_normal_df = pd.DataFrame(X_test_trns, columns=X_test.columns, index=X_test.index)
val_normal_df = pd.DataFrame(X_val_trns, columns=X_val.columns, index=X_val.index)

train_ds = CustomDataset(X_train_trns, y_train_trns)
val_ds = CustomDataset(X_val_trns, y_val_trns)
test_ds = CustomDataset(X_test_trns, y_test_trns)

params = {'batch_size': batch_size,
          'collate_fn':custom_collate, 
          'shuffle': False }

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, collate_fn=custom_collate, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_ds, **params)
test_loader = torch.utils.data.DataLoader(test_ds, **params)


'''
Model and train loop
'''
model = LSTM()

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)
val_losses = np.zeros(num_epochs)
best_test_loss = float('inf')  # Set to positive infinity initially

for epoch in range(num_epochs):  #201 best so far with .01 and batch 26
    # print(f"Epoch {epoch}\n---------") def train(device, model, criterion, optimizer, num_epochs, train_loader):

    train_loss = train(
        device = device,
        model = model,
        criterion = criterion,
        optimizer = optimizer,
        train_loader = train_loader
    )
    # test_loss = test_model(test_loader, model, criterion)
    val_loss = validate(val_loader, model, criterion)

    train_losses[epoch] = train_loss
    # test_losses[epoch] = test_loss
    val_losses[epoch] = val_loss

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4}, " 
             f"Val Loss: {val_loss:.4} \n---------")  #f"Test Loss: {test_loss:.3},
        
        # Save the model if val loss improves
        if val_loss < best_test_loss:
            best_test_loss = val_loss
            save_dir = 'saved_models'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'lsm_only_best_model.pt')
            torch.save(model.state_dict(), save_path)

X_train_df_copy, X_val_df_copy, X_test_df_copy = train_normal_df.copy(), val_normal_df.copy(), test_normal_df.copy()


'''
Predict and evaluate
'''
ystar_col, actuals_col = "Model forecast", "SNOW_lead7"
train_eval_loader = torch.utils.data.DataLoader(train_ds, collate_fn=custom_collate, shuffle=False) # need unshuffled train loader to see chron order
val_eval_loader = torch.utils.data.DataLoader(val_ds, collate_fn=custom_collate, drop_last=False, shuffle=False)
test_eval_loader = torch.utils.data.DataLoader(test_ds, collate_fn=custom_collate, drop_last=False, shuffle=False)

X_train_df_copy[ystar_col] = predict(train_eval_loader, model).squeeze()
X_val_df_copy[ystar_col] = predict(val_eval_loader, model).squeeze()
X_test_df_copy[ystar_col] = predict(test_eval_loader, model).squeeze()

#predicts_df = inverse_transform_df_out(df_out, scaler)
train_preds = inverse_transform_df_out(X_train_df_copy, scaler)
val_preds = inverse_transform_df_out(X_val_df_copy, scaler)
test_preds = inverse_transform_df_out(X_test_df_copy, scaler)

Xy_train_df = train_preds.assign(SNOW_lead7=y_train)
Xy_val_df = val_preds.assign(SNOW_lead7=y_val)
Xy_test_df = test_preds.assign(SNOW_lead7=y_test)

df_out = pd.concat((Xy_train_df , Xy_val_df, Xy_test_df))[[actuals_col, ystar_col]]
predicts_df = df_out


print(Xy_val_df[[actuals_col,ystar_col]])

mae = mean_absolute_error(Xy_train_df[actuals_col], Xy_train_df[ystar_col])
mse = mean_squared_error(Xy_train_df[actuals_col], Xy_train_df[ystar_col])
rmse = np.sqrt(mse)
r2 = r2_score(Xy_train_df[actuals_col], Xy_train_df[ystar_col])
print(f"Train: mae: {mae:.3}, mse: {mse:.3}, rmse: {rmse:.3}, r2: {r2:.3}\n")

mae = mean_absolute_error(Xy_val_df[actuals_col], Xy_val_df[ystar_col])
mse = mean_squared_error(Xy_val_df[actuals_col], Xy_val_df[ystar_col])
rmse = np.sqrt(mse)
r2 = r2_score(Xy_val_df[actuals_col], Xy_val_df[ystar_col])
print(f"Validation: mae: {mae:.3}, mse: {mse:.3}, rmse: {rmse:.3}, r2: {r2:.3}\n")

dfs = {'train_preds': train_preds, 'val_preds': val_preds, 'test_preds': test_preds}

plot_tricolor(predicts_df, Xy_train_dfal_with_y, Xy_test_df, dfs, actuals_col)
plot_losses(train_losses, val_losses)
