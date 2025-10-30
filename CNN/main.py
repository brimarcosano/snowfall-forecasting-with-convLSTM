
import logging
import os
import joblib
import pandas as pd
from CNN_LSTM_Thesis.CNN.Datasets.build_and_split_img_ds import split_and_load_data, create_dataloader_dict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import models

from Scripts.train_val import train_only, validate_only, test_only, predict
from Scripts.visualize_model_cnn import visualize_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Models.CNN_FC import CombinedModel,CNNImageOnly, CNNModel
from Datasets.snowfall_df import preprocess_data
from Scripts.save_dataloaders import load_torch_save, torch_save, datasets_exist
from torchvision.models import ResNet18_Weights


def load_model(model, load_path):
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    return model

def print_sample_data(dataloader, num_samples=5):
    for i, ((images, numeric_features), targets) in enumerate(dataloader):
        if i >= num_samples:
            break
        print(f"Sample {i + 1}:")
        print(f"Image Path: {dataloader.dataset.image_paths[i]}")
        print(f"Numeric Features: {numeric_features[i]}")
        print(f"Target: {targets[i]}")
        print(f"Date: {dataloader.dataset.dates[i]}")
        print("-" * 30)

num_epochs = 7
learning_rate = 0.00001
batch_size = 8

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
dataloader_dir = 'saved_dataloaders'
os.makedirs(dataloader_dir, exist_ok=True)
dataloader_path = os.path.join(dataloader_dir, 'dataloaders.pt')
og_val_path = os.path.join(dataloader_dir, 'og_val.pt')
og_train_path = os.path.join(dataloader_dir, 'og_train.pt')
og_test_path = os.path.join(dataloader_dir, 'og_test.pt')
numeric_data_path = os.path.join(dataloader_dir, 'numeric_data.pt')
save_dir = 'saved_models_cnn'

# rerun_data = input('Rerun data? (y/n): ')
try:
    if not datasets_exist(dataloader_path) or not datasets_exist(og_val_path) or not datasets_exist(og_train_path) or not datasets_exist(og_test_path):
        numeric_data = preprocess_data('2000-01-01')
        train_dataset, val_dataset, test_dataset = split_and_load_data(batch_size=batch_size, numeric_data_one=numeric_data)
        dataloaders = create_dataloader_dict(train_dataset, val_dataset, test_dataset, batch_size)
        torch_save(dataloaders, dataloader_path)
        torch_save(val_dataset, og_val_path)
        torch_save(train_dataset, og_train_path)
        torch_save(test_dataset, og_test_path)
    else:
        dataloaders = load_torch_save(dataloader_path)
        train_dataset = load_torch_save(og_train_path)
        val_dataset = load_torch_save(og_val_path)
        test_dataset = load_torch_save(og_test_path)
        logging.info('Dataloaders loaded successfully')
except Exception as e:
    logging.error(f'Error loading dataloaders: {e}')    

os.makedirs(save_dir, exist_ok=True)
save_path_cnn = os.path.join(save_dir, 'cnn_only_best_model.pt')
save_path_rmse = os.path.join(save_dir, 'cnn_best_model_rmse.pt')

#model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model = CNNModel()

# Load the model if a save path is provided
# if save_path_cnn is not None:
#     model = load_model(model, save_path_cnn)

# Loss and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)


best_val_loss = float('inf')
best_val_rmse = float('inf')

train_losses, test_losses, val_losses = np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs)
best_val_loss = float('inf')  # Set to positive infinity initially

# Implement early stopping
patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    train_loss = train_only(model, criterion, optimizer, dataloaders['train'])
    val_loss = validate_only(model, criterion, dataloaders['val'])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), save_path_rmse)
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 5 == 0:
        print(f"Epoch {epoch} - Train Loss: {train_loss:.3}, " 
             f"Val Loss: {val_loss:.3} \n---------")  #f"Test Loss: {test_loss:.3},
        
# model.load_state_dict(torch.load(save_path_cnn))
pred_df = predict(dataloaders['val'], model)
pred_df_test = predict(dataloaders['test'], model)
    
t = val_dataset.get_og_data()['target']
t = t.iloc[:len(pred_df)]
pred_df['Snow_og'] = t.values
og_ds = val_dataset.get_og_data().iloc[:len(pred_df)]
pred_df.index = og_ds.index

t2 = test_dataset.get_og_data()['target']
t2 = t2.iloc[:len(pred_df_test)]
pred_df_test['Snow_og'] = t2.values
og_ds_test = test_dataset.get_og_data().iloc[:len(pred_df_test)]
pred_df_test.index = og_ds_test.index

val_mae = mean_absolute_error(pred_df['Actuals'], pred_df['Predictions'])
val_rmse = np.sqrt(mean_squared_error(pred_df['Actuals'], pred_df['Predictions'])) 
val_r2 = r2_score(pred_df['Actuals'], pred_df['Predictions'])

test_mae = mean_absolute_error(pred_df_test['Actuals'], pred_df_test['Predictions'])
test_rmse = np.sqrt(mean_squared_error(pred_df_test['Actuals'], pred_df_test['Predictions']))
test_r2 = r2_score(pred_df_test['Actuals'], pred_df_test['Predictions'])

# Print metrics
print("\nValidation Metrics:")
print(f"MAE: {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"R2: {val_r2:.4f}")

print("\nTest Metrics:")
print(f"MAE: {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R2: {test_r2:.4f}")

    # Save the model if val RMSE improves
if val_rmse < best_val_rmse:
    best_val_rmse = val_rmse
    torch.save(model.state_dict(), save_path_rmse)

print(pred_df)
# visualize_model(device, model_tr, dataloaders, class_names)
#visualize_model(device, model_tr, dataloaders)
