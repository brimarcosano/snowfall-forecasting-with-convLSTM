import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import joblib
import pandas as pd
from CNN_LSTM_Thesis.CNN.Datasets.build_and_split_img_ds import split_image_datasets_only, split_and_load_data, create_dataloader_dict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from Scripts.train_val import train_only, validate_only, test_only, predict
from Scripts.visualize_model_cnn import visualize_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Models.CNN_FC import CombinedModel, CNNImageOnly, CNNModel
from Datasets.snowfall_df import preprocess_data
from Scripts.save_dataloaders import load_torch_save, torch_save, datasets_exist
from torchvision import models
from torch.utils.data import DataLoader
import multiprocessing
from torch.multiprocessing import freeze_support
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

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    num_epochs = 8
    learning_rate = 0.00001
    batch_size = 8

    dataloader_dir = 'saved_dataloaders'
    os.makedirs(dataloader_dir, exist_ok=True)
    save_dir = 'saved_models_cnn'

    numeric_data = preprocess_data('2000-01-01')
    train_dataset, val_dataset, test_dataset = split_and_load_data(batch_size=batch_size, numeric_data_one=numeric_data)
    dataloaders = create_dataloader_dict(train_dataset, val_dataset, test_dataset, batch_size)

    os.makedirs(save_dir, exist_ok=True)
    save_path_cnn = os.path.join(save_dir, 'cnn_only_best_model.pt')
    save_path_rmse = os.path.join(save_dir, 'cnn_best_model_rmse.pt')

    # print("Training dataset shape:", train_dataset[0][0][0].shape, train_dataset[0][0][1].shape, train_dataset[0][1].shape)
    # print("Validation dataset shape:", val_dataset[0][0][0].shape, val_dataset[0][0][1].shape, val_dataset[0][1].shape)
    # print("Test dataset shape:", test_dataset[0][0][0].shape, test_dataset[0][0][1].shape, test_dataset[0][1].shape)
    
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features  # Get the number of input features to the final layer
    print(num_features)
    model.fc = nn.Linear(num_features, 1)
    model = model.to(device)

    # sample_image = torch.randn(1, 3, 224, 224)  # Assuming input size of (3, 224, 224)
    # output = model(sample_image)
    # print("Output shape:", output.shape)
    # Load the model if a save path is provided
    # if save_path_cnn is not None:
    #     model = load_model(model, save_path_cnn)

    # Loss and optimizer
    criterion = nn.MSELoss()
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_val_loss = float('inf')
    best_val_rmse = float('inf')

    train_losses, test_losses, val_losses = np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs)
    best_val_loss = float('inf')  # Set to positive infinity initially

    for epoch in range(num_epochs):  #201 best so far with .01 and batch 26
        train_loss = train_only(
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            dataloader = dataloaders['train'],
            device = device
        )
        val_loss = validate_only(model, criterion, dataloaders['val'])
        scheduler.step()

        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        if epoch % 5 == 0:
            print(f"Epoch {epoch} - Train Loss: {train_loss:.3}, " 
                f"Val Loss: {val_loss:.3} \n---------")  #f"Test Loss: {test_loss:.3},

    # model.load_state_dict(torch.load(save_path_cnn))
        
    test_loss = test_only(model, criterion, dataloaders['test'])
    print(f"Test Loss: {test_loss:.4f}")

    # Get predictions
    val_pred_df = predict(dataloaders['val'], model)
    test_pred_df = predict(dataloaders['test'], model)

    val_mae = mean_absolute_error(val_pred_df['Actuals'], val_pred_df['Predictions'])
    val_mse = mean_squared_error(val_pred_df['Actuals'], val_pred_df['Predictions'])
    val_rmse = np.sqrt(mean_squared_error(val_pred_df['Actuals'], val_pred_df['Predictions'])) 
    val_r2 = r2_score(val_pred_df['Actuals'], val_pred_df['Predictions'])

    test_mae = mean_absolute_error(test_pred_df['Actuals'], test_pred_df['Predictions'])
    test_mse = mean_squared_error(test_pred_df['Actuals'], test_pred_df['Predictions'])
    test_rmse = np.sqrt(mean_squared_error(test_pred_df['Actuals'], test_pred_df['Predictions']))
    test_r2 = r2_score(test_pred_df['Actuals'], test_pred_df['Predictions'])

    # Print metrics
    print("\nValidation Metrics:")
    print(f"MAE: {val_mae:.4f}")
    print(f"MSE: {val_mse:.4f}")
    print(f"RMSE: {val_rmse:.4f}")
    print(f"R2: {val_r2:.4f}")

    print("\nTest Metrics:")
    print(f"MAE: {test_mae:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R2: {test_r2:.4f}")


        # Save the model if val RMSE improves
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), save_path_rmse)

    print(test_pred_df)
    # visualize_model(device, model_tr, dataloaders, class_names)
    #visualize_model(device, model_tr, dataloaders)

if __name__ == '__main__':
    # Add these lines to handle multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    freeze_support()
    main()