import gc
import os
import numpy as np
import torch
import torch.nn as nn
import time
from tempfile import TemporaryDirectory
import shutil
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_only(model, criterion, optimizer, dataloader):
    total_loss = 0.0
    model.train()
    with torch.set_grad_enabled(True):
        for (inputs, numeric_features), labels in dataloader:
            inputs, numeric_features, labels = inputs.float(), numeric_features.float(), labels.float()
            labels = labels.view(-1, 1)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs, numeric_features).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def validate_only(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (inputs, numeric_features), labels in dataloader:
            inputs = inputs.float()
            numeric_features = numeric_features.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            outputs = model(inputs, numeric_features)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

    # scheduler.step()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def test_only(model, criterion, scheduler, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (inputs, numeric_features), labels in dataloader:
            inputs = inputs.float()
            numeric_features = numeric_features.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            outputs = model(inputs, numeric_features)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def predict(dataloader, model):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0
    with torch.no_grad():
        for (inputs, numeric_features), labels in dataloader:
            inputs = inputs.float()
            numeric_features = numeric_features.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            outputs = model(inputs, numeric_features).squeeze(-1)

            predictions.extend(outputs.tolist())
            actuals.extend(labels.tolist())

        df = pd.DataFrame()
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        df['Predictions'] = predictions.flatten()
        df['Actuals'] = actuals.flatten()

    return df


def train_cnn(model, criterion, optimizer, dataloader):
    total_loss = 0.0
    model.train()
    with torch.set_grad_enabled(True):
        for (images, _), labels in dataloader:
            images = images.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # total_loss += loss.item() * images.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def validate_cnn(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images,  labels in dataloader:
            images = images.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # total_loss += loss.item() * images.size(0)
            total_loss += loss.item()

    # scheduler.step()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def test_cnn(model, criterion, scheduler, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images,  labels in dataloader:
            images = images.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # total_loss += loss.item() * images.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def train_lstm(model, criterion, optimizer, dataloader):
    total_loss = 0.0
    model.train()
    with torch.set_grad_enabled(True):
        for numeric_features, labels in dataloader:
            numeric_features = numeric_features.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(numeric_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def validate_lstm(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for numeric_features, labels in dataloader:
            numeric_features = numeric_features.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            outputs = model(numeric_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # scheduler.step()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def test_lstm(model, criterion, scheduler, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for numeric_features, labels in dataloader:
            numeric_features = numeric_features.float()
            labels = labels.float()
            labels = labels.view(-1, 1)
            outputs = model(numeric_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

# def predict(dataloader, model):
#     model.eval()
#     predictions = []
#     actuals = []
#     total_loss = 0
#     with torch.no_grad():
#         for images, labels in dataloader:
#             images = images.float()
#             # numeric_features = numeric_features.float()
#             labels = labels.float()
#             labels = labels.view(-1, 1)
#             outputs = model(images)

#             predictions.extend(outputs.tolist())
#             actuals.extend(labels.tolist())

#         df = pd.DataFrame()
#         predictions = np.array(predictions)
#         actuals = np.array(actuals)

#         df['Predictions'] = predictions.flatten()
#         df['Actuals'] = actuals.flatten()

#     return df


def train_models(cnn_model, lstm_model, cnn_dataloader, lstm_dataloader, 
                 cnn_optimizer, lstm_optimizer, 
                 cnn_criterion, lstm_criterion, 
                 cnn_epochs, lstm_epochs):

    # Histories for logging
    cnn_train_losses, cnn_val_losses = [], []
    lstm_train_losses, lstm_val_losses = [], []

    for epoch in range(max(cnn_epochs, lstm_epochs)):
        print(f"Epoch {epoch + 1}/{max(cnn_epochs, lstm_epochs)}")

        # Train CNN if within its epoch range
        if epoch < cnn_epochs:
            cnn_train_loss = train_cnn(cnn_model, cnn_criterion, cnn_optimizer, cnn_dataloader)
            cnn_val_loss = validate_cnn(cnn_model, cnn_criterion, cnn_dataloader)
            cnn_train_losses.append(cnn_train_loss)
            cnn_val_losses.append(cnn_val_loss)

        # Train LSTM if within its epoch range
        if epoch < lstm_epochs:
            lstm_train_loss = train_lstm(lstm_model, lstm_criterion, lstm_optimizer, lstm_dataloader)
            lstm_val_loss = validate_lstm(lstm_model, lstm_criterion, lstm_dataloader)
            lstm_train_losses.append(lstm_train_loss)
            lstm_val_losses.append(lstm_val_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} - CNN Train Loss: {cnn_train_loss:.4}, CNN Val Loss: {cnn_val_loss:.4}," 
                  f"LSTM Train Loss: {lstm_train_loss:.4}, LSTM Val Loss: {lstm_val_loss:.4} \n---------")
    # return {
    #     'cnn': {'train_losses': cnn_train_losses, 'val_losses': cnn_val_losses},
    #     'lstm': {'train_losses': lstm_train_losses, 'val_losses': lstm_val_losses},
    # }
    return cnn_train_losses, cnn_val_losses, lstm_train_losses, lstm_val_losses


def predict_both(cnn_dataloader, lstm_dataloader, cnn_model, lstm_model):
    cnn_model.eval()
    lstm_model.eval()
    
    cnn_predictions, lstm_predictions = [], []
    actuals = []
    
    # Ensure both dataloaders are iterated together
    cnn_iter = iter(cnn_dataloader)
    lstm_iter = iter(lstm_dataloader)
    
    with torch.no_grad():
        for _ in range(len(cnn_dataloader)):
            try:
                (cnn_images, cnn_labels) = next(cnn_iter)
                (lstm_numeric_features, lstm_labels) = next(lstm_iter)
            except StopIteration:
                break
            
            # Process CNN inputs
            cnn_images = cnn_images.float()
            cnn_labels = cnn_labels.float()
            cnn_labels = cnn_labels.view(-1, 1)
            cnn_outputs = cnn_model(cnn_images)
            cnn_predictions.extend(cnn_outputs.tolist())
            
            # Process LSTM inputs
            lstm_numeric_features = lstm_numeric_features.float()
            lstm_labels = lstm_labels.float()
            lstm_labels = lstm_labels.view(-1, 1)
            lstm_outputs = lstm_model(lstm_numeric_features)
            lstm_predictions.extend(lstm_outputs.tolist())
            
            # Store actual labels (assuming they are the same across dataloaders)
            actuals.extend(cnn_labels.tolist())

    # Combine predictions into a DataFrame
    df = pd.DataFrame()
    df['CNN_Predictions'] = np.array(cnn_predictions).flatten()
    df['LSTM_Predictions'] = np.array(lstm_predictions).flatten()
    df['Actuals'] = np.array(actuals).flatten()
    
    return df