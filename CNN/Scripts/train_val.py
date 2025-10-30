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

# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# def train_only(model, criterion, optimizer, dataloader):
#     total_loss = 0.0
#     model.train()
#     device = next(model.parameters()).device
    
#     for (images, numeric_features), labels in dataloader:
#         # Move data to device
#         images = images.float().to(device)
#         numeric_features = numeric_features.float().to(device)
#         labels = labels.float().to(device).view(-1, 1)
        
#         # Zero gradients
#         optimizer.zero_grad()
        
#         with torch.set_grad_enabled(True):
#             # Forward pass
#             outputs = model(images, numeric_features)
#             loss = criterion(outputs, labels)
            
#             # Backward pass
#             loss.backward()
#             optimizer.step()
            
#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# def validate_only(model, criterion, dataloader):
#     model.eval()
#     total_loss = 0
#     device = next(model.parameters()).device
    
#     with torch.no_grad():
#         for (images, numeric_features), labels in dataloader:
#             # Move data to device
#             images = images.float().to(device)
#             numeric_features = numeric_features.float().to(device)
#             labels = labels.float().to(device).view(-1, 1)
            
#             # Forward pass
#             outputs = model(images, numeric_features)
#             loss = criterion(outputs, labels)
            
#             total_loss += loss.item()
    
#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# def test_only(model, criterion, dataloader):
#     model.eval()
#     total_loss = 0
#     device = next(model.parameters()).device
    
#     with torch.no_grad():
#         for (images, numeric_features), labels in dataloader:
#             # Move data to device
#             images = images.float().to(device)
#             numeric_features = numeric_features.float().to(device)
#             labels = labels.float().to(device).view(-1, 1)
            
#             # Forward pass
#             outputs = model(images, numeric_features)
#             loss = criterion(outputs, labels)
            
#             total_loss += loss.item()
    
#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# def predict(dataloader, model):
#     model.eval()
#     predictions = []
#     actuals = []
#     device = next(model.parameters()).device
    
#     with torch.no_grad():
#         for (images, numeric_features), labels in dataloader:
#             # Move data to device
#             images = images.float().to(device)
#             numeric_features = numeric_features.float().to(device)
#             labels = labels.float().to(device).view(-1, 1)
            
#             # Forward pass
#             outputs = model(images, numeric_features)
            
#             # Store predictions and actuals
#             predictions.extend(outputs.cpu().numpy().tolist())
#             actuals.extend(labels.cpu().numpy().tolist())
    
#     # Create DataFrame
#     df = pd.DataFrame()
#     predictions = np.array(predictions)
#     actuals = np.array(actuals)
#     df['Predictions'] = predictions.flatten()
#     df['Actuals'] = actuals.flatten()
    
#     return df

def train_only(model, criterion, optimizer, dataloader, device):
    total_loss = 0.0
    model.train()
    
    for data in dataloader:
        # Unpack data - it comes as ((image, numeric_features), target)
        (images, numeric_features), targets = data
        # Move to device and ensure correct types
        images = images.float().to(device)  # Images are already batched
        numeric_features = numeric_features.float().to(device)
        labels = targets.float().to(device).view(-1, 1)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_only(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data in dataloader:
            # Unpack data
            (images, numeric_features), targets = data
            
            # Move to device and ensure correct types
            images = images.float().to(device)
            numeric_features = numeric_features.float().to(device)
            labels = targets.float().to(device).view(-1, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_only(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data in dataloader:
            # Unpack data
            (images, numeric_features), targets = data
            
            # Move to device and ensure correct types
            images = images.float().to(device)
            numeric_features = numeric_features.float().to(device)
            labels = targets.float().to(device).view(-1, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def predict(dataloader, model):
    model.eval()
    predictions = []
    actuals = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data in dataloader:
            # Unpack data
            (images, numeric_features), targets = data
            
            # Move to device and ensure correct types
            images = images.float().to(device)
            numeric_features = numeric_features.float().to(device)
            labels = targets.float().to(device).view(-1, 1)

            # Forward pass
            outputs = model(images)

            # Store predictions and actuals
            predictions.extend(outputs.cpu().numpy().tolist())
            actuals.extend(labels.cpu().numpy().tolist())

    # Create DataFrame
    df = pd.DataFrame()
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    df['Predictions'] = predictions.flatten()
    df['Actuals'] = actuals.flatten()

    return df

# def train_only(model, criterion, optimizer, dataloader):
#     total_loss = 0.0
#     model.train()
#     device = next(model.parameters()).device
    
#     for data in dataloader:
#         # Unpack data - it comes as ((image, numeric_features), target)
#         (images, numeric_features), targets = data
        
#         # Move to device and ensure correct types
#         images = images.float().to(device)
#         numeric_features = numeric_features.squeeze(1).float().to(device)  # Remove extra dimension
#         targets = targets.float().to(device)
        
#         # Zero gradients
#         optimizer.zero_grad()
        
#         with torch.set_grad_enabled(True):
#             # Forward pass
#             outputs = model(images, numeric_features)
#             loss = criterion(outputs, targets.view(-1, 1))
            
#             # Backward pass
#             loss.backward()
#             optimizer.step()
        
#         total_loss += loss.item()
    
#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# def validate_only(model, criterion, dataloader):
#     model.eval()
#     total_loss = 0.0
#     device = next(model.parameters()).device
    
#     with torch.no_grad():
#         for data in dataloader:
#             # Unpack data
#             (images, numeric_features), targets = data
            
#             # Move to device and ensure correct types
#             images = images.float().to(device)
#             numeric_features = numeric_features.squeeze(1).float().to(device)
#             targets = targets.float().to(device)
            
#             # Forward pass
#             outputs = model(images, numeric_features)
#             loss = criterion(outputs, targets.view(-1, 1))
            
#             total_loss += loss.item()
    
#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# def test_only(model, criterion, dataloader):
#     model.eval()
#     total_loss = 0.0
#     device = next(model.parameters()).device
    
#     with torch.no_grad():
#         for data in dataloader:
#             # Unpack data
#             (images, numeric_features), targets = data
            
#             # Move to device and ensure correct types
#             images = images.float().to(device)
#             numeric_features = numeric_features.squeeze(1).float().to(device)
#             targets = targets.float().to(device)
            
#             # Forward pass
#             outputs = model(images, numeric_features)
#             loss = criterion(outputs, targets.view(-1, 1))
            
#             total_loss += loss.item()
    
#     avg_loss = total_loss / len(dataloader)
#     return avg_loss

# def predict(dataloader, model):
#     model.eval()
#     predictions = []
#     actuals = []
#     device = next(model.parameters()).device
    
#     with torch.no_grad():
#         for data in dataloader:
#             # Unpack data
#             (images, numeric_features), targets = data
            
#             # Move to device and ensure correct types
#             images = images.float().to(device)
#             numeric_features = numeric_features.squeeze(1).float().to(device)
            
#             # Forward pass
#             outputs = model(images, numeric_features)
            
#             # Store predictions and targets
#             predictions.extend(outputs.cpu().numpy())
#             actuals.extend(targets.cpu().numpy())
    
#     # Create DataFrame
#     df = pd.DataFrame({
#         'Predictions': np.array(predictions).flatten(),
#         'Actuals': np.array(actuals).flatten()
#     })
    
#     return df
