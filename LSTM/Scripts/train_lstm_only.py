import gc
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import time
from tempfile import TemporaryDirectory
import shutil
import pandas as pd


def train(model, criterion, optimizer,train_loader):
    model.train()
    total_loss = 0
    for data, labels in train_loader:  
        optimizer.zero_grad()
        # labels = labels.view(-1, 1)
        # with torch.set_grad_enabled(True):
            #data = data.view(data.size(0), -1)
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def validate(val_loader, model, criterion, scheduler=None):
    model.eval()
    val_losses = [] 

    total_loss = 0
    # with torch.no_grad():
    for data, label in val_loader:
        outputs = model(data)
        loss = criterion(outputs, label)
        total_loss += loss.item() * data.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    val_losses.append(avg_loss)

    if scheduler is not None:
        scheduler.step(avg_loss)

    return avg_loss

def test_model(test_loader, model, loss_funct):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:  
            #data = data.view(data.size(0), -1)
            outputs = model(data)
            # labels = labels.view(-1, 1)
            loss = loss_funct(outputs, labels)
            total_loss += loss.item() * data.size(0)
    
    # avg_loss = 100 * total_loss / num_batches
    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

def predict(data_loader, model):
    model.eval()
    output = []
    y_list = []

    with torch.no_grad():
        # for X, y in data_loader:
        for i, (X, y) in enumerate(data_loader):
            y_star = model(X)
            output.extend(y_star.tolist())
            y_list.extend(y.tolist())
              
    return np.array(output), np.array(y_list)


def make_predictions_from_dataloader(unshuffled_dataloader, model):
    model.eval()
    predictions, actuals = [], []
    for x, y in unshuffled_dataloader:
        with torch.no_grad():
            p = model(x)
            predictions.append(p)
            actuals.append(y.squeeze())
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    return predictions.squeeze(), actuals