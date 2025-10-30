import os
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import glob
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torch.utils.data import SubsetRandomSampler, SequentialSampler

from CNN_LSTM_Thesis.CNN.Datasets.CustomDS_snowfall_target import CNNSnowfallDataset
from CNN_LSTM_Thesis.CNN.Datasets.snowfall_df import preprocess_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_subset(indices, start, end):
    return indices[start : start + end]

def split_and_load_data(batch_size, numeric_data_one):
    # labelled_imgs = '/Volumes/WWWTfilms-1/MODIS_NH_IMGS/COOS_Composites/ndsi_imgs_labels'
    #labelled_imgs = '/Users/briannamarcosano/Documents/SF_Testing_code/CNN_LSTM_Thesis/CNN/Datasets/ndsi_imgs_labels'
    ##### FOR CUSTOM DS vvvvv ######
    file_path = '/Users/briannamarcosano/SnowFall_Forecasting_with_NeuralNetworks/CNN_LSTM_Thesis/CNN/Datasets/COOS_Composites_December03_2024'
    imgs = []
    for filename in os.listdir(file_path):
        if filename.endswith('.tif'):  # Adjust the extensions as needed
            img_path = os.path.join(file_path, filename)
            #img = Image.open(img_path)
            imgs.append(img_path)

    labelled_imgs = sorted(imgs)

    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),  # Resize images to a consistent size
    #     transforms.ToTensor(),  # Convert images to tensors
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    # ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    numeric_data = get_closest_dates(labelled_imgs, numeric_data_one)
    indices = np.arange(len(numeric_data))
    
    # Shuffle indices
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(0.80 * len(indices))
    val_size = int(0.10 * len(indices))
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Print split sizes for verification
    print(f"Total samples: {len(indices)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Scale numeric features
    scaler = StandardScaler()
    
    # Split and scale numeric data
    train_numeric_data = numeric_data.iloc[train_indices]
    val_numeric_data = numeric_data.iloc[val_indices]
    test_numeric_data = numeric_data.iloc[test_indices]
    
    # Extract and store target variable
    train_labels = train_numeric_data['Target_Snowfall'].copy()
    val_labels = val_numeric_data['Target_Snowfall'].copy()
    test_labels = test_numeric_data['Target_Snowfall'].copy()
    
    # Drop target from features
    train_numeric_data = train_numeric_data.drop(columns=['Target_Snowfall'])
    val_numeric_data = val_numeric_data.drop(columns=['Target_Snowfall'])
    test_numeric_data = test_numeric_data.drop(columns=['Target_Snowfall'])
    
    # Fit scaler on training data only and transform all sets
    train_numeric_data_scaled = scaler.fit_transform(train_numeric_data)
    val_numeric_data_scaled = scaler.transform(val_numeric_data)
    test_numeric_data_scaled = scaler.transform(test_numeric_data)
    
    # Convert back to DataFrames with proper indices
    train_numeric_df = pd.DataFrame(
        train_numeric_data_scaled, 
        columns=train_numeric_data.columns,
        index=train_numeric_data.index
    )
    val_numeric_df = pd.DataFrame(
        val_numeric_data_scaled, 
        columns=val_numeric_data.columns,
        index=val_numeric_data.index
    )
    test_numeric_df = pd.DataFrame(
        test_numeric_data_scaled, 
        columns=test_numeric_data.columns,
        index=test_numeric_data.index
    )
    
    # Add back target variable
    train_numeric_df['Target_Snowfall'] = train_labels
    val_numeric_df['Target_Snowfall'] = val_labels
    test_numeric_df['Target_Snowfall'] = test_labels
    
    # Save scaler for later use
    joblib.dump(scaler, 'scaler.pkl')
    
    # Split image paths according to indices
    train_imgs = [labelled_imgs[i] for i in train_indices if i < len(labelled_imgs)]
    val_imgs = [labelled_imgs[i] for i in val_indices if i < len(labelled_imgs)]
    test_imgs = [labelled_imgs[i] for i in test_indices if i < len(labelled_imgs)]
    
    # Verify splits have matching lengths
    print("\nVerifying split sizes:")
    print(f"Train: {len(train_imgs)} images, {len(train_numeric_df)} numeric samples")
    print(f"Val: {len(val_imgs)} images, {len(val_numeric_df)} numeric samples")
    print(f"Test: {len(test_imgs)} images, {len(test_numeric_df)} numeric samples")
    
    # Create datasets
    train_dataset = CNNSnowfallDataset(train_imgs, train_numeric_df, transform=transform)
    val_dataset = CNNSnowfallDataset(val_imgs, val_numeric_df, transform=transform)
    test_dataset = CNNSnowfallDataset(test_imgs, test_numeric_df, transform=transform)
    
    return train_dataset, val_dataset, test_dataset

def create_dataloader_dict(train_dataset, val_dataset, test_dataset, batch_size):
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        ),
        "test": torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        ),
    }

    return dataloaders
    # filtered_samples = [s for s in dataset.samples if not s[0].endswith('_0.jpg')]  # Adjust file extension if needed
    # dataset.samples = filtered_samples
    
    # https://stackoverflow.com/questions/58105073/splitting-a-directory-with-images-into-sub-folders-using-pytorch-or-python
    # train_indices = get_subset(indices, 0, train_count)
    # validation_indices = get_subset(indices, train_count, validation_count)
    # # test_indices = get_subset(indices, train_count + validation_count, len(dataset))
    # test_indices = get_subset(indices, train_count + validation_count, remaining_samples)

def extract_date(img_path):
    date_pattern = re.compile(r'Composite_(\d{8})\.tif')
    filename = os.path.basename(img_path)
    match = date_pattern.search(filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None

def get_sunday_saturday_week(date):
    # Calculate the start of the week (Sunday)
    start_of_week = date - timedelta(days=date.weekday() + 1) if date.weekday() != 6 else date
    # Calculate the end of the week
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week

def get_closest_dates(labelled_imgs, numeric_data, threshold_days=1):
    img_dates = []
    img_dates_frmt = []
    for i, img in enumerate(labelled_imgs):
        t = os.path.basename(labelled_imgs[i]).split('_')[1].replace('.tif', '')
        f = datetime.strptime(t, '%Y%m%d')
        img_dates.append(f)
        img_dates_frmt.append(f.strftime('%Y-%m-%d'))

    max_diff = pd.Timedelta(days=6)

    new_num_df = pd.DataFrame(columns=numeric_data.columns)
    for img_date in img_dates:
        formatted_date_img = img_date.strftime('%Y-%m-%d')
        differences = np.abs(numeric_data.index - img_date)
        if differences.min() <= max_diff:
            closest_date_idx = differences.argmin()
            og_num_date = numeric_data.index.values[closest_date_idx]
            og_num_date = og_num_date.astype('M8[ms]').astype('O')
            og_num_date_str = og_num_date.strftime('%Y%m%d')
            og_num_date_dt = datetime.strptime(og_num_date_str, '%Y%m%d')
            
            numeric_data.index.values[closest_date_idx] = pd.to_datetime(formatted_date_img)
            row_to_add = numeric_data.iloc[[closest_date_idx]].reset_index(drop=False)
            new_num_df = pd.concat([new_num_df, row_to_add])
    
    new_num_df.index = new_num_df['DATE']       
    new_num_df = new_num_df.drop(['DATE'], axis=1)

    # for img_date in img_dates:
    #     formatted_date_img = img_date.strftime('%Y-%m-%d')
    #     differences = np.abs(numeric_data.index - img_date)
    #     closest_date_idx = differences.argmin()
    #     numeric_data.index.values[closest_date_idx] = pd.to_datetime(formatted_date_img)

        # numeric_data.index.values[closest_date_idx] = pd.to_datetime(formatted_date_img)        
    
    return new_num_df



def get_indices(dataset_size, splits):
    """Generate indices for train/val/test splits"""
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # Calculate split points
    train_split = int(np.floor(splits['train'] * dataset_size))
    val_split = int(np.floor(splits['val'] * dataset_size))
    
    # Create splits
    train_indices = indices[:train_split]
    val_indices = indices[train_split:train_split + val_split]
    test_indices = indices[train_split + val_split:]
    
    return train_indices, val_indices, test_indices

def split_image_datasets_only(batch_size=32):  
    data_dir = '/Users/briannamarcosano/SnowFall_Forecasting_with_NeuralNetworks/CNN_LSTM_Thesis/CNN/Datasets/COOS_Composites_December03_2024'

    splits = {
        'train': 0.8,  # 70% training
        'val': 0.1,   # 15% validation
        'test': 0.1   # 15% testing
    }
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization
    ])
    dataset = ImageDataset(data_dir, transform=transform)
    dataset_size = len(dataset)
    
    # Get indices for splits
    train_indices, val_indices, test_indices = get_indices(dataset_size, splits)
    
    # Create data loaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            dataset,
            sampler=SequentialSampler(train_indices),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        ),
        "val": torch.utils.data.DataLoader(
            dataset,
            sampler=SequentialSampler(val_indices),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        ),
        "test": torch.utils.data.DataLoader(
            dataset,
            sampler=SequentialSampler(test_indices),
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )
    }
    
    # Print split information
    print(f"\nDataset split info:")
    print(f"Total images: {dataset_size}")
    print(f"Training: {len(train_indices)} images")
    print(f"Validation: {len(val_indices)} images")
    print(f"Testing: {len(test_indices)} images")
    
    return dataloaders

class ImageDataset(Dataset):
    """Custom Dataset for loading images without class folders"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Get all tif files in directory
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No .tif files found in {data_dir}")
            
        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0 
    