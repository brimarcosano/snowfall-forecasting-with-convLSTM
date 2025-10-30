from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

from scipy.signal import savgol_filter

def plot_with_savgol_smoothing(train_losses, val_losses):
    """Plot with Savitzky-Golay filter for advanced smoothing."""
    plt.figure(figsize=(10, 6))
    
    # Original data
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, 'lightblue', alpha=0.3, label='Raw Training Loss')
    plt.plot(epochs, val_losses, 'lightcoral', alpha=0.3, label='Raw Validation Loss')
    
    # Apply Savitzky-Golay filter if we have enough data points
    if len(train_losses) > 1:
        window_size = min(len(train_losses) - (len(train_losses) % 2 - 1), 11)  # Must be odd
        polyorder = min(3, window_size - 1)
        
        smooth_train = savgol_filter(train_losses, window_size, 10)
        smooth_val = savgol_filter(val_losses, window_size, polyorder)
        
        plt.plot(epochs, smooth_train, 'b', linewidth=2, label='Smoothed Training Loss')
        plt.plot(epochs, smooth_val, 'r', linewidth=2, label='Smoothed Validation Loss')
    
    plt.title('Training and Validation Loss (Savitzky-Golay Filter)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/training_history_savgol_smoothed.png', dpi=300)
    plt.show()

def plot_predictions(actual, predicted, dates=None):
    plt.figure(figsize=(12, 6))
    if dates is not None:
        plt.plot(dates, actual, label='Actual', marker='o')
        plt.plot(dates, predicted, label='Predicted', marker='o')
        plt.xticks(rotation=45)
    else:
        plt.plot(actual, label='Actual', marker='o')
        plt.plot(predicted, label='Predicted', marker='o')
    
    plt.title('Actual vs Predicted Snowfall')
    plt.xlabel('Date')
    plt.ylabel('Snowfall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions_all_sets(train_actual=None, train_pred=None, val_actual=None, val_pred=None,
                            test_actual=None, test_pred=None, train_dates=None, val_dates=None, test_dates=None):
    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for better distinction
    actual_color = 'black'  # Bl
    pred_color = '#ff7f0e'    # Orange
    
    # Plot training data (subset)
    if train_actual is not None and len(train_actual) > 0:
        if train_dates is None:
            train_dates = range(len(train_actual))
        
        ax.plot(train_dates, train_actual, color=actual_color, alpha=0.5, 
                label='Training (Actual)', markersize=2)
        ax.plot(train_dates, train_pred, color=pred_color, alpha=0.5, 
                label='Training (Predicted)', markersize=2)
        
        # Add training label
        mid_idx = len(train_dates) // 2
        ax.text(train_dates[mid_idx], ax.get_ylim()[1] * 0.9, "TRAINING", 
                ha='center', va='center', alpha=0.2, fontsize=16, color='gray')
    
    # Plot validation data
    if val_actual is not None and len(val_actual) > 0:
        if val_dates is None:
            val_dates = range(len(train_actual), len(train_actual) + len(val_actual))
        
        ax.plot(val_dates, val_actual, color=actual_color, alpha=0.7, 
                label='Validation (Actual)', markersize=2)
        ax.plot(val_dates, val_pred, color='green', alpha=0.7, 
                label='Validation (Predicted)', markersize=2)
                
        # Add validation label
        mid_idx = len(val_dates) // 2
        ax.text(val_dates[mid_idx], ax.get_ylim()[1] * 0.9, "VALIDATION",
                ha='center', va='center', alpha=0.2, fontsize=16, color='gray')
                
        # Add separator line
        if train_actual is not None:
            ax.axvline(val_dates[0], color='gray', linestyle='--', alpha=0.5)
    
    # Plot test data with more emphasis
    if test_actual is not None and len(test_actual) > 0:
        if test_dates is None:
            test_dates = range(len(train_actual) + len(val_actual), 
                              len(train_actual) + len(val_actual) + len(test_actual))
        
        ax.plot(test_dates, test_actual, color=actual_color, 
                label='Test (Actual)', markersize=2)
        ax.plot(test_dates, test_pred, color='green', 
                label='Test (Predicted)', markersize=2)
                
        # Add test label
        mid_idx = len(test_dates) // 2
        ax.text(test_dates[mid_idx], ax.get_ylim()[1] * 0.9, "TEST",
                ha='center', va='center', alpha=0.2, fontsize=16, color='gray')
                
        # Add separator line
        if val_actual is not None:
            ax.axvline(test_dates[0], color='gray', linestyle='--', alpha=0.5)
    
    # Add labels and title
    ax.set_title('Snowfall Prediction Performance', fontsize=14, pad=20)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Snowfall (inches)', fontsize=12)
    
    # Format x-axis if using dates
    if isinstance(train_dates, pd.DatetimeIndex) or isinstance(val_dates, pd.DatetimeIndex) or isinstance(test_dates, pd.DatetimeIndex):
        fig.autofmt_xdate()
        ax.set_xlabel('Date', fontsize=12)
    
    # Add legend with better placement
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/snowfall_predictions.png', dpi=300, bbox_inches='tight')
    
    # Show plot
    plt.show()

def predict(dataloader, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for (images, numeric_features), labels in dataloader:
            # Move data to device
            images = images.float().to(device)
            numeric_features = numeric_features.float().to(device)
            labels = labels.float().to(device)
            
            # Get predictions
            outputs = model(images, numeric_features).squeeze(-1)
            
            # Store predictions and actuals
            predictions.extend(outputs.cpu().numpy().tolist())
            actuals.extend(labels.cpu().numpy().tolist())
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'Predictions': predictions,
        'Actuals': actuals
    })
    
    return df

def predict_with_dates(dataloader, model, dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for (images, numeric_features), labels in dataloader:
            images = images.float().to(device)
            numeric_features = numeric_features.float().to(device)
            labels = labels.float().to(device)
            
            outputs = model(images, numeric_features).squeeze(-1)
            
            predictions.extend(outputs.cpu().numpy().tolist())
            actuals.extend(labels.cpu().numpy().tolist())
    
    # Get original data with dates
    og_data = dataset.get_og_data()
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'Predictions': predictions,
        'Actuals': actuals,
        'Snow_og': og_data['target'].values[:len(predictions)]
    }, index=og_data.index[:len(predictions)])
    
    return df

def evaluate_predictions(predictions_df):
    metrics = {
        'mae': mean_absolute_error(predictions_df.Actuals, predictions_df.Predictions),
        'rmse': mean_squared_error(predictions_df.Actuals, predictions_df.Predictions) ** 0.5,
        'r2': r2_score(predictions_df.Actuals, predictions_df.Predictions)
    }
    
    return metrics

def calculate_metrics(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)

    return mae, mse, rmse, r2


def plot_losses(model):
    plt.plot(model.train_losses, label="Training loss")
    plt.plot(model.val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.show()
    plt.close()


def plot_losses2(train_losses, val_losses):
    plt.plot(train_losses, label='train loss')
    # plt.plot(test_losses, label='test loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('epoch no')
    plt.ylabel('loss')
    plt.legend()
    plt.show()