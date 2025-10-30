import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from CNN_LSTM_Thesis.CNN.Datasets.snowfall_df import preprocess_data
from CNN_LSTM_Thesis.CNN.Datasets.build_split_imgandnum_ds import split_and_load_data
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

from CNN_LSTM_Thesis.LSTM.Models.LSTM import ParallelCNNLSTM, SequentialCNNLSTMTrainer, SequentialCNNLSTM
from CNN_LSTM_Thesis.CNN.Datasets.CustomDS_snowfall_target import OptimizedCNNSnowfallDataset, create_optimized_dataloaders
from CNN_LSTM_Thesis.CNN.Datasets.snowfall_df import preprocess_data
from CNN_LSTM_Thesis.LSTM.Scripts.evaluate import plot_training_history, plot_predictions, plot_losses2, plot_with_savgol_smoothing,\
    plot_predictions_all_sets
from CNN_LSTM_Thesis.CNN.Datasets.build_split_imgandnum_ds import split_and_load_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.backends.mps.enable_noncontiguous_format = True  

def get_feature_count(data):
    """Determine the actual number of numeric features being used"""
    # Get only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # Exclude the target column
    feature_cols = [col for col in numeric_cols if col != 'Target_Snowfall']
    return len(feature_cols)

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_metrics(metrics_dict, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame([metrics_dict])
    df.to_csv(os.path.join(save_dir, f'metrics_{timestamp}.csv'), index=False)


def main():
    setup_logging()
    
    # Configuration
    config = {
        'batch_size': 8, # 18 for other
        'learning_rate': .0001,
        'num_epochs': 50,
        'hidden_size': 128,
        'num_layers': 2,
        'sequence_length': 7,
        'dropout': 0.4,
        'save_dir': 'models',
        'model_name': 'cnn_lstm_ndsi',
        'cache_dir': 'cached_data',
        'find_lr': True,  # Set to True to run the LR finder
        'max_lr': 0.002
    }
    
    # Create necessary directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['cache_dir'], exist_ok=True)
    train_cache_dir = os.path.join(config['cache_dir'], 'train')
    val_cache_dir = os.path.join(config['cache_dir'], 'val')
    test_cache_dir = os.path.join(config['cache_dir'], 'test')

    # try:
    logging.info("Preprocessing data...")
    numeric_data = preprocess_data('1997-01-01')

    numeric_columns = numeric_data.select_dtypes(include=[np.number]).columns
    numeric_data[numeric_columns] = numeric_data[numeric_columns].astype(np.float32)

    logging.info("Splitting data...")
    train_imgs, val_imgs, test_imgs, train_data, val_data, test_data, feature_scaler, target_scaler = split_and_load_data(
        batch_size=config['batch_size'],numeric_data_one=numeric_data)
            
    # transform = get_ndsi_transforms()
    
    train_dataset = OptimizedCNNSnowfallDataset(image_paths=train_imgs, 
                                        numeric_features=train_data, 
                                        seq_len=config['sequence_length'],
                                        cache_dir=train_cache_dir,
                                        name='Train set')
    
    val_dataset = OptimizedCNNSnowfallDataset(image_paths=val_imgs, 
                                    numeric_features=val_data, 
                                    seq_len=config['sequence_length'], 
                                    cache_dir=val_cache_dir,
                                    name='Validation set')
    
    test_dataset = OptimizedCNNSnowfallDataset(image_paths=test_imgs, 
                                        numeric_features=test_data, 
                                        seq_len=config['sequence_length'],
                                        cache_dir=test_cache_dir,
                                        name='Test set')
    # print("\nNumeric Features Summary:")
    # print(train_dataset.numeric_features.describe())
    
    logging.info("Creating dataloaders...")
    dataloaders = create_optimized_dataloaders(train_dataset=train_dataset, 
                                                val_dataset=val_dataset, 
                                                test_dataset=test_dataset,
                                                batch_size=config['batch_size'], 
                                                num_workers=0)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # input_size = get_feature_count(numeric_data)
    input_size=15
    logging.info(f"Number of numeric features (excluding target): {input_size}")
   
    model = SequentialCNNLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=0.01) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=4,verbose=True,cooldown=2)   
    trainer = SequentialCNNLSTMTrainer(model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler)

    # Training
    logging.info("Starting training...")
    save_path = os.path.join(config['save_dir'], f"{config['model_name']}_best.pt")
    train_losses, val_losses, train_losses_avg, val_losses_avg = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=config['num_epochs'],
        save_path=save_path)
    
    # Evaluation
    logging.info("Making predictions...")
    val_predictions = trainer.predict_with_dates(dataloaders['val'], val_dataset)
    test_predictions = trainer.predict_with_dates(dataloaders['test'], test_dataset)
    train_predictions = trainer.predict_with_dates(dataloaders['train'], train_dataset)

    # val_predictions['Predictions'] = target_scaler.inverse_transform(val_predictions[['Predictions']])
    # val_predictions['Actuals'] = target_scaler.inverse_transform(val_predictions[['Actuals']])

    # test_predictions['Predictions'] = target_scaler.inverse_transform(test_predictions[['Predictions']])
    # test_predictions['Actuals'] = target_scaler.inverse_transform(test_predictions[['Actuals']])
    # print(val_predictions)

    # print(test_predictions.head(15))

    # Calculate metrics
    metrics = {
        'val_mae': mean_absolute_error(val_predictions['Actuals'], val_predictions['Predictions']),
        'val_mse': mean_squared_error(val_predictions['Actuals'], val_predictions['Predictions']),
        'val_rmse': np.sqrt(mean_squared_error(val_predictions['Actuals'], val_predictions['Predictions'])),
        'val_r2': r2_score(val_predictions['Actuals'], val_predictions['Predictions']),
        'test_mae': mean_absolute_error(test_predictions['Actuals'], test_predictions['Predictions']),
        'test_mse': mean_squared_error(test_predictions['Actuals'], test_predictions['Predictions']),
        'test_rmse': np.sqrt(mean_squared_error(test_predictions['Actuals'], test_predictions['Predictions'])),
        'test_r2': r2_score(test_predictions['Actuals'], test_predictions['Predictions'])
    }
    
    logging.info("\nFinal Metrics:")
    for name, value in metrics.items():
        logging.info(f"{name}: {value:.4f}")
    
    save_metrics(metrics)
    os.makedirs('results', exist_ok=True)
    val_predictions.to_csv('results/validation_predictions.csv')
    test_predictions.to_csv('results/test_predictions.csv')
    
    # Save training history
    pd.DataFrame({
        'epoch': range(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
        }).to_csv('results/training_history.csv', index=False)
    
    logging.info("Training completed successfully!")
        
    # except Exception as e:
    #     logging.error(f"Error during training: {str(e)}", exc_info=True)
    #     raise

    # plot_training_history(train_losses, val_losses)
    plot_with_savgol_smoothing(train_losses, val_losses)

    # plot_predictions(
    #     test_predictions['Actuals'].values,
    #     test_predictions['Predictions'].values,
    #     test_predictions.index 
    # )

    plot_predictions_all_sets(
        train_actual=train_predictions['Actuals'][-30:].values,
        train_pred=train_predictions['Predictions'][-30:].values,
        train_dates=train_predictions.index[-30:],
        
        val_actual=val_predictions['Actuals'].values,
        val_pred=val_predictions['Predictions'].values,
        val_dates=val_predictions.index,
        
        test_actual=test_predictions['Actuals'].values,
        test_pred=test_predictions['Predictions'].values,
        test_dates=test_predictions.index
    )

if __name__ == "__main__":
    main()





# batch_size = 16 #changing batch size doesn't seem to do much?
# lr = .0001  #.0001 best so far  #val loss plateus more with .0001?
# num_epochs = 15 #40 for all months

# numeric_data = preprocess_data('2000-01-01')
# train_dataset, val_dataset, test_dataset = split_and_load_data(batch_size=batch_size, numeric_data_one=numeric_data)

# params = {'batch_size': batch_size,
#           'shuffle': False,
#           'drop_last': True }

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# val_loader = torch.utils.data.DataLoader(val_dataset, **params)
# test_loader = torch.utils.data.DataLoader(test_dataset, **params)

# model = CNNLSTM()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# train_losses, test_losses, val_losses = np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs)
# best_val_loss = float('inf')  # Set to positive infinity initially

# for epoch in range(num_epochs):  #201 best so far with .01 and batch 26
#     train_loss = train_only(
#         model = model,
#         criterion = criterion,
#         optimizer = optimizer,
#         dataloader = train_loader
#     )
#     # test_loss = test_model(test_loader, model, criterion)
#     val_loss = validate_only(model, criterion, val_loader)
#     scheduler.step()

#     train_losses[epoch] = train_loss
#     val_losses[epoch] = val_loss

#     if epoch % 3 == 0:
#         print(f"Epoch {epoch} - Train Loss: {train_loss:.3}, " 
#              f"Val Loss: {val_loss:.3} \n---------")  #f"Test Loss: {test_loss:.3},

# pred_df = predict(val_loader, model)
# test_pred_df = predict(test_loader, model)
    
# t = val_dataset.get_og_data()['target']
# t = t.iloc[:len(pred_df)]
# pred_df['Snow_og'] = t.values
# og_ds = val_dataset.get_og_data().iloc[:len(pred_df)]
# pred_df.index = og_ds.index

# t2 = test_dataset.get_og_data()['target']
# t2 = t2.iloc[:len(test_pred_df)]
# test_pred_df['Snow_og'] = t2.values
# og_ds = test_dataset.get_og_data().iloc[:len(test_pred_df)]
# test_pred_df.index = og_ds.index

# val_rmse = mean_squared_error(pred_df.Actuals, pred_df.Predictions) ** 0.5
# print('val mae :', mean_absolute_error(pred_df.Actuals, pred_df.Predictions))
# print('val rmse :', mean_squared_error(pred_df.Actuals, pred_df.Predictions) ** 0.5)
# print('val r2 :', r2_score(pred_df.Actuals, pred_df.Predictions))
# print(pred_df)

# test_rmse = mean_squared_error(test_pred_df.Actuals, test_pred_df.Predictions) ** 0.5
# print('test mae :', mean_absolute_error(test_pred_df.Actuals, test_pred_df.Predictions))
# print('test rmse :', mean_squared_error(test_pred_df.Actuals, test_pred_df.Predictions) ** 0.5)
# print('test r2 :', r2_score(test_pred_df.Actuals, test_pred_df.Predictions))
# print(test_pred_df)