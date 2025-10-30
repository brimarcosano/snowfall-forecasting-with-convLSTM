import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.nn.utils.rnn import pad_sequence

class SequentialCNNLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.early_stopping = EarlyStopping(patience=4, min_delta=0.001)
        
        # CNN Component - Remove the final FC layer
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the final FC layer with Identity to get the raw features
        self.cnn_features = self.cnn.fc.in_features  # Store the feature size (typically 512 for ResNet18)
        self.cnn.fc = nn.Identity()
        
        # Dimension reduction for CNN features (from cnn_features to 64)
        self.cnn_reduction = nn.Sequential(
            nn.Linear(self.cnn_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM Component - Updated input size to include the reduced CNN features
        self.lstm = nn.LSTM(
            input_size=input_size + 64,  # Original features + reduced CNN features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Final prediction layer
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    
    def forward(self, images, numeric_features):
        images = images.to(self.device)
        numeric_features = numeric_features.to(self.device)
        batch_size = images.shape[0]
        seq_length = numeric_features.shape[1]
        cnn_features_seq = []

        for t in range(seq_length):
            # Extract images at time t for all batches
            current_images = images[:, t]  # [batch_size, channels, height, width]
            cnn_out = self.cnn(current_images)  # [batch_size, cnn_features]
            reduced_cnn = self.cnn_reduction(cnn_out)  # [batch_size, 64]
            cnn_features_seq.append(reduced_cnn)
        
        # Stack CNN features along sequence dimension
        cnn_features = torch.stack(cnn_features_seq, dim=1)
        combined_input = torch.cat([cnn_features, numeric_features], dim=2)
        
        # Initialize LSTM hidden states
        h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        
        lstm_out, _ = self.lstm(combined_input, (h0, c0))  
        # features from last timestep
        final_features = lstm_out[:, -1, :]
        output = self.fusion(final_features)
        
        return output
    
class ParallelCNNLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.early_stopping = EarlyStopping(patience=5, min_delta=0.004)
        
        # final FC layer removed
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn_features = self.cnn.fc.in_features  # Store the feature size
        self.cnn.fc = nn.Identity()

        self.cnn_reduction = nn.Sequential(
            nn.Linear(self.cnn_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # setting combined feature size
        lstm_output_size = hidden_size * 2  # Bidirectional LSTM
        combined_size = 64 + lstm_output_size
        
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(combined_size),  # Normalize combined features
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU()
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                        
    def forward(self, images, numeric_features):
        images = images.to(self.device)
        numeric_features = numeric_features.to(self.device)
        batch_size = images.shape[0]
        
        seq_length = numeric_features.shape[1]

        cnn_features_seq = []
        for t in range(seq_length):
            # Extract images at time t for all batches
            current_images = images[:, t]  # [batch_size, channels, height, width]
            cnn_out = self.cnn(current_images)  # [batch_size, cnn_features]
            reduced_cnn = self.cnn_reduction(cnn_out)  # [batch_size, 64]
            cnn_features_seq.append(reduced_cnn)
        
        # Stack CNN features along sequence dimension
        cnn_features = torch.stack(cnn_features_seq, dim=1)
        cnn_final = cnn_features[:, -1, :]
        
        # Process LSTM
        h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        lstm_out, _ = self.lstm(numeric_features, (h0, c0))
        lstm_features = lstm_out[:, -1, :]  # features from last timestep

        combined_features = torch.cat([cnn_final, lstm_features], dim=1)
        output = self.fusion(combined_features)
        
        return output


class SequentialCNNLSTMTrainer:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_loss = float('inf')
        
    def train_model(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for idx, ((images, numeric_features), targets) in enumerate(train_loader):
            images = images.float().to(self.device)
            numeric_features = numeric_features.float().to(self.device)
            targets = targets.float().to(self.device).view(-1, 1)
            
            # Simple forward and backward
            self.optimizer.zero_grad()
            outputs = self.model(images, numeric_features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
                
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        return avg_loss
    
    def validate_model(self, val_loader):
        self.model.eval()
        total_loss = 0
        self.val_losses = []  # Track validation losses for smoothing

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

        with torch.no_grad():
            for (images, numeric_features), targets in val_loader:
                images = images.float().to(self.device)
                numeric_features = numeric_features.float().to(self.device)
                targets = targets.float().to(self.device).view(-1, 1)
                
                outputs = self.model(images, numeric_features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            smoothed_loss = sum(self.val_losses[-5:]) / min(len(self.val_losses), 5)
            self.scheduler.step(smoothed_loss)
        
        smoothed_loss = sum(self.val_losses[-5:]) / min(len(self.val_losses), 5)
            
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs, save_path=None):
        '''
            uses train_epoch and validate functions in epoch loop
        '''
        train_losses = []
        val_losses = []
        train_losses_avg = []
        val_losses_avg = []
        best_model = None
        best_val_loss = float('inf')
        
        # Early stopping parameters
        patience = 4
        min_delta = 0.001  # Increased minimum improvement threshold
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_model(train_loader)
            val_loss = self.validate_model(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Check if this is the best model
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                no_improve_count = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    best_model = self.model.state_dict()
            else:
                no_improve_count += 1
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                train_losses_avg.append(train_loss)
                val_losses_avg.append(val_loss)
                
            # Stop if no improvement for several epochs
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
                break
        
        if best_model and save_path:
            self.model.load_state_dict(best_model)
        
        return train_losses, val_losses, train_losses_avg, val_losses_avg
    
    def predict(self, loader):
        self.model.eval()
        predictions = []
        actuals = []
        dates = []
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:  # If targets are available
                    (images, numeric_features), targets = batch
                    actuals.extend(targets.cpu().numpy())
                else:
                    images, numeric_features = batch
                    
                images = images.float().to(self.device)
                numeric_features = numeric_features.float().to(self.device)
                
                outputs = self.model(images, numeric_features)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions).squeeze(), np.array(actuals).squeeze() if actuals else None

    def predict_with_dates(self, loader, dataset):
        predictions, actuals = self.predict(loader)
        
        dates = dataset.get_dates() if hasattr(dataset, 'get_dates') else None
        
        # Create DataFrame
        df = pd.DataFrame({'Predictions': predictions})
        if actuals is not None:
            df['Actuals'] = actuals
        if dates is not None:
            df.index = dates[:len(predictions)]
            
        return df

    # Optional helper function to get predictions as DataFrame
    def predict_to_df(self, loader, include_dates=True):        
        results = self.predict(loader)
        
        df = pd.DataFrame({'Predictions': results['predictions']})
        if 'actuals' in results:
            df['Actuals'] = results['actuals']
        
        return df


class ParallelCNNLSTMTrainer:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_val_loss = float('inf')
        
    def train_model(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for idx, ((images, numeric_features), targets) in enumerate(train_loader):
            images = images.float().to(self.device)
            numeric_features = numeric_features.float().to(self.device)
            targets = targets.float().to(self.device).view(-1, 1)
            
            # Simple forward and backward
            self.optimizer.zero_grad()
            outputs = self.model(images, numeric_features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
                
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        return avg_loss
    
    def validate_model(self, val_loader):
        self.model.eval()
        total_loss = 0
        self.val_losses = []  # Track validation losses for smoothing

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

        with torch.no_grad():
            for (images, numeric_features), targets in val_loader:
                images = images.float().to(self.device)
                numeric_features = numeric_features.float().to(self.device)
                targets = targets.float().to(self.device).view(-1, 1)
                
                outputs = self.model(images, numeric_features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            smoothed_loss = sum(self.val_losses[-5:]) / min(len(self.val_losses), 5)
            self.scheduler.step(smoothed_loss)
        
        smoothed_loss = sum(self.val_losses[-5:]) / min(len(self.val_losses), 5)

        # if self.scheduler is not None:
        #     smoothed_loss = sum(self.val_losses[-5:]) / min(len(self.val_losses), 5)
        #     self.scheduler.step(smoothed_loss)
            
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs, save_path=None):
        '''
            uses train_epoch and validate functions in epoch loop
        '''
        train_losses = []
        val_losses = []
        train_losses_avg = []
        val_losses_avg = []
        best_model = None
        best_val_loss = float('inf')
        
        # Early stopping parameters
        patience = 5
        min_delta = 0.004 
        no_improve_count = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_model(train_loader)
            val_loss = self.validate_model(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                train_losses_avg.append(train_loss)
                val_losses_avg.append(val_loss)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                #print(f"New best model at epoch {epoch} with val_loss = {val_loss:.4f}")
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
                break
        
        # loading best model at the end
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with val_loss = {best_val_loss:.4f}")
            
        return train_losses, val_losses, train_losses_avg, val_losses_avg
    
    def predict(self, loader):
        self.model.eval()
        predictions = []
        actuals = []
        dates = []
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:  # If targets are available
                    (images, numeric_features), targets = batch
                    actuals.extend(targets.cpu().numpy())
                else:
                    images, numeric_features = batch
                    
                images = images.float().to(self.device)
                numeric_features = numeric_features.float().to(self.device)
                
                outputs = self.model(images, numeric_features)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions).squeeze(), np.array(actuals).squeeze() if actuals else None

    def predict_with_dates(self, loader, dataset):
        predictions, actuals = self.predict(loader)
        
        dates = dataset.get_dates() if hasattr(dataset, 'get_dates') else None
        
        # Create DataFrame
        df = pd.DataFrame({'Predictions': predictions})
        if actuals is not None:
            df['Actuals'] = actuals
        if dates is not None:
            df.index = dates[:len(predictions)]
            
        return df

    # Optional helper function to get predictions as DataFrame
    def predict_to_df(self, loader, include_dates=True):        
        results = self.predict(loader)
        
        df = pd.DataFrame({'Predictions': results['predictions']})
        if 'actuals' in results:
            df['Actuals'] = results['actuals']
        
        return df


class EarlyStopping:
   def __init__(self, patience, min_delta, restore_best_weights=True):
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.best_loss = None
       self.early_stop = False
       self.restore_best_weights = restore_best_weights

   def __call__(self, val_loss):
       if self.best_loss is None:
           self.best_loss = val_loss
       elif val_loss > self.best_loss - self.min_delta:
           self.counter += 1
           if self.counter >= self.patience:
               self.early_stop = True
       else:
           self.best_loss = val_loss
           self.counter = 0

   def restore_weights(self, model):
        """Restore the best weights to the model"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored best model with validation loss: {self.best_loss:.4f}")


class LSTM(nn.Module):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # https://medium.com/analytics-vidhya/time-series-forecasting-lstm-f45fbc7796e1
    def __init__(self, layer_dim=2, inputs=11, n_hidden=60, bidirectional=True): 
        super(LSTM, self).__init__()
        
        self.input_size = inputs
        self.hidden_layers = n_hidden
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional
        self.input_bn = nn.BatchNorm1d(self.input_size)
        self.hidden_bn = nn.BatchNorm1d(self.hidden_layers)

        self.lstm1 = nn.LSTM(self.input_size, self.hidden_layers, self.layer_dim, 
                             bidirectional=self.bidirectional, batch_first=True, 
                             dropout=0.2)

        lstm_output_size = self.hidden_layers * 2 if bidirectional else self.hidden_layers

        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=lstm_output_size, out_features=64)
        self.output_linear = nn.Linear(64, 1)

    def forward(self, data):
        batch_size = data.size()[0]
        seq_len = data.size()[1]

        self.h0 = torch.zeros(self.layer_dim * (2 if self.bidirectional else 1), batch_size, self.hidden_layers).to(data.device)
        self.c0 = torch.zeros(self.layer_dim * (2 if self.bidirectional else 1), batch_size, self.hidden_layers).to(data.device)
    
        lstm_out, (hidden_cell, _) = self.lstm1(data, (self.h0, self.c0))

        x = self.linear(lstm_out[:, -1, :] ) # last time step
        x = self.dropout(x)
        out = self.output_linear(x)

        return out
    




class CNNLSTM(nn.Module):
    def __init__(self, layer_dim=1, inputs=11, n_hidden=100, seq_len=5): 
        super(CNNLSTM, self).__init__()
        self.input_size = inputs
        self.hidden_layers = n_hidden
        self.layer_dim = layer_dim
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_layers, 1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        #self.cnn.fc = nn.Linear(512, 256) 
        self.cnn.fc = nn.Identity()  # remove the final fully connected layer

        self.fc_numeric = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=16), 
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )

        self.fc_final = nn.Sequential(
        nn.Linear(512+self.hidden_layers, 128),  # Match the concatenated feature size
        nn.ReLU(),
        nn.Linear(128, 1),
    )
        self.output_linear = nn.Linear(self.input_size, 1)
        self.relu = nn.LeakyReLU()

        # Initialize hidden and cell states
        self.h0 = torch.zeros(self.layer_dim, 1, self.hidden_layers)
        self.c0 = torch.zeros(self.layer_dim, 1, self.hidden_layers)

    def forward(self, image, dataframe):
        image = image.unsqueeze(1)
        batch_size, idk, seq_len, channels, height, width = image.size()
        image = image.view(batch_size * seq_len, channels, height, width)
        image_features = self.cnn(image)
        image_features = image_features.view(batch_size, seq_len, -1)

        # dataframe = dataframe.view(batch_size, seq_len, -1)
        dataframe = dataframe.view(batch_size, seq_len, self.input_size)
        # h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_layers)
        # c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_layers)
        h0 = self.h0.repeat(1, batch_size, 1)
        c0 = self.c0.repeat(1, batch_size, 1)
        lstm_out, (hidden_cell, _) = self.lstm1(dataframe, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        combined_features = torch.cat((image_features, lstm_out), dim=2)  # Shape: (batch_size, seq_len, 512 + hidden_layers)
        
        out = self.fc_final(combined_features)  # Shape: (batch_size, 1)
        out = out.view(batch_size, seq_len, -1)

        return out

def custom_collate(batch):
    sequences, labels = zip(*batch)

    # pdding sequences to the max length in the batch
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [torch.cat((torch.zeros(max_len - len(seq), seq.size()[1]), seq), dim=0) for seq in sequences]

    # stacking padded sequences and labels
    stacked_sequences = torch.stack(padded_sequences, dim=0)
    stacked_labels = torch.stack(labels, dim=0)

    return stacked_sequences, stacked_labels