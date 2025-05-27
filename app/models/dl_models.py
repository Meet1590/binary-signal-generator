import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from .base_model import BaseSignalModel
import joblib
from tqdm import tqdm

class LSTMNet(nn.Module):
    """
    LSTM Neural Network architecture
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=3):
        super(LSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step output
        last_time_step = lstm_out[:, -1, :]
        
        # Fully connected layers with dropout
        x = self.fc1(last_time_step)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class GRUNet(nn.Module):
    """
    GRU Neural Network architecture
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=3):
        super(GRUNet, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Apply attention
        attention_weights = self.attention(gru_out)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        
        # Fully connected layers with dropout
        x = self.fc1(context_vector)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNLSTMNet(nn.Module):
    """
    CNN-LSTM hybrid architecture
    """
    def __init__(self, input_size, seq_length, hidden_size=64, num_layers=2, dropout=0.2, output_size=3):
        super(CNNLSTMNet, self).__init__()
        
        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)
        
        # Calculate CNN output size
        cnn_output_size = 64 * (seq_length // 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        # Reshape for CNN [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Reshape back for LSTM [batch, features, seq_len] -> [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get the last time step output
        last_time_step = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_time_step)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class LSTMModel(BaseSignalModel):
    """
    LSTM model implementation for signal prediction
    """
    def __init__(self, input_size, seq_length, hidden_size=64, num_layers=2, dropout=0.3, lr=0.001, epochs=50, batch_size=64, output_size=3):
        super().__init__(name="LSTM")
        self.input_size = input_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.model = LSTMNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        """
        Train the LSTM model
        
        Args:
            X: Features array with shape [n_samples, seq_length, n_features]
            y: Target array
        """
        # Scale features
        n_samples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(n_samples * seq_length, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_length, n_features)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for inputs, targets in progress_bar:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
            
            # Print epoch statistics
            epoch_loss = running_loss / total
            epoch_acc = 100 * correct / total
            print(f'Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions with the LSTM model
        
        Args:
            X: Features array with shape [n_samples, seq_length, n_features]
        """
        self.model.eval()
        
        # Scale features
        n_samples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(n_samples * seq_length, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_length, n_features)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """
        Predict class probabilities with the LSTM model
        
        Args:
            X: Features array with shape [n_samples, seq_length, n_features]
        """
        self.model.eval()
        
        # Scale features
        n_samples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(n_samples * seq_length, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_length, n_features)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Save the PyTorch model and other attributes
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'seq_length': self.seq_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'output_size': self.output_size,
            'name': self.name
        }
        
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Load the saved state
        state = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create a new instance of the class
        model = cls(
            input_size=state['input_size'],
            seq_length=state['seq_length'],
            hidden_size=state['hidden_size'],
            num_layers=state['num_layers'],
            dropout=state['dropout'],
            output_size=state['output_size']
        )
        
        # Load the model state dict
        model.model.load_state_dict(state['model_state_dict'])
        model.optimizer.load_state_dict(state['optimizer_state_dict'])
        model.scaler = state['scaler']
        model.name = state['name']
        model.is_fitted = True
        
        return model

class GRUModel(LSTMModel):
    """
    GRU model implementation for signal prediction
    """
    def __init__(self, input_size, seq_length, hidden_size=64, num_layers=2, dropout=0.3, lr=0.001, epochs=50, batch_size=64, output_size=3):
        super().__init__(input_size, seq_length, hidden_size, num_layers, dropout, lr, epochs, batch_size, output_size)
        self.name = "GRU"
        
        # Initialize GRU network instead of LSTM
        self.model = GRUNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

class CNNLSTMModel(LSTMModel):
    """
    CNN-LSTM hybrid model implementation for signal prediction
    """
    def __init__(self, input_size, seq_length, hidden_size=64, num_layers=2, dropout=0.3, lr=0.001, epochs=50, batch_size=64, output_size=3):
        super().__init__(input_size, seq_length, hidden_size, num_layers, dropout, lr, epochs, batch_size, output_size)
        self.name = "CNN-LSTM"
        
        # Initialize CNN-LSTM network
        self.model = CNNLSTMNet(
            input_size=input_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) 