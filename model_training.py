import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from preprocessing import preprocess_data


class FinancialDataset(Dataset):
    """Custom Dataset for financial time series data"""
    
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM model for financial time series prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data, target_col, window_size=10, step_size=1):
    """
    Create sequences for LSTM model with improved memory handling
    
    Parameters:
        data (pd.DataFrame): Input DataFrame
        target_col (str): Name of the target column
        window_size (int): Size of the window for features
        step_size (int): Step size for sliding window
        
    Returns:
        np.ndarray: Feature sequences
        np.ndarray: Target values
    """
    features = []
    targets = []
    
    # Drop timestamp column if it exists
    if 'timestamp' in data.columns:
        X = data.drop(columns=['timestamp'])
    else:
        X = data.copy()
    
    # Get target column
    y = X[target_col]
    X = X.drop(columns=[target_col])
    
    # Create sequences with specified step size
    for i in range(0, len(X) - window_size, step_size):
        # Convert window to numpy and handle missing values
        window = X.iloc[i:i+window_size].values
        if np.isnan(window).any():
            continue
        
        # Get target value
        target = y.iloc[i+window_size]
        if np.isnan(target):
            continue
            
        # Map target values to non-negative indices: -1 -> 0, 0 -> 1, 1 -> 2
        if target_col.startswith('trend_'):
            # For trend columns: -1 -> 0 (DOWN), 0 -> 1 (NEUTRAL), 1 -> 2 (UP)
            target_mapped = int(target) + 1
        else:
            # For label columns: keep as is (already 0, 1, 2)
            target_mapped = int(target)
            
        features.append(window)
        targets.append(target_mapped)
        
        # Batch processing to save memory
        if len(features) >= 10000:
            yield np.array(features), np.array(targets)
            features = []
            targets = []
    
    # Return any remaining data
    if features:
        yield np.array(features), np.array(targets)


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """Train the model"""
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%')
    
    return model


def evaluate_model(model, test_loader, device, classes=['Down', 'Neutral', 'Up']):
    """Evaluate the model"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=classes))
    
    return accuracy, precision, recall, f1


def generate_signal(model, input_data, target_timeframe, device):
    """
    Generate a trading signal based on model prediction
    
    Parameters:
        model: Trained model
        input_data: Preprocessed input features
        target_timeframe: Timeframe for prediction ('1min', '3min', '5min')
        device: Device to run prediction on
        
    Returns:
        dict: Signal information
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
    # Map predicted class back to original trend values
    # 0 -> DOWN, 1 -> NEUTRAL, 2 -> UP
    if predicted_class == 0:
        signal = "DOWN"
    elif predicted_class == 2:
        signal = "UP"
    else:
        signal = "NEUTRAL"
        
    # Format for output
    signal_info = {
        "pair": "EUR/USD",
        "timeframe": target_timeframe,
        "signal": signal,
        "confidence": float(confidence)
    }
    
    return signal_info


def save_model(model, scaler, model_path, scaler_path, config):
    """Save model and related data"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), model_path)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, scaler_path)
    
    # Save configuration with unique name for each target
    config_dir = os.path.dirname(model_path)
    target_col = config.get('target_col', 'unknown')
    config_file = os.path.join(config_dir, f'config_{target_col}.json')
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Config saved to {config_file}")


def load_model(model_path, scaler_path, config_path=None):
    """Load trained model and related data"""
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = LSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes']
    )
    
    # Load model state with proper device mapping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load scaler
    import joblib
    scaler = joblib.load(scaler_path)
    
    return model, scaler, config


def main(data_path=None, target_col='trend_1min', window_size=10, batch_size=64, 
         hidden_size=64, num_layers=2, learning_rate=0.001, epochs=50, 
         model_output_dir='models'):
    """
    Main function to train the model
    
    Parameters:
        data_path: Path to preprocessed data CSV file
        target_col: Target column for prediction
        window_size: Size of sliding window for features
        batch_size: Batch size for training
        hidden_size: Hidden size of LSTM
        num_layers: Number of LSTM layers
        learning_rate: Learning rate
        epochs: Number of training epochs
        model_output_dir: Directory to save model files
    """
    # Create output directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or preprocess data
    if data_path and os.path.exists(data_path):
        print(f"Loading preprocessed data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Preprocessing data...")
        excel_file_path = "DATA_FOREX/1.EURUSD/Recent Data - (2023 - Latest)/RECENT_DATA_FILE_DUMP_EURUSD_M1.xlsx"
        text_file_directory = "DATA_FOREX/1.EURUSD/Recent Data - (2023 - Latest)/Text Files/"
        df = preprocess_data(excel_file_path, text_file_directory)
        
        # Save preprocessed data
        df.to_csv("processed_data.csv", index=False)
    
    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Scale numerical features
    feature_cols = [col for col in train_df.columns if col not in ['timestamp', target_col, 'label_1min', 'label_3min', 'label_5min', 'trend_1min', 'trend_3min', 'trend_5min']]
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Create sequences for training
    print("Creating sequences for training...")
    train_features, train_targets = [], []
    for batch_features, batch_targets in create_sequences(train_df, target_col, window_size):
        train_features.append(batch_features)
        train_targets.append(batch_targets)
    
    if train_features:
        train_features = np.vstack(train_features)
        train_targets = np.concatenate(train_targets)
    else:
        raise ValueError("No valid sequences could be created from the training data")
    
    # Create sequences for testing (use all at once for small test sets)
    print("Creating sequences for testing...")
    test_features, test_targets = [], []
    for batch_features, batch_targets in create_sequences(test_df, target_col, window_size):
        test_features.append(batch_features)
        test_targets.append(batch_targets)
    
    if test_features:
        test_features = np.vstack(test_features)
        test_targets = np.concatenate(test_targets)
    else:
        raise ValueError("No valid sequences could be created from the test data")
    
    print(f"Train sequences: {train_features.shape}")
    print(f"Test sequences: {test_features.shape}")
    
    # Create datasets and data loaders
    train_dataset = FinancialDataset(train_features, train_targets)
    test_dataset = FinancialDataset(test_features, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = train_features.shape[2]  # Number of features
    num_classes = len(np.unique(train_targets))
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("Training model...")
    model = train_model(model, train_loader, criterion, optimizer, device, epochs)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device)
    
    # Save model
    model_path = os.path.join(model_output_dir, f'lstm_{target_col}.pth')
    scaler_path = os.path.join(model_output_dir, f'scaler_{target_col}.pkl')
    
    config = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'window_size': window_size,
        'target_col': target_col,
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    save_model(model, scaler, model_path, scaler_path, config)
    
    # Generate example signal
    example_input = test_features[0]
    timeframe = target_col.split('_')[1] if '_' in target_col else '1min'
    signal = generate_signal(model, example_input, timeframe, device)
    
    print("\nExample signal:")
    print(json.dumps(signal, indent=2))
    
    return model, scaler, config


if __name__ == "__main__":
    main(data_path="processed_data.csv", target_col="trend_1min") 