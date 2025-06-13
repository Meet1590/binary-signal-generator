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
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import time


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
    
    # Remove all target columns from features
    target_cols = [col for col in X.columns if col.startswith('label_') or col.startswith('trend_')]
    X = X.drop(columns=target_cols)
    
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


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, patience=5):
    """Train the model with early stopping"""
    model.train()
    
    # For tracking training progress
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        model.train()
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
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation phase
        val_loss, val_acc = evaluate_model_during_training(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | '
              f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model if found
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 
                  'train_accs': train_accs, 'val_accs': val_accs}


def evaluate_model_during_training(model, data_loader, criterion, device):
    """Evaluate model during training"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    val_loss = running_loss / len(data_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def evaluate_model(model, test_loader, device, classes=['Down', 'Neutral', 'Up']):
    """Evaluate the model with detailed metrics"""
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
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, cm


def confusion_matrix(y_true, y_pred):
    """Create a confusion matrix from true and predicted labels"""
    # Get number of classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return cm


def plot_confusion_matrix(cm, target_col, output_dir='models'):
    """Plot and save confusion matrix"""
    classes = ['Down', 'Neutral', 'Up']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {target_col}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{target_col}_confusion_matrix.png")
    plt.close()


def generate_signal(model, input_data, scaler, target_timeframe, device):
    """
    Generate a trading signal based on model prediction
    
    Parameters:
        model: Trained model
        input_data: Raw input data
        scaler: Fitted scaler for features
        target_timeframe: Timeframe for prediction ('1min', '3min', '5min')
        device: Device to run prediction on
        
    Returns:
        dict: Signal information
    """
    model.eval()
    
    # Skip scaling if input_data is already a numpy array from sequence data
    if isinstance(input_data, np.ndarray):
        scaled_data = input_data
    else:
        # Scale input data for DataFrame input
        scaled_data = scaler.transform(input_data)
    
    # Convert to tensor
    with torch.no_grad():
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)
        # Add batch dimension if not present
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        # Add sequence dimension if we're directly using the final step of a sequence
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(1)
        
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


def save_model(model, scaler, training_history, model_path, scaler_path, history_path, config):
    """Save model, scaler, training history and configuration"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), model_path)
    
    # Save scaler
    dump(scaler, scaler_path)
    
    # Save training history
    with open(history_path, 'wb') as f:
        import pickle
        pickle.dump(training_history, f)
    
    # Save configuration
    config_dir = os.path.dirname(model_path)
    target_col = config.get('target_col', 'unknown')
    config_file = os.path.join(config_dir, f'config_{target_col}.json')
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Training history saved to {history_path}")
    print(f"Config saved to {config_file}")


def load_model(model_path, config_path, device=None):
    """Load trained model and configuration"""
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = LSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.2)
    ).to(device)
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, config


def train_all_models(data_path, fast_train=True, output_dir='models_lstm'):
    """
    Train LSTM models for all timeframes
    
    Parameters:
        data_path (str): Path to preprocessed data
        fast_train (bool): If True, use less data and epochs for faster training
        output_dir (str): Directory to save models
    """
    # Set common parameters
    window_size = 10
    # batch_size = 64 if not fast_train else 128
    batch_size = 128
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    epochs = 100 if not fast_train else 10
    patience = 5
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # If fast_train is True, use only a subset of data
    if fast_train:
        print("Fast train mode enabled. Using only 15% of the data.")
        # Convert timestamp to datetime for sorting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Use the most recent 15% of data
        df = df.sort_values('timestamp')
        df = df.iloc[-int(len(df) * 0.15):]
    
    # Target columns to train models for
    target_columns = ['trend_1min', 'trend_3min', 'trend_5min']
    
    # Train a model for each target
    for target_col in target_columns:
        print(f"\n{'='*50}")
        print(f"Training LSTM model for {target_col}")
        print(f"{'='*50}")
        
        # Split data into train, validation, and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
        train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Validation data shape: {val_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Scale features (exclude target columns and timestamp)
        feature_cols = [col for col in train_df.columns 
                       if col not in ['timestamp'] + target_columns]
        
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        val_df[feature_cols] = scaler.transform(val_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        
        # Create sequences for training
        print("Creating sequences for training...")
        train_features, train_targets = [], []
        step_size = 5 if fast_train else 1  # Use larger step size for fast training
        
        for batch_features, batch_targets in create_sequences(train_df, target_col, window_size, step_size):
            train_features.append(batch_features)
            train_targets.append(batch_targets)
        
        if train_features:
            train_features = np.vstack(train_features)
            train_targets = np.concatenate(train_targets)
        else:
            raise ValueError("No valid sequences could be created from the training data")
        
        # Create sequences for validation
        print("Creating sequences for validation...")
        val_features, val_targets = [], []
        for batch_features, batch_targets in create_sequences(val_df, target_col, window_size, step_size):
            val_features.append(batch_features)
            val_targets.append(batch_targets)
        
        if val_features:
            val_features = np.vstack(val_features)
            val_targets = np.concatenate(val_targets)
        else:
            raise ValueError("No valid sequences could be created from the validation data")
        
        # Create sequences for testing
        print("Creating sequences for testing...")
        test_features, test_targets = [], []
        for batch_features, batch_targets in create_sequences(test_df, target_col, window_size, step_size):
            test_features.append(batch_features)
            test_targets.append(batch_targets)
        
        if test_features:
            test_features = np.vstack(test_features)
            test_targets = np.concatenate(test_targets)
        else:
            raise ValueError("No valid sequences could be created from the test data")
        
        print(f"Train sequences: {train_features.shape}")
        print(f"Validation sequences: {val_features.shape}")
        print(f"Test sequences: {test_features.shape}")
        
        # Create datasets and data loaders
        train_dataset = FinancialDataset(train_features, train_targets)
        val_dataset = FinancialDataset(val_features, val_targets)
        test_dataset = FinancialDataset(test_features, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = train_features.shape[2]  # Number of features
        num_classes = len(np.unique(train_targets))
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=0.2
        ).to(device)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Using device: {device}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        print("Training model...")
        start_time = time.time()
        model, training_history = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate model on test set
        print("\nEvaluating model on test set...")
        accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader, device)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, target_col, output_dir)
        
        # Save model and related data
        model_path = os.path.join(output_dir, f'lstm_{target_col}.pth')
        scaler_path = os.path.join(output_dir, f'scaler_{target_col}.joblib')
        history_path = os.path.join(output_dir, f'history_{target_col}.pkl')
        
        config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'window_size': window_size,
            'target_col': target_col,
            'feature_cols': feature_cols,
            'dropout': 0.2,
            'training_time': training_time,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        }
        
        save_model(model, scaler, training_history, model_path, scaler_path, history_path, config)
        
        # Generate example signal
        try:
            print("\nGenerating example signal...")
            timeframe = target_col.split('_')[1] if '_' in target_col else '1min'
            
            # Get the input shape expected by the model
            input_shape = (1, window_size, input_size)  # batch_size, seq_length, num_features
            
            # Create an example sequence directly - using the last sequence from test_features
            example_sequence = test_features[-1:].copy()  # Use the entire sequence (already scaled)
            
            # Generate a signal using the example sequence
            input_tensor = torch.tensor(example_sequence, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]
                
                # Map predicted class to signal
                if pred_class == 0:
                    signal_value = "DOWN"
                elif pred_class == 2:
                    signal_value = "UP"
                else:
                    signal_value = "NEUTRAL"
                    
                signal_info = {
                    "pair": "EUR/USD",
                    "timeframe": timeframe,
                    "signal": signal_value,
                    "confidence": float(confidence)
                }
                
            print("Example signal:")
            print(json.dumps(signal_info, indent=2))
        except Exception as e:
            print(f"Error generating example signal: {str(e)}")
        
        print(f"{'='*50}\n")


def main():
    """Main function"""
    try:
        # Define parameters directly
        data_path = "final_data/preprocessed_train.csv"
        fast_train = False  # Set to False for full training
        output_dir = "models_lstm"
        
        print(f"Starting LSTM training with settings:")
        print(f"Data path: {data_path}")
        print(f"Fast train: {fast_train}")
        print(f"Output directory: {output_dir}")
        
        # Train models for all timeframes
        train_all_models(data_path, fast_train, output_dir)
        
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTraining failed. Please check the error message above.")


if __name__ == "__main__":
    main() 