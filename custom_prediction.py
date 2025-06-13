import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
from joblib import load
import random
from datetime import datetime
import time

# Import the LSTM model class from train_lstm.py
from train_lstm import LSTMModel


def load_model_and_scaler(model_path, config_path, scaler_path, device=None):
    """
    Load trained model, configuration and scaler
    
    Parameters:
        model_path (str): Path to the saved model file
        config_path (str): Path to the model configuration file
        scaler_path (str): Path to the saved scaler file
        device (torch.device): Device to load the model on
        
    Returns:
        tuple: (model, config, scaler)
    """
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
    
    # Load scaler
    scaler = load(scaler_path)
    
    return model, config, scaler


def prepare_input_sequence(input_data, config, scaler):
    """
    Prepare input data for prediction
    
    Parameters:
        input_data (pd.DataFrame): Input data
        config (dict): Model configuration
        scaler (object): Fitted scaler
        
    Returns:
        torch.Tensor: Prepared input sequence
    """
    # Get the expected input size from the model config
    expected_input_size = config.get('input_size')
    
    # Get all feature columns from config (including label columns for scaler compatibility)
    all_feature_cols = config.get('feature_cols', [])
    
    # Make sure all required columns exist in the input data
    missing_cols = [col for col in all_feature_cols if col not in input_data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")
    
    # Extract window size
    window_size = config.get('window_size', 10)
    
    # Get last window_size rows with all features (including labels) for scaling
    if len(input_data) < window_size:
        raise ValueError(f"Input data has {len(input_data)} rows, but window_size is {window_size}")
    
    # First scale using all columns the scaler was trained on
    window_data_all = input_data[all_feature_cols].values
    scaled_data_all = scaler.transform(window_data_all)
    
    # Now filter out just the columns we need for the model input
    # (exclude labels which were included for scaling but not used in the model)
    model_features = [i for i, col in enumerate(all_feature_cols) 
                      if not col.startswith('label_') and not col.startswith('trend_')]
    
    # Ensure we have the right number of features
    if len(model_features) != expected_input_size:
        print(f"Warning: Number of features ({len(model_features)}) doesn't match expected input size ({expected_input_size})")
        print(f"Using only the first {expected_input_size} features")
        model_features = model_features[:expected_input_size]
    
    # Select only the relevant columns for the model input
    scaled_data = scaled_data_all[:, model_features]
    
    # Convert to tensor
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    
    return input_tensor


def predict_signal(model, input_tensor, device):
    """
    Make a prediction using the model
    
    Parameters:
        model (LSTMModel): Trained model
        input_tensor (torch.Tensor): Input tensor
        device (torch.device): Device to run prediction on
        
    Returns:
        tuple: (predicted_class, confidence, probabilities)
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
    
    return predicted_class, confidence, probabilities


def format_prediction(predicted_class, confidence, timeframe, pair="EUR/USD"):
    """
    Format prediction into the required output format
    
    Parameters:
        predicted_class (int): Predicted class (0, 1, or 2)
        confidence (float): Confidence of the prediction
        timeframe (str): Timeframe of the prediction
        pair (str): Currency pair
        
    Returns:
        dict: Formatted prediction
    """
    # Map class back to signal: 0 -> DOWN, 1 -> NEUTRAL, 2 -> UP
    if predicted_class == 0:
        signal = "DOWN"
    elif predicted_class == 2:
        signal = "UP"
    else:
        signal = "NEUTRAL"
    
    return {
        "pair": pair,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": round(float(confidence), 3)
    }


def get_random_chunk(df, window_size=10):
    """
    Get a random chunk of consecutive rows from the dataframe
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        window_size (int): Size of the chunk
        
    Returns:
        pd.DataFrame: Random chunk of data
    """
    if len(df) <= window_size:
        return df
    
    # Get a random starting index
    max_start_idx = len(df) - window_size
    start_idx = random.randint(0, max_start_idx)
    
    # Return the chunk
    return df.iloc[start_idx:start_idx + window_size]


def predict_with_model(input_file, timeframe, models_dir='models_lstm', shuffle=False, confidence_threshold=0.6):
    """
    Make predictions using a specific model timeframe
    
    Parameters:
        input_file (str): Path to CSV file with preprocessed data
        timeframe (str): Timeframe to use for prediction ('1min', '3min', '5min')
        models_dir (str): Directory containing trained models
        shuffle (bool): Whether to use a random chunk of data instead of the last rows
        confidence_threshold (float): Threshold for prediction confidence
        
    Returns:
        dict: Prediction result
    """
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate timeframe
    if timeframe not in ['1min', '3min', '5min']:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of: '1min', '3min', '5min'")
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set up model paths
    target_col = f'trend_{timeframe}'
    model_path = os.path.join(models_dir, f'lstm_{target_col}.pth')
    config_path = os.path.join(models_dir, f'config_{target_col}.json')
    scaler_path = os.path.join(models_dir, f'scaler_{target_col}.joblib')
    
    # Check if files exist
    if not all(os.path.exists(path) for path in [model_path, config_path, scaler_path]):
        raise FileNotFoundError(f"Model files for {timeframe} not found.")
    
    # Load model
    model, config, scaler = load_model_and_scaler(model_path, config_path, scaler_path, device)
    
    # Print debug information
    print(f"\nProcessing {timeframe} timeframe:")
    print(f"- Model input size: {config['input_size']}")
    print(f"- Model window size: {config['window_size']}")
    
    # Get window size from config
    window_size = config.get('window_size', 10)
    
    # Get data chunk based on shuffle flag
    if shuffle:
        print("Using random chunk of data...")
        input_chunk = get_random_chunk(df, window_size)
        chunk_start = input_chunk.iloc[0]['timestamp'] if 'timestamp' in input_chunk.columns else "N/A"
        chunk_end = input_chunk.iloc[-1]['timestamp'] if 'timestamp' in input_chunk.columns else "N/A"
        print(f"- Random chunk from {chunk_start} to {chunk_end}")
    else:
        print("Using last rows of data...")
        input_chunk = df.iloc[-window_size:]
    
    # Prepare input
    try:
        input_tensor = prepare_input_sequence(input_chunk, config, scaler)
        print(f"- Input tensor shape: {input_tensor.shape}")
    except Exception as e:
        print(f"Error preparing input: {str(e)}")
        return None
    
    # Make prediction
    try:
        predicted_class, confidence, probabilities = predict_signal(model, input_tensor, device)
        
        # Adjust confidence based on threshold
        if confidence < confidence_threshold and predicted_class != 1:  # 1 is NEUTRAL
            print(f"Low confidence prediction. Changing to NEUTRAL.")
            predicted_class = 1
            confidence = probabilities[1]
        
        # Format prediction
        prediction = format_prediction(predicted_class, confidence, timeframe)
        
        # Print detailed results
        print("\nPrediction Details:")
        print(f"- Signal: {prediction['signal']}")
        print(f"- Confidence: {prediction['confidence']:.3f}")
        print(f"- Class probabilities: DOWN={probabilities[0]:.3f}, NEUTRAL={probabilities[1]:.3f}, UP={probabilities[2]:.3f}")
        
        return prediction
    except Exception as e:
        print(f"Error predicting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main(timeframe='1min', input_file="final_data/preprocessed_test.csv", 
         models_dir="models_lstm", shuffle=False, confidence_threshold=0.6):
    """
    Main function to run predictions with specified parameters
    
    Parameters:
        timeframe (str): Timeframe to predict ('1min', '3min', or '5min')
        input_file (str): Path to input CSV file
        models_dir (str): Directory containing trained models
        shuffle (bool): Whether to use random chunk of data instead of last rows
        confidence_threshold (float): Confidence threshold for predictions
    """
    try:
        print(f"Starting prediction with settings:")
        print(f"- Timeframe: {timeframe}")
        print(f"- Input file: {input_file}")
        print(f"- Models directory: {models_dir}")
        print(f"- Shuffle: {shuffle}")
        print(f"- Confidence threshold: {confidence_threshold}")
        
        # Make prediction
        result = predict_with_model(
            input_file, 
            timeframe, 
            models_dir, 
            shuffle, 
            confidence_threshold
        )
        
        if result:
            print("\nFinal Prediction:")
            print(json.dumps(result, indent=2))
            print("\nPrediction completed successfully!")
            return result
        else:
            print("\nPrediction failed.")
            return None
            
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPrediction failed. Please check the error message above.")
        return None


if __name__ == "__main__":
    # Example usage
    main(timeframe='5min', shuffle=True, confidence_threshold=0.4) 