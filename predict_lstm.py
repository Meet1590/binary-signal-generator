import pandas as pd
import numpy as np
import json
import os
import torch
import torch.nn as nn
from joblib import load
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
    window_data_all = input_data.iloc[-window_size:][all_feature_cols].values
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


def predict_from_csv(input_file, models_dir='models_lstm', output_file=None, confidence_threshold=0.5):
    """
    Make predictions using the latest data from a CSV file
    
    Parameters:
        input_file (str): Path to CSV file with preprocessed data
        models_dir (str): Directory containing trained models
        output_file (str): Path to save prediction results
        confidence_threshold (float): Threshold for prediction confidence
        
    Returns:
        dict: Prediction results
    """
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Target timeframes
    timeframes = ['1min', '3min', '5min']
    
    # Results
    results = {}
    
    # Process each timeframe
    for timeframe in timeframes:
        target_col = f'trend_{timeframe}'
        
        # Paths to model files
        model_path = os.path.join(models_dir, f'lstm_{target_col}.pth')
        config_path = os.path.join(models_dir, f'config_{target_col}.json')
        scaler_path = os.path.join(models_dir, f'scaler_{target_col}.joblib')
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [model_path, config_path, scaler_path]):
            print(f"Warning: Model files for {timeframe} not found. Skipping...")
            continue
        
        # Load model
        model, config, scaler = load_model_and_scaler(model_path, config_path, scaler_path, device)
        
        # Print debug information
        print(f"\nProcessing {timeframe} timeframe:")
        print(f"- Model input size: {config['input_size']}")
        print(f"- Model window size: {config['window_size']}")
        print(f"- All feature columns in config: {len(config.get('feature_cols', []))}")
        
        # Show which columns will be used for prediction
        all_features = config.get('feature_cols', [])
        model_features = [col for col in all_features 
                          if not col.startswith('label_') and not col.startswith('trend_')]
        print(f"- Features used for model input: {model_features}")
        
        # Prepare input
        try:
            input_tensor = prepare_input_sequence(df, config, scaler)
            print(f"- Input tensor shape: {input_tensor.shape}")
        except Exception as e:
            print(f"Error preparing input for {timeframe}: {str(e)}")
            continue
        
        # Make prediction
        try:
            predicted_class, confidence, probabilities = predict_signal(model, input_tensor, device)
            
            # Adjust confidence based on threshold
            if confidence < confidence_threshold and predicted_class != 1:  # 1 is NEUTRAL
                print(f"Low confidence prediction for {timeframe}. Changing to NEUTRAL.")
                predicted_class = 1
                confidence = probabilities[1]
            
            # Format prediction
            prediction = format_prediction(predicted_class, confidence, timeframe)
            results[timeframe] = prediction
            
            print(f"Prediction for {timeframe}: {prediction['signal']} (Confidence: {prediction['confidence']:.3f})")
        except Exception as e:
            print(f"Error predicting for {timeframe}: {str(e)}")
    
    # Save to file if requested
    if output_file and results:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results


def main():
    """Main function"""
    try:
        # Define parameters directly
        input_file = "final_data/preprocessed_test.csv"
        models_dir = "models_lstm"
        output_file = "lstm_predictions.json"
        confidence_threshold = 0.6  # Higher threshold for more reliable signals
        
        print(f"Starting predictions with settings:")
        print(f"Input file: {input_file}")
        print(f"Models directory: {models_dir}")
        print(f"Output file: {output_file}")
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Make predictions
        results = predict_from_csv(input_file, models_dir, output_file, confidence_threshold)
        
        # Print final results
        print("\nFinal Predictions:")
        print(json.dumps(results, indent=2))
        
        print("\nPrediction completed successfully!")
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPrediction failed. Please check the error message above.")


if __name__ == "__main__":
    main() 