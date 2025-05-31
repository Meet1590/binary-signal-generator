import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
from datetime import datetime, timedelta
from preprocessing import preprocess_data
from model_training import load_model, generate_signal


def load_latest_data(excel_file_path, window_size=10):
    """
    Load and preprocess the latest forex data
    
    Parameters:
        excel_file_path: Path to the Excel file with forex data
        window_size: Number of data points needed for prediction
        
    Returns:
        pd.DataFrame: Processed data
    """
    # Load and preprocess data
    df = preprocess_data(excel_file_path)
    
    # Get the last window_size rows
    latest_data = df.tail(window_size)
    
    return latest_data


def prepare_input_sequence(data, feature_cols, scaler, window_size):
    """
    Prepare input sequence for model prediction
    
    Parameters:
        data: DataFrame with latest data
        feature_cols: List of feature columns
        scaler: Fitted scaler for normalization
        window_size: Size of the input window
        
    Returns:
        np.ndarray: Normalized input sequence
    """
    # Check if we have enough data
    if len(data) < window_size:
        raise ValueError(f"Not enough data points. Need {window_size}, got {len(data)}")
    
    # Get the most recent window_size data points
    recent_data = data.tail(window_size)
    
    # Extract features and normalize
    features = recent_data[feature_cols].values
    features = scaler.transform(features)
    
    return features


def predict_signals(model_dir, data, timeframes=['1min', '3min', '5min']):
    """
    Generate trading signals for multiple timeframes
    
    Parameters:
        model_dir: Directory containing trained models
        data: DataFrame with latest data
        timeframes: List of timeframes to generate signals for
        
    Returns:
        list: List of signal dictionaries
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signals = []
    
    for timeframe in timeframes:
        target_col = f"trend_{timeframe}"
        model_path = os.path.join(model_dir, f'lstm_{target_col}.pth')
        scaler_path = os.path.join(model_dir, f'scaler_{target_col}.pkl')
        config_path = os.path.join(model_dir, f'config_{target_col}.json')
        
        # Check if model exists
        if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(config_path):
            print(f"Model files for {timeframe} not found. Skipping...")
            continue
            
        try:
            # Load model and scaler
            model, scaler, config = load_model(model_path, scaler_path, config_path)
            model = model.to(device)
            
            # Prepare input
            feature_cols = config['feature_cols']
            window_size = config['window_size']
            
            # Validate that required features exist in data
            missing_features = [col for col in feature_cols if col not in data.columns]
            if missing_features:
                print(f"Missing features for {timeframe}: {missing_features}. Skipping...")
                continue
            
            # Create input sequence
            input_seq = prepare_input_sequence(data, feature_cols, scaler, window_size)
            
            # Generate signal
            signal = generate_signal(model, input_seq, timeframe, device)
            signals.append(signal)
            print(f"Generated signal for {timeframe}: {signal['signal']} with confidence {signal['confidence']:.4f}")
            
        except Exception as e:
            print(f"Error generating signal for {timeframe}: {str(e)}")
    
    return signals


def save_signals(signals, output_file):
    """
    Save signals to a JSON file
    
    Parameters:
        signals: List of signal dictionaries
        output_file: Path to output file
    """
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = {
        "timestamp": timestamp,
        "signals": signals
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Signals saved to {output_file}")


def main():
    """Main function to generate signals"""
    parser = argparse.ArgumentParser(description='Generate forex trading signals')
    parser.add_argument('--data-file', type=str, required=True, help='Path to forex data file')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--output-file', type=str, default='signals.json', help='Output file for signals')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1min', '3min', '5min'], 
                        help='Timeframes to generate signals for')
    
    args = parser.parse_args()
    
    # Load latest data
    print(f"Loading data from {args.data_file}")
    data = load_latest_data(args.data_file)
    
    # Generate signals
    print("Generating signals...")
    signals = predict_signals(args.model_dir, data, args.timeframes)
    
    # Save signals
    save_signals(signals, args.output_file)
    
    print("Done!")


if __name__ == "__main__":
    main() 