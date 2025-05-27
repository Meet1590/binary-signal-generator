import os
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

# Import local modules
from data_processor import (
    load_forex_data, 
    add_technical_indicators, 
    calculate_trend,
    create_lag_features
)

def load_model(model_path):
    """
    Load a trained model
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}")
    
    # Determine model type from filename
    if 'ensemble' in model_path.lower():
        from models import EnsembleModel
        model = EnsembleModel.load(model_path)
    elif 'arima' in model_path.lower():
        from models import ARIMAModel, AutoARIMAModel
        if 'auto' in model_path.lower():
            model = AutoARIMAModel.load(model_path)
        else:
            model = ARIMAModel.load(model_path)
    elif any(x in model_path.lower() for x in ['lstm', 'gru', 'cnn']):
        from models import LSTMModel, GRUModel, CNNLSTMModel
        if 'gru' in model_path.lower():
            model = GRUModel.load(model_path)
        elif 'cnn' in model_path.lower():
            model = CNNLSTMModel.load(model_path)
        else:
            model = LSTMModel.load(model_path)
    else:
        # Assume it's a standard ML model
        from models import BaseSignalModel
        model = BaseSignalModel.load(model_path)
    
    return model

def prepare_data_for_prediction(data, window_size=10):
    """
    Prepare data for prediction
    
    Args:
        data: DataFrame with raw OHLC data
        window_size: Size of the lookback window
        
    Returns:
        Processed data ready for prediction
    """
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Add trend features
    data = calculate_trend(data)
    
    # Add lag features for key indicators
    key_indicators = ['rsi_14', 'macd', 'adx', 'boll_width', 'ma_cross']
    data = create_lag_features(data, columns=key_indicators)
    
    # Get features (exclude 'label' if present)
    features = data.select_dtypes(include=[np.number])
    if 'label' in features.columns:
        features = features.drop(columns=['label'])
    
    # Create a rolling window
    X = []
    for i in range(len(features) - window_size + 1):
        X.append(features.iloc[i:i+window_size].values)
    
    # Return the most recent window
    return np.array(X[-1:])

def plot_prediction_chart(data, signal, save_path=None):
    """
    Plot a chart with the prediction
    
    Args:
        data: DataFrame with OHLC data
        signal: Signal dictionary
        save_path: Path to save the chart image
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the close price
    plt.plot(data.index, data['close'], label='Close Price')
    
    # Add a marker for the prediction
    last_idx = data.index[-1]
    last_price = data['close'].iloc[-1]
    
    if signal['signal'] == 'UP':
        plt.scatter(last_idx, last_price, color='green', s=100, marker='^', label='UP Signal')
    elif signal['signal'] == 'DOWN':
        plt.scatter(last_idx, last_price, color='red', s=100, marker='v', label='DOWN Signal')
    else:
        plt.scatter(last_idx, last_price, color='gray', s=100, marker='o', label='NEUTRAL Signal')
    
    # Add a text annotation with the confidence
    plt.annotate(f"Confidence: {signal['confidence']:.2f}", 
                 (last_idx, last_price), 
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=12)
    
    plt.title(f"Forex Signal Prediction: {signal['pair']} - {signal['timeframe']}")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Chart saved to {save_path}")
    
    plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate forex trading signals')
    
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the input data file')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    
    parser.add_argument('--window_size', type=int, default=10,
                        help='Size of the lookback window')
    
    parser.add_argument('--output_file', type=str, default='signal.json',
                        help='Path to save the signal output')
    
    parser.add_argument('--rows', type=int, default=1000,
                        help='Number of most recent rows to use')
    
    parser.add_argument('--chart', action='store_true',
                        help='Generate a chart visualization')
    
    parser.add_argument('--chart_path', type=str, default='signal_chart.png',
                        help='Path to save the chart image')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    data = load_forex_data(args.data_file)
    
    # Use only the most recent rows
    if args.rows and args.rows < len(data):
        data = data.iloc[-args.rows:]
    
    # Load model
    model = load_model(args.model_path)
    
    # Prepare data for prediction
    X = prepare_data_for_prediction(data, window_size=args.window_size)
    
    # Generate signal
    signal = model.generate_signal(X)
    
    # Print the signal
    print("\nGenerated Signal:")
    print(json.dumps(signal, indent=2))
    
    # Save signal to file
    with open(args.output_file, 'w') as f:
        json.dump(signal, f, indent=2)
    
    print(f"Signal saved to {args.output_file}")
    
    # Generate chart if requested
    if args.chart:
        plot_prediction_chart(data, signal, save_path=args.chart_path)

if __name__ == "__main__":
    main() 