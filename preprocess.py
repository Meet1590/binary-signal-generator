import pandas as pd
import numpy as np
import re
from stockstats import wrap
import os

def load_data(file_path):
    """
    Load data from CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    # Read data from CSV (semicolon-delimited without headers)
    col_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(file_path, header=None, sep=';', names=col_names)
    
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y%m%d %H%M%S")
    df.set_index('timestamp', inplace=True)
    
    # Make sure data has no duplicate values
    if not df.index.is_unique:
        print("Removing duplicate timestamps...")
        duplicated_data = df.index[df.index.duplicated()]
        df = df[~df.index.duplicated(keep='first')]
    
    return df

def parse_gap_report(file_path):
    """
    Parse gap report from text file.
    
    Parameters:
        file_path (str): Path to the gap report file
        
    Returns:
        list: List of dictionaries containing gap information
    """
    gaps = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"Gap of (\d+)s found between (\d{14}) and (\d{14})\.", line)
            if match:
                duration = int(match.group(1))
                start = pd.to_datetime(match.group(2), format="%Y%m%d%H%M%S")
                end = pd.to_datetime(match.group(3), format="%Y%m%d%H%M%S")
                gaps.append({"start": start, "end": end, "duration_s": duration})
    return gaps

def handle_gaps(clean_df, gap_file_path):
    """
    Handle gaps in the time series data.
    
    Parameters:
        clean_df (pd.DataFrame): Input dataframe
        gap_file_path (str): Path to the gap report file
        
    Returns:
        pd.DataFrame: Dataframe with gaps handled
    """
    if gap_file_path and os.path.exists(gap_file_path):
        gaps = parse_gap_report(gap_file_path)
        # Forward fill small gaps (<= 300s) at 1-minute level
        clean_df = clean_df.asfreq('1T')
        # Flag large gaps (> 300s)
        for gap in gaps:
            if gap['duration_s'] > 300:
                # Mark period as unreliable (set to NaN)
                clean_df.loc[gap['start']:gap['end']] = None
    else:
        # Forward fill all gaps if no gap report
        clean_df = clean_df.asfreq('1T', method='ffill')
    
    # Drop NaN values
    clean_df = clean_df.dropna()
    return clean_df

def create_labels(df):
    """
    Create labels for 1-minute, 3-minute, and 5-minute candles.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with labels added
    """
    df = df.copy()
    
    # 1-minute label
    prev_close_1_min = df['close'].shift(1)
    conditions = [
        df['close'] > prev_close_1_min,
        df['close'] < prev_close_1_min
    ]
    choices = [1, 2]
    df.loc[:, 'label_1min'] = np.select(conditions, choices, default=0)
    df.loc[df.index[0], 'label_1min'] = 0
    
    # 3-minute label
    prev_close_3_min = df['close'].shift(2)
    conditions = [
        df['close'] > prev_close_3_min,
        df['close'] < prev_close_3_min
    ]
    df.loc[:, 'label_3min'] = np.select(conditions, choices, default=0)
    df.loc[df.index[0], 'label_3min'] = 0
    
    # 5-minute label
    prev_close_5_min = df['close'].shift(4)  # Fixed bug: was using wrong variable name
    conditions = [
        df['close'] > prev_close_5_min,
        df['close'] < prev_close_5_min
    ]
    df.loc[:, 'label_5min'] = np.select(conditions, choices, default=0)
    df.loc[df.index[0], 'label_5min'] = 0
    
    return df

def calculate_trend(df, label_col, trend_col, window=3, exclude_current=False, neutral_value=int(0)):
    """
    Calculate trend from an existing label column using rolling sum of the last `window` directions.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        label_col (str): Name of existing label column (e.g., 'label_1min').
        trend_col (str): Name of new output trend column.
        window (int): Number of candles to use for trend (default=3).
        exclude_current (bool): If True, exclude current candle in trend calculation.
        neutral_value: Value for neutral trend (default=0).

    Returns:
        pd.Series: Series containing trend values
    """
    # Map labels to direction values
    direction = pd.Series(
        np.select(
            [
                df[label_col] == 1,
                df[label_col] == 2
            ],
            [1, -1],
            default=0
        ),
        index=df.index
    )

    # Shift if current candle is to be excluded
    if exclude_current:
        direction = direction.shift(1)

    # Calculate rolling trend sum
    trend_sum = direction.rolling(window=window, min_periods=window).sum()

    # Assign trend label: 1 = UP, -1 = DOWN, else = neutral_value
    trend = pd.Series(
        np.select(
            [
                trend_sum >= 2,
                trend_sum <= -2
            ],
            [1, -1],
            default=neutral_value
        ),
        index=df.index
    )

    return trend

def add_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with indicators added
    """
    # Reset index to make stockstats work properly
    df = df.reset_index()
    
    # Wrap with stockstats
    sdf = wrap(df)
    
    # Compute indicators
    sdf['close_5_ema']     # EMA(5)
    sdf['close_10_ema']    # EMA(10)
    sdf['rsi_14']          # RSI(14)
    sdf['macdh']           # MACD Histogram
    sdf['adx']             # ADX
    sdf['atr']             # ATR
    sdf['boll_ub']         # Bollinger Upper Band
    sdf['boll_lb']         # Bollinger Lower Band
    sdf['boll_width'] = sdf['boll_ub'] - sdf['boll_lb']  # Bollinger Band Width
    
    # Manually compute Candle Body/Wick Ratio
    sdf['body'] = abs(sdf['close'] - sdf['open'])
    sdf['wick'] = sdf['high'] - sdf['low']
    sdf['body_wick_ratio'] = sdf['body'] / sdf['wick'].replace(0, 1e-9)
    
    # Create final dataframe with selected columns
    final_df = sdf[[
        'timestamp', 'open', 'high', 'low', 'close',
        'close_5_ema', 'close_10_ema', 'rsi_14', 'macdh',
        'adx', 'atr', 'boll_width', 'body_wick_ratio', 'label_1min', 'label_3min', 'label_5min',
        'trend_1min', 'trend_3min', 'trend_5min'
    ]]
    
    return final_df

def preprocess_data(input_file, output_file, gap_file=None):
    """
    Main preprocessing function that processes input data and saves the result.
    
    Parameters:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the preprocessed CSV file
        gap_file (str, optional): Path to the gap report file
    """
    print(f"Loading data from {input_file}...")
    df = load_data(input_file)
    
    print("Handling gaps...")
    df = handle_gaps(df, gap_file)
    
    print("Creating labels...")
    df = create_labels(df)
    
    print("Calculating trends...")
    df['trend_1min'] = calculate_trend(df, label_col='label_1min', trend_col='trend_1min')
    df['trend_3min'] = calculate_trend(df, label_col='label_3min', trend_col='trend_3min')
    df['trend_5min'] = calculate_trend(df, label_col='label_5min', trend_col='trend_5min')
    
    print("Adding indicators...")
    df = add_indicators(df)
    
    # Save to CSV
    print(f"Saving preprocessed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

def main():
    # File paths
    train_input = "final_data/training_data.csv"
    train_output = "final_data/preprocessed_train.csv"
    test_input = "final_data/test_data.csv"
    test_output = "final_data/preprocessed_test.csv"
    gap_file = "final_data/combined_txt_file.txt"
    
    # Process training data
    print("Processing training data...")
    preprocess_data(train_input, train_output, gap_file)
    
    # Process test data
    print("Processing test data...")
    preprocess_data(test_input, test_output, gap_file)

if __name__ == "__main__":
    main() 