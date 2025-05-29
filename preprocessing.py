import pandas as pd
import os
import re
import numpy as np
from stockstats import wrap

def load_data(excel_file_path):
    """
    Load data from Excel file and clean it
    """
    # Data extraction from Excel file
    raw_df = pd.read_excel(excel_file_path, header=None)
    
    # Preparing clean dataframe from raw dataframe
    col_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = raw_df[0].str.split(";", expand=True)
    df.columns = col_names

    # Removing extra data
    clean_df = df.drop(columns=['volume'])

    # Convert price columns to float
    clean_df[['open', 'high', 'low', 'close']] = clean_df[['open', 'high', 'low', 'close']].astype(float)
    clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'], format="%Y%m%d %H%M%S")

    # Creating index from timestamp
    clean_df.set_index('timestamp', inplace=True)

    # Make sure data has no duplicate values
    duplicated_data = clean_df.index[clean_df.index.duplicated()]
    clean_df = clean_df[~clean_df.index.duplicated(keep='first')]
    
    return clean_df

def combine_text_files(text_file_directory):
    """
    Combine all text files in a directory into one file
    """
    all_files = os.listdir(text_file_directory)
    with open(f'{text_file_directory}/combined_txt_file.txt', 'a') as file:
        for a_file in all_files:
            if a_file != 'combined_txt_file.txt':  # Skip the output file if it exists
                with open(f'{text_file_directory}/{a_file}', 'r') as temp_file:
                    file.write(temp_file.read() + '\n')
    return f'{text_file_directory}/combined_txt_file.txt'

def parse_gap_report(file_path):
    """
    Parse gap report from text file
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

def handling_gaps(clean_df, gap_file_path=None):
    """
    Handle gaps in the data
    """
    if gap_file_path:
        gaps = parse_gap_report(gap_file_path)
        # Forward fill small gaps (<= 300s) at 1-minute level
        clean_df = clean_df.asfreq('1T')
        # Flag large gaps (> 300s)
        for gap in gaps:
            if gap['duration_s'] > 300:
                # Mark period as unreliable (e.g., set to NaN or flag)
                clean_df.loc[gap['start']:gap['end']] = None
    else:
        # Forward fill all gaps if no gap report
        clean_df = clean_df.asfreq('1T', method='ffill')    
    return clean_df

def create_labels(df):
    """
    Create labels for different timeframes
    """
    df = df.copy()  

    # 1 minute candle labels
    prev_close_1_min = df['close'].shift(1)
    conditions = [
        df['close'] > prev_close_1_min,
        df['close'] < prev_close_1_min
    ]
    choices = [1, 2]  # 1 for up, 2 for down, 0 for neutral
    df.loc[:, 'label_1min'] = np.select(conditions, choices, default=0)
    df.loc[df.index[0], 'label_1min'] = 0

    # 3 minutes candle labels
    prev_close_3_min = df['close'].shift(2)
    conditions = [
        df['close'] > prev_close_3_min,
        df['close'] < prev_close_3_min
    ]
    df.loc[:, 'label_3min'] = np.select(conditions, choices, default=0)
    df.loc[df.index[0], 'label_3min'] = 0

    # 5 minutes candle labels
    prev_close_5_min = df['close'].shift(4)
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
        pd.DataFrame: Original DataFrame with only one additional `trend_col`.
    """
    df = df.copy()

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

    # Assign trend label: 1 = UP, 0 = DOWN, else = neutral_value
    df[trend_col] = pd.Series(
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

    return df[[trend_col]]

def add_trends(df):
    """
    Add trend indicators for different timeframes
    """
    df = df.copy()
    df['trend_1min'] = calculate_trend(df, label_col='label_1min', trend_col='trend_1min')['trend_1min']
    df['trend_3min'] = calculate_trend(df, label_col='label_3min', trend_col='trend_3min')['trend_3min']
    df['trend_5min'] = calculate_trend(df, label_col='label_5min', trend_col='trend_5min')['trend_5min']
    return df

def calculate_indicators(df):
    """
    Calculate technical indicators
    """
    df = df.reset_index()  # stockstats requires 'timestamp' as a column
    
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

    # Create final DataFrame with selected features
    final_df = sdf[[
        'timestamp', 'open', 'high', 'low', 'close',
        'close_5_ema', 'close_10_ema', 'rsi_14', 'macdh',
        'adx', 'atr', 'boll_width', 'body_wick_ratio', 'label_1min', 'label_3min', 'label_5min',
        'trend_1min', 'trend_3min', 'trend_5min'
    ]]

    return final_df

def remove_highly_correlated_features(df, threshold=0.8):
    """
    Removes features that are highly correlated with each other
    """
    # Work on numerical columns only
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Drop features from original DataFrame
    reduced_df = df.drop(columns=to_drop)
    
    print(f"Dropped features due to high correlation: {to_drop}")
    return reduced_df

def preprocess_data(excel_file_path, text_file_directory=None):
    """
    Main function to preprocess data
    """
    # Load and clean data
    clean_df = load_data(excel_file_path)
    
    # Handle gaps if text files are provided
    if text_file_directory:
        combine_text_file_path = combine_text_files(text_file_directory)
        clean_df = handling_gaps(clean_df, combine_text_file_path)
    else:
        clean_df = handling_gaps(clean_df)
    
    # Drop NaN values
    df_1min = clean_df.dropna()
    
    # Create labels
    df_1min = create_labels(df_1min)
    
    # Add trends
    df_1min = add_trends(df_1min)
    
    # Calculate indicators
    final_df = calculate_indicators(df_1min)
    
    # Remove highly correlated features
    final_df = remove_highly_correlated_features(final_df)
    
    return final_df

if __name__ == "__main__":
    # Example usage
    excel_file_path = "DATA_FOREX/1.EURUSD/Recent Data - (2023 - Latest)/RECENT_DATA_FILE_DUMP_EURUSD_M1.xlsx"
    text_file_directory = "DATA_FOREX/1.EURUSD/Recent Data - (2023 - Latest)/Text Files/"
    
    df = preprocess_data(excel_file_path, text_file_directory)
    print(df.head())
    
    # Save processed data
    df.to_csv("processed_data.csv", index=False) 