import pandas as pd
import numpy as np
import os
from stockstats import wrap
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

def load_forex_data(file_path, limit=None):
    """
    Load forex data from an Excel file
    
    Args:
        file_path: Path to the Excel file
        limit: Optional limit on number of rows to load (for memory constraints)
        
    Returns:
        Pandas DataFrame with processed data
    """
    print(f"Loading data from {file_path}...")
    col_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Data extraction from Excel file
    raw_df = pd.read_excel(file_path, header=None, nrows=limit)
    
    # Preparing clean dataframe from raw dataframe
    df = raw_df[0].str.split(";", expand=True)
    df.columns = col_names
    
    # Removing extra data
    clean_df = df.drop(columns=['volume'])
    
    # Convert price columns to float
    clean_df[['open', 'high', 'low', 'close']] = clean_df[['open', 'high', 'low', 'close']].astype(float)
    clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'], format="%Y%m%d %H%M%S")
    
    # Creating index from timestamp
    clean_df.set_index('timestamp', inplace=True)
    
    # Remove duplicates
    if not clean_df.index.is_unique:
        print("Removing duplicate timestamps...")
        clean_df = clean_df[~clean_df.index.duplicated(keep='first')]
    
    # Handle gaps - using a simple forward fill approach
    clean_df = clean_df.asfreq('1T', method='ffill')
    
    # Drop any remaining NaN values
    df_clean = clean_df.dropna()
    
    print(f"Loaded {len(df_clean)} rows of data")
    return df_clean

def create_labels(df, lookahead=1, threshold=0.0):
    """
    Create binary labels for price direction
    
    Args:
        df: DataFrame with OHLC data
        lookahead: Number of periods to look ahead for price movement
        threshold: Minimum price movement threshold (as percentage)
        
    Returns:
        DataFrame with added labels
    """
    df = df.copy()
    
    # Calculate future price
    future_close = df['close'].shift(-lookahead)
    
    # Calculate percentage change
    pct_change = (future_close - df['close']) / df['close'] * 100
    
    # Create labels based on price movement and threshold
    conditions = [
        pct_change > threshold,      # UP
        pct_change < -threshold      # DOWN
    ]
    choices = [1, -1]                # 1=UP, -1=DOWN, 0=NEUTRAL
    df['label'] = np.select(conditions, choices, default=0)
    
    # Drop rows with NaN future prices
    df = df.dropna(subset=['label'])
    
    # Count label distribution
    label_counts = df['label'].value_counts()
    print(f"Label distribution: UP={label_counts.get(1, 0)}, DOWN={label_counts.get(-1, 0)}, NEUTRAL={label_counts.get(0, 0)}")
    
    return df

def calculate_trend(df, window=3):
    """
    Calculate trend indicators from price data
    
    Args:
        df: DataFrame with OHLC data
        window: Window size for trend calculation
        
    Returns:
        DataFrame with trend features
    """
    df = df.copy()
    
    # Direction of candles
    df['candle_direction'] = np.where(df['close'] > df['open'], 1, 
                                      np.where(df['close'] < df['open'], -1, 0))
    
    # Candle trend (rolling sum of directions)
    df['candle_trend'] = df['candle_direction'].rolling(window=window, min_periods=1).sum()
    
    # Price momentum (rate of change)
    df['price_momentum'] = df['close'].pct_change(periods=window) * 100
    
    # Volatility
    df['volatility'] = (df['high'] - df['low']) / df['open'] * 100
    
    # Average candle size
    df['candle_size'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['avg_candle_size'] = df['candle_size'].rolling(window=window).mean()
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with technical indicators
    """
    print("Adding technical indicators...")
    
    # Convert index to column for stockstats
    df_reset = df.reset_index()
    
    # Wrap with stockstats
    sdf = wrap(df_reset)
    
    # Compute indicators
    # Moving averages
    sdf['close_5_sma']     # 5-period Simple Moving Average
    sdf['close_10_sma']    # 10-period SMA
    sdf['close_20_sma']    # 20-period SMA
    sdf['close_5_ema']     # 5-period Exponential Moving Average
    sdf['close_10_ema']    # 10-period EMA
    sdf['close_20_ema']    # 20-period EMA
    
    # Oscillators
    sdf['rsi_14']          # 14-period RSI
    sdf['rsi_5']           # 5-period RSI (faster)
    sdf['cci_20']          # Commodity Channel Index
    sdf['stochrsi_14']     # Stochastic RSI
    
    # MACD
    sdf['macd']            # MACD
    sdf['macds']           # MACD Signal Line
    sdf['macdh']           # MACD Histogram
    
    # Trend indicators
    sdf['adx']             # Average Directional Index
    sdf['atr']             # Average True Range
    
    # Bollinger Bands
    sdf['boll']            # Bollinger Middle Band
    sdf['boll_ub']         # Bollinger Upper Band
    sdf['boll_lb']         # Bollinger Lower Band
    sdf['boll_width'] = sdf['boll_ub'] - sdf['boll_lb']  # Bollinger Band Width
    
    # Manually calculate Keltner Channels instead of using stockstats
    # Typical price
    sdf['tp'] = (sdf['high'] + sdf['low'] + sdf['close']) / 3
    # EMA of typical price (middle line)
    sdf['kc_middle'] = sdf['tp'].ewm(span=20, adjust=False).mean()
    # ATR for channel width
    if 'atr' not in sdf.columns:
        sdf['tr'] = np.maximum(
            sdf['high'] - sdf['low'],
            np.maximum(
                abs(sdf['high'] - sdf['close'].shift(1)),
                abs(sdf['low'] - sdf['close'].shift(1))
            )
        )
        sdf['atr'] = sdf['tr'].rolling(window=10).mean()
    # Upper and lower bands
    sdf['kc_upper'] = sdf['kc_middle'] + 2 * sdf['atr']
    sdf['kc_lower'] = sdf['kc_middle'] - 2 * sdf['atr']
    sdf['kc_width'] = sdf['kc_upper'] - sdf['kc_lower']  # Keltner Channel Width
    
    # Ichimoku Cloud
    sdf['senkou_a']        # Ichimoku Senkou Span A
    sdf['senkou_b']        # Ichimoku Senkou Span B
    sdf['cloud_green'] = np.where(sdf['senkou_a'] > sdf['senkou_b'], 1, 0)  # Green cloud
    
    # Volume indicators (if volume data is available)
    if 'volume' in sdf.columns:
        sdf['vr']          # Volume Ratio
        sdf['obv']         # On Balance Volume
    
    # Manually compute Candle Body/Wick Ratio
    sdf['body'] = abs(sdf['close'] - sdf['open'])
    sdf['wick'] = sdf['high'] - sdf['low']
    sdf['body_wick_ratio'] = sdf['body'] / sdf['wick'].replace(0, 1e-9)
    
    # Candle position relative to range
    sdf['close_position'] = (sdf['close'] - sdf['low']) / (sdf['high'] - sdf['low'] + 1e-9)
    
    # Additional derived features
    sdf['ma_cross'] = np.where(sdf['close_5_ema'] > sdf['close_20_ema'], 1, -1)  # EMA cross
    sdf['bb_position'] = (sdf['close'] - sdf['boll_lb']) / (sdf['boll_ub'] - sdf['boll_lb'] + 1e-9)  # Position within Bollinger Bands
    
    # Price distance from moving averages (%)
    sdf['dist_ma20'] = (sdf['close'] - sdf['close_20_ema']) / sdf['close_20_ema'] * 100
    
    # Drop intermediate columns
    sdf.drop(columns=['body', 'wick', 'tp', 'tr'], inplace=True, errors='ignore')
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(sdf)
    
    # Some indicators create NaN values, especially at the beginning
    # Replace all NaN with 0 for simplicity (or with more sophisticated imputation if needed)
    result_df = result_df.fillna(0)
    
    return result_df

def create_lag_features(df, columns=None, lags=[1, 2, 3]):
    """
    Create lagged features for time series analysis
    
    Args:
        df: DataFrame
        columns: List of columns to create lags for (None for all numeric columns)
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    df = df.copy()
    
    # If no columns specified, use all numeric columns except 'label'
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in columns:
            columns.remove('label')
    
    # Create lag features for selected columns
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Remove rows with NaN values created by lag
    df = df.dropna()
    
    return df

def remove_highly_correlated_features(df, threshold=0.95):
    """
    Remove features that are highly correlated with each other
    
    Args:
        df: DataFrame with features
        threshold: Correlation threshold for removal
        
    Returns:
        DataFrame with reduced features
    """
    print("Removing highly correlated features...")
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features")
    
    # Drop features
    df_reduced = df.drop(columns=to_drop)
    
    return df_reduced

def process_data_for_training(file_path, output_dir="processed_data", lookahead=1, limit=None):
    """
    Complete pipeline to process data for model training
    
    Args:
        file_path: Path to raw data file
        output_dir: Directory to save processed data
        lookahead: Periods to look ahead for labeling
        limit: Optional row limit for testing
        
    Returns:
        Processed DataFrame ready for modeling
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_forex_data(file_path, limit=limit)
    
    # Add basic trend features
    df = calculate_trend(df)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Create binary labels
    df = create_labels(df, lookahead=lookahead)
    
    # Create lag features for key indicators
    key_indicators = ['rsi_14', 'macd', 'adx', 'boll_width', 'close_position', 'ma_cross']
    df = create_lag_features(df, columns=key_indicators)
    
    # Remove highly correlated features
    df = remove_highly_correlated_features(df)
    
    # Save processed data
    df.to_csv(f"{output_dir}/processed_data.csv")
    
    print(f"Processed data saved to {output_dir}/processed_data.csv")
    print(f"Total features: {len(df.columns) - 1}")  # -1 for label column
    
    # Create and save feature scaler
    features = df.drop(columns=['label']).select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaler.fit(features)
    
    # Save scaler for later use
    joblib.dump(scaler, f"{output_dir}/feature_scaler.pkl")
    print(f"Feature scaler saved to {output_dir}/feature_scaler.pkl")
    
    return df

def create_train_val_test_split(df, train_size=0.7, val_size=0.15):
    """
    Split data into training, validation and test sets chronologically
    
    Args:
        df: DataFrame with processed data
        train_size: Proportion for training set
        val_size: Proportion for validation set
        
    Returns:
        train_df, val_df, test_df
    """
    # Sort by index (timestamp) to ensure chronological order
    df = df.sort_index()
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Data split - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def create_windowed_dataset(df, window_size=10, batch_size=None):
    """
    Create windowed arrays for sequence modeling
    
    Args:
        df: DataFrame with features and labels
        window_size: Size of the lookback window
        batch_size: Optional batch size (for memory-constrained environments)
        
    Returns:
        X, y arrays for modeling
    """
    features = df.drop(columns=['label']).select_dtypes(include=[np.number]).values
    labels = df['label'].values
    
    X, y = [], []
    
    # Process in batches if specified
    if batch_size:
        # Calculate number of batches
        n_samples = len(features) - window_size
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        for batch in tqdm(range(n_batches), desc="Creating windowed dataset"):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            for i in range(start_idx, end_idx):
                X.append(features[i:i+window_size])
                y.append(labels[i+window_size-1])  # Label for the last point in window
                
            # Optionally yield each batch for very large datasets
            # yield np.array(X), np.array(y)
            # X, y = [], []  # Clear lists after yielding
    else:
        # Process all at once
        for i in tqdm(range(len(features) - window_size), desc="Creating windowed dataset"):
            X.append(features[i:i+window_size])
            y.append(labels[i+window_size-1])
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Example usage
    data_path = "../DATA_FOREX/1.EURUSD/Recent Data - (2023 - Latest)/RECENT_DATA_FILE_DUMP_EURUSD_M1.xlsx"
    
    # Process a small sample for testing
    df = process_data_for_training(data_path, limit=10000)
    
    # Show feature names
    print("\nFeature names:")
    print(df.columns.tolist()) 