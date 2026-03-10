import pandas as pd
import numpy as np
import os

def load_and_clean_data(ticker):
    """
    Loads raw data from CSV, cleans it, and returns a DataFrame.
    """
    file_path = f'data/{ticker}_raw.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
        
    # Read CSV, skip the second row which is often metadata in yfinance
    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    
    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Remove any duplicate rows
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle missing values - forward fill then backward fill for any remaining
    df = df.ffill().bfill()
    
    return df

def feature_engineering(df):
    """
    Calculates returns, moving averages, and volatility.
    """
    # Daily returns (percent change)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 20-day moving average
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 50-day moving average
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Rolling 30-day volatility
    df['Volatility_30d'] = df['Daily_Return'].rolling(window=30).std()
    
    # --- Winning Edge Additions ---
    
    # RSI (Relative Strength Index) - Measures overbought/oversold levels
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands - Measures price relative to volatility
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    return df

def process_all_tickers(tickers):
    """
    Processes all tickers and aligns them.
    """
    processed_data = {}
    
    for ticker in tickers:
        df = load_and_clean_data(ticker)
        if df is not None:
            df = feature_engineering(df)
            processed_data[ticker] = df
            
    # Align to common date range
    common_index = None
    for ticker, df in processed_data.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
            
    # Filter each DataFrame to common index
    for ticker in processed_data:
        processed_data[ticker] = processed_data[ticker].loc[common_index]
        
    return processed_data

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    processed_data = process_all_tickers(tickers)
    
    # Create directory for processed data
    os.makedirs('processed_data', exist_ok=True)
    
    for ticker, df in processed_data.items():
        file_path = f'processed_data/{ticker}_processed.csv'
        df.to_csv(file_path)
        print(f"Processed and saved {ticker} data to {file_path}")
        
    print("Data processing and feature engineering complete.")
