import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(tickers, years=3):
    """
    Downloads historical daily price data for the given tickers.
    
    Args:
        tickers (list): List of stock ticker symbols.
        years (int): Number of years of historical data to fetch.
        
    Returns:
        dict: A dictionary where keys are tickers and values are DataFrames.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data_dict = {}
    
    # Create a directory for raw data if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        # Fetch data
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            continue
            
        # Save to CSV
        file_path = f'data/{ticker}_raw.csv'
        df.to_csv(file_path)
        print(f"Saved {ticker} data to {file_path}")
        
        data_dict[ticker] = df
        
    return data_dict

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    downloaded_data = download_stock_data(tickers)
    print("Data extraction complete.")
