import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_processed_data(ticker):
    """
    Loads processed data from CSV.
    """
    file_path = f'processed_data/{ticker}_processed.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
        
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def compute_analytics(ticker, df):
    """
    Computes analytics for a single stock.
    """
    mean_return = df['Daily_Return'].mean()
    std_return = df['Daily_Return'].std()
    annualized_volatility = std_return * np.sqrt(252)
    
    # Store results
    results = {
        'Mean Daily Return': mean_return,
        'Std Dev of Returns': std_return,
        'Annualized Volatility': annualized_volatility
    }
    return results

def plot_price_history(processed_data):
    """
    Plots price history for all stocks.
    """
    plt.figure(figsize=(12, 6))
    for ticker, df in processed_data.items():
        plt.plot(df.index, df['Close'], label=ticker)
    
    plt.title('Stock Price History (3 Years)')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/price_history.png')
    plt.close()

def plot_moving_averages(ticker, df):
    """
    Plots moving averages against price for a single stock.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Price', alpha=0.5)
    plt.plot(df.index, df['MA20'], label='20-Day MA')
    plt.plot(df.index, df['MA50'], label='50-Day MA')
    
    plt.title(f'{ticker} Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f'visualizations/{ticker}_ma.png')
    plt.close()

def plot_correlation_heatmap(correlation_matrix):
    """
    Plots a heatmap of correlations.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Daily Returns')
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    processed_data = {}
    analytics_results = {}
    
    for ticker in tickers:
        df = load_processed_data(ticker)
        if df is not None:
            processed_data[ticker] = df
            analytics_results[ticker] = compute_analytics(ticker, df)
            
    # Print analytics
    print("\n--- Basic Analytics for each stock ---")
    for ticker, results in analytics_results.items():
        print(f"\n{ticker}:")
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")
            
    # Compute correlation matrix
    returns_df = pd.DataFrame({ticker: df['Daily_Return'] for ticker, df in processed_data.items()})
    correlation_matrix = returns_df.corr()
    
    print("\n--- Correlation Matrix of Daily Returns ---")
    print(correlation_matrix)
    
    # Visualizations
    plot_price_history(processed_data)
    # Highlight one stock's MA
    plot_moving_averages('AAPL', processed_data['AAPL'])
    plot_correlation_heatmap(correlation_matrix)
    
    print("\nAnalytics and Visualizations complete. Check 'visualizations' folder.")
