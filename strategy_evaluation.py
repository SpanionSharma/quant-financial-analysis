import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def backtest_ma_crossover(ticker, df):
    """
    Backtests a simple moving average crossover strategy.
    Buy when MA20 > MA50, Sell when MA20 < MA50.
    """
    df = df.copy()
    
    # Generate signals
    df['Signal'] = 0.0
    df.loc[df.index[50:], 'Signal'] = np.where(df.loc[df.index[50:], 'MA20'] > df.loc[df.index[50:], 'MA50'], 1.0, 0.0)
    
    # Calculate daily returns for the strategy
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    
    # Calculate cumulative returns
    df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    
    return df

def plot_strategy_performance(ticker, df):
    """
    Plots the performance of the strategy vs buy and hold.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative_Market_Return'], label='Buy and Hold')
    plt.plot(df.index, df['Cumulative_Strategy_Return'], label='MA Crossover Strategy')
    
    plt.title(f'{ticker} Strategy Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/{ticker}_strategy_performance.png')
    plt.close()

if __name__ == "__main__":
    ticker = 'AAPL'
    file_path = f'processed_data/{ticker}_processed.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        results_df = backtest_ma_crossover(ticker, df)
        
        # Performance summary
        final_bh_return = results_df['Cumulative_Market_Return'].iloc[-1] - 1
        final_strat_return = results_df['Cumulative_Strategy_Return'].iloc[-1] - 1
        
        print(f"\n--- Strategy Evaluation for {ticker} ---")
        print(f"Buy and Hold Cumulative Return: {final_bh_return:.2%}")
        print(f"Strategy Cumulative Return: {final_strat_return:.2%}")
        
        plot_strategy_performance(ticker, results_df)
        print(f"\nPerformance plot saved to visualizations/{ticker}_strategy_performance.png")
    else:
        print(f"Error: {file_path} not found. Run data_processing.py first.")
