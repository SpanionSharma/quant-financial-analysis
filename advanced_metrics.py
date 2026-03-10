import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculates the Sharpe Ratio of the return series.
    Formula: (Mean Return - Risk Free Rate) / Std Dev of Return
    Annualized by multiplying by sqrt(252).
    """
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0:
        return 0
        
    sharpe = (mean_return - risk_free_rate) / std_return
    return sharpe * np.sqrt(252)

def calculate_max_drawdown(prices):
    """
    Calculates the Maximum Drawdown (MDD) of the price series.
    MDD is the maximum observed loss from a peak to a trough of a portfolio.
    """
    cumulative_returns = (1 + prices.pct_change().dropna()).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

if __name__ == "__main__":
    import os
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    for ticker in tickers:
        file_path = f'processed_data/{ticker}_processed.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            sharpe = calculate_sharpe_ratio(df['Daily_Return'])
            mdd = calculate_max_drawdown(df['Close'])
            print(f"{ticker}: Sharpe Ratio = {sharpe:.2f}, Max Drawdown = {mdd:.2%}")
