import os
import pandas as pd
from data_extraction import download_stock_data
from data_processing import process_all_tickers
from analysis_and_visualization import (
    compute_analytics, 
    plot_price_history, 
    plot_moving_averages, 
    plot_correlation_heatmap,
    load_processed_data
)
from strategy_evaluation import backtest_ma_crossover, plot_strategy_performance
from advanced_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from ml_trend_prediction import train_and_predict

def run_pipeline():
    print("=== Starting Financial Data Analysis Pipeline ===\n")
    
    # 1. Data Extraction
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    print("Step 1: Extracting data...")
    download_stock_data(tickers)
    
    # 2. Data Processing & Feature Engineering
    print("\nStep 2: Processing data and engineering features...")
    processed_data = process_all_tickers(tickers)
    
    # Ensure processed_data directory and files are save
    os.makedirs('processed_data', exist_ok=True)
    for ticker, df in processed_data.items():
        df.to_csv(f'processed_data/{ticker}_processed.csv')
    
    # 3. Analytics & Visualization
    print("\nStep 3: Computing analytics and generating visualizations...")
    analytics_results = {}
    for ticker in tickers:
        df = processed_data[ticker]
        analytics_results[ticker] = compute_analytics(ticker, df)
        
    # Print results
    print("\n--- Analytics Summary ---")
    for ticker, results in analytics_results.items():
        print(f"\n{ticker}:")
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")
            
    # Correlation Matrix
    returns_df = pd.DataFrame({ticker: df['Daily_Return'] for ticker, df in processed_data.items()})
    correlation_matrix = returns_df.corr()
    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)
    
    # Visualizations
    plot_price_history(processed_data)
    plot_moving_averages('AAPL', processed_data['AAPL'])
    plot_correlation_heatmap(correlation_matrix)
    
    # 4. Advanced Metrics
    print("\nStep 4: Computing advanced risk metrics...")
    for ticker in tickers:
        df = processed_data[ticker]
        sharpe = calculate_sharpe_ratio(df['Daily_Return'])
        mdd = calculate_max_drawdown(df['Close'])
        print(f"{ticker}: Sharpe Ratio = {sharpe:.2f}, Max Drawdown = {mdd:.2%}")
    
    # 5. ML Strategy Evaluation
    print("\nStep 5: Evaluating MA Crossover Strategy for AAPL...")
    strategy_results = backtest_ma_crossover('AAPL', processed_data['AAPL'])
    
    # ... Strategy Results ...
    final_bh_return = strategy_results['Cumulative_Market_Return'].iloc[-1] - 1
    final_strat_return = strategy_results['Cumulative_Strategy_Return'].iloc[-1] - 1
    
    print(f"\nAAPL Strategy Results:")
    print(f"  Buy and Hold Return: {final_bh_return:.2%}")
    print(f"  Strategy Return: {final_strat_return:.2%}")
    
    plot_strategy_performance('AAPL', strategy_results)
    
    # 6. ML Prediction
    print("\nStep 6: Running Machine Learning Trend Prediction for AAPL...")
    train_and_predict('AAPL', processed_data['AAPL'])
    
    print("\n=== Pipeline Execution Complete ===")
    print("All outputs are saved in 'data/', 'processed_data/', and 'visualizations/' folders.")

if __name__ == "__main__":
    run_pipeline()
