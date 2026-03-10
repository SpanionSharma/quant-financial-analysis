# Quantitative Developer Assignment: Financial Data Analysis

This project is a Python-based financial data analysis pipeline that fetches historical stock data from Yahoo Finance, processes it, computes basic analytics, and evaluates a simple trading strategy.

## Features
- **Data Extraction**: Downloads 3 years of historical daily price data for AAPL, MSFT, and GOOGL using `yfinance`.
- **Data Cleaning**: Handles missing values, removes duplicates, and aligns datasets to a common date range.
- **Feature Engineering**: Calculates daily returns, log returns, moving averages (20-day & 50-day), and 30-day rolling volatility.
- **Basic Analytics**: Computes mean daily returns, standard deviation, annualized volatility, and a correlation matrix.
- **Visualizations**: Generates price history plots, moving average crossover charts, and correlation heatmaps.
- **Strategy Evaluation**: Implements a 20/50-day moving average crossover strategy for AAPL and compares its performance against a buy-and-hold strategy.

## Project Structure
- `data/`: Raw CSV files from Yahoo Finance.
- `processed_data/`: Cleaned and feature-enriched datasets.
- `visualizations/`: Generated PNG charts.
- `data_extraction.py`: Handles downloading data.
- `data_processing.py`: Handles cleaning and feature engineering.
- `analysis_and_visualization.py`: Computes metrics and generates plots.
- `strategy_evaluation.py`: Backtests the MA crossover strategy.
- `main.py`: Orchestrates the entire pipeline.
- `README.md`: This file.

## Requirements
To run this project, you need Python 3 and the following libraries:
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

You can install them using:
```bash
pip install yfinance pandas numpy matplotlib seaborn
```

## How to Run
Execute the main script to run the entire pipeline:
```bash
python main.py
```

## Methodology and Assumptions
- **Missing Values**: Handled using forward-fill followed by backward-fill to ensure a continuous time series without introducing look-ahead bias in a way that breaks early data.
- **Annualization**: Annualized volatility is calculated using 252 trading days.
- **Strategy**: The MA crossover strategy assumes trades are executed at the closing price on the day the signal is generated (or the next day, depending on the shift). Here, we use the signal from the previous day to determine the current day's return to avoid look-ahead bias.
- **Date Range**: The common period across all three stocks is used for aligned analysis.

## Results Summary
The analytics and strategy results are printed to the console when running `main.py`. Detailed visualizations can be found in the `visualizations/` directory.
