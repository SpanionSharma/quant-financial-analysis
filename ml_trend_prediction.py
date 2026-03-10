import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def prepare_ml_data(df):
    """
    Prepares features (X) and target (y) for the ML model.
    Target: 1 if tomorrow's price > today's price, else 0.
    Features: RSI, Daily Return, MA Spread, Volatility.
    """
    df = df.copy()
    
    # Create target (Binary: Up=1, Down=0)
    df['Target'] = np.where(df['Daily_Return'].shift(-1) > 0, 1, 0)
    
    # Create features
    df['MA_Spread'] = (df['MA20'] - df['MA50']) / df['MA50']
    
    # Select feature columns (drop NAs from rolling calculations)
    features = ['Daily_Return', 'RSI', 'MA_Spread', 'Volatility_30d']
    df = df.dropna(subset=features + ['Target'])
    
    X = df[features]
    y = df['Target']
    
    return X, y, df

def train_and_predict(ticker, df):
    """
    Trains a simple Random Forest model to predict price direction.
    """
    X, y, clean_df = prepare_ml_data(df)
    
    # Time-series split (no shuffling to prevent look-ahead bias)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- ML Prediction Results for {ticker} ---")
    print(f"Accuracy of directional prediction: {acc:.2%}")
    
    # Predict for tomorrow (the very last available row)
    last_features = X.tail(1)
    prediction = model.predict(last_features)[0]
    prob = model.predict_proba(last_features)[0]
    
    direction = "UP" if prediction == 1 else "DOWN"
    print(f"Model prediction for next trading day: {direction} (Confidence: {max(prob):.2%})")
    
    return model, acc

if __name__ == "__main__":
    ticker = 'AAPL'
    file_path = f'processed_data/{ticker}_processed.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        train_and_predict(ticker, df)
    else:
        print(f"Error: {file_path} not found.")
