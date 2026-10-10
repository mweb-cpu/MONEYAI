import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def fetch_and_prepare_data(ticker="BTC-USD", start_date="2023-01-01", end_date="2026-01-01"):
    """Downloads price data, calculates technical indicators, and returns a clean DataFrame."""
    df = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Daily Return and Lags
    df['Return'] = df['Close'].pct_change()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag3'] = df['Return'].shift(3)

    # Rolling Volatility
    df['Volatility_10'] = df['Return'].rolling(window=10).std()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema12 - ema26) / df['Close'] 
    macd_signal = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - macd_signal 

    # Bollinger Bands
    bb_middle = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    df['BB_PctB'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    df['BB_Width'] = (bb_upper - bb_lower) / bb_middle

    # Regression Target
    df['Target_Return'] = df['Return'].shift(-1)
    df.dropna(inplace=True)

    return df

# Local execution block (runs only when executing predict_market.py directly)
if __name__ == "__main__":
    ticker = "BTC-USD"
    df = fetch_and_prepare_data(ticker)
    
    # Export engineered indicators to CSV for R statistical feature selection
    df.to_csv('btc_features.csv')
    print("✅ Saved 'btc_features.csv' for R feature selection.")

    features = [
        'Return', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 
        'Volatility_10', 'Volume', 'RSI', 
        'MACD', 'MACD_Hist', 'BB_PctB', 'BB_Width'
    ]

    X = df[features]
    y = df['Target_Return']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n--- PERCENTAGE RETURN REGRESSION RESULTS ---")
    print(f"Mean Absolute Error (MAE): {mae:.4%}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4%}")
    print(f"R-squared Score (R2): {r2:.4f}\n")