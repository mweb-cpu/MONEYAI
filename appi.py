import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Crypto Forecast Interface", layout="wide")
st.title("🤖 AI Crypto Percentage Return Forecaster")

# Sidebar Configuration
st.sidebar.header("Model Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")
n_trees = st.sidebar.slider("Number of Trees", 50, 300, 100, step=25)
train_ratio = st.sidebar.slider("Training Split", 0.6, 0.9, 0.8, step=0.05)

if st.sidebar.button("Run Model & Forecast"):
    with st.spinner("Fetching data and training Random Forest..."):
        # 1. Download Data
        df = yf.download(ticker, start="2023-01-01")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Feature Engineering
        df['Return'] = df['Close'].pct_change()
        df['Return_Lag1'] = df['Return'].shift(1)
        df['Return_Lag2'] = df['Return'].shift(2)
        df['Return_Lag3'] = df['Return'].shift(3)
        df['Volatility_10'] = df['Return'].rolling(10).std()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (ema12 - ema26) / df['Close']
        df['MACD_Hist'] = df['MACD'] - df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_PctB'] = (df['Close'] - (bb_middle - bb_std * 2)) / (bb_std * 4)
        df['BB_Width'] = (bb_std * 4) / bb_middle

        df['Target_Return'] = df['Return'].shift(-1)
        df.dropna(inplace=True)

        # 3. Model Training (Update this features list based on R Boruta results)
        features = [
            'Return', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 
            'Volatility_10', 'Volume', 'RSI', 'MACD', 'MACD_Hist', 'BB_PctB', 'BB_Width'
        ]
        X = df[features]
        y = df['Target_Return']

        split_idx = int(len(df) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # 4. Display Outputs
        latest_sample = X.tail(1)
        next_pred = model.predict(latest_sample)[0]

        col1, col2 = st.columns(2)
        col1.metric("Predicted Next-Day Return", f"{next_pred:.2%}")
        col2.metric("Trading Signal", "BUY 🟢" if next_pred > 0 else "SELL 🔴")

        st.subheader("Predicted vs Actual Returns (Test Set)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(X_test.index, y_test, label="Actual Return", color="gray", alpha=0.6)
        ax.plot(X_test.index, predictions, label="Predicted Return", color="crimson", linestyle="--")
        ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        ax.legend()
        st.pyplot(fig)