import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from predict_market import fetch_and_prepare_data

# Page Config
st.set_page_config(page_title="Crypto AI Forecaster", layout="wide")

st.title("📈 Crypto AI Market Forecaster")
st.write("Predicting next-day cryptocurrency percentage returns using Random Forest and R Boruta feature selection.")

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("Model Settings")
ticker = st.sidebar.text_input("Ticker Symbol", "BTC-USD")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100)
test_size = st.sidebar.slider("Test Set Ratio", 0.10, 0.40, 0.20, step=0.05)

# Default list of all 11 engineered technical indicators
DEFAULT_FEATURES = [
    'Return', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 
    'Volatility_10', 'Volume', 'RSI', 
    'MACD', 'MACD_Hist', 'BB_PctB', 'BB_Width'
]

def load_selected_features():
    """Reads features chosen by R Boruta if selected_features.txt exists."""
    if os.path.exists("selected_features.txt"):
        with open("selected_features.txt", "r") as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        if features:
            return features, True
    return DEFAULT_FEATURES, False

# ---------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------
if st.button("Run Model & Forecast", type="primary"):
    with st.spinner("Downloading price data and training model..."):
        # 1. Fetch & prepare technical indicators from predict_market.py
        df = fetch_and_prepare_data(ticker)
        
        # 2. Check for R Boruta features
        features, uses_boruta = load_selected_features()
        
        if uses_boruta:
            st.info(f"**Using {len(features)} R Boruta Confirmed Features:** `{', '.join(features)}`")
        else:
            st.warning("`selected_features.txt` not found in repo. Using all 11 default indicators.")
        
        X = df[features]
        y = df['Target_Return']
        
        # 3. Chronological Time-Series Split
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 4. Train Random Forest Model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # 5. Next-Day Live Forecast
        latest_features = X.tail(1)
        next_day_pred = model.predict(latest_features)[0]
        
        # Display Signal Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Predicted Next-Day Return", 
                value=f"{next_day_pred * 100:.2f}%"
            )
        with col2:
            if next_day_pred > 0:
                st.success("### Trading Signal: BUY / LONG 🟢")
            else:
                st.error("### Trading Signal: SELL / CASH 🔴")
        
        # 6. Test Set Performance Chart
        st.subheader("Test Set Performance: Actual vs Predicted Daily Returns")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual Return", color="gray", alpha=0.6)
        ax.plot(y_test.index, predictions, label="Predicted Return", color="crimson", linestyle="--")
        ax.axhline(0, color="black", linestyle=":", alpha=0.5)
        ax.set_ylabel("Daily Return %")
        ax.set_xlabel("Date")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)