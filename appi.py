import os

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from predict_market import fetch_and_prepare_data


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Crypto AI Forecaster",
    layout="wide"
)

st.title("📈 Crypto AI Market Forecaster")

st.write(
    "Predicting next-day cryptocurrency percentage returns "
    "using Random Forest and R Boruta feature selection."
)


# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------

st.sidebar.header("Model Settings")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    "BTC-USD"
)

n_estimators = st.sidebar.slider(
    "Number of Trees (n_estimators)",
    10,
    200,
    100
)

test_size = st.sidebar.slider(
    "Test Set Ratio",
    0.10,
    0.40,
    0.20,
    step=0.05
)


# ---------------------------------------------------------
# DEFAULT FEATURES
# ---------------------------------------------------------

DEFAULT_FEATURES = [
    "Return",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Volatility_10",
    "Volume",
    "RSI",
    "MACD",
    "MACD_Hist",
    "BB_PctB",
    "BB_Width"
]


# ---------------------------------------------------------
# LOAD R BORUTA FEATURES
# ---------------------------------------------------------

def load_selected_features():

    # Check whether R created selected_features.txt
    if os.path.exists("selected_features.txt"):

        with open("selected_features.txt", "r") as f:

            features = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]

        if features:
            return features, True

    return DEFAULT_FEATURES, False


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

if st.button("Run Model & Forecast", type="primary"):

    with st.spinner(
        "Downloading price data and training model..."
    ):

        # ---------------------------------------------------------
        # 1. FETCH + PREPARE DATA
        # ---------------------------------------------------------

        df = fetch_and_prepare_data(ticker)

        # ---------------------------------------------------------
        # 2. LOAD R BORUTA FEATURES
        # ---------------------------------------------------------

        features, uses_boruta = load_selected_features()

        if uses_boruta:

            st.info(
                f"**Using {len(features)} R Boruta Confirmed "
                f"Features:** `{', '.join(features)}`"
            )

        else:

            st.warning(
                "`selected_features.txt` was not found. "
                "Using all 11 default indicators."
            )

        # ---------------------------------------------------------
        # 3. CREATE X AND Y
        # ---------------------------------------------------------

        X = df[features]

        y = df["Target_Return"]

        # ---------------------------------------------------------
        # 4. CHRONOLOGICAL TRAIN/TEST SPLIT
        # ---------------------------------------------------------

        split_idx = int(
            len(df) * (1 - test_size)
        )

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]

        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # ---------------------------------------------------------
        # 5. TRAIN RANDOM FOREST
        # ---------------------------------------------------------

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        # ---------------------------------------------------------
        # 6. TEST SET PREDICTIONS
        # ---------------------------------------------------------

        predictions = model.predict(X_test)

        # ---------------------------------------------------------
        # 7. MODEL EVALUATION
        # ---------------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # ---------------------------------------------------------
        # 8. NEXT-DAY FORECAST
        # ---------------------------------------------------------

        latest_features = X.tail(1)

        next_day_pred = model.predict(
            latest_features
        )[0]

        # ---------------------------------------------------------
        # 9. CURRENT PRICE
        # ---------------------------------------------------------

        current_price = float(
            df["Close"].iloc[-1]
        )

        predicted_price = (
            current_price *
            (1 + next_day_pred)
        )

        expected_change = (
            predicted_price -
            current_price
        )

        # ---------------------------------------------------------
        # 10. DISPLAY FORECAST
        # ---------------------------------------------------------

        st.subheader("🔮 Next-Day Forecast")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                label="Current BTC Price",
                value=f"${current_price:,.2f}"
            )

        with col2:

            st.metric(
                label="Predicted Next-Day Price",
                value=f"${predicted_price:,.2f}",
                delta=f"{expected_change:+,.2f}"
            )

        with col3:

            st.metric(
                label="Predicted Next-Day Return",
                value=f"{next_day_pred * 100:.2f}%"
            )

        # ---------------------------------------------------------
        # 11. TRADING SIGNAL
        # ---------------------------------------------------------

        if next_day_pred > 0:

            st.success(
                "### Trading Signal: BUY / LONG 🟢"
            )

        else:

            st.error(
                "### Trading Signal: SELL / CASH 🔴"
            )

        # ---------------------------------------------------------
        # 12. MODEL PERFORMANCE
        # ---------------------------------------------------------

        st.subheader("📊 Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "MAE",
                f"{mae:.4%}"
            )

        with col2:

            st.metric(
                "RMSE",
                f"{rmse:.4%}"
            )

        with col3:

            st.metric(
                "R²",
                f"{r2:.4f}"
            )

        # ---------------------------------------------------------
        # 13. TEST SET PERFORMANCE CHART
        # ---------------------------------------------------------

        st.subheader(
            "Test Set Performance: "
            "Actual vs Predicted Daily Returns"
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(
            y_test.index,
            y_test,
            label="Actual Return",
            color="gray",
            alpha=0.6
        )

        ax.plot(
            y_test.index,
            predictions,
            label="Predicted Return",
            color="crimson",
            linestyle="--"
        )

        ax.axhline(
            0,
            color="black",
            linestyle=":",
            alpha=0.5
        )

        ax.set_ylabel(
            "Daily Return %"
        )

        ax.set_xlabel(
            "Date"
        )

        ax.legend(
            loc="upper right"
        )

        ax.grid(
            True,
            linestyle="--",
            alpha=0.3
        )

        st.pyplot(fig)

        # ---------------------------------------------------------
        # 14. FEATURE IMPORTANCE
        # ---------------------------------------------------------

        st.subheader(
            "🌲 Random Forest Feature Importance"
        )

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=True
        )

        fig2, ax2 = plt.subplots(
            figsize=(9, 5)
        )

        ax2.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax2.set_xlabel(
            "Relative Importance"
        )

        ax2.set_ylabel(
            "Feature"
        )

        ax2.set_title(
            "Random Forest Feature Importances"
        )

        plt.tight_layout()

        st.pyplot(fig2)