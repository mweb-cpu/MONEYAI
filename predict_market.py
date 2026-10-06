import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. DATA INGESTION
# ---------------------------------------------------------
ticker = "BTC-USD"
df = yf.download(ticker, start="2023-01-01", end="2026-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
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

# --- NEW INDICATOR: MACD ---
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = (ema12 - ema26) / df['Close']  # Scale-normalized MACD line
macd_signal = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - macd_signal  # Scale-normalized MACD Histogram

# --- NEW INDICATOR: BOLLINGER BANDS ---
bb_middle = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
bb_upper = bb_middle + (bb_std * 2)
bb_lower = bb_middle - (bb_std * 2)

# %B: Relative position of price within the bands (0 = at lower band, 1 = at upper band)
df['BB_PctB'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
# Band Width: Standardized measure of volatility expansion/contraction
df['BB_Width'] = (bb_upper - bb_lower) / bb_middle

# ---------------------------------------------------------
# 3. REGRESSION TARGET
# ---------------------------------------------------------
df['Target_Return'] = df['Return'].shift(-1)
df.dropna(inplace=True)

# Export engineered indicators to CSV for R statistical feature selection
df.to_csv('btc_features.csv')
print("✅ Saved 'btc_features.csv' for R feature selection.") 
# ---------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------

# Default list of all engineered features
default_features = [
    'Return', 'Return_Lag1', 'Return_Lag2', 'Return_Lag3', 
    'Volatility_10', 'Volume', 'RSI', 
    'MACD', 'MACD_Hist', 'BB_PctB', 'BB_Width'
]

# Check if R has generated selected_features.txt
if os.path.exists('selected_features.txt'):
    with open('selected_features.txt', 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✅ Loaded {len(features)} Boruta confirmed features from 'selected_features.txt':")
    print(features)
else:
    features = default_features
    print("⚠️ 'selected_features.txt' not found. Using all default features.")

# Subset feature matrix X using selected features
X = df[features]
y = df['Target_Return']

# Time-series chronological split (80% train, 20% test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# ---------------------------------------------------------
# 5. EVALUATION & IMPORTANCES
# ---------------------------------------------------------
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- PERCENTAGE RETURN REGRESSION RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.4%}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4%}")
print(f"R-squared Score (R2): {r2:.4f}\n")

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

print("--- Feature Importances ---")
print(feature_importance_df.sort_values(by='Importance', ascending=False).to_string(index=False))

# Plot Feature Importances
plt.figure(figsize=(9, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.title('Random Forest Regressor Feature Importances')
plt.xlabel('Relative Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Plot Predictions
results_df = pd.DataFrame({'Actual Return': y_test, 'Predicted Return': predictions}, index=X_test.index)
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df['Actual Return'], label='Actual Return', color='gray', alpha=0.6)
plt.plot(results_df.index, results_df['Predicted Return'], label='Predicted Return', color='crimson', linestyle='--')
plt.title(f'{ticker} - Predicted vs Actual Daily Percentage Returns')
plt.xlabel('Date')
plt.ylabel('Daily Percentage Return')
plt.axhline(0, color='black', linestyle=':', alpha=0.5)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()