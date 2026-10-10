# ---------------------------------------------------------
# FEATURE SELECTION IN R (Boruta)
# ---------------------------------------------------------

# Install Boruta package if not already present
if (!require("Boruta")) {
  install.packages("Boruta", repos = "https://cloud.r-project.org")
}

library(Boruta)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

# Load the dataset created by predict_market.py
data <- read.csv("btc_features.csv", row.names = 1)

# Remove missing values
data <- na.omit(data)

# ---------------------------------------------------------
# 2. RUN BORUTA FEATURE SELECTION
# ---------------------------------------------------------

set.seed(42)

boruta_output <- Boruta(
  Target_Return ~
    Return +
    Return_Lag1 +
    Return_Lag2 +
    Return_Lag3 +
    Volatility_10 +
    Volume +
    RSI +
    MACD +
    MACD_Hist +
    BB_PctB +
    BB_Width,
  data = data,
  doTrace = 2
)

# ---------------------------------------------------------
# 3. DISPLAY BORUTA DECISIONS
# ---------------------------------------------------------

print("--- BORUTA FEATURE SELECTION DECISIONS ---")

print(attStats(boruta_output))

# ---------------------------------------------------------
# 4. PLOT FEATURE IMPORTANCE
# ---------------------------------------------------------

plot(
  boruta_output,
  las = 2,
  main = "Boruta Feature Importance Rankings",
  cex.axis = 0.7
)

# ---------------------------------------------------------
# 5. EXTRACT CONFIRMED FEATURES
# ---------------------------------------------------------

stats <- attStats(boruta_output)

confirmed_features <- rownames(
  stats[stats$decision == "Confirmed", ]
)

# ---------------------------------------------------------
# 6. DEFAULT TO ALL FEATURES IF NONE ARE CONFIRMED
# ---------------------------------------------------------

if (length(confirmed_features) == 0) {

  confirmed_features <- c(
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
  )
}

# ---------------------------------------------------------
# 7. SAVE SELECTED FEATURES
# ---------------------------------------------------------

writeLines(
  confirmed_features,
  "selected_features.txt"
)

print("Saved confirmed features to 'selected_features.txt'.")

print("--- FEATURES USED BY RANDOM FOREST ---")
print(confirmed_features)