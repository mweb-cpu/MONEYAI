# ---------------------------------------------------------
# FEATURE SELECTION IN R (Boruta)
# ---------------------------------------------------------

# Install Boruta package if not already present
if (!require("Boruta")) install.packages("Boruta", repos = "https://cloud.r-project.org")
library(Boruta)

# Load the dataset created by predict_market.py
data <- read.csv("btc_features.csv", row.names = 1)
data <- na.omit(data)

# Run Boruta algorithm on Target_Return using all 11 features
set.seed(42)
boruta_output <- Boruta(
  Target_Return ~ Return + Return_Lag1 + Return_Lag2 + Return_Lag3 + 
                  Volatility_10 + Volume + RSI + MACD + MACD_Hist + BB_PctB + BB_Width, 
  data = data, 
  doTrace = 2
)

# Display Decisions (Confirmed, Tentative, or Rejected)
print("--- BORUTA FEATURE SELECTION DECISIONS ---")
print(attStats(boruta_output))

# Plot feature importance rankings
plot(boruta_output, las = 2, main = "Boruta Feature Importance Rankings", cex.axis = 0.7)