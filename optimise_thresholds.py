import pandas as pd
import numpy as np
import joblib
from local_utils import add_technical_features  # Ensure this path is correct

# Step 1: Load Historical CSV Data
file_path = "eth_1h_binance.csv"
df = pd.read_csv(file_path)

# Step 2: Ensure 'close' is numeric and preprocess
df['close'] = pd.to_numeric(df['close'], errors='coerce')
df.dropna(inplace=True)

# Step 3: Add Technical Indicators (same as in training)
df = add_technical_features(df)
df.dropna(inplace=True)

# Step 4: Select Features
features = df[['returns', 'rolling_mean', 'rolling_std', 'upper_band', 'lower_band']]

# Step 5: Load Trained Model
model = joblib.load("model.pkl")

# Step 6: Predict Probabilities
proba = model.predict_proba(features)

# Step 7: Add probabilities to DataFrame for analysis
df["proba_sell"] = proba[:, 0]
df["proba_hold"] = proba[:, 1]
df["proba_buy"] = proba[:, 2]

# Optimization Logic: Try various thresholds and calculate cumulative return
best_return = -np.inf
best_buy_threshold = None
best_sell_threshold = None

results = []

for buy_thresh in np.arange(0.2, 0.6, 0.01):
    for sell_thresh in np.arange(0.2, 0.6, 0.01):
        def classify(row):
            if row["proba_buy"] > buy_thresh:
                return 1  # BUY
            elif row["proba_sell"] > sell_thresh:
                return -1  # SELL
            else:
                return 0  # HOLD

        df["predicted_signal"] = df.apply(classify, axis=1)

        # Strategy return = future return * signal
        df["strategy_return"] = df["returns"].shift(-1) * df["predicted_signal"]
        cumulative_return = df["strategy_return"].cumsum().iloc[-2]  # exclude NaN due to shift

        results.append((buy_thresh, sell_thresh, cumulative_return))

        if cumulative_return > best_return:
            best_return = cumulative_return
            best_buy_threshold = buy_thresh
            best_sell_threshold = sell_thresh

# Save all results for analysis
pd.DataFrame(results, columns=["buy_threshold", "sell_threshold", "cumulative_return"]).to_csv("threshold_optimization_results.csv", index=False)

# Final message
print("✅ Optimization complete.")
print(f"📈 Best Buy Threshold: {best_buy_threshold}")
print(f"📉 Best Sell Threshold: {best_sell_threshold}")
print(f"💰 Max Cumulative Return: {best_return:.4f}")
