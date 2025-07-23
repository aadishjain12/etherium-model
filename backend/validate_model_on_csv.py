import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from local_utils import add_technical_features  # Ensure this is correct

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

# Step 8: Custom Threshold-Based Classification
buy_threshold = 0.2
sell_threshold = 0.21

def classify(row):
    if row["proba_buy"] > buy_threshold:
        return 1   # BUY
    elif row["proba_sell"] > sell_threshold:
        return -1  # SELL
    else:
        return 0   # HOLD

df["predicted_signal"] = df.apply(classify, axis=1)

# Step 9: Save Output to CSV
output_path = "predicted_results.csv"
df.to_csv(output_path, index=False)

print(f"✅ Validation complete using threshold logic. Results saved to {output_path}")

# Step 10: Backtest Strategy Performance
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['strategy_return'] = df['predicted_signal'].shift(1) * df['log_return']  # Use previous signal

# Cumulative Returns
df['cum_strategy_return'] = df['strategy_return'].cumsum()
df['cum_buy_hold'] = df['log_return'].cumsum()

# Step 11: Plot Performance
plt.figure(figsize=(14, 6))
plt.plot(df['cum_strategy_return'], label='📈 Strategy Returns', color='green')
plt.plot(df['cum_buy_hold'], label='💰 Buy & Hold Returns', color='blue')
plt.title("Backtest: Model Strategy vs Buy & Hold")
plt.xlabel("Time (Hourly candles)")
plt.ylabel("Cumulative Log Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("strategy_backtest.png")
plt.show()

print("📊 Backtest complete. Chart saved as 'strategy_backtest.png'")
