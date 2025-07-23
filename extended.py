import pandas as pd
import time
from datetime import datetime
from data_fetcher import fetch_ohlcv_binance
import joblib
from local_utils import preprocess_live_data, display_output
import os

wallet_balance = 1000.0  # starting capital
cash_balance = wallet_balance  # in USD
eth_balance = 0.0  # in ETH
current_position = "NONE"
entry_price = None

# Load trained model
model = joblib.load("model.pkl")
print("✅ Model loaded.")

# Set optimized thresholds
BUY_THRESHOLD = 0.15
SELL_THRESHOLD = 0.18

# CSV logging setup
csv_file = "eth_predictions_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w") as f:
        f.write("timestamp,price,signal,sell_prob,hold_prob,buy_prob,wallet_balance\n")

print("🔁 Running model every 5 minutes. Press Ctrl+C to stop.\n")

try:
    while True:
        print("🔄 Fetching Live ETH 5m Data...")
        df = fetch_ohlcv_binance(interval="1h")

        if df is not None and not df.empty:
            print(f"✅ Fetched {len(df)} rows of ETHUSDT 5m data")

            live_features = preprocess_live_data(df)
            print("🚨 Checking live input features:")
            print(live_features.tail(3))

            if live_features is not None and not live_features.empty:
                latest_row = live_features.iloc[[-1]]
                proba = model.predict_proba(latest_row)
                print(f"🔍 Raw probabilities (class order: {model.classes_}): {proba}")

                prob_array = proba[0]
                probabilities = dict(zip(model.classes_, prob_array))

                buy_prob = probabilities.get('BUY', 0)
                sell_prob = probabilities.get('SELL', 0)

                if buy_prob >= BUY_THRESHOLD:
                    signal = "BUY"
                elif sell_prob >= SELL_THRESHOLD:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                latest_price = df['close'].iloc[-1]
                previous_price = df['close'].iloc[-2]
                latest_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # --- Wallet Simulation Logic ---
                if signal == "BUY":
                    if current_position != "LONG":
                        # Close short if any
                        if current_position == "SHORT":
                            cash_balance += abs(eth_balance) * latest_price
                            eth_balance = 0.0

                        # Enter LONG
                        eth_balance = cash_balance / latest_price
                        cash_balance = 0.0
                        current_position = "LONG"

                elif signal == "SELL":
                    if current_position != "SHORT":
                        # Close long if any
                        if current_position == "LONG":
                            cash_balance += eth_balance * latest_price
                            eth_balance = 0.0

                        # Enter SHORT
                        eth_balance = - (cash_balance / latest_price)
                        cash_balance = 0.0
                        current_position = "SHORT"

                elif signal == "HOLD":
                    pass

                # Update wallet value in USD
                wallet_balance = cash_balance + (eth_balance * latest_price)

                # Display
                display_output(
                    signal=signal,
                    price=latest_price,
                    time=latest_time,
                    initial_balance=wallet_balance,
                    previous_price=previous_price,
                    probabilities=probabilities
                )

                # Save to CSV
                with open(csv_file, "a") as f:
                    f.write(f"{latest_time},{latest_price},{signal},"
                            f"{probabilities.get('SELL', 0):.4f},"
                            f"{probabilities.get('HOLD', 0):.4f},"
                            f"{probabilities.get('BUY', 0):.4f},"
                            f"{wallet_balance:.2f}\n")
            else:
                print("❌ Failed to preprocess live features.")
        else:
            print("❌ Failed to fetch live ETH data.")

        print("\n⏳ Sleeping for 5 minutes...\n")
        time.sleep(300)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
