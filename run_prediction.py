from data_fetcher import fetch_eth_data
from local_utils import prepare_data, predict_signal, add_bollinger_bands

# Step 1: Fetch latest data
df = fetch_eth_data()
print(f"✅ Fetched {len(df)} rows of ETHUSDT 1h data")
print(df.head())

if df.empty:
    print("❌ No data fetched. Check the API or internet connection.")
    exit()

# Step 2: Prepare features
df = prepare_data(df)

if df.empty:
    print("❌ No data left after preparation. Try adjusting your prepare_data logic.")
    exit()

# Step 3: Add Bollinger Bands
df = add_bollinger_bands(df)

# ✅ Step 4: Drop rows with NaNs in critical features
df.dropna(subset=['returns', 'rolling_mean', 'rolling_std', 'upper_band', 'lower_band'], inplace=True)

if df.empty:
    print("❌ Bollinger Band columns contain NaNs. Try checking the rolling window size or data sufficiency.")
    exit()

# Step 5: Use latest row for prediction
latest_row = df.iloc[-1]
features = latest_row[['returns', 'rolling_mean', 'rolling_std', 'upper_band', 'lower_band']].to_dict()

# Step 6: Predict signal
signal = predict_signal(features)
print(f"📈 Predicted Signal: {signal.upper()}")
