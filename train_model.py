import pandas as pd
import joblib
from data_fetcher import fetch_ohlcv_binance
import ccxt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from local_utils import prepare_data

# --------- Step 1: Fetch Data from Binance ---------

# Fetch 5-minute interval data for ETHUSDT
df = fetch_ohlcv_binance(symbol="ETHUSDT", interval="5m", limit=10000)

if df is None:
    print("❌ Failed to fetch 5-minute data.")
    exit()

print(f"✅ Fetched {len(df)} rows of 5-minute ETH/USDT data")

# --------- Step 2: Prepare Data using your prepare_data() function ---------
df = prepare_data(df)
print(f"✅ Prepared {len(df)} rows with features")
# (Optional) Print the tail to inspect features
print(df[['timestamp', 'close', 'returns', 'rolling_mean', 'rolling_std', 'upper_band', 'lower_band']].tail())


# --------- Step 3: Create Labels (Signal) ---------
# Here, we define a simple signal based on returns thresholds.
def get_signal(row):
    # These thresholds can be tuned to get more BUY/SELL signals.
    if row['returns'] > 0.002:
        return 'BUY'
    elif row['returns'] < -0.002:
        return 'SELL'
    else:
        return 'HOLD'

df['signal'] = df.apply(get_signal, axis=1)

# Show label distribution (this helps to see how imbalanced it is)
print("Label distribution before splitting:")
print(df['signal'].value_counts())

# --------- Step 4: Define Features and Labels ---------
features = ['returns', 'rolling_mean', 'rolling_std', 'upper_band', 'lower_band']
X = df[features]
y = df['signal']

# --------- Step 5: Split Data (Stratified) ---------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nBefore oversampling, training label distribution:")
print(y_train.value_counts())

# --------- Step 6: Handle Imbalance via Oversampling ---------
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print("After oversampling, training label distribution:")
print(pd.Series(y_train_res).value_counts())

# --------- Step 7: Train the Model ---------
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

importances = model.feature_importances_
for name, importance in zip(X.columns, importances):
    print(f"{name}: {importance:.4f}")


# Evaluate on test set
y_pred = model.predict(X_test)
print("\n📊 Classification Report on Test Data:\n", classification_report(y_test, y_pred))
print("🎯 Accuracy on Test Data:", accuracy_score(y_test, y_pred))

# --------- Step 8: Save the Model ---------
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
