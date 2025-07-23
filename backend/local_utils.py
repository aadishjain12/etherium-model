import pandas as pd
import numpy as np


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Feature Engineering
    df = add_technical_features(df)

    # Drop NaNs after rolling
    df.dropna(inplace=True)
    return df


def add_technical_features(df):
    """Feature engineering using Bollinger Bands and returns"""
    df['returns'] = df['close'].pct_change()
    df['rolling_mean'] = df['close'].rolling(window=10).mean()  # Reduced for 5m data
    df['rolling_std'] = df['close'].rolling(window=10).std()
    df['upper_band'] = df['rolling_mean'] + 2 * df['rolling_std']
    df['lower_band'] = df['rolling_mean'] - 2 * df['rolling_std']
    return df


def create_labels(df, threshold=0.001):  # More sensitive threshold for 5m data
    df['future_return'] = df['close'].pct_change().shift(-1)
    df['label'] = 0  # Default Hold

    df.loc[df['future_return'] > threshold, 'label'] = 1   # Buy
    df.loc[df['future_return'] < -threshold, 'label'] = -1  # Sell

    df.drop(columns=['future_return'], inplace=True)
    return df


def preprocess_live_data(df):
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Reuse same features as training
    df = add_technical_features(df)
    df.dropna(inplace=True)

    features = df[['returns', 'rolling_mean', 'rolling_std', 'upper_band', 'lower_band']].iloc[-1:]
    print("Live features:", features.columns.tolist())

    return features


def predict_signal(model, features):
    proba = model.predict_proba(features)[0]
    predicted_label = np.argmax(proba)  # 0 = SELL, 1 = HOLD, 2 = BUY
    label_map = {0: -1, 1: 0, 2: 1}  # Match your original labels
    inverse_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    prediction = label_map[predicted_label]
    predicted_class = inverse_map[prediction]

    probabilities = {
        "SELL": proba[0],
        "HOLD": proba[1],
        "BUY": proba[2]
    }

    print(f"🧠 Prediction probabilities: [Sell: {proba[0]:.4f}, Hold: {proba[1]:.4f}, Buy: {proba[2]:.4f}]")
    print(f"🔮 Predicted label: {prediction} ({predicted_class})")

    return prediction, probabilities


def display_output(signal, price, time, initial_balance, previous_price, probabilities):
    buy_prob = probabilities.get('BUY', 0)
    sell_prob = probabilities.get('SELL', 0)

    if signal == "BUY":
        icon = "🟢"
        prob = f"(↑ {buy_prob:.2f})"
    elif signal == "SELL":
        icon = "🔴"
        prob = f"(↓ {sell_prob:.2f})"
    else:
        icon = "⚪"
        prob = ""

    print(f"🕒 {time} | ${price:.2f} | {icon} {signal} {prob} | 💼 ${initial_balance:.2f}")


def print_label_distribution(df):
    print("✅ Label distribution:")
    print(df['label'].value_counts())
