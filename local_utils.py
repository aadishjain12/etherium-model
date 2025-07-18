import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(df, window=5):
    df['returns'] = df['close'].pct_change()
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df = df.dropna()
    return df

def extract_features(df):
    latest = df.iloc[-1]
    return {
        'latest_return': latest['returns'],
        'rolling_mean': latest['rolling_mean'],
        'rolling_std': latest['rolling_std']
    }

def predict_signal(features):
    # Simple logic-based prediction
    if features['latest_return'] > 0 and features['latest_return'] > features['rolling_std']:
        return 'buy'
    elif features['latest_return'] < 0 and abs(features['latest_return']) > features['rolling_std']:
        return 'sell'
    else:
        return 'hold'

def simulate_portfolio(df, initial_balance=1000):
    balance = initial_balance
    position = 0
    signal_history = []
    price_history = []
    correct_predictions = 0
    total_predictions = 0

    df = prepare_data(df)

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # Extract features up to current index
        window_df = df.iloc[:i]
        features = extract_features(window_df)
        signal = predict_signal(features)

        signal_history.append(signal)
        price_history.append(current_row['close'])

        price_change = current_row['returns']

        # For accuracy tracking
        actual_movement = 'buy' if price_change > 0 else 'sell' if price_change < 0 else 'hold'
        if signal in ['buy', 'sell']:
            if signal == actual_movement:
                correct_predictions += 1
            total_predictions += 1

        # Simulate trading
        if signal == 'buy' and balance > 0:
            position = balance / current_row['close']
            balance = 0
        elif signal == 'sell' and position > 0:
            balance = position * current_row['close']
            position = 0

    # Final value
    final_value = balance + (position * df.iloc[-1]['close'])
    df['signal'] = [''] * (len(df) - len(signal_history)) + signal_history

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"\n📈 Final Portfolio Value: ${final_value:.2f}")
    print(f"✅ Model Accuracy (directional): {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions} correct predictions)")

    return df

def generate_signals(df, short_window=5, long_window=20):
    """
    Generates buy/sell signals based on moving average crossover.
    Adds a 'signal' column to the dataframe with 'buy', 'sell', or None.
    """
    df = df.copy()
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()

    # Generate signal: 1 for buy, -1 for sell, 0 for hold
    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1

    # Only mark points where signal changes
    df['signal_shifted'] = df['signal'].shift(1)
    df['action'] = None
    df.loc[(df['signal'] == 1) & (df['signal_shifted'] != 1), 'action'] = 'buy'
    df.loc[(df['signal'] == -1) & (df['signal_shifted'] != -1), 'action'] = 'sell'

    df.drop(columns=['signal', 'signal_shifted'], inplace=True)
    df.rename(columns={'action': 'signal'}, inplace=True)

    return df



def plot_backtest(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['close'], label='Price', linewidth=2)
    buy_signals = df[df['signal'] == 'buy']
    sell_signals = df[df['signal'] == 'sell']
    plt.scatter(buy_signals['date'], buy_signals['close'], marker='^', color='green', label='Buy Signal')
    plt.scatter(sell_signals['date'], sell_signals['close'], marker='v', color='red', label='Sell Signal')
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
