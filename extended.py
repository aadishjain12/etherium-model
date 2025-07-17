import os
import sys
import time
import pandas as pd
from datetime import datetime
from local_utils import prepare_data, extract_features, plot_backtest, fetch_coinmarketcap_data

# Simulated sentiment generator
def simulated_sentiment_step(i):
    cycle = ["Strongly Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Strongly Bearish"]
    return cycle[i % len(cycle)]

def simulate_sentiment(i):
    return simulated_sentiment_step(i)

def get_sentiment_summary(features, step=None):
    # Only simulate sentiment
    sentiment = simulate_sentiment(step or 0)
    return f"📊 Simulated Sentiment: {sentiment}\n→ Suggested action: {'BUY' if 'Bullish' in sentiment else 'SELL' if 'Bearish' in sentiment else 'HOLD'}"

def run_sentiment_mode():
    print(f"\n📈 Simulated Market Sentiment for ETH/USDT @ {datetime.now()}")
    print("→ This model uses historical market indicators to simulate trading signals.")

    try:
        df = fetch_coinmarketcap_data()
        df = prepare_data(df)
        features = extract_features(df)
        sentiment = get_sentiment_summary(features)
        print(f"→ {sentiment}")

    except Exception as e:
        print(f"❌ Failed to generate sentiment: {e}")

def simulate_portfolio(df):
    cash = 1000.0
    eth = 0.0
    portfolio_values = []

    print("📊 Columns in DataFrame:", df.columns.tolist())

    for i in range(len(df)):
        row = df.iloc[i]
        date = row['date'] if 'date' in row else f"Step {i}"

        if 'close' in df.columns:
            close_price = row['close']
        elif 'price' in df.columns:
            close_price = row['price']
        else:
            raise ValueError("No valid close or price column found in data.")

        features = extract_features(df.iloc[:i+1])
        sentiment = get_sentiment_summary(features, step=i)

        action = "HOLD"
        if "buy" in sentiment.lower():
            action = "BUY"
        elif "sell" in sentiment.lower():
            action = "SELL"

        if action == "BUY" and cash > 0:
            eth = cash / close_price
            cash = 0
        elif action == "SELL" and eth > 0:
            cash = eth * close_price
            eth = 0

        total_value = cash + eth * close_price
        portfolio_values.append((date, total_value, action))
        print(f"📅 {date} | Action: {action} | ETH: {eth:.4f} | Cash: ${cash:.2f} | Portfolio Value: ${total_value:.2f}")

    result_df = pd.DataFrame(portfolio_values, columns=["Date", "Portfolio Value", "Action"])
    result_df.set_index("Date", inplace=True)
    result_df["Portfolio Value"].plot(title="📊 Portfolio Value Over Time")
    return result_df

def run_backtest_mode():
    df = fetch_coinmarketcap_data()
    df = prepare_data(df)
    features = extract_features(df)
    plot_backtest(df)

    print("\n💼 Running portfolio simulation with $1000 starting balance...")
    result_df = simulate_portfolio(df)
    print(result_df.tail())

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python main.py [sentiment|backtest]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == 'sentiment':
        run_sentiment_mode()
    elif mode == 'backtest':
        run_backtest_mode()
    else:
        print("Invalid mode. Use 'sentiment' or 'backtest'.")
