import os
import sys
import time
import pandas as pd
from datetime import datetime
from local_utils import prepare_data, extract_features, plot_backtest, generate_signals, simulate_portfolio
import requests

# Fetch historical candle data from Binance
def fetch_binance_candle_data(symbol="ETHUSDT", interval="1h", limit=100):
    url = f'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()

    if isinstance(data, dict) and data.get("code"):
        raise ValueError(f"Binance API error: {data['msg']}")

    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])

    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df[['date', 'close']]

# Fetch hourly historical ETH prices from CoinMarketCap
def fetch_coinmarketcap_data(api_key):
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical'
    symbol = 'ETH'

    params = {
        'symbol': symbol,
        'interval': 'hourly',
        'count': 100
    }

    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if 'data' not in data or 'quotes' not in data['data']:
        raise ValueError("Invalid API response from CoinMarketCap")

    quotes = data['data']['quotes']
    df = pd.DataFrame([{
        'date': q['timestamp'],
        'close': q['quote']['USD']['price']
    } for q in quotes])
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = df['close'].astype(float)
    return df[['date', 'close']]

# Simulated sentiment generator
def simulated_sentiment_step(i):
    cycle = ["Strongly Bullish", "Slightly Bullish", "Neutral", "Slightly Bearish", "Strongly Bearish"]
    return cycle[i % len(cycle)]

def simulate_sentiment(i):
    return simulated_sentiment_step(i)

def get_sentiment_summary(features, step=None):
    sentiment = simulate_sentiment(step or 0)
    return f"\U0001F4CA Simulated Sentiment: {sentiment}\n→ Suggested action: {'BUY' if 'Bullish' in sentiment else 'SELL' if 'Bearish' in sentiment else 'HOLD'}"

def run_sentiment_mode():
    print(f"\n\U0001F4C8 Simulated Market Sentiment for ETH/USDT @ {datetime.now()}")
    print("→ This model uses historical market indicators to simulate trading signals.")
    try:
        df = fetch_coinmarketcap_data(api_key="cf1a3abc-4e7a-4f73-8c57-13fc66fb6a76")
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

    print("\U0001F4CA Columns in DataFrame before simulation:", df.columns.tolist())

    if 'close' not in df.columns:
        raise ValueError("No 'close' column found in data.")

    for i in range(len(df)):
        row = df.iloc[i]
        date = row['date']

        close_price = row['close']
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
    result_df["Portfolio Value"].plot(title="\U0001F4CA Portfolio Value Over Time")
    return result_df

def run_backtest_mode(source="binance"):
    if source == "binance":
        df = fetch_binance_candle_data()
    elif source == "coinmarketcap":
        df = fetch_coinmarketcap_data(api_key="cf1a3abc-4e7a-4f73-8c57-13fc66fb6a76")
    else:
        raise ValueError("Invalid data source. Use 'binance' or 'coinmarketcap'.")

    print(f"✅ Fetched {len(df)} rows from {source}.")
    print(f"🧾 Initial columns: {df.columns.tolist()}")

    # Prepare and validate data
    df = prepare_data(df)
    print("✅ Columns after prepare_data:", df.columns.tolist())

    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column after prepare_data(). Check your pipeline.")

    # Extract features (used for any future ML-based model)
    features = extract_features(df)

    df = generate_signals(df)

    print("\n📊 Plotting backtest results...")
    plot_backtest(df)  # now also prints accuracy

    print("\n💼 Running portfolio simulation with $1000 starting balance...")
    result_df = simulate_portfolio(df)  # also includes accuracy if available
    print(result_df.tail())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extended.py [sentiment|backtest] [binance|coinmarketcap]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    source = sys.argv[2].lower() if len(sys.argv) > 2 else "coinmarketcap"

    if mode == 'sentiment':
        run_sentiment_mode()
    elif mode == 'backtest':
        run_backtest_mode(source)
    else:
        print("Invalid mode. Use 'sentiment' or 'backtest'.")
