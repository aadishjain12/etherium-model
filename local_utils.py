import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_coinmarketcap_data():
    url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical"
    params = {
        "id": 1027,  # ETH
        "convertId": 2781,  # USDT
        "timeStart": int((datetime.now().timestamp()) - 86400 * 15),  # last 15 days
        "timeEnd": int(datetime.now().timestamp())
    }
    headers = {
        "Accepts": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, params=params, headers=headers)
    data = r.json()

    prices = data['data']['quotes']
    df = pd.DataFrame([{
        'timestamp': quote['timeOpen'],
        'price': quote['quote']['close']
    } for quote in prices])

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def prepare_data(df):
    df['returns'] = df['price'].pct_change()
    df['rolling_mean'] = df['price'].rolling(window=3).mean()
    df['rolling_std'] = df['price'].rolling(window=3).std()
    df.dropna(inplace=True)
    return df

def extract_features(df):
    latest = df.iloc[-1]
    features = {
        'latest_return': latest['returns'],
        'rolling_mean': latest['rolling_mean'],
        'rolling_std': latest['rolling_std']
    }
    return features

def plot_backtest(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['price'], label='Price', linewidth=2)
    plt.plot(df.index, df['rolling_mean'], label='Rolling Mean (3)', linestyle='--')
    plt.fill_between(df.index,
                     df['rolling_mean'] - df['rolling_std'],
                     df['rolling_mean'] + df['rolling_std'],
                     color='gray', alpha=0.2, label='Rolling ±1 STD')
    plt.title('ETH/USDT Price with Rolling Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
