import requests
import pandas as pd

def fetch_ohlcv_binance(symbol="ETHUSDT", interval="5m", limit=10000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore']

        df = pd.DataFrame(data, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        print("❌ Error fetching Binance data:", e)
        return None

# Test it directly
if __name__ == "__main__":
    df = fetch_ohlcv_binance()
    if df is not None:
        print("✅ Data fetched successfully:\n", df.tail())
    else:
        print("❌ No data returned.")
