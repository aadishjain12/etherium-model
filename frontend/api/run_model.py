
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from joblib import load
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data_fetcher import fetch_ohlcv_binance
from local_utils import preprocess_live_data

app = FastAPI()

model = load("model.pkl")

@app.get("/api/run_model")
def run_model():
    df = fetch_ohlcv_binance('ETHUSDT', interval='1h', lookback='2 days ago UTC')
    if df is None or df.empty:
        return JSONResponse(content={"error": "Failed to fetch data"}, status_code=500)

    X_live = preprocess_live_data(df)
    prediction = model.predict(X_live.tail(1))[0]
    return {"prediction": prediction}
