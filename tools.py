import time
import pandas as pd
import yfinance as yf

def fetch_prices(ticker: str, period: str = "5y", retries: int = 3, pause: float = 1.5) -> pd.DataFrame:
    "Download OHLCV daily data; returns a dataframe indexed by date."
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.rename(columns=str.lower)
                df = df.dropna()
                return df
        except Exception as e:
            last_err = e
        time.sleep(pause)
    raise RuntimeError(f"Failed to fetch data for {ticker}: {last_err}")

def fetch_news_titles(ticker: str, max_items: int = 20):
    "Lightweight news via yfinance.Ticker.news (if available)."
    try:
        news = yf.Ticker(ticker).news or []
        titles = [n.get("title","") for n in news if n.get("title")]
        return titles[:max_items]
    except Exception:
        return []
