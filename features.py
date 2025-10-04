import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    "Compute technical features and the next-day up/down label."
    data = df.copy()
    close = data["close"]

    # Returns
    data["ret_1d"] = close.pct_change()
    data["log_ret_1d"] = np.log1p(data["ret_1d"])
    data["ret_5d"] = close.pct_change(5)
    data["ret_21d"] = close.pct_change(21)

    # Rolling stats
    data["roll_mean_5"] = close.rolling(5).mean() / close - 1
    data["roll_mean_21"] = close.rolling(21).mean() / close - 1
    data["roll_std_21"] = close.rolling(21).std()

    # TA indicators
    rsi = RSIIndicator(close, window=14)
    data["rsi_14"] = rsi.rsi()

    macd = MACD(close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()

    stoch = StochasticOscillator(high=data["high"], low=data["low"], close=close)
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    bb = BollingerBands(close)
    data["bb_high"] = bb.bollinger_hband() / close - 1
    data["bb_low"] = bb.bollinger_lband() / close - 1
    data["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close

    # Forward return & label
    data["fwd_ret_1d"] = close.pct_change().shift(-1)
    data["target_up"] = (data["fwd_ret_1d"] > 0).astype(int)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data

FEATURE_COLUMNS = [
    "ret_1d","log_ret_1d","ret_5d","ret_21d",
    "roll_mean_5","roll_mean_21","roll_std_21",
    "rsi_14","macd","macd_signal","stoch_k","stoch_d",
    "bb_high","bb_low","bb_width"
]
