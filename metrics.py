import numpy as np

def compute_metrics(strategy_returns):
    arr = strategy_returns.dropna().values
    if len(arr) == 0:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_drawdown": float("nan"), "win_rate": float("nan")}

    # Daily to annual
    daily_mean = np.mean(arr)
    daily_std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else float("nan")

    equity = (1.0 + arr).cumprod()
    peak = np.maximum.accumulate(equity)
    dd = equity/peak - 1.0
    max_dd = float(np.min(dd))

    # CAGR
    total_return = equity[-1] - 1.0
    years = len(arr)/252.0
    cagr = float((equity[-1])**(1.0/years) - 1.0) if years > 0 and equity[-1] > 0 else float("nan")

    win_rate = float(np.mean(arr > 0))

    return {"cagr": cagr, "sharpe": float(sharpe), "max_drawdown": max_dd, "win_rate": win_rate}
