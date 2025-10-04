# Agentic Stock Predictor (Python)

An end‑to‑end starter project for a **stock prediction AI agent** in Python. It:
- Pulls historical prices (Yahoo Finance via `yfinance`)
- Builds technical features (RSI, MACD, Bollinger, rolling stats)
- Trains a classifier (XGBoost by default) to predict **next‑day up/down**
- Runs a simple walk‑forward backtest with a rules‑based agent
- Outputs metrics and plots (equity curve & drawdown)

> This is a learning scaffold—not financial advice. Past performance ≠ future results.

## Quickstart

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
python main.py --tickers AAPL MSFT NVDA --period 5y --model xgb --threshold 0.55
```

Artifacts are saved under `./artifacts/<TICKER>/`:
- `metrics.json` – summary stats
- `equity_curve.png`, `drawdown.png` – charts
- `predictions.csv` – daily predictions & positions

## Design

- `agents/agent.py` – A tiny goal‑directed loop that plans → acts (train, backtest) → reflects.
- `agents/tools.py` – Tools to fetch prices and lightweight news.
- `models/features.py` – Feature engineering.
- `models/train.py` – Model training & inference helpers.
- `backtest/backtest.py` – Walk‑forward backtest & metrics.
- `main.py` – CLI entry point.
- `config.py` – Tunables (train/test splits, seeds, etc.).

## Notes
- You can switch to `--model linear` to avoid XGBoost.
- yfinance free data may throttle; retry logic is included.
- Feel free to extend the agent with additional tools (options flow, macro series, risk parity, etc.).
