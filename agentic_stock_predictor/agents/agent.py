import os, json
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
from rich import print as rprint

from .tools import fetch_prices, fetch_news_titles
from ..models.features import make_features, FEATURE_COLUMNS
from ..models.train import get_model, fit_predict_proba, evaluate_auc
from ..backtest.backtest import walk_forward, save_artifacts
from ..config import Config

@dataclass
class Agent:
    model_name: str = "xgb"
    threshold: float = 0.55
    out_dir: str = "artifacts"

    def run(self, ticker: str, period: str = "5y") -> Dict[str, Any]:
        cfg = Config()
        art_dir = os.path.join(self.out_dir, ticker)
        os.makedirs(art_dir, exist_ok=True)

        # 1) Observe: data + simple news context
        rprint("[bold]Observing market data...[/]")
        prices = fetch_prices(ticker, period=period)
        news = fetch_news_titles(ticker, max_items=10)

        # 2) Think: feature engineering, model choice
        rprint("[bold]Engineering features...[/]")
        df = make_features(prices)
        feature_cols = FEATURE_COLUMNS

        # Decide minimal start based on config
        min_train = max(cfg.min_train_days, 200)

        # 3) Act: walk-forward training + predictions
        rprint(f"[bold]Training {self.model_name.upper()} with walk-forward backtest...[/]")
        model = get_model(self.model_name)

        def fit_fn(X_train, y_train, X_test):
            return fit_predict_proba(model, X_train, y_train, X_test)

        preds, metrics = walk_forward(
            df, feature_cols,
            threshold=self.threshold,
            min_train_days=min_train,
            step_days=cfg.step_days,
            transaction_cost_bp=cfg.transaction_cost_bp,
            fit_fn=fit_fn
        )

        # 4) Reflect: basic validation
        auc = evaluate_auc(df.loc[preds.index, "target_up"], preds["p_up"])
        metrics["auc"] = float(auc)

        # Annotate news in report (for human context)
        report = {
            "ticker": ticker,
            "n_samples": int(len(df)),
            "metrics": metrics,
            "recent_news": news[:5],
            "artifact_dir": art_dir
        }

        save_artifacts(art_dir, preds, metrics)
        with open(os.path.join(art_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return report
