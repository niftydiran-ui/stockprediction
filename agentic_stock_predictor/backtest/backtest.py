import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import compute_metrics

def walk_forward(
    data: pd.DataFrame,
    feature_cols,
    threshold: float = 0.55,
    min_train_days: int = 400,
    step_days: int = 1,
    transaction_cost_bp: float = 2.0,
    fit_fn=None
):
    rows = []
    for i in range(min_train_days, len(data)-1, step_days):
        train = data.iloc[:i]
        test_row = data.iloc[i:i+1]

        X_train = train[feature_cols].values
        y_train = train["target_up"].values
        X_test = test_row[feature_cols].values

        model, proba = fit_fn(X_train, y_train, X_test)
        p_up = float(proba[0])
        position = 1.0 if p_up >= threshold else 0.0
        rows.append({
            "date": test_row.index[0],
            "p_up": p_up,
            "position": position,
            "fwd_ret_1d": float(test_row["fwd_ret_1d"].values[0])
        })
    df_pred = pd.DataFrame(rows).set_index("date")

    # Apply P&L with simple cost on position changes
    df_pred["position_prev"] = df_pred["position"].shift(1).fillna(0.0)
    df_pred["turnover"] = (df_pred["position"] - df_pred["position_prev"]).abs()
    cost = (transaction_cost_bp / 10000.0) * df_pred["turnover"]
    df_pred["strategy_ret"] = df_pred["position"] * df_pred["fwd_ret_1d"] - cost
    df_pred["cum_equity"] = (1.0 + df_pred["strategy_ret"]).cumprod()

    metrics = compute_metrics(df_pred["strategy_ret"])
    return df_pred, metrics

def save_artifacts(artifact_dir: str, preds: pd.DataFrame, metrics: dict):
    os.makedirs(artifact_dir, exist_ok=True)
    preds.to_csv(os.path.join(artifact_dir, "predictions.csv"))
    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Equity curve
    plt.figure()
    preds["cum_equity"].plot(title="Equity Curve")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "equity_curve.png"))
    plt.close()

    # Drawdown
    roll_max = preds["cum_equity"].cummax()
    dd = preds["cum_equity"]/roll_max - 1.0
    plt.figure()
    dd.plot(title="Drawdown")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, "drawdown.png"))
    plt.close()
