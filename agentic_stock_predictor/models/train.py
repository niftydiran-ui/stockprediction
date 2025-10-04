import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS

def get_model(name: str):
    if name == "linear":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=200, n_jobs=None))
        ])
    elif name == "xgb":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate=0.05,
            reg_lambda=1.0,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def fit_predict_proba(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1]
    return model, proba

def evaluate_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")
