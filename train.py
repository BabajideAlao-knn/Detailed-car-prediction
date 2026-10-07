"""Train the final car-price model and save it for the Streamlit app.

Reproduces the best model from the notebook (tuned XGBoost, 70/30 split, random_state=42).

    python train.py
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from carprice import CAT, NUM, add_features, clean_raw

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "autochek_cars_ng.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "car_price_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.02, max_depth=6, min_child_weight=3, subsample=0.8,
                  colsample_bytree=1.0, n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)


def build_pipeline() -> TransformedTargetRegressor:
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), CAT),
    ])
    return TransformedTargetRegressor(regressor=Pipeline([("prep", preprocess), ("model", XGBRegressor(**XGB_PARAMS))]),
                                      func=np.log, inverse_func=np.exp)


def train(save: bool = True):
    df, display_names = clean_raw(pd.read_csv(DATA_RAW), return_display_names=True)
    df = add_features(df)
    X, y = df[NUM + CAT], df["price_ngn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)

    model = build_pipeline().fit(X_train, y_train)
    pred = model.predict(X_test)
    ape = np.abs(pred - y_test) / y_test
    log_ratio = np.log(y_test / pred)  # actual vs predicted, used for the price range shown in the app

    meta = {
        "model": "XGBoost (tuned)",
        "trained_on_rows": int(len(X_train)), "tested_on_rows": int(len(X_test)),
        "metrics": {"r2": r2_score(y_test, pred), "r2_log": r2_score(np.log(y_test), np.log(pred)),
                    "mae": mean_absolute_error(y_test, pred), "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "mape": mean_absolute_percentage_error(y_test, pred), "median_ape": float(np.median(ape)),
                    **{f"within_{t}pct": float((ape <= t / 100).mean()) for t in (10, 20, 30)}},
        # 80% of test-set listings fell between pred*exp(q10) and pred*exp(q90)
        "interval_log_ratio": {"q10": float(np.quantile(log_ratio, 0.10)), "q90": float(np.quantile(log_ratio, 0.90))},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "display_names": display_names,
    }
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH, compress=3)
        META_PATH.write_text(json.dumps(meta, indent=1))
    return model, meta


if __name__ == "__main__":
    _, m = train()
    k = m["metrics"]
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}  |  test MAPE {k['mape']:.1%}  median error {k['median_ape']:.1%}  "
          f"R2(log) {k['r2_log']:.3f}  within ±20%: {k['within_20pct']:.0%}")
