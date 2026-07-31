"""
train.py
--------
Trains two separate models:
  1. XGBoost regressor for 7-day hourly forecasting (real ML forecast).
  2. Prophet for 30-day daily trend outlook (statistical/seasonal projection,
     explicitly NOT treated as a precise hourly forecast).

Both are trained per-state on the state-averaged series, since a single
national model would blur very different climates (Rajasthan vs Kerala).

Run:
    python -m app.train
"""

import os

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_FILE = os.path.join(DATA_DIR, "features.parquet")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "roll_mean_24h", "roll_std_24h",
    "temp_ensemble_spread",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
]
TARGET_COL = "target_temp_next_1h"

TEST_HOLDOUT_DAYS = 180  # last ~6 months held out, chronological — no shuffling


def chronological_split(df: pd.DataFrame):
    cutoff = df["time"].max() - pd.Timedelta(days=TEST_HOLDOUT_DAYS)
    train = df[df["time"] < cutoff]
    test = df[df["time"] >= cutoff]
    return train, test


def train_xgboost_7day(df: pd.DataFrame):
    """One model trained across all grid points (lat/lon implicitly captured
    via the ensemble/lag features which already reflect local conditions)."""
    train, test = chronological_split(df)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLS], train[TARGET_COL])

    preds = model.predict(test[FEATURE_COLS])
    mae = mean_absolute_error(test[TARGET_COL], preds)
    rmse = np.sqrt(mean_squared_error(test[TARGET_COL], preds))
    print(f"[7-day XGBoost] MAE={mae:.3f}°C  RMSE={rmse:.3f}°C  (n_test={len(test)})")

    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_7day.pkl"))
    return model, {"mae": mae, "rmse": rmse}


def train_prophet_30day_per_state(df: pd.DataFrame):
    """Separate lightweight Prophet model per state, on the daily state-average
    series. Framed as a trend outlook, not an hourly forecast."""
    daily_state = (
        df.groupby(["state", df["time"].dt.date])["temp_ensemble_mean"]
        .mean()
        .reset_index()
        .rename(columns={"time": "ds", "temp_ensemble_mean": "y"})
    )
    daily_state["ds"] = pd.to_datetime(daily_state["ds"])

    results = {}
    for state, group in daily_state.groupby("state"):
        group = group[["ds", "y"]].sort_values("ds")
        if len(group) < 60:
            continue  # not enough history for a meaningful seasonal fit

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.8,  # gives us the confidence band for the UI
        )
        m.fit(group)

        model_path = os.path.join(MODEL_DIR, f"prophet_{state.replace(' ', '_')}.pkl")
        joblib.dump(m, model_path)
        results[state] = model_path
        print(f"[30-day Prophet] trained for {state} ({len(group)} daily points)")

    return results


if __name__ == "__main__":
    df = pd.read_parquet(FEATURES_FILE)
    df["time"] = pd.to_datetime(df["time"])

    print("=== Training 7-day XGBoost model ===")
    train_xgboost_7day(df)

    print("\n=== Training 30-day Prophet models (per state) ===")
    train_prophet_30day_per_state(df)

    print("\nAll models saved to", MODEL_DIR)
