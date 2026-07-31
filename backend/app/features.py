"""
features.py
------------
Combines the per-grid-point raw parquet files into one dataset and builds
model-ready features: lags, rolling stats, cyclic date encoding, and the
multi-model ensemble mean/spread.

Run:
    python -m app.features
"""

import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FEATURES_OUT = os.path.join(DATA_DIR, "features.parquet")

MODELS = ["best_match", "ecmwf_ifs", "gfs_seamless", "icon_seamless"]
BASE_VAR = "temperature_2m"


def load_all_points() -> pd.DataFrame:
    files = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw data found in {RAW_DIR}. Run fetch_data.py first.")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df


def add_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/spread across models is often a stronger predictor than any single model."""
    model_cols = [f"{BASE_VAR}_{m}" for m in MODELS if f"{BASE_VAR}_{m}" in df.columns]
    if not model_cols:
        # Fallback: only best_match was returned unsuffixed
        model_cols = [BASE_VAR] if BASE_VAR in df.columns else []

    df["temp_ensemble_mean"] = df[model_cols].mean(axis=1)
    df["temp_ensemble_spread"] = df[model_cols].std(axis=1)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["lat", "lon", "time"])
    group = df.groupby(["lat", "lon"])["temp_ensemble_mean"]

    df["lag_1h"] = group.shift(1)
    df["lag_24h"] = group.shift(24)
    df["lag_168h"] = group.shift(168)  # same hour, previous week

    df["roll_mean_24h"] = group.transform(lambda s: s.rolling(24, min_periods=6).mean())
    df["roll_std_24h"] = group.transform(lambda s: s.rolling(24, min_periods=6).std())

    return df


def add_cyclic_date_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df["time"].dt.hour
    day_of_year = df["time"].dt.dayofyear
    month = df["time"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["month"] = month

    return df


def build_features() -> pd.DataFrame:
    df = load_all_points()
    df = add_ensemble_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_cyclic_date_features(df)

    # Target: next-hour temperature at this grid point
    df["target_temp_next_1h"] = df.groupby(["lat", "lon"])["temp_ensemble_mean"].shift(-1)

    df = df.dropna(subset=["lag_1h", "lag_24h", "lag_168h", "target_temp_next_1h"])
    return df


if __name__ == "__main__":
    df = build_features()
    df.to_parquet(FEATURES_OUT, index=False)
    print(f"Saved {len(df)} feature rows to {FEATURES_OUT}")
    print(df.columns.tolist())
