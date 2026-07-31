"""
update_latest.py
-----------------
Keeps a small "current conditions" feature cache fresh for every grid point,
using Open-Meteo's forecast endpoint (which also returns recent past days),
instead of the full multi-year archive. This is what predict_grid() and
predict_point() read from — cheap to run frequently (e.g. hourly via cron),
unlike recomputing features.py's full historical join on every API request.

Run manually:
    python -m app.update_latest

In production, schedule this hourly (cron on the Render service, or a
Render Cron Job hitting the same script) so the site reflects fresh data.
"""

import os
import time

import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GRID_FILE = os.path.join(DATA_DIR, "india_grid.csv")
LATEST_OUT = os.path.join(DATA_DIR, "latest_features.parquet")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "pressure_msl", "wind_speed_10m", "cloud_cover",
]
MODELS = ["best_match", "ecmwf_ifs", "gfs_seamless", "icon_seamless"]
BASE_VAR = "temperature_2m"

PAST_DAYS = 8       # need 168h (7 days) of lag history, +1 day buffer
FORECAST_DAYS = 7   # also grab the model's own 7-day forecast as a feature/fallback


def fetch_point_latest(lat: float, lon: float, retries: int = 3) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "models": ",".join(MODELS),
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(FORECAST_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Failed to fetch latest data for {lat},{lon}")

    hourly = payload["hourly"]
    df = pd.DataFrame({"time": hourly["time"]})
    for key, values in hourly.items():
        if key == "time":
            continue
        df[key] = values
    df["lat"] = lat
    df["lon"] = lon
    return df


def compute_features_for_point(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature logic as app/features.py, applied to this point's recent window."""
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    model_cols = [f"{BASE_VAR}_{m}" for m in MODELS if f"{BASE_VAR}_{m}" in df.columns]
    if not model_cols:
        model_cols = [BASE_VAR] if BASE_VAR in df.columns else []
    df["temp_ensemble_mean"] = df[model_cols].mean(axis=1)
    df["temp_ensemble_spread"] = df[model_cols].std(axis=1)

    s = df["temp_ensemble_mean"]
    df["lag_1h"] = s.shift(1)
    df["lag_24h"] = s.shift(24)
    df["lag_168h"] = s.shift(168)
    df["roll_mean_24h"] = s.rolling(24, min_periods=6).mean()
    df["roll_std_24h"] = s.rolling(24, min_periods=6).std()

    hour = df["time"].dt.hour
    doy = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df["time"].dt.month

    return df


def main():
    grid = pd.read_csv(GRID_FILE)
    print(f"Refreshing latest features for {len(grid)} grid points...")

    all_latest = []
    for i, row in grid.iterrows():
        try:
            raw = fetch_point_latest(row.lat, row.lon)
            feats = compute_features_for_point(raw)
            # Keep only the most recent fully-lagged row = "now"
            now_row = feats.dropna(subset=["lag_1h", "lag_24h", "lag_168h"]).tail(1).copy()
            if now_row.empty:
                continue
            now_row["state"] = row.state
            all_latest.append(now_row)
        except Exception as exc:
            print(f"  skipping ({row.lat},{row.lon}): {exc}")
        if i % 20 == 0:
            print(f"  {i + 1}/{len(grid)} done")
        time.sleep(0.1)

    result = pd.concat(all_latest, ignore_index=True)
    result.to_parquet(LATEST_OUT, index=False)
    print(f"Saved {len(result)} live grid-point feature rows to {LATEST_OUT}")


if __name__ == "__main__":
    main()
