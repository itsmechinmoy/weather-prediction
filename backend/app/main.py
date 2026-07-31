"""
main.py
-------
FastAPI backend serving:
  GET /predict/grid         -> [{lat, lon, temp}, ...] for the whole India grid (heatmap layer)
  GET /predict/point        -> 7-day hourly + 30-day trend for a clicked lat/lon
  GET /predict/state-avg    -> average predicted temp per state (choropleth)
  GET /accuracy             -> backtested error metrics for the accuracy page

Reads from data/latest_features.parquet (kept fresh by app/update_latest.py)
rather than recomputing from the full historical dataset per request.

Run:
    uvicorn app.main:app --reload
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "roll_mean_24h", "roll_std_24h",
    "temp_ensemble_spread",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
]

app = FastAPI(title="India Weather Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load artifacts once at startup ---------------------------------------
grid_path = os.path.join(DATA_DIR, "india_grid.csv")
if os.path.exists(grid_path):
    grid_df = pd.read_csv(grid_path)
else:
    # Don't crash the whole app over this — endpoints that need it will
    # return a clear 503 instead of the service failing to boot at all.
    print(f"WARNING: {grid_path} not found. Run app/grid.py and commit the output.")
    grid_df = pd.DataFrame(columns=["lat", "lon", "state"])

xgb_model = None
xgb_path = os.path.join(MODEL_DIR, "xgb_7day.pkl")
if os.path.exists(xgb_path):
    xgb_model = joblib.load(xgb_path)

prophet_models = {}
if os.path.isdir(MODEL_DIR):
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("prophet_") and fname.endswith(".pkl"):
            state_name = fname[len("prophet_"):-len(".pkl")].replace("_", " ")
            prophet_models[state_name] = os.path.join(MODEL_DIR, fname)


def load_latest_features() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "latest_features.parquet")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail="Live feature cache not found. Run app/update_latest.py first.",
        )
    return pd.read_parquet(path)


# --- Endpoints --------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "grid_points": len(grid_df),
        "states_with_30day_model": len(prophet_models),
        "xgb_model_loaded": xgb_model is not None,
    }


@app.get("/predict/grid")
def predict_grid():
    """
    Current/next-hour temperature per grid point, for the heatmap layer.
    (Grid layer intentionally shows "now" rather than every hour offset —
    use /predict/point for the full 7-day hourly detail on a single spot.)
    """
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run app/train.py first.")

    latest = load_latest_features()
    preds = xgb_model.predict(latest[FEATURE_COLS])

    points = [
        {
            "lat": float(row.lat),
            "lon": float(row.lon),
            "state": row.state,
            "temp": round(float(pred), 2),
        }
        for row, pred in zip(latest.itertuples(), preds)
    ]
    return {"points": points, "count": len(points)}


@app.get("/predict/point")
def predict_point(lat: float, lon: float):
    """7-day recursive hourly forecast + 30-day trend for the nearest grid point."""
    if grid_df.empty:
        raise HTTPException(status_code=503, detail="Grid not generated yet. Run app/grid.py and commit data/india_grid.csv.")

    grid = grid_df.copy()
    grid["dist"] = (grid["lat"] - lat) ** 2 + (grid["lon"] - lon) ** 2
    nearest = grid.loc[grid["dist"].idxmin()]
    state = nearest["state"]

    response = {
        "requested": {"lat": lat, "lon": lon},
        "nearest_grid_point": {"lat": float(nearest["lat"]), "lon": float(nearest["lon"])},
        "state": state,
        "forecast_7day_hourly": [],
        "forecast_30day_trend": [],
    }

    # --- 7-day hourly, recursive forecasting ---
    if xgb_model is not None:
        latest = load_latest_features()
        row = latest[(latest["lat"] == nearest["lat"]) & (latest["lon"] == nearest["lon"])]
        if not row.empty:
            state_vec = row.iloc[0].to_dict()
            base_time = pd.Timestamp(state_vec["time"])

            preds = []
            # lag_24h / lag_168h held from last known actuals rather than
            # recursively predicted — a simplification that avoids compounding
            # error on the weekly/daily lags, while lag_1h updates every step.
            for h in range(1, 169):  # next 168 hours = 7 days
                future_time = base_time + pd.Timedelta(hours=h)
                feats = {
                    "lag_1h": state_vec["lag_1h"],
                    "lag_24h": state_vec["lag_24h"],
                    "lag_168h": state_vec["lag_168h"],
                    "roll_mean_24h": state_vec["roll_mean_24h"],
                    "roll_std_24h": state_vec["roll_std_24h"],
                    "temp_ensemble_spread": state_vec["temp_ensemble_spread"],
                    "hour_sin": np.sin(2 * np.pi * future_time.hour / 24),
                    "hour_cos": np.cos(2 * np.pi * future_time.hour / 24),
                    "doy_sin": np.sin(2 * np.pi * future_time.dayofyear / 365.25),
                    "doy_cos": np.cos(2 * np.pi * future_time.dayofyear / 365.25),
                    "month": future_time.month,
                }
                x = pd.DataFrame([feats])[FEATURE_COLS]
                pred = float(xgb_model.predict(x)[0])
                preds.append({"time": future_time.isoformat(), "predicted_temp": round(pred, 2)})

                # feed prediction back in as next step's lag_1h
                state_vec["lag_1h"] = pred

            response["forecast_7day_hourly"] = preds

    # --- 30-day trend ---
    if state in prophet_models:
        m = joblib.load(prophet_models[state])
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future).tail(30)
        response["forecast_30day_trend"] = [
            {
                "date": r.ds.strftime("%Y-%m-%d"),
                "predicted_temp": round(r.yhat, 2),
                "lower": round(r.yhat_lower, 2),
                "upper": round(r.yhat_upper, 2),
            }
            for r in forecast.itertuples()
        ]

    return response


@app.get("/predict/state-avg")
def predict_state_avg():
    """Average current predicted temp per state (grid-point average, not population-weighted)."""
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    latest = load_latest_features()
    preds = xgb_model.predict(latest[FEATURE_COLS])
    latest = latest.copy()
    latest["pred_temp"] = preds

    agg = latest.groupby("state")["pred_temp"].mean().round(2).reset_index()
    return {"states": [{"state": r.state, "avg_temp": r.pred_temp} for r in agg.itertuples()]}


@app.get("/accuracy")
def accuracy():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="No metrics saved yet. Run app/train.py first.")
    with open(metrics_path) as f:
        return json.load(f)
