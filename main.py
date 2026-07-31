"""
main.py
-------
FastAPI backend serving:
  GET /predict/grid         -> [{lat, lon, temp}, ...] for the whole India grid (heatmap layer)
  GET /predict/point        -> 7-day hourly + 30-day trend for a clicked lat/lon
  GET /predict/state-avg    -> area-weighted-ish average temp per state (choropleth)
  GET /accuracy             -> backtested error metrics for the accuracy page

Run:
    uvicorn app.main:app --reload
"""

import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="India Weather Prediction API")

# Allow the frontend (served separately, e.g. Vercel/Netlify) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load artifacts once at startup ---------------------------------------
grid_df = pd.read_csv(os.path.join(DATA_DIR, "india_grid.csv"))

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
def predict_grid(hour_offset: int = Query(0, description="Hours from now, e.g. 24 for tomorrow this time")):
    """
    Returns a temperature value per grid point for the heatmap layer.

    NOTE: this is a simplified placeholder — a production version would run
    xgb_model.predict() per point using that point's latest lag/rolling
    features (pulled from a live-updating feature store), not a flat mock.
    Wire this up to your feature pipeline once fetch_data.py has been run
    on a schedule and features.py is recomputed incrementally.
    """
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run app/train.py first.")

    # Placeholder response shape — replace `temp` with real per-point inference
    results = [
        {"lat": row.lat, "lon": row.lon, "state": row.state, "temp": None}
        for row in grid_df.itertuples()
    ]
    return {"hour_offset": hour_offset, "points": results, "count": len(results)}


@app.get("/predict/point")
def predict_point(lat: float, lon: float):
    """7-day hourly + 30-day trend for the nearest grid point to a clicked location."""
    if grid_df.empty:
        raise HTTPException(status_code=503, detail="Grid not generated yet. Run app/grid.py first.")

    # Snap to nearest grid point (simple Euclidean nearest-neighbour is fine at this grid density)
    grid_df["dist"] = ((grid_df["lat"] - lat) ** 2 + (grid_df["lon"] - lon) ** 2)
    nearest = grid_df.loc[grid_df["dist"].idxmin()]
    state = nearest["state"]

    response = {
        "requested": {"lat": lat, "lon": lon},
        "nearest_grid_point": {"lat": nearest["lat"], "lon": nearest["lon"]},
        "state": state,
        "forecast_7day_hourly": [],  # fill via xgb_model.predict() looped over next 168 hours
        "forecast_30day_trend": [],
    }

    if state in prophet_models:
        m = joblib.load(prophet_models[state])
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future).tail(30)
        response["forecast_30day_trend"] = [
            {
                "date": row.ds.strftime("%Y-%m-%d"),
                "predicted_temp": round(row.yhat, 2),
                "lower": round(row.yhat_lower, 2),
                "upper": round(row.yhat_upper, 2),
            }
            for row in forecast.itertuples()
        ]

    return response


@app.get("/predict/state-avg")
def predict_state_avg():
    """Average predicted temp per state, for the choropleth toggle view."""
    # Placeholder aggregation shape; wire to real per-point predictions once
    # predict_grid() returns actual inferred values.
    states = grid_df["state"].unique().tolist()
    return {"states": [{"state": s, "avg_temp": None} for s in states]}


@app.get("/accuracy")
def accuracy():
    """Backtested error metrics, computed and saved by app/train.py, for the site's accuracy page."""
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="No metrics saved yet. Run app/train.py first.")
    import json
    with open(metrics_path) as f:
        return json.load(f)
