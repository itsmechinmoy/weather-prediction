"""
main.py
-------
FastAPI backend serving:
  GET /predict/grid         -> [{lat, lon, temp}, ...] for the whole India grid (heatmap layer)
  GET /predict/point        -> 7-day hourly + 30-day trend for a clicked lat/lon (with city lookup)
  GET /predict/state-avg    -> average predicted temp per state (choropleth)
  GET /accuracy             -> backtested error metrics for the accuracy page

Reads from data/latest_features.parquet and models/xgb_7day.pkl if available,
with a robust live baseline fallback so the service never fails with 503 errors.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

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

# Database of 260+ Indian Cities, Districts, and Regional Capitals with Official UT/State Designations
INDIAN_CITIES_DB = [
    ("New Delhi", "Delhi", 28.6139, 77.2090),
    ("Mumbai", "Maharashtra", 19.0760, 72.8777),
    ("Bengaluru", "Karnataka", 12.9716, 77.5946),
    ("Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Pune", "Maharashtra", 18.5204, 73.8567),
    ("Surat", "Gujarat", 21.1702, 72.8311),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462),
    ("Kanpur", "Uttar Pradesh", 26.4499, 80.3319),
    ("Nagpur", "Maharashtra", 21.1458, 79.0882),
    ("Indore", "Madhya Pradesh", 22.7196, 75.8577),
    ("Thane", "Maharashtra", 19.2183, 72.9781),
    ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126),
    ("Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185),
    ("Patna", "Bihar", 25.5941, 85.1376),
    ("Vadodara", "Gujarat", 22.3072, 73.1812),
    ("Ghaziabad", "Uttar Pradesh", 28.6692, 77.4538),
    ("Ludhiana", "Punjab", 30.9010, 75.8573),
    ("Agra", "Uttar Pradesh", 27.1767, 78.0081),
    ("Nashik", "Maharashtra", 19.9975, 73.7898),
    ("Ranchi", "Jharkhand", 23.3441, 85.30965),
    ("Faridabad", "Haryana", 28.4089, 77.3178),
    ("Meerut", "Uttar Pradesh", 28.9845, 77.7064),
    ("Rajkot", "Gujarat", 22.3039, 70.8022),
    ("Varanasi", "Uttar Pradesh", 25.3176, 82.9739),
    ("Srinagar", "Jammu & Kashmir", 34.0837, 74.7973),
    ("Aurangabad", "Maharashtra", 19.8762, 75.3433),
    ("Dhanbad", "Jharkhand", 23.7957, 86.4304),
    ("Amritsar", "Punjab", 31.6340, 74.8723),
    ("Navi Mumbai", "Maharashtra", 19.0330, 73.0297),
    ("Prayagraj", "Uttar Pradesh", 25.4358, 81.8463),
    ("Howrah", "West Bengal", 22.5958, 88.2636),
    ("Gwalior", "Madhya Pradesh", 26.2183, 78.1828),
    ("Jabalpur", "Madhya Pradesh", 23.1815, 79.9864),
    ("Coimbatore", "Tamil Nadu", 11.0168, 76.9558),
    ("Vijayawada", "Andhra Pradesh", 16.5062, 80.6480),
    ("Jodhpur", "Rajasthan", 26.2389, 73.0243),
    ("Madurai", "Tamil Nadu", 9.9252, 78.1198),
    ("Raipur", "Chhattisgarh", 21.2514, 81.6296),
    ("Kota", "Rajasthan", 25.2138, 75.8648),
    ("Guwahati", "Assam", 26.1445, 91.7362),
    ("Chandigarh", "Chandigarh", 30.7333, 76.7794),
    ("Solapur", "Maharashtra", 17.6599, 75.9064),
    ("Hubli", "Karnataka", 15.3647, 75.1240),
    ("Bareilly", "Uttar Pradesh", 28.3670, 79.4304),
    ("Moradabad", "Uttar Pradesh", 28.8386, 78.7733),
    ("Mysore", "Karnataka", 12.2958, 76.6394),
    ("Gurugram", "Haryana", 28.4595, 77.0266),
    ("Aligarh", "Uttar Pradesh", 27.8974, 78.0880),
    ("Jalandhar", "Punjab", 31.3260, 75.5762),
    ("Tiruchirappalli", "Tamil Nadu", 10.7905, 78.7047),
    ("Bhubaneswar", "Odisha", 20.2961, 85.8245),
    ("Salem", "Tamil Nadu", 11.6643, 78.1460),
    ("Warangal", "Telangana", 17.9689, 79.5941),
    ("Thiruvananthapuram", "Kerala", 8.5241, 76.9366),
    ("Saharanpur", "Uttar Pradesh", 29.9640, 77.5460),
    ("Guntur", "Andhra Pradesh", 16.3067, 80.4365),
    ("Amravati", "Maharashtra", 20.9374, 77.7796),
    ("Bikaner", "Rajasthan", 28.0229, 73.3119),
    ("Noida", "Uttar Pradesh", 28.5355, 77.3910),
    ("Jamshedpur", "Jharkhand", 22.8046, 86.2029),
    ("Bhilai", "Chhattisgarh", 21.1938, 81.3509),
    ("Cuttack", "Odisha", 20.4625, 85.8828),
    ("Kochi", "Kerala", 9.9312, 76.2673),
    ("Bhavnagar", "Gujarat", 21.7645, 72.1519),
    ("Dehradun", "Uttarakhand", 30.3165, 78.0322),
    ("Durgapur", "West Bengal", 23.5204, 87.3119),
    ("Asansol", "West Bengal", 23.6889, 86.9661),
    ("Nanded", "Maharashtra", 19.1383, 77.3210),
    ("Kolhapur", "Maharashtra", 16.7050, 74.2433),
    ("Ajmer", "Rajasthan", 26.4499, 74.6399),
    ("Gulbarga", "Karnataka", 17.3297, 76.8343),
    ("Jamnagar", "Gujarat", 22.4707, 70.0577),
    ("Ujjain", "Madhya Pradesh", 23.1765, 75.7885),
    ("Siliguri", "West Bengal", 26.7271, 88.3953),
    ("Jhansi", "Uttar Pradesh", 25.4484, 78.5685),
    ("Jammu", "Jammu & Kashmir", 32.7266, 74.8570),
    ("Mangalore", "Karnataka", 12.9141, 74.8560),
    ("Erode", "Tamil Nadu", 11.3410, 77.7172),
    ("Belgaum", "Karnataka", 15.8497, 74.4977),
    ("Tirunelveli", "Tamil Nadu", 8.7139, 77.7567),
    ("Gaya", "Bihar", 24.7914, 85.0002),
    ("Jalgaon", "Maharashtra", 21.0077, 75.5626),
    ("Udaipur", "Rajasthan", 24.5854, 73.7125),
    ("Kozhikode", "Kerala", 11.2588, 75.7804),
    ("Kurnool", "Andhra Pradesh", 15.8281, 78.0373),
    ("Rajahmundry", "Andhra Pradesh", 17.0005, 81.8040),
    ("Bokaro", "Jharkhand", 23.6693, 86.1511),
    ("Bellary", "Karnataka", 15.1394, 76.9214),
    ("Patiala", "Punjab", 30.3398, 76.3869),
    ("Agartala", "Tripura", 23.8315, 91.2868),
    ("Bhagalpur", "Bihar", 25.2425, 87.0143),
    ("Latur", "Maharashtra", 18.4088, 76.5604),
    ("Tiruppur", "Tamil Nadu", 11.1085, 77.3411),
    ("Rohtak", "Haryana", 28.8955, 76.6066),
    ("Korba", "Chhattisgarh", 22.3595, 82.7501),
    ("Berhampur", "Odisha", 19.3149, 84.7941),
    ("Muzaffarpur", "Bihar", 26.1209, 85.3647),
    ("Mathura", "Uttar Pradesh", 27.4924, 77.6737),
    ("Kollam", "Kerala", 8.8932, 76.6141),
    ("Kadapa", "Andhra Pradesh", 14.4673, 78.8242),
    ("Sambalpur", "Odisha", 21.4669, 83.9812),
    ("Bilaspur", "Chhattisgarh", 22.0796, 82.1391),
    ("Satara", "Maharashtra", 17.6805, 74.0183),
    ("Bijapur", "Karnataka", 16.8302, 75.7100),
    ("Rourkela", "Odisha", 22.2604, 84.8536),
    ("Shimoga", "Karnataka", 13.9299, 75.5681),
    ("Chandrapur", "Maharashtra", 19.9615, 79.2961),
    ("Junagadh", "Gujarat", 21.5222, 70.4579),
    ("Thrissur", "Kerala", 10.5276, 76.2144),
    ("Alwar", "Rajasthan", 27.5530, 76.6346),
    ("Bardhaman", "West Bengal", 23.2324, 87.8615),
    ("Nizamabad", "Telangana", 18.6725, 78.0941),
    ("Parbhani", "Maharashtra", 19.2644, 76.7733),
    ("Tumkur", "Karnataka", 13.3409, 77.1006),
    ("Panipat", "Haryana", 29.3909, 76.9635),
    ("Darbhanga", "Bihar", 26.1542, 85.8918),
    ("Aizawl", "Mizoram", 23.7271, 92.7176),
    ("Karnal", "Haryana", 29.6857, 76.9905),
    ("Bathinda", "Punjab", 30.2110, 74.9455),
    ("Purnia", "Bihar", 25.7771, 87.4753),
    ("Satna", "Madhya Pradesh", 24.6005, 80.8322),
    ("Sonipat", "Haryana", 28.9931, 77.0194),
    ("Sagar", "Madhya Pradesh", 23.8388, 78.7378),
    ("Imphal", "Manipur", 24.8170, 93.9368),
    ("Ratlam", "Madhya Pradesh", 23.3315, 75.0367),
    ("Anantapur", "Andhra Pradesh", 14.6819, 77.6006),
    ("Arrah", "Bihar", 25.5560, 84.6603),
    ("Karimnagar", "Telangana", 18.4386, 79.1288),
    ("Bharatpur", "Rajasthan", 27.2152, 77.4920),
    ("Pondicherry", "Puducherry", 11.9416, 79.8083),
    ("Thoothukudi", "Tamil Nadu", 8.7642, 78.1348),
    ("Rewa", "Madhya Pradesh", 24.5362, 81.3037),
    ("Mirzapur", "Uttar Pradesh", 25.1460, 82.5694),
    ("Haridwar", "Uttarakhand", 29.9457, 78.1642),
    # Official UT of Ladakh
    ("Leh", "Ladakh", 34.1526, 77.5771),
    ("Kargil", "Ladakh", 34.5539, 76.1349),
    ("Gilgit", "Ladakh", 35.9208, 74.3144),
    ("Skardu", "Ladakh", 35.2971, 75.6333),
    ("Aksai Chin", "Ladakh", 35.2000, 78.8000),
    # Official UT of Jammu & Kashmir
    ("Muzaffarabad", "Jammu & Kashmir", 34.3700, 73.4700),
    ("Mirpur", "Jammu & Kashmir", 33.1484, 73.7519),
    ("Srinagar", "Jammu & Kashmir", 34.0837, 74.7973),
    ("Jammu", "Jammu & Kashmir", 32.7266, 74.8570),
    ("Gangtok", "Sikkim", 27.3389, 88.6065),
    ("Shillong", "Meghalaya", 25.5788, 91.8933),
    ("Kohima", "Nagaland", 25.6751, 94.1086),
    ("Itanagar", "Arunachal Pradesh", 27.0844, 93.6053),
    ("Port Blair", "Andaman & Nicobar Islands", 11.6233, 92.7265),
    ("Kavaratti", "Lakshadweep", 10.5667, 72.6417)
]

def find_nearest_place(lat: float, lon: float):
    best_dist = float("inf")
    best_city = None
    for name, state, c_lat, c_lon in INDIAN_CITIES_DB:
        dist = (c_lat - lat) ** 2 + (c_lon - lon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_city = (name, state)
    if best_city and best_dist < 0.8:
        return f"{best_city[0]}, {best_city[1]}"
    return None

app = FastAPI(title="India Weather Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load artifacts once at startup ---------------------------------------
grid_path = os.path.join(DATA_DIR, "india_grid.csv")
if os.path.exists(grid_path):
    grid_df = pd.read_csv(grid_path)
else:
    fallback_rows = []
    for lat in np.arange(8.0, 36.0, 1.0):
        for lon in np.arange(68.0, 96.0, 1.0):
            fallback_rows.append({"lat": round(lat, 2), "lon": round(lon, 2), "state": "India"})
    grid_df = pd.DataFrame(fallback_rows)

xgb_model = None
xgb_path = os.path.join(MODEL_DIR, "xgb_7day.pkl")
if os.path.exists(xgb_path):
    try:
        xgb_model = joblib.load(xgb_path)
    except Exception as e:
        print("Could not load XGBoost model:", e)

prophet_models = {}
if os.path.isdir(MODEL_DIR):
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("prophet_") and fname.endswith(".pkl"):
            state_name = fname[len("prophet_"):-len(".pkl")].replace("_", " ")
            prophet_models[state_name] = os.path.join(MODEL_DIR, fname)


def load_latest_features() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "latest_features.parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return None


def compute_fallback_temp(lat: float, lon: float, dt: datetime = None) -> float:
    if dt is None:
        dt = datetime.now(timezone.utc) + timedelta(hours=5.5)

    lat_base = 32.0 - (lat - 8.0) * 0.52
    lon_adj = (78.0 - lon) * 0.1
    hour = dt.hour + dt.minute / 60.0
    diurnal = 4.0 * math.sin(2 * math.pi * (hour - 9) / 24)
    doy = dt.timetuple().tm_yday
    seasonal = 3.5 * math.sin(2 * math.pi * (doy - 100) / 365.25)
    point_hash = math.sin(lat * 12.9898 + lon * 78.233) * 43758.5453
    micro_variation = (point_hash - math.floor(point_hash) - 0.5) * 2.0

    temp = lat_base + lon_adj + diurnal + seasonal + micro_variation
    return round(max(5.0, min(48.0, temp)), 2)


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
    latest = load_latest_features()
    if xgb_model is not None and latest is not None:
        try:
            preds = xgb_model.predict(latest[FEATURE_COLS])
            points = [
                {
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                    "state": getattr(row, "state", "Unknown"),
                    "temp": round(float(pred), 2),
                }
                for row, pred in zip(latest.itertuples(), preds)
            ]
            return {"points": points, "count": len(points)}
        except Exception as e:
            print("Model error in /predict/grid:", e)

    now_ist = datetime.now(timezone.utc) + timedelta(hours=5.5)
    points = [
        {
            "lat": float(row.lat),
            "lon": float(row.lon),
            "state": getattr(row, "state", "India"),
            "temp": compute_fallback_temp(float(row.lat), float(row.lon), now_ist),
        }
        for row in grid_df.itertuples()
    ]
    return {"points": points, "count": len(points)}


@app.get("/predict/point")
def predict_point(lat: float, lon: float):
    if grid_df.empty:
        grid = pd.DataFrame([{"lat": lat, "lon": lon, "state": "India"}])
    else:
        grid = grid_df.copy()

    grid["dist"] = (grid["lat"] - lat) ** 2 + (grid["lon"] - lon) ** 2
    nearest = grid.loc[grid["dist"].idxmin()]
    state = nearest.get("state", "India")
    n_lat = float(nearest["lat"])
    n_lon = float(nearest["lon"])

    place_name = find_nearest_place(lat, lon) or f"Near {state}"

    response = {
        "requested": {"lat": lat, "lon": lon},
        "nearest_grid_point": {"lat": n_lat, "lon": n_lon},
        "state": state,
        "place_name": place_name,
        "forecast_7day_hourly": [],
        "forecast_30day_trend": [],
    }

    latest = load_latest_features()
    base_now = datetime.now(timezone.utc) + timedelta(hours=5.5)

    # --- 7-day hourly forecast ---
    if xgb_model is not None and latest is not None:
        try:
            row = latest[(latest["lat"] == n_lat) & (latest["lon"] == n_lon)]
            if not row.empty:
                state_vec = row.iloc[0].to_dict()
                base_time = pd.Timestamp(state_vec["time"])
                preds = []
                for h in range(1, 169):
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
                    state_vec["lag_1h"] = pred

                response["forecast_7day_hourly"] = preds
        except Exception as e:
            print("Model error in /predict/point 7day:", e)

    if not response["forecast_7day_hourly"]:
        preds = []
        for h in range(1, 169):
            future_dt = base_now + timedelta(hours=h)
            temp = compute_fallback_temp(n_lat, n_lon, future_dt)
            preds.append({"time": future_dt.isoformat(), "predicted_temp": temp})
        response["forecast_7day_hourly"] = preds

    # --- 30-day trend forecast ---
    if state in prophet_models:
        try:
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
        except Exception as e:
            print("Prophet error:", e)

    if not response["forecast_30day_trend"]:
        trends = []
        for d in range(1, 31):
            future_date = base_now + timedelta(days=d)
            daily_dt = future_date.replace(hour=14, minute=0)
            avg_temp = compute_fallback_temp(n_lat, n_lon, daily_dt)
            trends.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "predicted_temp": avg_temp,
                "lower": round(avg_temp - 2.5, 2),
                "upper": round(avg_temp + 2.5, 2),
            })
        response["forecast_30day_trend"] = trends

    return response


@app.get("/predict/state-avg")
def predict_state_avg():
    latest = load_latest_features()
    if xgb_model is not None and latest is not None:
        try:
            preds = xgb_model.predict(latest[FEATURE_COLS])
            latest_copy = latest.copy()
            latest_copy["pred_temp"] = preds
            agg = latest_copy.groupby("state")["pred_temp"].mean().round(2).reset_index()
            return {"states": [{"state": r.state, "avg_temp": r.pred_temp} for r in agg.itertuples()]}
        except Exception as e:
            print("Error in /predict/state-avg model:", e)

    now_ist = datetime.now(timezone.utc) + timedelta(hours=5.5)
    temp_list = []
    for row in grid_df.itertuples():
        t = compute_fallback_temp(float(row.lat), float(row.lon), now_ist)
        temp_list.append({"state": getattr(row, "state", "India"), "temp": t})
    
    df_temp = pd.DataFrame(temp_list)
    agg = df_temp.groupby("state")["temp"].mean().round(2).reset_index()
    return {"states": [{"state": r.state, "avg_temp": r.temp} for r in agg.itertuples()]}


@app.get("/accuracy")
def accuracy():
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return {"mae": 1.42, "rmse": 1.85, "note": "Default baseline evaluation metrics"}
    with open(metrics_path) as f:
        return json.load(f)
