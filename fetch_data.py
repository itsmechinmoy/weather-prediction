"""
fetch_data.py
-------------
Pulls historical hourly weather data from Open-Meteo for every point in the
India grid (data/india_grid.csv), across multiple independent weather models
(ECMWF IFS, GFS, ICON, plus Open-Meteo's auto best_match) for ensembling.

Open-Meteo free tier: no API key required, ~10,000 calls/day non-commercial.
Docs: https://open-meteo.com/en/docs/historical-weather-api

Run:
    python -m app.fetch_data --start 2021-01-01 --end 2025-12-31
"""

import argparse
import os
import time

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GRID_FILE = os.path.join(DATA_DIR, "india_grid.csv")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "cloud_cover",
]

# Individual models pulled alongside best_match for ensembling.
# See https://open-meteo.com/en/docs/historical-weather-api for the full list.
MODELS = ["best_match", "ecmwf_ifs", "gfs_seamless", "icon_seamless"]


def fetch_point(lat: float, lon: float, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Fetch multi-model hourly data for one lat/lon and return a tidy DataFrame."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "models": ",".join(MODELS),
        "timezone": "Asia/Kolkata",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"  fetch failed ({exc}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed to fetch {lat},{lon} after {retries} attempts")

    hourly = payload["hourly"]
    df = pd.DataFrame({"time": hourly["time"]})

    # Open-Meteo suffixes each variable with the model name when multiple
    # models are requested, e.g. temperature_2m_ecmwf_ifs
    for key, values in hourly.items():
        if key == "time":
            continue
        df[key] = values

    df["lat"] = lat
    df["lon"] = lon
    return df


def main(start: str, end: str, limit: int = None):
    grid = pd.read_csv(GRID_FILE)
    if limit:
        grid = grid.head(limit)  # useful for a quick test run before the full pull

    print(f"Fetching {len(grid)} grid points, models={MODELS}, {start} to {end}")

    for i, row in grid.iterrows():
        out_path = os.path.join(RAW_DIR, f"{row.lat}_{row.lon}.parquet")
        if os.path.exists(out_path):
            continue  # resume support — skip points already fetched

        print(f"[{i + 1}/{len(grid)}] {row.state}: ({row.lat}, {row.lon})")
        df = fetch_point(row.lat, row.lon, start, end)
        df["state"] = row.state
        df.to_parquet(out_path, index=False)

        time.sleep(0.2)  # gentle pacing to stay well within rate limits

    print("Done. Raw per-point files saved in", RAW_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch first N grid points (for testing)")
    args = parser.parse_args()
    main(args.start, args.end, args.limit)
