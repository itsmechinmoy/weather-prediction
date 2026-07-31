"""
grid.py
--------
Generates a uniform lat/lon grid covering India and assigns each grid
point to the state/UT it falls inside, using a public boundary GeoJSON.

Output: data/india_grid.csv with columns [lat, lon, state]

Run:
    python -m app.grid
"""

import json
import os

import pandas as pd
import requests
from shapely.geometry import Point, shape

# Bounding box roughly covering mainland India + islands
LAT_MIN, LAT_MAX = 6.0, 37.5
LON_MIN, LON_MAX = 68.0, 97.5

# Grid spacing in degrees. ~0.5 degrees ≈ 55 km at the equator.
# Lower this (e.g. 0.25) for a denser grid once you've validated the pipeline
# on the coarser one — it multiplies your row count roughly 4x per halving.
GRID_STEP = 0.5

STATE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
GEOJSON_CACHE = os.path.join(DATA_DIR, "india_state.geojson")
GRID_OUTPUT = os.path.join(DATA_DIR, "india_grid.csv")


def fetch_state_geojson() -> dict:
    """Download (and cache) the India state boundaries GeoJSON."""
    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE, "r") as f:
            return json.load(f)

    resp = requests.get(STATE_GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    geojson = resp.json()

    with open(GEOJSON_CACHE, "w") as f:
        json.dump(geojson, f)

    return geojson


def build_state_lookup(geojson: dict):
    """Return list of (state_name, shapely_polygon) for point-in-polygon tests."""
    lookup = []
    for feature in geojson["features"]:
        # Property key varies by source; try common ones used in this dataset
        props = feature["properties"]
        name = props.get("NAME_1") or props.get("st_nm") or props.get("name")
        geom = shape(feature["geometry"])
        lookup.append((name, geom))
    return lookup


def assign_state(lat: float, lon: float, state_lookup) -> str:
    point = Point(lon, lat)  # shapely uses (x=lon, y=lat)
    for name, geom in state_lookup:
        if geom.contains(point):
            return name
    return None  # outside all state polygons (ocean, border gaps, etc.)


def generate_grid() -> pd.DataFrame:
    geojson = fetch_state_geojson()
    state_lookup = build_state_lookup(geojson)

    rows = []
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon = LON_MIN
        while lon <= LON_MAX:
            state = assign_state(lat, lon, state_lookup)
            if state is not None:  # keep only points that fall inside India
                rows.append({"lat": round(lat, 3), "lon": round(lon, 3), "state": state})
            lon += GRID_STEP
        lat += GRID_STEP

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_grid()
    df.to_csv(GRID_OUTPUT, index=False)
    print(f"Generated {len(df)} grid points across {df['state'].nunique()} states/UTs")
    print(f"Saved to {GRID_OUTPUT}")
    print(df.groupby("state").size().sort_values(ascending=False))
