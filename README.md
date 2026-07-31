# India Weather Prediction Map

Grid-based weather prediction site for India — 7-day hourly forecast (ML) +
30-day trend outlook (statistical), rendered as an interactive heatmap with
click-to-drill-down, built entirely on the free Open-Meteo API using a
multi-model ensemble (ECMWF IFS, GFS, ICON, best_match).

## Project structure

```
weather-project/
├── backend/
│   ├── requirements.txt
│   ├── data/                  # created at runtime (grid, raw pulls, features)
│   ├── models/                 # created at runtime (trained model files)
│   └── app/
│       ├── grid.py             # 1. generate India grid + assign states
│       ├── fetch_data.py       # 2. pull multi-model historical data from Open-Meteo
│       ├── features.py         # 3. build lag/rolling/cyclic/ensemble features
│       ├── train.py            # 4. train XGBoost (7-day) + Prophet (30-day per state)
│       └── main.py             # 5. FastAPI server
└── frontend/
    └── index.html               # Leaflet heatmap + click-to-drill-down + Chart.js
```

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline — run in this order

```bash
# 1. Generate the India grid (few hundred points, not thousands of towns)
python -m app.grid

# 2. Pull historical multi-model data (start with --limit 10 to test quickly,
#    since fetching the full grid over multiple years will take a while)
python -m app.fetch_data --start 2021-01-01 --end 2025-12-31 --limit 10

# 3. Build features (lags, rolling stats, cyclic dates, ensemble mean/spread)
python -m app.features

# 4. Train both models
python -m app.train

# 5. Serve the API
uvicorn app.main:app --reload
```

Then open `frontend/index.html` in a browser (or serve it with any static
server) — it talks to the API at `http://localhost:8000` by default; change
`API_BASE` in `index.html` once you deploy.

## Known placeholders to fill in before this is demo-ready

- `main.py`'s `predict_grid()` currently returns `temp: null` for every
  point — wire it to `xgb_model.predict()` using each point's latest
  lag/rolling features (you'll want a small scheduled job that recomputes
  these hourly, similar to `fetch_data.py`, rather than recomputing from
  scratch on every request).
- `predict_point()`'s `forecast_7day_hourly` list is empty — loop
  `xgb_model.predict()` forward 168 hours, feeding each prediction back in
  as the next hour's lag feature (standard recursive forecasting).
- `predict_state_avg()` needs real aggregation once grid predictions are live.
- `train.py` doesn't yet save `models/metrics.json` for the `/accuracy`
  endpoint — add that after computing MAE/RMSE per model.
- Grid density (`GRID_STEP` in `grid.py`) is set to 0.5° to start — tighten
  it once you've confirmed the full pipeline runs end-to-end on the coarse
  grid, since cost scales roughly with the square of density.

## Data source

All weather data from [Open-Meteo](https://open-meteo.com) — free,
no API key, non-commercial use, aggregating ECMWF, NOAA GFS, DWD ICON and
other national weather service models. State boundaries from
[geohacker/india](https://github.com/geohacker/india).
