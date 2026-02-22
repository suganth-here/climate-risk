# Climate Lending System: Full Scoring, Model, Dataset, and Interest Logic

This document explains exactly how the current website computes risk, score, approval, and interest adjustments.

## 1) What the live website actually uses

The React frontend calls Django API endpoints:
- `POST /api/predict/`
- `POST /api/portfolio/analyze/`
- `GET /api/metadata/`

Runtime logic is in:
- `backend/api/views.py`
- `backend/api/logic.py`
- `src/climate_intelligence.py`
- `src/data_loader.py`
- `src/lending_rules.py`

Important: the live API does **not** load `artifacts/loan_approval_model.json` or `artifacts/basic_linear_model.json` during prediction.  
Those artifacts come from offline training scripts and are currently separate from online inference.

## 2) Datasets used in live scoring

Loaded by `src/data_loader.py` (required files):
- `dataset/cyclone_clean.csv` (13,535 lines incl. header)
- `dataset/flood_points_clean.csv` (64 lines incl. header)
- `dataset/india_annual_rainfall.csv` (21,907 lines incl. header)
- `dataset/india_tmax_final.csv` (14,955 lines incl. header)
- `dataset/coastline_points.csv` (410,958 lines incl. header)

Additionally used by `backend/api/logic.py` when available:
- `dataset/india_extreme_low_temp.csv` (21,907 lines incl. header)
- `dataset/india_extreme_high_temp.csv` (21,907 lines incl. header)
- Elevation file (optional): one of:
  - `dataset/india_elevation.csv`
  - `dataset/elevation_points.csv`
  - `dataset/elevation.csv`

## 3) Data cleaning and feature construction

### 3.1 Cleaning (`src/data_loader.py`)

Main operations:
- Column renaming to canonical names.
- Numeric coercion (`to_numeric(..., errors="coerce")`).
- Drop rows missing critical fields.
- Latitude/longitude range filter:
  - latitude in `[-90, 90]`
  - longitude in `[-180, 180]`
- Deduplication.
- Interpolation/fill for some fields:
  - Cyclone wind interpolated then median-filled.
  - Rainfall and Tmax interpolated.

### 3.2 Historical table (`ClimateLendingIntelligence.build_historical_feature_table`)

Location bins:
- `lat_bin = round(latitude, 0)`
- `lon_bin = round(longitude, 0)`

Computed features include:
- `cyclone_events`, `cyclone_mean_wind`
- `flood_events`
- Static nearest-neighbor rainfall (`rainfall_mm`) and Tmax (`tmax_c`)
- Spatial local densities:
  - Flood density within radius `1.8`
  - Cyclone density within radius `2.2`

Normalized features:
- `wind_severity`, `rain_severity`, `heat_severity`
- `cyclone_local_wind_severity`

Risk indices:
- `flood_risk_index = 0.55 * norm(flood_events) + 0.45 * norm(flood_local_density)`
- `cyclone_risk_index = 0.40 * norm(cyclone_events) + 0.25 * wind_severity + 0.20 * norm(cyclone_local_density) + 0.15 * cyclone_local_wind_severity`
- `heat_risk_index = heat_severity`
- `location_risk_index = norm(mean disaster frequency by location)`

Core historical climate risk:
- `climate_risk_score = 0.35*flood_risk_index + 0.35*cyclone_risk_index + 0.20*heat_risk_index + 0.10*location_risk_index`
- Clipped to `[0, 1]`

## 4) Projection model used (50-year risk)

Implemented in `ClimateLendingIntelligence.project_risk_50_years`.

Model type:
- `LinearRegression` from scikit-learn.

How it is trained:
- A global linear model is fit on all historical rows (`year -> climate_risk_score`).
- For each location bin `(lat_bin, lon_bin)`:
  - If location has at least 3 unique years, fit location-specific linear regression.
  - Else fallback to global model.

Output:
- `predicted_climate_risk` for each year over horizon (default 50 years from start year).
- Predicted values clipped to `[0,1]`.

## 5) Tenure risk calculation

`tenure_risk(latitude, longitude, tenure_years, start_year)`:
- Find nearest projected location bin.
- Slice projected years from `start_year` to `start_year + tenure_years - 1`.
- `tenure_risk_score = mean(predicted_climate_risk over tenure window)`
- `tenure_risk_percent = tenure_risk_score * 100`

This is returned to UI, but it is **not** directly the climate credit score.

## 6) Annual risk point calculation (0-100 style points)

Built by `build_annual_risk_points(...)` in `backend/api/logic.py`.

### 6.1 Raw hazard signals at input location

- Rainfall raw value: nearest rainfall point.
- Coastline risk proxy:
  - `coast_risk = exp(-distance_to_nearest_coast / 2.0)` clipped to `[0,1]`.
- Flood raw composite:
  - `flood_risk = 0.88*flood_model + 0.12*flood_proximity`
- Cyclone raw composite:
  - `cyclone_risk = 0.72*cyclone_model + 0.13*cyclone_proximity + 0.15*wind_norm`
- Temperature signal:
  - nearest extreme low temp and high temp are read;
  - `temperature_signal = max(abs(low_temp), high_temp)`

### 6.2 Converting raw signals into points (`pattern_score`)

Each signal is compared against reference distributions built from all known location bins.

For value percentile `pct` in reference:
- if `pct <= alert_quantile`:
  - `score = 35 * (pct / alert_quantile)`
- else:
  - `score = 35 + 65 * ((pct - alert_quantile) / (1 - alert_quantile))`
- clip to `[0,100]`

Quantiles used:
- Flood history internal score: `0.97` (computed but currently unused in final output)
- Cyclone: `0.92`
- Temperature: `0.90`
- Rainfall: `0.92`
- Sea Level (from coastline risk): `0.92`

Final annual points returned:
- `Cyclone = cyclone_points`
- `Temperature = temperature_points`
- `Rainfall = rainfall_points`
- `Sea Level = sea_level_points`
- `Flood = average(Rainfall, Cyclone, Sea Level, Temperature)`

So flood point is a composite average of four components.

## 7) Final climate credit score

`climate_credit_score_from_annual_points`:
- `avg_risk = mean([Cyclone, Temperature, Rainfall, Flood, Sea Level])`
- `score = round(100 - avg_risk)`
- clipped to `[0,100]`

Interpretation:
- Higher annual risk points -> lower climate credit score.

## 8) Approval / rejection policy logic

`build_policy_decision(...)` in `backend/api/logic.py`.

Loan is auto-`Not Approved` if **any** condition is true:
1. `Flood > 45`
2. `Temperature > 50`
3. At least two annual points are `> 65`
4. Extreme temperature hit:
   - low temp is in the lower 15% tail (`<= 15th percentile`) or
   - high temp is in the upper 85% tail (`>= 85th percentile`)
   for nearest points from extreme temp datasets.

Else decision is `Approved`.

`safe` flag mirrors this policy outcome.

## 9) Interest adjustment and risk band mapping

Interest mapping from `src/lending_rules.py` via `lending_adjustment_from_score(score)`:

- `score >= 80`:
  - interest delta `+0.00%`
  - tenure delta `0 years`
  - insurance premium delta `+0.0%`
  - band `Low Risk`
- `65 <= score < 80`:
  - interest delta `+0.30%`
  - tenure delta `-1 years`
  - insurance premium delta `+4.0%`
  - band `Moderate Risk`
- `50 <= score < 65`:
  - interest delta `+0.70%`
  - tenure delta `-3 years`
  - insurance premium delta `+9.0%`
  - band `Elevated Risk`
- `35 <= score < 50`:
  - interest delta `+1.20%`
  - tenure delta `-5 years`
  - insurance premium delta `+15.0%`
  - band `High Risk`
- `score < 35`:
  - interest delta `+1.75%`
  - tenure delta `-7 years`
  - insurance premium delta `+22.0%`
  - band `Severe Risk`

In single-loan API response, the UI text currently focuses on interest rate increase message.

## 10) Portfolio scoring flow

`analyze_portfolio(...)`:
- Validates CSV columns (`property_id`, `latitude`, `longitude`, `tenure_years`).
- Runs same per-loan evaluation + score logic.
- Portfolio summary:
  - `approved`, `not_approved`, `average_tenure_risk`
- Per-row outputs:
  - score (`x/100`)
  - short interest adjustment text
  - reason

Portfolio alert text is computed from:
- share of low-score loans among coastal subset (or full set if no coastal rows),
- mean flood and cyclone points,
- and then mapped to default-exposure range text.

## 11) Inputs currently accepted but not used in scoring

In `evaluate_single_application(...)`:
- `loan_amount` argument is accepted but not used in current risk/score formula.
- `property_amount` is also accepted but not used.

Current live score is climate-hazard driven from geospatial + tenure inputs.

## 12) Offline model artifacts in `artifacts/`

These exist but are not used by live API right now:
- `artifacts/loan_approval_model.json`
  - Logistic regression artifact trained by `train_loan_approval_model.py`
  - Uses synthetic labels (`loan_approved` generated from constructed risk formula)
- `artifacts/basic_linear_model.json`
  - Numpy linear regression from `train_basic_model.py`

`train_climate_lending_model.py` can also train a RandomForest classifier if a real labeled file exists at `data/loan_training_data.csv`.

## 13) End-to-end request flow (single prediction)

1. Frontend sends lat/lon/tenure/loan/start year to `/api/predict/`.
2. Backend loads cached runtime engine and cleaned datasets.
3. 50-year projection is rebuilt for requested start year.
4. Tenure risk is computed from projected risk series.
5. Annual hazard points are computed from spatial+projection+pattern scoring.
6. Climate credit score is computed as `100 - mean(annual points)`.
7. Policy rules decide `Approved` vs `Not Approved`.
8. Interest-rate delta text is generated from score bands.
9. API returns score, points, tenure stats, decision safety flag, statements, and 50-year series.

---

If you want, I can also add a second file with worked numerical examples (sample inputs -> intermediate values -> final score and interest delta) so team members can verify outputs manually.

## 14) Worked numerical example (single loan)

This example uses the exact runtime formula shape to show how final outputs are produced.

Assume annual points after pattern scoring are:
- `Cyclone = 58.40`
- `Temperature = 62.10`
- `Rainfall = 55.30`
- `Sea Level = 49.20`
- `Flood = (Cyclone + Temperature + Rainfall + Sea Level)/4`
- `Flood = (58.40 + 62.10 + 55.30 + 49.20)/4 = 56.25`

### 14.1 Climate credit score

`avg_risk = mean([Cyclone, Temperature, Rainfall, Flood, Sea Level])`

`avg_risk = (58.40 + 62.10 + 55.30 + 56.25 + 49.20)/5 = 56.25`

`score = round(100 - avg_risk) = round(43.75) = 44`

Final:
- `climate_credit_score = 44/100`

### 14.2 Approval policy check

Rules:
- reject if `Flood > 45` -> here `56.25 > 45` (true)
- reject if `Temperature > 50` -> here `62.10 > 50` (true)
- reject if any two points `> 65` -> false in this example
- reject on extreme temp hit if percentile rule matches location

Since at least one reject condition is true, decision becomes:
- `Not Approved`
- `safe = false`

### 14.3 Interest-rate adjustment

Score `44` falls in band `35 <= score < 50`:
- interest delta: `+1.20%`
- tenure delta: `-5 years`
- insurance premium delta: `+15.0%`
- risk band: `High Risk`

Frontend short message:
- `"Interest rate increased by 1.2%."`

## 15) Worked numerical example (portfolio rollup)

Suppose portfolio scores for 5 loans are:
- `[82, 71, 48, 42, 63]`

Interest text by score band:
- `82 -> Unchanged interest rate.`
- `71 -> Interest rate increased by 0.3%.`
- `48 -> Interest rate increased by 1.2%.`
- `42 -> Interest rate increased by 1.2%.`
- `63 -> Interest rate increased by 0.7%.`

If this set has:
- approved count = 3
- not approved count = 2
- tenure risk percents = `[18.5, 27.1, 46.2, 52.8, 35.0]`

Then:
- `average_tenure_risk = mean(...) = 35.92%` (rounded to 2 decimals in API)

Portfolio alert is derived from:
- coastal subset (Sea Level >= 60) when available, else all rows
- share of rows with score < 50
- mean Flood and Cyclone values of chosen subset
- mapped into the alert text range

## 16) Offline training scripts (what each one does)

### `train_loan_approval_model.py`
- Builds synthetic labeled training data from climate + finance features.
- Trains tuned logistic regression.
- Saves artifact to `artifacts/loan_approval_model.json`.
- Artifact includes:
  - feature columns
  - normalization stats
  - weights
  - threshold
  - evaluation metrics

Important:
- labels are synthetic (`loan_approved` is generated in script), not real bank labels.

### `train_climate_lending_model.py`
- Builds historical/projection tables.
- Writes summary metrics to `artifacts/climate_lending_metrics.json` (if script is run).
- Can train RandomForest only if real labeled dataset exists:
  - `data/loan_training_data.csv`
  - required columns: `latitude, longitude, loan_amount, tenure_years, loan_approved`

### `train_basic_model.py`
- Trains simple linear regression baseline.
- Saves to `artifacts/basic_linear_model.json`.
- Intended as basic demonstration benchmark.

## 17) Exact API payload examples

### 17.1 Single prediction request

`POST /api/predict/`

```json
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "tenure_years": 15,
  "loan_amount": 5000000,
  "property_id": "98122",
  "projection_start_year": 2026
}
```

### 17.2 Portfolio request (JSON mode)

`POST /api/portfolio/analyze/`

```json
{
  "projection_start_year": 2026,
  "rows": [
    {
      "property_id": "98122",
      "latitude": 13.0827,
      "longitude": 80.2707,
      "tenure_years": 20
    },
    {
      "property_id": "98123",
      "latitude": 19.0760,
      "longitude": 72.8777,
      "tenure_years": 18
    }
  ]
}
```
