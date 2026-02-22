from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.climate_intelligence import ClimateLendingIntelligence  # noqa: E402
from src.data_loader import validate_portfolio_df  # noqa: E402
from src.lending_rules import lending_adjustment_from_score  # noqa: E402


DEFAULT_PROJECTION_START_YEAR = 2026
DEFAULT_PROJECTION_HORIZON = 50

INDIA_STATE_COORDS: Dict[str, Tuple[float, float]] = {
    "Andhra Pradesh": (16.5062, 80.6480),
    "Arunachal Pradesh": (27.0844, 93.6053),
    "Assam": (26.1445, 91.7362),
    "Bihar": (25.5941, 85.1376),
    "Chhattisgarh": (21.2514, 81.6296),
    "Goa": (15.4909, 73.8278),
    "Gujarat": (23.2156, 72.6369),
    "Haryana": (30.7333, 76.7794),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.3441, 85.3096),
    "Karnataka": (12.9716, 77.5946),
    "Kerala": (8.5241, 76.9366),
    "Madhya Pradesh": (23.2599, 77.4126),
    "Maharashtra": (19.0760, 72.8777),
    "Manipur": (24.8170, 93.9368),
    "Meghalaya": (25.5788, 91.8933),
    "Mizoram": (23.7271, 92.7176),
    "Nagaland": (25.6751, 94.1086),
    "Odisha": (20.2961, 85.8245),
    "Punjab": (30.7333, 76.7794),
    "Rajasthan": (26.9124, 75.7873),
    "Sikkim": (27.3389, 88.6065),
    "Tamil Nadu": (13.0827, 80.2707),
    "Telangana": (17.3850, 78.4867),
    "Tripura": (23.8315, 91.2868),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.3165, 78.0322),
    "West Bengal": (22.5726, 88.3639),
}

REQUIRED_PORTFOLIO_COLUMNS = [
    "property_id",
    "latitude",
    "longitude",
    "tenure_years",
]


def climate_credit_score_from_annual_points(annual_points: Dict[str, float]) -> int:
    vals = [float(v) for v in annual_points.values()]
    avg_risk = float(np.mean(vals)) if vals else 0.0
    score = int(round(100.0 - avg_risk))
    return int(max(0, min(100, score)))


def _top_hazards(annual_points: Dict[str, float], top_n: int = 3) -> List[str]:
    return [k for k, _ in sorted(annual_points.items(), key=lambda x: float(x[1]), reverse=True)[:top_n]]


def build_pricing_adjustment_text(score: int, annual_points: Optional[Dict[str, float]] = None) -> str:
    adj = lending_adjustment_from_score(int(score))
    annual = annual_points or {}
    phrase_map = {
        "Flood": "high flood recurrence probability",
        "Cyclone": "elevated cyclone intensity risk",
        "Sea Level": "long-term property vulnerability in coastal zone",
        "Temperature": "persistent heat-stress escalation",
        "Rainfall": "rainfall volatility and runoff stress",
    }
    top2 = sorted(annual.items(), key=lambda x: float(x[1]), reverse=True)[:2]
    if len(top2) >= 2:
        p1 = phrase_map.get(top2[0][0], "elevated climate stress")
        p2 = phrase_map.get(top2[1][0], "collateral vulnerability pressure")
        driver_text = f"{p1} and {p2}"
    elif len(top2) == 1:
        p1 = phrase_map.get(top2[0][0], "elevated climate stress")
        driver_text = p1
    else:
        driver_text = "elevated climate exposure under long-horizon hazard projections"

    if float(adj.interest_rate_delta_pct) > 0:
        return (
            "Loan Pricing Adjustment: Interest rate increased by "
            f"{float(adj.interest_rate_delta_pct):.1f}% due to {driver_text}."
        )
    return "Loan Pricing Adjustment: Interest rate unchanged due to low projected climate exposure and stable collateral risk outlook."


def build_interest_adjustment_short_text(score: int) -> str:
    delta = float(lending_adjustment_from_score(int(score)).interest_rate_delta_pct)
    delta = max(0.0, delta)
    if delta <= 0.0:
        return "Unchanged interest rate."
    return f"Interest rate increased by {delta:.1f}%."


def build_explainability_log_text(annual_points: Dict[str, float]) -> str:
    if not annual_points:
        return "Explainability Log: Not available."
    sea_level = float(annual_points.get("Sea Level", 0.0))
    rainfall = float(annual_points.get("Rainfall", 0.0))
    flood = float(annual_points.get("Flood", 0.0))
    cyclone = float(annual_points.get("Cyclone", 0.0))
    total = max(1e-6, float(sum(float(v) for v in annual_points.values())))
    storm_weight = int(round((cyclone / total) * 100.0))
    flood_return_period = max(2, int(round(12.0 - (flood / 8.0))))
    return (
        "Explainability Log: Score derived from elevated sea-level risk index "
        f"({sea_level:.1f}/100), rainfall anomaly pressure ({rainfall:.1f}/100), and high flood recurrence "
        f"(approx. 1-in-{flood_return_period} year return period). "
        f"Storm surge exposure contributes {storm_weight}% of total risk weighting."
    )


def build_portfolio_alert_text(
    concentration_pct: int, flood_mean: float, cyclone_mean: float, years: int = 10
) -> str:
    # Exposure projection proxy scales with joint flood/cyclone pressure.
    joint_hazard = float(np.clip((flood_mean + cyclone_mean) / 2.0, 0.0, 100.0))
    low_exp = max(4, int(round(joint_hazard * 0.20)))
    high_exp = max(low_exp + 2, int(round(joint_hazard * 0.30)))
    return (
        "Portfolio Risk Alert: Portfolio analysis indicates "
        f"{int(concentration_pct)}% of active loans are concentrated in high flood and cyclone-prone coastal zones. "
        f"Estimated climate-linked default exposure may rise by {low_exp}-{high_exp}% over the next {int(years)} years. "
        "Risk diversification recommended."
    )


def build_risk_reason_text(annual_points: Dict[str, float], tenure: Dict[str, object]) -> str:
    top2 = _top_hazards(annual_points, top_n=2)
    t0 = int(tenure["start_year"])
    t1 = int(tenure["end_year"])
    tenure_risk_pct = float(tenure["tenure_risk_percent"])
    h1 = top2[0] if len(top2) > 0 else "Flood"
    h2 = top2[1] if len(top2) > 1 else "Cyclone"
    h1_v = float(annual_points.get(h1, 0.0))
    h2_v = float(annual_points.get(h2, 0.0))
    composite = float(np.mean([float(v) for v in annual_points.values()])) if annual_points else 0.0
    return (
        f"Loan climate risk is primarily driven by {h1.lower()} ({h1_v:.1f}/100) and {h2.lower()} ({h2_v:.1f}/100) "
        f"signals across the {t0}-{t1} horizon. Composite hazard pressure is {composite:.1f}/100 with "
        f"tenure climate stress at {tenure_risk_pct:.2f}%."
    )


def build_property_score_text(property_id: str, score: int, annual_points: Dict[str, float]) -> str:
    top2 = _top_hazards(annual_points, top_n=2)
    if len(top2) >= 2:
        hazard_text = f"{top2[0].lower()} and {top2[1].lower()} risk projected over 30-50 years"
    elif len(top2) == 1:
        hazard_text = f"{top2[0].lower()} risk projected over 30-50 years"
    else:
        hazard_text = "climate risk projected over 30-50 years"
    return f"Property-ID {property_id} assigned Climate Credit Score: {score}/100-high {hazard_text}."


def nearest_value(df: pd.DataFrame, latitude: float, longitude: float, value_col: str) -> float:
    d2 = (df["latitude"] - latitude) ** 2 + (df["longitude"] - longitude) ** 2
    idx = int(d2.idxmin())
    return float(df.at[idx, value_col])


def min_max_score(value: float, lo: float, hi: float) -> float:
    if np.isclose(hi, lo):
        return 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


def coastline_risk_score(coast_df: pd.DataFrame, latitude: float, longitude: float) -> float:
    d2 = (coast_df["latitude"] - latitude) ** 2 + (coast_df["longitude"] - longitude) ** 2
    dist = float(np.sqrt(d2.min()))
    return float(max(0.0, min(1.0, np.exp(-dist / 2.0))))


def nearest_distance_score(df: pd.DataFrame, latitude: float, longitude: float, scale: float) -> float:
    d2 = (df["latitude"] - latitude) ** 2 + (df["longitude"] - longitude) ** 2
    dist = float(np.sqrt(d2.min()))
    return float(max(0.0, min(1.0, np.exp(-dist / scale))))


@lru_cache(maxsize=1)
def load_elevation_dataset() -> Optional[pd.DataFrame]:
    candidates = [
        PROJECT_ROOT / "dataset" / "india_elevation.csv",
        PROJECT_ROOT / "dataset" / "elevation_points.csv",
        PROJECT_ROOT / "dataset" / "elevation.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            lat_col = cols.get("latitude", cols.get("lat"))
            lon_col = cols.get("longitude", cols.get("lon"))
            elev_col = cols.get("elevation_m", cols.get("elevation", cols.get("elev_m")))
            if lat_col and lon_col and elev_col:
                out = pd.DataFrame(
                    {
                        "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
                        "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
                        "elevation_m": pd.to_numeric(df[elev_col], errors="coerce"),
                    }
                ).dropna()
                if not out.empty:
                    return out
    return None


def _read_lat_lon_points(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    lat_col = cols.get("latitude", cols.get("lat"))
    lon_col = cols.get("longitude", cols.get("lon"))
    if not lat_col or not lon_col:
        return None
    temp_col = None
    for c in df.columns:
        lc = c.lower()
        if "temp" in lc:
            temp_col = c
            break
    out = pd.DataFrame(
        {
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )
    if temp_col is not None:
        out["temp_c"] = pd.to_numeric(df[temp_col], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"])
    if out.empty:
        return None
    return out


@lru_cache(maxsize=1)
def load_extreme_temperature_datasets() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    root = PROJECT_ROOT / "dataset"
    low_candidates = sorted(root.glob("*extreme*low*temp*.csv")) + sorted(root.glob("*low*temperature*.csv"))
    high_candidates = sorted(root.glob("*extreme*high*temp*.csv")) + sorted(root.glob("*high*temperature*.csv"))

    low_df = None
    high_df = None
    for p in low_candidates:
        low_df = _read_lat_lon_points(p)
        if low_df is not None:
            break
    for p in high_candidates:
        high_df = _read_lat_lon_points(p)
        if high_df is not None:
            break
    return low_df, high_df


def nearest_extreme_temp_value(df: Optional[pd.DataFrame], latitude: float, longitude: float) -> Optional[float]:
    if df is None or "temp_c" not in df.columns:
        return None
    d2 = (df["latitude"] - latitude) ** 2 + (df["longitude"] - longitude) ** 2
    idx = int(d2.idxmin())
    v = pd.to_numeric(df.at[idx, "temp_c"], errors="coerce")
    if pd.isna(v):
        return None
    return float(v)


def extreme_temperature_hits(
    latitude: float, longitude: float, low_df: Optional[pd.DataFrame], high_df: Optional[pd.DataFrame]
) -> Tuple[bool, str]:
    reasons = []
    low_val = nearest_extreme_temp_value(low_df, latitude, longitude)
    high_val = nearest_extreme_temp_value(high_df, latitude, longitude)

    if low_df is not None and low_val is not None and "temp_c" in low_df.columns:
        low_cutoff = float(pd.to_numeric(low_df["temp_c"], errors="coerce").quantile(0.15))
        if low_val <= low_cutoff:
            reasons.append(f"extreme low temperature ({low_val:.2f}C <= {low_cutoff:.2f}C)")

    if high_df is not None and high_val is not None and "temp_c" in high_df.columns:
        high_cutoff = float(pd.to_numeric(high_df["temp_c"], errors="coerce").quantile(0.85))
        if high_val >= high_cutoff:
            reasons.append(f"extreme high temperature ({high_val:.2f}C >= {high_cutoff:.2f}C)")
    if reasons:
        return True, "Location matches " + " and ".join(reasons) + "."
    return False, ""


def elevation_risk_from_lat_lon(latitude: float, longitude: float) -> float:
    himalaya = np.exp(-(((latitude - 30.5) ** 2) / 20.0 + ((longitude - 82.0) ** 2) / 80.0))
    western_ghats = np.exp(-(((latitude - 12.5) ** 2) / 40.0 + ((longitude - 75.0) ** 2) / 6.0))
    eastern_ghats = np.exp(-(((latitude - 17.0) ** 2) / 60.0 + ((longitude - 83.0) ** 2) / 10.0))
    elevation_norm = float(np.clip(0.62 * himalaya + 0.23 * western_ghats + 0.15 * eastern_ghats, 0.0, 1.0))
    return float(1.0 - elevation_norm)


def elevation_risk_score(elevation_df: Optional[pd.DataFrame], latitude: float, longitude: float) -> float:
    if elevation_df is None:
        return elevation_risk_from_lat_lon(latitude, longitude)
    elev = nearest_value(elevation_df, latitude, longitude, "elevation_m")
    elev_norm = min_max_score(elev, float(elevation_df["elevation_m"].min()), float(elevation_df["elevation_m"].max()))
    return float(1.0 - elev_norm)


def elevation_value_for_ui(elevation_df: Optional[pd.DataFrame], latitude: float, longitude: float) -> float:
    if elevation_df is not None:
        return float(nearest_value(elevation_df, latitude, longitude, "elevation_m"))
    elev_proxy = (1.0 - elevation_risk_from_lat_lon(latitude, longitude)) * 3500.0
    return float(max(0.0, elev_proxy))


def pattern_score(value: float, reference: pd.Series, alert_quantile: float = 0.80) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    if ref.empty:
        return 0.0
    arr = np.sort(ref.to_numpy(dtype=float))
    rank = np.searchsorted(arr, float(value), side="right")
    pct = float(rank / max(1, len(arr)))

    if pct <= alert_quantile:
        score = 35.0 * (pct / max(alert_quantile, 1e-6))
    else:
        score = 35.0 + 65.0 * ((pct - alert_quantile) / max(1.0 - alert_quantile, 1e-6))
    return float(np.clip(score, 0.0, 100.0))


def build_pattern_references(
    engine: ClimateLendingIntelligence, low_extreme_df: Optional[pd.DataFrame], high_extreme_df: Optional[pd.DataFrame]
) -> Dict[str, pd.Series]:
    assert engine.historical_features is not None

    locs = engine.historical_features[["lat_bin", "lon_bin"]].drop_duplicates().reset_index(drop=True)
    coast_df = engine.cleaned["coastline"][["latitude", "longitude"]]
    flood_df = engine.cleaned["flood"][["latitude", "longitude"]]
    cyclone_df = engine.cleaned["cyclone"][["latitude", "longitude", "wind"]].copy()
    cyclone_df["wind"] = pd.to_numeric(cyclone_df["wind"], errors="coerce").fillna(0.0)

    loc_risk = (
        engine.historical_features.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(
            flood_idx=("flood_risk_index", "mean"),
            cyclone_idx=("cyclone_risk_index", "mean"),
        )
    )

    flood_raw_vals = []
    cyclone_raw_vals = []
    temperature_raw_vals = []
    rainfall_raw_vals = []
    coast_raw_vals = []
    rain_df = engine.cleaned["rainfall"][["latitude", "longitude", "rainfall_mm"]]

    wind_min = float(cyclone_df["wind"].min())
    wind_max = float(cyclone_df["wind"].max())

    for _, r in locs.iterrows():
        lat = float(r["lat_bin"])
        lon = float(r["lon_bin"])
        row = loc_risk[(loc_risk["lat_bin"] == lat) & (loc_risk["lon_bin"] == lon)].iloc[0]

        flood_prox = nearest_distance_score(flood_df, lat, lon, scale=0.9)
        flood_raw_vals.append((0.88 * float(row["flood_idx"])) + (0.12 * flood_prox))

        cyclone_prox = nearest_distance_score(cyclone_df, lat, lon, scale=1.1)
        d2_cyc = (cyclone_df["latitude"] - lat) ** 2 + (cyclone_df["longitude"] - lon) ** 2
        cyc_idx = int(d2_cyc.idxmin())
        nearest_wind = float(cyclone_df.at[cyc_idx, "wind"])
        wind_norm = min_max_score(nearest_wind, wind_min, wind_max)
        cyclone_raw_vals.append((0.72 * float(row["cyclone_idx"])) + (0.13 * cyclone_prox) + (0.15 * wind_norm))

        low_prox = 0.0 if low_extreme_df is None else nearest_distance_score(low_extreme_df, lat, lon, scale=1.0)
        high_prox = 0.0 if high_extreme_df is None else nearest_distance_score(high_extreme_df, lat, lon, scale=1.0)
        low_val = nearest_extreme_temp_value(low_extreme_df, lat, lon)
        high_val = nearest_extreme_temp_value(high_extreme_df, lat, lon)
        low_abs = abs(low_val) if low_val is not None else 0.0
        high_abs = high_val if high_val is not None else 0.0
        temperature_raw_vals.append(max(low_abs, high_abs))
        rainfall_raw_vals.append(nearest_value(rain_df, lat, lon, "rainfall_mm"))
        coast_raw_vals.append(coastline_risk_score(coast_df, lat, lon))

    return {
        "flood_raw": pd.Series(flood_raw_vals),
        "cyclone_raw": pd.Series(cyclone_raw_vals),
        "temperature_raw": pd.Series(temperature_raw_vals),
        "rainfall_raw": pd.Series(rainfall_raw_vals),
        "coastline_raw": pd.Series(coast_raw_vals),
    }


def build_annual_risk_points(
    engine: ClimateLendingIntelligence,
    latitude: float,
    longitude: float,
    lat_bin: float,
    lon_bin: float,
    start_year: int,
    references: Dict[str, pd.Series],
    elevation_df: Optional[pd.DataFrame],
    low_extreme_df: Optional[pd.DataFrame],
    high_extreme_df: Optional[pd.DataFrame],
) -> Dict[str, float]:
    assert engine.projection_features is not None
    proj_slice = engine.projection_features[
        (engine.projection_features["lat_bin"] == lat_bin)
        & (engine.projection_features["lon_bin"] == lon_bin)
        & (engine.projection_features["year"] == start_year)
    ]
    if proj_slice.empty:
        proj_slice = engine.projection_features[
            (engine.projection_features["lat_bin"] == lat_bin) & (engine.projection_features["lon_bin"] == lon_bin)
        ]
    row = proj_slice.iloc[0]

    rain_df = engine.cleaned["rainfall"][["latitude", "longitude", "rainfall_mm"]]
    coast_df = engine.cleaned["coastline"][["latitude", "longitude"]]
    flood_df = engine.cleaned["flood"][["latitude", "longitude"]]
    cyclone_df = engine.cleaned["cyclone"][["latitude", "longitude", "wind"]].copy()
    cyclone_df["wind"] = pd.to_numeric(cyclone_df["wind"], errors="coerce").fillna(0.0)

    rain_val = nearest_value(rain_df, latitude, longitude, "rainfall_mm")
    rain_risk = min_max_score(rain_val, float(rain_df["rainfall_mm"].min()), float(rain_df["rainfall_mm"].max()))
    coast_risk = coastline_risk_score(coast_df, latitude, longitude)
    elev_risk = elevation_risk_score(elevation_df, latitude, longitude)

    flood_proximity = nearest_distance_score(flood_df, latitude, longitude, scale=0.9)
    flood_model = float(row["flood_risk_index"])
    flood_risk = (0.88 * flood_model) + (0.12 * flood_proximity)

    cyclone_proximity = nearest_distance_score(cyclone_df, latitude, longitude, scale=1.1)
    d2_cyc = (cyclone_df["latitude"] - latitude) ** 2 + (cyclone_df["longitude"] - longitude) ** 2
    cyc_idx = int(d2_cyc.idxmin())
    nearest_wind = float(cyclone_df.at[cyc_idx, "wind"])
    wind_norm = min_max_score(nearest_wind, float(cyclone_df["wind"].min()), float(cyclone_df["wind"].max()))
    cyclone_model = float(row["cyclone_risk_index"])
    cyclone_risk = (0.72 * cyclone_model) + (0.13 * cyclone_proximity) + (0.15 * wind_norm)

    flood_history_points = pattern_score(flood_risk, references["flood_raw"], alert_quantile=0.97)
    cyclone_points = pattern_score(cyclone_risk, references["cyclone_raw"], alert_quantile=0.92)
    low_val = nearest_extreme_temp_value(low_extreme_df, latitude, longitude)
    high_val = nearest_extreme_temp_value(high_extreme_df, latitude, longitude)
    low_abs = abs(low_val) if low_val is not None else 0.0
    high_abs = high_val if high_val is not None else 0.0
    temperature_signal = max(low_abs, high_abs)
    temperature_points = pattern_score(temperature_signal, references["temperature_raw"], alert_quantile=0.90)
    rainfall_points = pattern_score(rain_val, references["rainfall_raw"], alert_quantile=0.92)
    sea_level_points = pattern_score(coast_risk, references["coastline_raw"], alert_quantile=0.92)

    flood_points = round((rainfall_points + cyclone_points + sea_level_points + temperature_points) / 4.0, 2)
    _ = flood_history_points

    return {
        "Cyclone": round(cyclone_points, 2),
        "Temperature": round(temperature_points, 2),
        "Rainfall": round(rainfall_points, 2),
        "Flood": flood_points,
        "Sea Level": round(sea_level_points, 2),
    }


def get_50_year_series(engine: ClimateLendingIntelligence, lat_bin: float, lon_bin: float, start_year: int) -> pd.DataFrame:
    assert engine.projection_features is not None
    end_year = start_year + 49
    series = engine.projection_features[
        (engine.projection_features["lat_bin"] == lat_bin)
        & (engine.projection_features["lon_bin"] == lon_bin)
        & (engine.projection_features["year"] >= start_year)
        & (engine.projection_features["year"] <= end_year)
    ][["year", "predicted_climate_risk"]].copy()
    return series.sort_values("year").reset_index(drop=True)


def build_policy_decision(
    annual_points: Dict[str, float],
    latitude: float,
    longitude: float,
    low_extreme_df: Optional[pd.DataFrame],
    high_extreme_df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    policy_reasons = []
    if annual_points["Flood"] > 45.0:
        policy_reasons.append(f"flood result {annual_points['Flood']:.2f} is greater than 45")
    if annual_points["Temperature"] > 50.0:
        policy_reasons.append(f"temperature score {annual_points['Temperature']:.2f} is greater than 50")
    very_high_risk_items = [f"{k} {v:.2f}" for k, v in annual_points.items() if v > 65.0]
    if len(very_high_risk_items) >= 2:
        policy_reasons.append("any two annual risk points above 65: " + ", ".join(very_high_risk_items))
    extreme_hit, extreme_reason = extreme_temperature_hits(float(latitude), float(longitude), low_extreme_df, high_extreme_df)
    if extreme_hit:
        policy_reasons.append(extreme_reason.rstrip("."))

    if policy_reasons:
        return {
            "decision": "Not Approved",
            "reason": "Auto-rejected: " + "; ".join(policy_reasons) + ".",
            "safe": False,
        }
    return {
        "decision": "Approved",
        "reason": f"Approved: flood result {annual_points['Flood']:.2f} is not greater than 45 and no extreme temperature hit.",
        "safe": True,
    }


def evaluate_loan_application(
    engine: ClimateLendingIntelligence,
    latitude: float,
    longitude: float,
    tenure_years: int,
    start_year: int,
    references: Dict[str, pd.Series],
    elevation_df: Optional[pd.DataFrame],
    low_extreme_df: Optional[pd.DataFrame],
    high_extreme_df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    tenure = engine.tenure_risk(
        latitude=float(latitude),
        longitude=float(longitude),
        tenure_years=int(tenure_years),
        start_year=int(start_year),
    )
    annual_points = build_annual_risk_points(
        engine=engine,
        latitude=float(latitude),
        longitude=float(longitude),
        lat_bin=float(tenure["lat_bin"]),
        lon_bin=float(tenure["lon_bin"]),
        start_year=int(start_year),
        references=references,
        elevation_df=elevation_df,
        low_extreme_df=low_extreme_df,
        high_extreme_df=high_extreme_df,
    )
    policy = build_policy_decision(
        annual_points=annual_points,
        latitude=float(latitude),
        longitude=float(longitude),
        low_extreme_df=low_extreme_df,
        high_extreme_df=high_extreme_df,
    )
    risk_reason = build_risk_reason_text(annual_points=annual_points, tenure=tenure)

    return {
        "tenure": tenure,
        "annual_points": annual_points,
        "decision": str(policy["decision"]),
        "reason": risk_reason,
        "safe": bool(policy["safe"]),
    }


@lru_cache(maxsize=1)
def load_runtime() -> Dict[str, object]:
    engine = ClimateLendingIntelligence(dataset_dir=str(PROJECT_ROOT / "dataset"))
    engine.load_and_clean()
    engine.build_historical_feature_table()
    engine.project_risk_50_years(start_year=DEFAULT_PROJECTION_START_YEAR, horizon_years=DEFAULT_PROJECTION_HORIZON)
    elevation_df = load_elevation_dataset()
    low_df, high_df = load_extreme_temperature_datasets()
    references = build_pattern_references(engine, low_df, high_df)
    return {
        "engine": engine,
        "elevation_df": elevation_df,
        "low_df": low_df,
        "high_df": high_df,
        "references": references,
    }


def evaluate_single_application(
    latitude: float,
    longitude: float,
    tenure_years: int,
    loan_amount: float,
    property_amount: float = 0.0,
    projection_start_year: int = DEFAULT_PROJECTION_START_YEAR,
    property_id: str = "98122",
) -> Dict[str, object]:
    runtime = load_runtime()
    engine: ClimateLendingIntelligence = runtime["engine"]  # type: ignore[assignment]
    elevation_df: Optional[pd.DataFrame] = runtime["elevation_df"]  # type: ignore[assignment]
    low_df: Optional[pd.DataFrame] = runtime["low_df"]  # type: ignore[assignment]
    high_df: Optional[pd.DataFrame] = runtime["high_df"]  # type: ignore[assignment]
    references: Dict[str, pd.Series] = runtime["references"]  # type: ignore[assignment]

    engine.project_risk_50_years(start_year=int(projection_start_year), horizon_years=DEFAULT_PROJECTION_HORIZON)

    eval_out = evaluate_loan_application(
        engine=engine,
        latitude=float(latitude),
        longitude=float(longitude),
        tenure_years=int(tenure_years),
        start_year=int(projection_start_year),
        references=references,
        elevation_df=elevation_df,
        low_extreme_df=low_df,
        high_extreme_df=high_df,
    )
    tenure = eval_out["tenure"]
    annual_points = eval_out["annual_points"]
    reason = str(eval_out["reason"])
    safe = bool(eval_out["safe"])
    elevation_ui = elevation_value_for_ui(elevation_df, float(latitude), float(longitude))
    climate_credit_score = climate_credit_score_from_annual_points(annual_points)

    series_50 = get_50_year_series(
        engine=engine,
        lat_bin=float(tenure["lat_bin"]),
        lon_bin=float(tenure["lon_bin"]),
        start_year=int(projection_start_year),
    )

    return {
        "reason": reason,
        "safe": safe,
        "safety_status": "SAFE" if safe else "NOT SAFE",
        "tenure": {
            "lat_bin": float(tenure["lat_bin"]),
            "lon_bin": float(tenure["lon_bin"]),
            "start_year": int(tenure["start_year"]),
            "end_year": int(tenure["end_year"]),
            "tenure_risk_score": float(tenure["tenure_risk_score"]),
            "tenure_risk_percent": float(tenure["tenure_risk_percent"]),
        },
        "annual_points": {
            "Cyclone": float(annual_points["Cyclone"]),
            "Temperature": float(annual_points["Temperature"]),
            "Rainfall": float(annual_points["Rainfall"]),
            "Flood": float(annual_points["Flood"]),
            "Sea Level": float(annual_points["Sea Level"]),
        },
        "climate_credit_score": int(climate_credit_score),
        "output_statements": {
            "property_climate_credit_score": build_property_score_text(str(property_id), int(climate_credit_score), annual_points),
            "loan_pricing_adjustment": build_pricing_adjustment_text(int(climate_credit_score), annual_points),
            "portfolio_risk_alert": build_portfolio_alert_text(
                concentration_pct=int(round(float(annual_points.get("Sea Level", 0.0)))),
                flood_mean=float(annual_points.get("Flood", 0.0)),
                cyclone_mean=float(annual_points.get("Cyclone", 0.0)),
                years=10,
            ),
            "explainability_log": build_explainability_log_text(annual_points),
        },
        "elevation_m": float(elevation_ui),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "series_50": [
            {"year": int(row.year), "predicted_climate_risk": float(row.predicted_climate_risk)}
            for row in series_50.itertuples(index=False)
        ],
    }


def analyze_portfolio(portfolio_df: pd.DataFrame, projection_start_year: int) -> Dict[str, object]:
    runtime = load_runtime()
    engine: ClimateLendingIntelligence = runtime["engine"]  # type: ignore[assignment]
    elevation_df: Optional[pd.DataFrame] = runtime["elevation_df"]  # type: ignore[assignment]
    low_df: Optional[pd.DataFrame] = runtime["low_df"]  # type: ignore[assignment]
    high_df: Optional[pd.DataFrame] = runtime["high_df"]  # type: ignore[assignment]
    references: Dict[str, pd.Series] = runtime["references"]  # type: ignore[assignment]

    clean_df = validate_portfolio_df(portfolio_df)
    engine.project_risk_50_years(start_year=int(projection_start_year), horizon_years=DEFAULT_PROJECTION_HORIZON)

    internal_rows = []
    output_rows = []
    approved_count = 0
    rejected_count = 0
    for row in clean_df.itertuples(index=False):
        eval_out = evaluate_loan_application(
            engine=engine,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            tenure_years=int(row.tenure_years),
            start_year=int(projection_start_year),
            references=references,
            elevation_df=elevation_df,
            low_extreme_df=low_df,
            high_extreme_df=high_df,
        )
        tenure = eval_out["tenure"]
        annual_points = eval_out["annual_points"]
        if str(eval_out["decision"]) == "Approved":
            approved_count += 1
        elif str(eval_out["decision"]) == "Not Approved":
            rejected_count += 1

        score_val = climate_credit_score_from_annual_points(annual_points)
        interest_adjustment_text = build_interest_adjustment_short_text(score_val)

        internal_rows.append(
            {
                "property_id": str(row.property_id),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "tenure_years": int(row.tenure_years),
                "tenure_risk_percent": round(float(tenure["tenure_risk_percent"]), 2),
                "Cyclone": round(float(annual_points["Cyclone"]), 2),
                "Temperature": round(float(annual_points["Temperature"]), 2),
                "Rainfall": round(float(annual_points["Rainfall"]), 2),
                "Flood": round(float(annual_points["Flood"]), 2),
                "Sea Level": round(float(annual_points["Sea Level"]), 2),
                "reason": str(eval_out["reason"]),
                "score": int(score_val),
                "interest_adjustment": interest_adjustment_text,
            }
        )
        output_rows.append(
            {
                "Property id": str(row.property_id),
                "Latitude": float(row.latitude),
                "Longitude": float(row.longitude),
                "Tenure years": int(row.tenure_years),
                "Interest adjustment": interest_adjustment_text,
                "Reason": str(eval_out["reason"]),
                "Score": f"{int(score_val)}/100",
            }
        )

    results_df = pd.DataFrame(internal_rows)
    avg_tenure_risk = float(results_df["tenure_risk_percent"].mean()) if not results_df.empty else np.nan
    portfolio_alert = "Portfolio Risk Alert: Not available."
    if not results_df.empty:
        coastal = results_df[results_df["Sea Level"] >= 60.0]
        base = coastal if not coastal.empty else results_df
        high = base[base["score"] < 50]
        pct = int(round((len(high) / max(len(base), 1)) * 100.0))
        flood_mean = float(base["Flood"].mean()) if "Flood" in base.columns else 0.0
        cyclone_mean = float(base["Cyclone"].mean()) if "Cyclone" in base.columns else 0.0
        portfolio_alert = build_portfolio_alert_text(
            concentration_pct=pct,
            flood_mean=flood_mean,
            cyclone_mean=cyclone_mean,
            years=10,
        )

    return {
        "total_records": int(len(clean_df)),
        "approved": approved_count,
        "not_approved": rejected_count,
        "average_tenure_risk": None if np.isnan(avg_tenure_risk) else round(avg_tenure_risk, 2),
        "portfolio_risk_alert": portfolio_alert,
        "results": output_rows,
    }


def metadata_payload() -> Dict[str, object]:
    return {
        "default_projection_start_year": DEFAULT_PROJECTION_START_YEAR,
        "default_projection_horizon_years": DEFAULT_PROJECTION_HORIZON,
        "states": INDIA_STATE_COORDS,
        "required_portfolio_columns": REQUIRED_PORTFOLIO_COLUMNS,
    }
