from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "property_id",
    "latitude",
    "longitude",
    "tenure_years",
}


COLUMN_ALIASES = {
    "propertyid": "property_id",
    "property_id": "property_id",
    "latitude": "latitude",
    "longitude": "longitude",
    "tenureyears": "tenure_years",
    "tenure_years": "tenure_years",
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("-", "").replace("__", "_")


def _rename_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        normalized = _normalize_name(col)
        mapped = COLUMN_ALIASES.get(normalized)
        if mapped:
            rename_map[col] = mapped
    renamed = df.rename(columns=rename_map)
    if not renamed.columns.duplicated().any():
        return renamed

    # Merge duplicate canonical columns by taking the first non-null value row-wise.
    merged = pd.DataFrame(index=renamed.index)
    for col_name in pd.unique(renamed.columns):
        col_block = renamed.loc[:, renamed.columns == col_name]
        if col_block.shape[1] == 1:
            merged[col_name] = col_block.iloc[:, 0]
        else:
            merged[col_name] = col_block.bfill(axis=1).iloc[:, 0]
    return merged


def validate_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_with_aliases(df)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["property_id"] = result["property_id"].astype(str)
    numeric_cols = ["latitude", "longitude", "tenure_years"]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    if result[numeric_cols].isna().any().any():
        raise ValueError("Portfolio CSV contains invalid numeric values.")

    return result[["property_id", "latitude", "longitude", "tenure_years"]].copy()


def _range_filter_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)].copy()


def _dedupe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def load_climate_datasets(dataset_dir: str = "dataset") -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, int]]]:
    """Load and strictly clean climate datasets using only provided files."""
    root = Path(dataset_dir)
    files = {
        "cyclone": root / "cyclone_clean.csv",
        "flood": root / "flood_points_clean.csv",
        "rainfall": root / "india_annual_rainfall.csv",
        "tmax": root / "india_tmax_final.csv",
        "coastline": root / "coastline_points.csv",
    }
    missing_files = [str(p) for p in files.values() if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required climate datasets: {missing_files}")

    quality: Dict[str, Dict[str, int]] = {}
    out: Dict[str, pd.DataFrame] = {}

    cyclone = pd.read_csv(files["cyclone"])
    cyclone = cyclone.rename(columns={"LAT": "latitude", "LON": "longitude", "ISO_TIME": "time", "WMO_WIND": "wind"})
    cyclone["time"] = pd.to_datetime(cyclone["time"], errors="coerce")
    cyclone["year"] = cyclone["time"].dt.year
    cyclone["latitude"] = pd.to_numeric(cyclone["latitude"], errors="coerce")
    cyclone["longitude"] = pd.to_numeric(cyclone["longitude"], errors="coerce")
    cyclone["wind"] = pd.to_numeric(cyclone["wind"], errors="coerce")
    cyclone_missing = int(cyclone[["latitude", "longitude", "year"]].isna().any(axis=1).sum())
    cyclone = cyclone.dropna(subset=["latitude", "longitude", "year"]).copy()
    cyclone["wind"] = cyclone["wind"].interpolate(limit_direction="both")
    cyclone["wind"] = cyclone["wind"].fillna(cyclone["wind"].median())
    cyclone = _range_filter_lat_lon(cyclone)
    cyclone, cyclone_dups = _dedupe(cyclone)
    out["cyclone"] = cyclone
    quality["cyclone"] = {"dropped_missing_critical": cyclone_missing, "duplicates_removed": cyclone_dups}

    flood = pd.read_csv(files["flood"])
    flood["latitude"] = pd.to_numeric(flood["latitude"], errors="coerce")
    flood["longitude"] = pd.to_numeric(flood["longitude"], errors="coerce")
    flood["year"] = pd.to_numeric(flood["year"], errors="coerce")
    flood_missing = int(flood[["latitude", "longitude", "year"]].isna().any(axis=1).sum())
    flood = flood.dropna(subset=["latitude", "longitude", "year"]).copy()
    flood["year"] = flood["year"].astype(int)
    flood = _range_filter_lat_lon(flood)
    flood, flood_dups = _dedupe(flood)
    out["flood"] = flood
    quality["flood"] = {"dropped_missing_critical": flood_missing, "duplicates_removed": flood_dups}

    rainfall = pd.read_csv(files["rainfall"]).rename(
        columns={"Latitude": "latitude", "Longitude": "longitude", "Rainfall_mm": "rainfall_mm"}
    )
    rainfall["latitude"] = pd.to_numeric(rainfall["latitude"], errors="coerce")
    rainfall["longitude"] = pd.to_numeric(rainfall["longitude"], errors="coerce")
    rainfall["rainfall_mm"] = pd.to_numeric(rainfall["rainfall_mm"], errors="coerce")
    rain_missing = int(rainfall[["latitude", "longitude", "rainfall_mm"]].isna().any(axis=1).sum())
    rainfall = rainfall.dropna(subset=["latitude", "longitude", "rainfall_mm"]).copy()
    rainfall["rainfall_mm"] = rainfall["rainfall_mm"].interpolate(limit_direction="both")
    rainfall = _range_filter_lat_lon(rainfall)
    rainfall, rain_dups = _dedupe(rainfall)
    out["rainfall"] = rainfall
    quality["rainfall"] = {"dropped_missing_critical": rain_missing, "duplicates_removed": rain_dups}

    tmax = pd.read_csv(files["tmax"]).rename(columns={"Latitude": "latitude", "Longitude": "longitude", "Tmax_C": "tmax_c"})
    tmax["latitude"] = pd.to_numeric(tmax["latitude"], errors="coerce")
    tmax["longitude"] = pd.to_numeric(tmax["longitude"], errors="coerce")
    tmax["tmax_c"] = pd.to_numeric(tmax["tmax_c"], errors="coerce")
    tmax_missing = int(tmax[["latitude", "longitude", "tmax_c"]].isna().any(axis=1).sum())
    tmax = tmax.dropna(subset=["latitude", "longitude", "tmax_c"]).copy()
    tmax["tmax_c"] = tmax["tmax_c"].interpolate(limit_direction="both")
    tmax = _range_filter_lat_lon(tmax)
    tmax, tmax_dups = _dedupe(tmax)
    out["tmax"] = tmax
    quality["tmax"] = {"dropped_missing_critical": tmax_missing, "duplicates_removed": tmax_dups}

    coastline = pd.read_csv(files["coastline"]).rename(columns={"LAT": "latitude", "LON": "longitude"})
    coastline["latitude"] = pd.to_numeric(coastline["latitude"], errors="coerce")
    coastline["longitude"] = pd.to_numeric(coastline["longitude"], errors="coerce")
    coast_missing = int(coastline[["latitude", "longitude"]].isna().any(axis=1).sum())
    coastline = coastline.dropna(subset=["latitude", "longitude"]).copy()
    coastline = _range_filter_lat_lon(coastline)
    coastline, coast_dups = _dedupe(coastline)
    out["coastline"] = coastline
    quality["coastline"] = {"dropped_missing_critical": coast_missing, "duplicates_removed": coast_dups}

    return out, quality


def normalize_series(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if np.isclose(hi, lo):
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)
