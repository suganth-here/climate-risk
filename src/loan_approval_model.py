import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _rounded_bins(df: pd.DataFrame, lat_col: str, lon_col: str, decimals: int = 0) -> pd.DataFrame:
    out = df.copy()
    out["lat_bin"] = pd.to_numeric(out[lat_col], errors="coerce").round(decimals)
    out["lon_bin"] = pd.to_numeric(out[lon_col], errors="coerce").round(decimals)
    return out.dropna(subset=["lat_bin", "lon_bin"])


class LoanApprovalModel:
    def __init__(self, artifact_path: str = "artifacts/loan_approval_model.json", dataset_dir: str = "dataset"):
        self.artifact_path = Path(artifact_path)
        self.dataset_dir = Path(dataset_dir)
        self.model: Dict = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self._load_climate_sources()

    def _load_climate_sources(self) -> None:
        rainfall = pd.read_csv(self.dataset_dir / "india_annual_rainfall.csv").rename(
            columns={"Latitude": "latitude", "Longitude": "longitude", "Rainfall_mm": "rainfall_mm"}
        )
        tmax = pd.read_csv(self.dataset_dir / "india_tmax_final.csv").rename(
            columns={"Latitude": "latitude", "Longitude": "longitude", "Tmax_C": "tmax_c"}
        )
        self.climate = rainfall.merge(tmax, on=["latitude", "longitude"], how="inner")
        self.climate = self.climate.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

        flood = pd.read_csv(self.dataset_dir / "flood_points_clean.csv")
        flood_bins = _rounded_bins(flood, "latitude", "longitude", decimals=0)
        self.flood_count = flood_bins.groupby(["lat_bin", "lon_bin"]).size().rename("flood_event_count").reset_index()

        cyclone = pd.read_csv(self.dataset_dir / "cyclone_clean.csv").rename(
            columns={"LAT": "latitude", "LON": "longitude", "WMO_WIND": "wmo_wind"}
        )
        cyclone["wmo_wind"] = pd.to_numeric(cyclone["wmo_wind"], errors="coerce").fillna(0.0)
        cyclone_bins = _rounded_bins(cyclone, "latitude", "longitude", decimals=0)
        self.cyclone_agg = (
            cyclone_bins.groupby(["lat_bin", "lon_bin"])
            .agg(
                cyclone_event_count=("latitude", "size"),
                cyclone_mean_wind=("wmo_wind", "mean"),
            )
            .reset_index()
        )

        coastline = pd.read_csv(self.dataset_dir / "coastline_points.csv")
        coast_bins = _rounded_bins(coastline, "latitude", "longitude", decimals=0)
        coast_india = coast_bins[
            coast_bins["lat_bin"].between(5, 38) & coast_bins["lon_bin"].between(67, 98)
        ]
        self.coast_density = (
            coast_india.groupby(["lat_bin", "lon_bin"]).size().rename("coast_density").reset_index()
        )

    def _nearest_climate(self, lat: float, lon: float) -> Dict[str, float]:
        d = (self.climate["latitude"] - lat) ** 2 + (self.climate["longitude"] - lon) ** 2
        idx = int(d.idxmin())
        return {
            "rainfall_mm": float(self.climate.at[idx, "rainfall_mm"]),
            "tmax_c": float(self.climate.at[idx, "tmax_c"]),
        }

    @staticmethod
    def _lookup_count(df: pd.DataFrame, lat: float, lon: float, col: str) -> float:
        lat_bin = round(float(lat), 0)
        lon_bin = round(float(lon), 0)
        row = df[(df["lat_bin"] == lat_bin) & (df["lon_bin"] == lon_bin)]
        if row.empty:
            return 0.0
        return float(row.iloc[0][col])

    def build_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
        df["property_value"] = pd.to_numeric(df["property_value"], errors="coerce")
        df["tenure_years"] = pd.to_numeric(df["tenure_years"], errors="coerce")
        df["asset_type"] = df["asset_type"].astype(str)

        nearest = df.apply(lambda r: self._nearest_climate(float(r["latitude"]), float(r["longitude"])), axis=1)
        nearest_df = pd.DataFrame(list(nearest))
        df["rainfall_mm"] = nearest_df["rainfall_mm"]
        df["tmax_c"] = nearest_df["tmax_c"]

        df["flood_event_count"] = df.apply(
            lambda r: self._lookup_count(self.flood_count, r["latitude"], r["longitude"], "flood_event_count"), axis=1
        )
        df["cyclone_event_count"] = df.apply(
            lambda r: self._lookup_count(
                self.cyclone_agg, r["latitude"], r["longitude"], "cyclone_event_count"
            ),
            axis=1,
        )
        df["cyclone_mean_wind"] = df.apply(
            lambda r: self._lookup_count(self.cyclone_agg, r["latitude"], r["longitude"], "cyclone_mean_wind"), axis=1
        )
        df["coast_density"] = df.apply(
            lambda r: self._lookup_count(self.coast_density, r["latitude"], r["longitude"], "coast_density"), axis=1
        )
        df["loan_to_value"] = df["loan_amount"] / df["property_value"].replace(0, np.nan)
        df["loan_to_value"] = df["loan_to_value"].fillna(0.0)

        asset_ohe = pd.get_dummies(df["asset_type"], prefix="asset", dtype=float)
        X = pd.concat([df, asset_ohe], axis=1)

        feature_columns = self.model["feature_columns"]
        for col in feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        return X[feature_columns].copy()

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        X_df = self.build_features(input_df)
        mean = pd.Series(self.model["normalization"]["mean"])
        std = pd.Series(self.model["normalization"]["std"]).replace(0, 1.0)
        X_norm = (X_df - mean) / std
        X = np.c_[np.ones(len(X_norm)), X_norm.to_numpy(dtype=float)]

        weights = np.array(self.model["weights"], dtype=float)
        probs = _sigmoid(X @ weights)
        threshold = float(self.model.get("threshold", 0.5))
        decision = np.where(probs >= threshold, "Loan Approved", "Loan Rejected")

        out = input_df.copy()
        out["approval_probability"] = probs
        out["loan_decision"] = decision
        return out
