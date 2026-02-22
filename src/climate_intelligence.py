from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_climate_datasets, normalize_series


@dataclass
class ModelBundle:
    model: object
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    feature_columns: List[str]


class ClimateLendingIntelligence:
    """End-to-end climate-informed intelligence built only on provided datasets."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.cleaned: Dict[str, pd.DataFrame] = {}
        self.quality_report: Dict[str, Dict[str, int]] = {}
        self.historical_features: Optional[pd.DataFrame] = None
        self.projection_features: Optional[pd.DataFrame] = None
        self.classifier_bundle: Optional[ModelBundle] = None

    @staticmethod
    def _bin_lat_lon(df: pd.DataFrame, decimals: int = 0) -> pd.DataFrame:
        out = df.copy()
        out["lat_bin"] = out["latitude"].round(decimals)
        out["lon_bin"] = out["longitude"].round(decimals)
        return out

    @staticmethod
    def _nearest_static_value(static_df: pd.DataFrame, lat: float, lon: float, value_col: str) -> float:
        d2 = (static_df["latitude"] - lat) ** 2 + (static_df["longitude"] - lon) ** 2
        idx = int(d2.idxmin())
        return float(static_df.at[idx, value_col])

    def load_and_clean(self) -> None:
        self.cleaned, self.quality_report = load_climate_datasets(str(self.dataset_dir))

    def build_historical_feature_table(self) -> pd.DataFrame:
        if not self.cleaned:
            self.load_and_clean()

        cyclone = self._bin_lat_lon(self.cleaned["cyclone"][["latitude", "longitude", "year", "wind"]].copy())
        flood = self._bin_lat_lon(self.cleaned["flood"][["latitude", "longitude", "year"]].copy())
        rain = self.cleaned["rainfall"][["latitude", "longitude", "rainfall_mm"]].copy()
        tmax = self.cleaned["tmax"][["latitude", "longitude", "tmax_c"]].copy()

        cyc_year = (
            cyclone.groupby(["lat_bin", "lon_bin", "year"])
            .agg(
                cyclone_events=("year", "size"),
                cyclone_mean_wind=("wind", "mean"),
            )
            .reset_index()
        )
        flood_year = (
            flood.groupby(["lat_bin", "lon_bin", "year"])
            .agg(flood_events=("year", "size"))
            .reset_index()
        )

        years = sorted(set(cyc_year["year"].unique()) | set(flood_year["year"].unique()))
        locs = pd.concat(
            [
                cyc_year[["lat_bin", "lon_bin"]],
                flood_year[["lat_bin", "lon_bin"]],
            ],
            axis=0,
        ).drop_duplicates()

        if len(locs) == 0 or len(years) == 0:
            raise ValueError("Insufficient temporal disaster data for yearly feature generation.")

        grid = (
            locs.assign(key=1)
            .merge(pd.DataFrame({"year": years, "key": 1}), on="key", how="inner")
            .drop(columns=["key"])
        )

        hist = grid.merge(cyc_year, on=["lat_bin", "lon_bin", "year"], how="left")
        hist = hist.merge(flood_year, on=["lat_bin", "lon_bin", "year"], how="left")
        hist["cyclone_events"] = hist["cyclone_events"].fillna(0.0)
        hist["cyclone_mean_wind"] = hist["cyclone_mean_wind"].fillna(0.0)
        hist["flood_events"] = hist["flood_events"].fillna(0.0)

        unique_locs = hist[["lat_bin", "lon_bin"]].drop_duplicates().reset_index(drop=True)
        rain_points = rain[["latitude", "longitude", "rainfall_mm"]].to_numpy(dtype=float)
        tmax_points = tmax[["latitude", "longitude", "tmax_c"]].to_numpy(dtype=float)
        loc_rows: List[Dict[str, float]] = []
        for _, r in unique_locs.iterrows():
            lat = float(r["lat_bin"])
            lon = float(r["lon_bin"])
            rain_d2 = (rain_points[:, 0] - lat) ** 2 + (rain_points[:, 1] - lon) ** 2
            tmax_d2 = (tmax_points[:, 0] - lat) ** 2 + (tmax_points[:, 1] - lon) ** 2
            rain_idx = int(np.argmin(rain_d2))
            tmax_idx = int(np.argmin(tmax_d2))
            loc_rows.append(
                {
                    "lat_bin": lat,
                    "lon_bin": lon,
                    "rainfall_mm": float(rain_points[rain_idx, 2]),
                    "tmax_c": float(tmax_points[tmax_idx, 2]),
                }
            )
        loc_static = pd.DataFrame(loc_rows)
        hist = hist.merge(loc_static, on=["lat_bin", "lon_bin"], how="left")

        flood_points = flood[["latitude", "longitude"]].to_numpy(dtype=float)
        cyclone_points = cyclone[["latitude", "longitude", "wind"]].to_numpy(dtype=float)
        spatial_rows: List[Dict[str, float]] = []
        flood_radius2 = 1.8 * 1.8
        cyclone_radius2 = 2.2 * 2.2
        for _, r in unique_locs.iterrows():
            lat = float(r["lat_bin"])
            lon = float(r["lon_bin"])

            flood_d2 = (flood_points[:, 0] - lat) ** 2 + (flood_points[:, 1] - lon) ** 2
            flood_local_density = float(np.sum(flood_d2 <= flood_radius2))

            cyc_d2 = (cyclone_points[:, 0] - lat) ** 2 + (cyclone_points[:, 1] - lon) ** 2
            cyc_mask = cyc_d2 <= cyclone_radius2
            cyclone_local_density = float(np.sum(cyc_mask))
            cyclone_local_wind = float(np.mean(cyclone_points[cyc_mask, 2])) if cyclone_local_density > 0 else 0.0

            spatial_rows.append(
                {
                    "lat_bin": lat,
                    "lon_bin": lon,
                    "flood_local_density": flood_local_density,
                    "cyclone_local_density": cyclone_local_density,
                    "cyclone_local_wind": cyclone_local_wind,
                }
            )
        spatial_df = pd.DataFrame(spatial_rows)
        hist = hist.merge(spatial_df, on=["lat_bin", "lon_bin"], how="left")

        hist["disaster_frequency"] = hist["cyclone_events"] + hist["flood_events"]
        hist["wind_severity"] = normalize_series(hist["cyclone_mean_wind"])
        hist["cyclone_local_wind_severity"] = normalize_series(hist["cyclone_local_wind"])
        hist["rain_severity"] = normalize_series(hist["rainfall_mm"])
        hist["heat_severity"] = normalize_series(hist["tmax_c"])
        hist["disaster_intensity"] = (hist["wind_severity"] + hist["rain_severity"] + hist["heat_severity"]) / 3.0

        flood_event_norm = normalize_series(hist["flood_events"])
        flood_local_norm = normalize_series(hist["flood_local_density"])
        cyclone_event_norm = normalize_series(hist["cyclone_events"])
        cyclone_local_norm = normalize_series(hist["cyclone_local_density"])
        hist["flood_risk_index"] = (0.55 * flood_event_norm) + (0.45 * flood_local_norm)
        hist["cyclone_risk_index"] = (
            0.40 * cyclone_event_norm
            + 0.25 * hist["wind_severity"]
            + 0.20 * cyclone_local_norm
            + 0.15 * hist["cyclone_local_wind_severity"]
        )
        hist["flood_risk_index"] = hist["flood_risk_index"].clip(0.0, 1.0)
        hist["cyclone_risk_index"] = hist["cyclone_risk_index"].clip(0.0, 1.0)
        hist["heat_risk_index"] = hist["heat_severity"]

        location_totals = (
            hist.groupby(["lat_bin", "lon_bin"])["disaster_frequency"].mean().reset_index(name="location_disaster_mean")
        )
        location_totals["location_risk_index"] = normalize_series(location_totals["location_disaster_mean"])
        hist = hist.merge(location_totals[["lat_bin", "lon_bin", "location_risk_index"]], on=["lat_bin", "lon_bin"], how="left")

        hist["climate_risk_score"] = (
            0.35 * hist["flood_risk_index"]
            + 0.35 * hist["cyclone_risk_index"]
            + 0.20 * hist["heat_risk_index"]
            + 0.10 * hist["location_risk_index"]
        )
        hist["climate_risk_score"] = hist["climate_risk_score"].clip(0.0, 1.0)

        self.historical_features = hist.sort_values(["lat_bin", "lon_bin", "year"]).reset_index(drop=True)
        return self.historical_features

    def project_risk_50_years(self, start_year: int = 2026, horizon_years: int = 50) -> pd.DataFrame:
        if self.historical_features is None:
            self.build_historical_feature_table()
        assert self.historical_features is not None
        hist = self.historical_features

        projection_years = np.arange(start_year, start_year + horizon_years)
        rows: List[Dict[str, float]] = []

        global_model = LinearRegression()
        global_model.fit(hist[["year"]], hist["climate_risk_score"])
        year_frame = pd.DataFrame({"year": projection_years})

        for (lat_bin, lon_bin), g in hist.groupby(["lat_bin", "lon_bin"]):
            g = g.sort_values("year")
            if g["year"].nunique() >= 3:
                model = LinearRegression()
                model.fit(g[["year"]], g["climate_risk_score"])
                pred = model.predict(year_frame)
            else:
                pred = global_model.predict(year_frame)

            pred = np.clip(pred, 0.0, 1.0)
            static_vals = g.iloc[-1]
            for y, p in zip(projection_years, pred):
                rows.append(
                    {
                        "lat_bin": float(lat_bin),
                        "lon_bin": float(lon_bin),
                        "year": int(y),
                        "predicted_climate_risk": float(p),
                        "location_risk_index": float(static_vals["location_risk_index"]),
                        "flood_risk_index": float(static_vals["flood_risk_index"]),
                        "cyclone_risk_index": float(static_vals["cyclone_risk_index"]),
                        "heat_risk_index": float(static_vals["heat_risk_index"]),
                        "disaster_frequency": float(static_vals["disaster_frequency"]),
                    }
                )

        self.projection_features = pd.DataFrame(rows)
        return self.projection_features

    def nearest_projection_location(self, latitude: float, longitude: float) -> Tuple[float, float]:
        if self.projection_features is None:
            self.project_risk_50_years()
        assert self.projection_features is not None
        locs = self.projection_features[["lat_bin", "lon_bin"]].drop_duplicates().reset_index(drop=True)
        d2 = (locs["lat_bin"] - latitude) ** 2 + (locs["lon_bin"] - longitude) ** 2
        row = locs.iloc[int(d2.to_numpy().argmin())]
        return float(row["lat_bin"]), float(row["lon_bin"])

    def tenure_risk(self, latitude: float, longitude: float, tenure_years: int, start_year: int = 2026) -> Dict[str, object]:
        if self.projection_features is None:
            self.project_risk_50_years(start_year=start_year, horizon_years=50)
        assert self.projection_features is not None
        lat_bin, lon_bin = self.nearest_projection_location(latitude, longitude)

        end_year = start_year + int(tenure_years) - 1
        min_year = int(self.projection_features["year"].min())
        max_year = int(self.projection_features["year"].max())
        if start_year < min_year or end_year > max_year:
            # Rebuild projections to match the requested tenure window.
            horizon = max(50, int(tenure_years))
            self.project_risk_50_years(start_year=start_year, horizon_years=horizon)
            assert self.projection_features is not None
            lat_bin, lon_bin = self.nearest_projection_location(latitude, longitude)

        subset = self.projection_features[
            (self.projection_features["lat_bin"] == lat_bin)
            & (self.projection_features["lon_bin"] == lon_bin)
            & (self.projection_features["year"] >= start_year)
            & (self.projection_features["year"] <= end_year)
        ].copy()
        if subset.empty:
            raise ValueError("No projected risk records found for selected tenure window.")

        tenure_risk_score = float(subset["predicted_climate_risk"].mean())
        return {
            "lat_bin": lat_bin,
            "lon_bin": lon_bin,
            "start_year": start_year,
            "end_year": end_year,
            "tenure_risk_score": tenure_risk_score,
            "tenure_risk_percent": round(tenure_risk_score * 100.0, 2),
            "series": subset[["year", "predicted_climate_risk"]].reset_index(drop=True),
        }

    @staticmethod
    def explain_decision(tenure_risk_score: float, flood_risk: float, cyclone_risk: float, heat_risk: float) -> Tuple[str, str]:
        # Base model decision; final policy rejections are applied in app layer.
        if tenure_risk_score > 0.62:
            decision = "Not Approved"
        elif tenure_risk_score > 0.45:
            decision = "Conditional Approval"
        else:
            decision = "Approved"

        top_driver = max(
            [("flood", flood_risk), ("cyclone", cyclone_risk), ("heat", heat_risk)],
            key=lambda x: x[1],
        )[0]

        if decision == "Not Approved":
            reason = f"High projected {top_driver} risk during loan tenure."
        elif decision == "Conditional Approval":
            reason = f"Moderate climate risk; {top_driver} is the dominant hazard."
        else:
            reason = "Projected climate risk is within safe lending threshold."
        return decision, reason

    def train_loan_classifier(self, loan_csv_path: str) -> ModelBundle:
        """Train classifier only when a labeled real loan dataset is provided."""
        loan_path = Path(loan_csv_path)
        if not loan_path.exists():
            raise FileNotFoundError(
                f"Labeled loan dataset not found: {loan_csv_path}. "
                "Provide a real dataset with target column loan_approved."
            )
        if self.historical_features is None:
            self.build_historical_feature_table()

        loan_df = pd.read_csv(loan_path)
        required = {"latitude", "longitude", "loan_amount", "tenure_years", "loan_approved"}
        missing = required - set(loan_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in loan dataset: {sorted(missing)}. "
                "Need these to train classification model using real labels."
            )

        loan_df["latitude"] = pd.to_numeric(loan_df["latitude"], errors="coerce")
        loan_df["longitude"] = pd.to_numeric(loan_df["longitude"], errors="coerce")
        loan_df["loan_amount"] = pd.to_numeric(loan_df["loan_amount"], errors="coerce")
        loan_df["tenure_years"] = pd.to_numeric(loan_df["tenure_years"], errors="coerce")
        loan_df["loan_approved"] = pd.to_numeric(loan_df["loan_approved"], errors="coerce")
        loan_df = loan_df.dropna(subset=["latitude", "longitude", "loan_amount", "tenure_years", "loan_approved"]).copy()

        feat_rows: List[Dict[str, float]] = []
        for _, row in loan_df.iterrows():
            tenure = self.tenure_risk(float(row["latitude"]), float(row["longitude"]), int(row["tenure_years"]))
            loc_slice = self.projection_features[
                (self.projection_features["lat_bin"] == tenure["lat_bin"])
                & (self.projection_features["lon_bin"] == tenure["lon_bin"])
            ].iloc[0]
            feat_rows.append(
                {
                    "tenure_risk_score": tenure["tenure_risk_score"],
                    "location_risk_index": float(loc_slice["location_risk_index"]),
                    "flood_risk_index": float(loc_slice["flood_risk_index"]),
                    "cyclone_risk_index": float(loc_slice["cyclone_risk_index"]),
                    "heat_risk_index": float(loc_slice["heat_risk_index"]),
                    "disaster_frequency": float(loc_slice["disaster_frequency"]),
                    "loan_amount": float(row["loan_amount"]),
                    "tenure_years": float(row["tenure_years"]),
                    "loan_approved": int(row["loan_approved"]),
                }
            )

        ds = pd.DataFrame(feat_rows)
        features = [c for c in ds.columns if c != "loan_approved"]
        X = ds[features]
        y = ds["loan_approved"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        metrics = {
            "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
            "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
            "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
            "f1": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        }
        cm = confusion_matrix(y_test, y_pred).tolist()

        pi = permutation_importance(rf, X_test, y_test, n_repeats=8, random_state=42)
        importance = dict(sorted(zip(features, pi.importances_mean), key=lambda x: x[1], reverse=True))

        self.classifier_bundle = ModelBundle(
            model=rf,
            metrics=metrics,
            confusion_matrix=cm,
            feature_importance={k: float(v) for k, v in importance.items()},
            feature_columns=features,
        )
        return self.classifier_bundle

    def baseline_logistic(self, loan_csv_path: str) -> ModelBundle:
        """Optional baseline model for benchmark comparison."""
        loan_path = Path(loan_csv_path)
        if not loan_path.exists():
            raise FileNotFoundError(f"Labeled loan dataset not found: {loan_csv_path}")
        if self.classifier_bundle is None:
            self.train_loan_classifier(loan_csv_path)
        assert self.classifier_bundle is not None

        loan_df = pd.read_csv(loan_path)
        loan_df = loan_df.dropna(subset=["latitude", "longitude", "loan_amount", "tenure_years", "loan_approved"]).copy()
        rows = []
        for _, row in loan_df.iterrows():
            tenure = self.tenure_risk(float(row["latitude"]), float(row["longitude"]), int(row["tenure_years"]))
            rows.append(
                {
                    "tenure_risk_score": tenure["tenure_risk_score"],
                    "loan_amount": float(row["loan_amount"]),
                    "tenure_years": float(row["tenure_years"]),
                    "loan_approved": int(row["loan_approved"]),
                }
            )
        ds = pd.DataFrame(rows)
        X = ds[["tenure_risk_score", "loan_amount", "tenure_years"]]
        y = ds["loan_approved"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = {
            "accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
            "precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
            "recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
            "f1": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
        }
        cm = confusion_matrix(y_test, y_pred).tolist()
        return ModelBundle(model=clf, metrics=metrics, confusion_matrix=cm, feature_importance={}, feature_columns=list(X.columns))
