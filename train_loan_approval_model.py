import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split


DATASET_DIR = Path("dataset")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "loan_approval_model.json"


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_columns: List[str]
    target_column: str


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        return pd.Series(default, index=series.index, dtype=float)
    return out.fillna(out.median())


def _rounded_bins(df: pd.DataFrame, lat_col: str, lon_col: str, decimals: int = 0) -> pd.DataFrame:
    out = df.copy()
    out["lat_bin"] = pd.to_numeric(out[lat_col], errors="coerce").round(decimals)
    out["lon_bin"] = pd.to_numeric(out[lon_col], errors="coerce").round(decimals)
    return out.dropna(subset=["lat_bin", "lon_bin"])


def build_training_dataframe(random_seed: int = 42) -> DatasetBundle:
    rainfall = pd.read_csv(DATASET_DIR / "india_annual_rainfall.csv")
    tmax = pd.read_csv(DATASET_DIR / "india_tmax_final.csv")
    flood = pd.read_csv(DATASET_DIR / "flood_points_clean.csv")
    cyclone = pd.read_csv(DATASET_DIR / "cyclone_clean.csv")
    coastline = pd.read_csv(DATASET_DIR / "coastline_points.csv")

    rain_df = rainfall.rename(columns={"Latitude": "latitude", "Longitude": "longitude", "Rainfall_mm": "rainfall_mm"})
    tmax_df = tmax.rename(columns={"Latitude": "latitude", "Longitude": "longitude", "Tmax_C": "tmax_c"})
    climate = rain_df.merge(tmax_df, on=["latitude", "longitude"], how="inner")
    climate = climate.dropna(subset=["latitude", "longitude"])

    flood_bins = _rounded_bins(flood, "latitude", "longitude", decimals=0)
    flood_count = (
        flood_bins.groupby(["lat_bin", "lon_bin"]).size().rename("flood_event_count").reset_index()
    )

    cyclone = cyclone.rename(columns={"LAT": "latitude", "LON": "longitude", "WMO_WIND": "wmo_wind"})
    cyclone["wmo_wind"] = _safe_numeric(cyclone["wmo_wind"], default=0.0)
    cyclone_bins = _rounded_bins(cyclone, "latitude", "longitude", decimals=0)
    cyclone_agg = (
        cyclone_bins.groupby(["lat_bin", "lon_bin"])
        .agg(
            cyclone_event_count=("latitude", "size"),
            cyclone_mean_wind=("wmo_wind", "mean"),
        )
        .reset_index()
    )

    coast_bins = _rounded_bins(coastline, "latitude", "longitude", decimals=0)
    coast_india = coast_bins[
        coast_bins["lat_bin"].between(5, 38) & coast_bins["lon_bin"].between(67, 98)
    ]
    coast_density = coast_india.groupby(["lat_bin", "lon_bin"]).size().rename("coast_density").reset_index()

    base = _rounded_bins(climate, "latitude", "longitude", decimals=0)
    base = base.merge(flood_count, on=["lat_bin", "lon_bin"], how="left")
    base = base.merge(cyclone_agg, on=["lat_bin", "lon_bin"], how="left")
    base = base.merge(coast_density, on=["lat_bin", "lon_bin"], how="left")

    base["flood_event_count"] = base["flood_event_count"].fillna(0.0)
    base["cyclone_event_count"] = base["cyclone_event_count"].fillna(0.0)
    base["cyclone_mean_wind"] = base["cyclone_mean_wind"].fillna(0.0)
    base["coast_density"] = base["coast_density"].fillna(0.0)

    rng = np.random.default_rng(random_seed)
    n = len(base)
    base["tenure_years"] = rng.integers(5, 31, size=n)
    base["loan_to_value"] = rng.uniform(0.35, 0.95, size=n)
    base["loan_amount"] = rng.uniform(2_000_000, 20_000_000, size=n)
    base["property_value"] = base["loan_amount"] / base["loan_to_value"].clip(lower=0.1)
    asset_types = np.array(["Residential", "Commercial", "Infrastructure", "Industrial"])
    base["asset_type"] = rng.choice(asset_types, size=n, p=[0.45, 0.25, 0.15, 0.15])

    rain_norm = (base["rainfall_mm"] - base["rainfall_mm"].min()) / (
        (base["rainfall_mm"].max() - base["rainfall_mm"].min()) + 1e-9
    )
    heat_norm = (base["tmax_c"] - base["tmax_c"].min()) / ((base["tmax_c"].max() - base["tmax_c"].min()) + 1e-9)
    flood_norm = base["flood_event_count"] / (base["flood_event_count"].max() + 1e-9)
    cyclone_norm = base["cyclone_event_count"] / (base["cyclone_event_count"].max() + 1e-9)
    coast_norm = base["coast_density"] / (base["coast_density"].max() + 1e-9)

    climate_risk = (
        0.35 * rain_norm
        + 0.20 * heat_norm
        + 0.20 * flood_norm
        + 0.15 * cyclone_norm
        + 0.10 * coast_norm
    )
    finance_risk = 0.60 * base["loan_to_value"] + 0.40 * (base["tenure_years"] / 30.0)
    asset_risk_map = {
        "Residential": 0.25,
        "Commercial": 0.40,
        "Infrastructure": 0.55,
        "Industrial": 0.50,
    }
    asset_risk = base["asset_type"].map(asset_risk_map).fillna(0.40)
    noise = rng.normal(0.0, 0.05, size=n)

    overall_risk = (0.65 * climate_risk) + (0.25 * finance_risk) + (0.10 * asset_risk) + noise
    approval_score = 1.0 - overall_risk
    base["loan_approved"] = ((approval_score > 0.50) & (base["loan_to_value"] <= 0.90)).astype(int)

    asset_ohe = pd.get_dummies(base["asset_type"], prefix="asset", dtype=float)
    feature_columns = [
        "rainfall_mm",
        "tmax_c",
        "flood_event_count",
        "cyclone_event_count",
        "cyclone_mean_wind",
        "coast_density",
        "loan_to_value",
        "loan_amount",
        "property_value",
        "tenure_years",
    ]
    X = pd.concat([base[feature_columns], asset_ohe], axis=1)
    y = base["loan_approved"].astype(int)

    dataset = pd.concat([X, y.rename("loan_approved")], axis=1).dropna()
    return DatasetBundle(df=dataset, feature_columns=X.columns.tolist(), target_column="loan_approved")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fit_best_logistic(
    X_train_df: pd.DataFrame, y_train: np.ndarray, random_state: int = 42
) -> Tuple[LogisticRegression, float, float]:
    """Tune C and threshold on a validation split for stronger generalization."""
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train_df, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )

    c_grid = [0.2, 0.5, 1.0, 2.0, 5.0]
    best_c = 1.0
    best_f1 = -1.0

    # Small CV loop on sub-train for stable C selection.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for c in c_grid:
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_sub, y_sub):
            X_tr = X_sub.iloc[tr_idx]
            X_va = X_sub.iloc[va_idx]
            y_tr = y_sub[tr_idx]
            y_va = y_sub[va_idx]

            model = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", solver="lbfgs")
            model.fit(X_tr, y_tr)
            p_va = model.predict_proba(X_va)[:, 1]
            m = classification_metrics(y_va, p_va, threshold=0.5)
            fold_scores.append(m["f1"])

        score = float(np.mean(fold_scores))
        if score > best_f1:
            best_f1 = score
            best_c = c

    base_model = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", solver="lbfgs")
    base_model.fit(X_sub, y_sub)
    val_prob = base_model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_val_f1 = -1.0
    for th in np.linspace(0.30, 0.70, 81):
        f1 = classification_metrics(y_val, val_prob, threshold=float(th))["f1"]
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = float(th)

    final_model = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", solver="lbfgs")
    final_model.fit(X_train_df, y_train)
    return final_model, best_c, best_threshold


def main() -> None:
    bundle = build_training_dataframe(random_seed=42)
    df = bundle.df
    feature_columns = [c for c in df.columns if c != bundle.target_column]

    X_all = df[feature_columns]
    y_all = df[bundle.target_column].to_numpy(dtype=float)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    mean = X_train_df.mean()
    std = X_train_df.std(ddof=0).replace(0, 1.0)
    X_train_norm = (X_train_df - mean) / std
    X_test_norm = (X_test_df - mean) / std

    model, best_c, threshold = fit_best_logistic(X_train_norm, y_train, random_state=42)
    y_prob = model.predict_proba(X_test_norm)[:, 1]
    metrics = classification_metrics(y_test, y_prob, threshold=threshold)

    coeff = model.coef_.ravel()
    intercept = float(model.intercept_[0])
    # Keep artifact format compatible with runtime scorer: [intercept, feature weights...]
    weights = np.r_[intercept, coeff]

    payload = {
        "model_type": "logistic_regression_numpy",
        "target": bundle.target_column,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "weights": weights.tolist(),
        "normalization": {
            "mean": mean.to_dict(),
            "std": std.to_dict(),
        },
        "metrics": metrics,
        "train_rows": int(len(X_train_df)),
        "test_rows": int(len(X_test_df)),
        "best_c": best_c,
        "note": "loan_approved labels are synthetically derived from climate and finance risk. Metrics are for this synthetic task.",
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
