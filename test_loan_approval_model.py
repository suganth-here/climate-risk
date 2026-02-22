import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from train_loan_approval_model import DATASET_DIR, build_training_dataframe


MODEL_PATH = Path("artifacts/loan_approval_model.json")
SAMPLE_PROPERTIES_PATH = Path("data/sample_properties.csv")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run train_loan_approval_model.py first."
        )
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def evaluate_on_holdout() -> None:
    model = load_model()
    bundle = build_training_dataframe(random_seed=2026)
    df = bundle.df
    feature_columns = model["feature_columns"]

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in test dataset: {missing}")

    X_df = df[feature_columns].copy()
    y_true = df[model["target"]].to_numpy(dtype=int)

    mean = pd.Series(model["normalization"]["mean"])
    std = pd.Series(model["normalization"]["std"]).replace(0, 1.0)
    X_norm = (X_df - mean) / std
    X = np.c_[np.ones(len(X_norm)), X_norm.to_numpy(dtype=float)]

    weights = np.array(model["weights"], dtype=float)
    probs = _sigmoid(X @ weights)
    preds = (probs >= model["threshold"]).astype(int)

    accuracy = float((preds == y_true).mean())
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    print("Test dataset evaluation")
    print(f"Rows: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix => TP:{tp} TN:{tn} FP:{fp} FN:{fn}")


def predict_sample_properties() -> None:
    if not SAMPLE_PROPERTIES_PATH.exists():
        print(f"Skipping sample prediction. File not found: {SAMPLE_PROPERTIES_PATH}")
        return

    model = load_model()

    rainfall = pd.read_csv(DATASET_DIR / "india_annual_rainfall.csv").rename(
        columns={"Latitude": "latitude", "Longitude": "longitude", "Rainfall_mm": "rainfall_mm"}
    )
    tmax = pd.read_csv(DATASET_DIR / "india_tmax_final.csv").rename(
        columns={"Latitude": "latitude", "Longitude": "longitude", "Tmax_C": "tmax_c"}
    )
    climate = rainfall.merge(tmax, on=["latitude", "longitude"], how="inner")

    props = pd.read_csv(SAMPLE_PROPERTIES_PATH)
    props = props.copy()

    def nearest_value(row: pd.Series, value_col: str) -> float:
        d = (climate["latitude"] - row["latitude"]) ** 2 + (climate["longitude"] - row["longitude"]) ** 2
        idx = int(d.idxmin())
        return float(climate.at[idx, value_col])

    props["rainfall_mm"] = props.apply(lambda r: nearest_value(r, "rainfall_mm"), axis=1)
    props["tmax_c"] = props.apply(lambda r: nearest_value(r, "tmax_c"), axis=1)
    props["flood_event_count"] = 0.0
    props["cyclone_event_count"] = 0.0
    props["cyclone_mean_wind"] = 0.0
    props["coast_density"] = 0.0
    props["loan_to_value"] = props["loan_amount"] / props["property_value"].replace(0, np.nan)
    props["loan_to_value"] = props["loan_to_value"].fillna(0.0)

    asset_ohe = pd.get_dummies(props["asset_type"], prefix="asset", dtype=float)
    feature_columns = model["feature_columns"]
    X_df = pd.concat([props, asset_ohe], axis=1)
    for col in feature_columns:
        if col not in X_df.columns:
            X_df[col] = 0.0
    X_df = X_df[feature_columns]

    mean = pd.Series(model["normalization"]["mean"])
    std = pd.Series(model["normalization"]["std"]).replace(0, 1.0)
    X_norm = (X_df - mean) / std
    X = np.c_[np.ones(len(X_norm)), X_norm.to_numpy(dtype=float)]
    weights = np.array(model["weights"], dtype=float)

    probs = _sigmoid(X @ weights)
    decisions = np.where(probs >= model["threshold"], "Loan Approved", "Loan Rejected")

    out = props[["property_id", "latitude", "longitude", "loan_amount", "property_value"]].copy()
    out["approval_probability"] = np.round(probs, 4)
    out["decision"] = decisions

    print("\nPredictions for sample properties")
    print(out.to_string(index=False))


if __name__ == "__main__":
    evaluate_on_holdout()
    predict_sample_properties()
