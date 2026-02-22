import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.climate_engine import ClimateRiskEngine, PropertyInput
from src.data_loader import validate_portfolio_df

DATA_PATH = Path("data/sample_properties.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "basic_linear_model.json"


def build_target(df: pd.DataFrame) -> pd.Series:
    if "climate_credit_score" in df.columns:
        return pd.to_numeric(df["climate_credit_score"], errors="coerce")

    engine = ClimateRiskEngine(horizon_years=50)
    scores = []
    for _, row in df.iterrows():
        p = PropertyInput(
            property_id=str(row["property_id"]),
            location_name="",
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            loan_amount=float(row["loan_amount"]),
            property_value=float(row["property_value"]),
            tenure_years=int(row["tenure_years"]),
            asset_type=str(row["asset_type"]),
        )
        score, _, _ = engine.climate_credit_score(p)
        scores.append(score)
    return pd.Series(scores, name="climate_credit_score")


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = df[["latitude", "longitude", "loan_amount", "property_value", "tenure_years"]].copy()
    base["loan_to_value"] = base["loan_amount"] / base["property_value"].replace(0, np.nan)
    base["loan_to_value"] = base["loan_to_value"].fillna(0.0)

    asset_ohe = pd.get_dummies(df["asset_type"], prefix="asset", dtype=float)
    X = pd.concat([base, asset_ohe], axis=1)

    metadata = {
        "feature_columns": X.columns.tolist(),
        "asset_categories": sorted(df["asset_type"].astype(str).unique().tolist()),
    }
    return X, metadata


def train_linear_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    # Add intercept column.
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
    intercept = float(coeffs[0])
    weights = coeffs[1:]
    return weights, intercept


def predict(X: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return X @ weights + intercept


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    df = validate_portfolio_df(raw_df)

    y = build_target(raw_df)
    if y.isna().any():
        raise ValueError("Target column has invalid values.")

    X_df, metadata = build_features(df)

    # Basic train/test split for demonstration.
    rng = np.random.default_rng(42)
    idx = np.arange(len(X_df))
    rng.shuffle(idx)

    split = max(1, int(0.8 * len(idx)))
    train_idx = idx[:split]
    test_idx = idx[split:]
    if len(test_idx) == 0:
        test_idx = train_idx

    X_train = X_df.iloc[train_idx].to_numpy(dtype=float)
    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    X_test = X_df.iloc[test_idx].to_numpy(dtype=float)
    y_test = y.iloc[test_idx].to_numpy(dtype=float)

    weights, intercept = train_linear_regression(X_train, y_train)
    y_pred = predict(X_test, weights, intercept)

    metrics = {
        "test_mae": round(mae(y_test, y_pred), 4),
        "test_r2": round(r2(y_test, y_pred), 4),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "linear_regression_numpy",
        "intercept": intercept,
        "weights": {name: float(w) for name, w in zip(X_df.columns.tolist(), weights)},
        "metadata": metadata,
        "metrics": metrics,
    }

    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Saved model artifact: {MODEL_PATH}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
