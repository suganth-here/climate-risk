import json
from pathlib import Path

from src.climate_intelligence import ClimateLendingIntelligence


ARTIFACT_DIR = Path("artifacts")
METRICS_PATH = ARTIFACT_DIR / "climate_lending_metrics.json"


def main() -> None:
    engine = ClimateLendingIntelligence(dataset_dir="dataset")
    engine.load_and_clean()
    hist = engine.build_historical_feature_table()
    proj = engine.project_risk_50_years(start_year=2026, horizon_years=50)

    payload = {
        "historical_rows": int(len(hist)),
        "projection_rows": int(len(proj)),
        "quality_report": engine.quality_report,
        "note": (
            "RandomForest classifier requires real labeled dataset at data/loan_training_data.csv "
            "with columns latitude, longitude, loan_amount, tenure_years, loan_approved."
        ),
    }

    loan_path = Path("data/loan_training_data.csv")
    if loan_path.exists():
        bundle = engine.train_loan_classifier(str(loan_path))
        payload["classification_metrics"] = bundle.metrics
        payload["confusion_matrix"] = bundle.confusion_matrix
        payload["feature_importance"] = bundle.feature_importance
    else:
        payload["classification_metrics"] = None

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {METRICS_PATH}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
