from dataclasses import dataclass


@dataclass
class LendingAdjustment:
    interest_rate_delta_pct: float
    tenure_delta_years: int
    insurance_premium_delta_pct: float
    risk_band: str


def lending_adjustment_from_score(score: int) -> LendingAdjustment:
    """Translate climate score to loan pricing/structure adjustments."""

    if score >= 80:
        return LendingAdjustment(0.00, 0, 0.0, "Low Risk")
    if score >= 65:
        return LendingAdjustment(0.30, -1, 4.0, "Moderate Risk")
    if score >= 50:
        return LendingAdjustment(0.70, -3, 9.0, "Elevated Risk")
    if score >= 35:
        return LendingAdjustment(1.20, -5, 15.0, "High Risk")
    return LendingAdjustment(1.75, -7, 22.0, "Severe Risk")
