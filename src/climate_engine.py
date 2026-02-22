from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class PropertyInput:
    property_id: str
    location_name: str
    latitude: float
    longitude: float
    loan_amount: float
    property_value: float
    tenure_years: int
    asset_type: str


class ClimateRiskEngine:
    """Dataset-aware climate risk engine for lending use-cases.

    Scores range from 0 to 100 where higher is better (lower climate risk).
    """

    def __init__(self, horizon_years: int = 50, dataset_dir: str = "dataset"):
        self.horizon_years = horizon_years
        self.dataset_dir = Path(dataset_dir)
        self.weights = {
            "flood": 0.30,
            "storm": 0.25,
            "heat": 0.25,
            "sea_level": 0.20,
        }
        self._loaded = False
        self._load_sources()

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(max(lo, min(hi, value)))

    @staticmethod
    def _min_max_norm(value: float, lo: float, hi: float) -> float:
        return float((value - lo) / ((hi - lo) + 1e-9))

    @staticmethod
    def _nearest_index(points: np.ndarray, lat: float, lon: float) -> int:
        d2 = (points[:, 0] - lat) ** 2 + (points[:, 1] - lon) ** 2
        return int(np.argmin(d2))

    @staticmethod
    def _distance_to_nearest(points: np.ndarray, lat: float, lon: float) -> float:
        d2 = (points[:, 0] - lat) ** 2 + (points[:, 1] - lon) ** 2
        return float(np.sqrt(np.min(d2)))

    def _local_counts(self, points: np.ndarray, radius: float) -> np.ndarray:
        if len(points) == 0:
            return np.array([0.0], dtype=float)
        sample_size = min(500, len(points))
        sample_idx = np.linspace(0, len(points) - 1, num=sample_size, dtype=int)
        sample = points[sample_idx]
        r2 = radius * radius
        counts = []
        for lat, lon in sample:
            d2 = (points[:, 0] - lat) ** 2 + (points[:, 1] - lon) ** 2
            counts.append(float(np.sum(d2 <= r2)))
        return np.array(counts, dtype=float)

    def _load_sources(self) -> None:
        try:
            rain = pd.read_csv(self.dataset_dir / "india_annual_rainfall.csv").rename(
                columns={"Latitude": "latitude", "Longitude": "longitude", "Rainfall_mm": "rainfall_mm"}
            )
            tmax = pd.read_csv(self.dataset_dir / "india_tmax_final.csv").rename(
                columns={"Latitude": "latitude", "Longitude": "longitude", "Tmax_C": "tmax_c"}
            )
            climate = rain.merge(tmax, on=["latitude", "longitude"], how="inner")
            climate = climate.dropna(subset=["latitude", "longitude", "rainfall_mm", "tmax_c"]).reset_index(drop=True)
            self._climate_latlon = climate[["latitude", "longitude"]].to_numpy(dtype=float)
            self._rain = climate["rainfall_mm"].to_numpy(dtype=float)
            self._tmax = climate["tmax_c"].to_numpy(dtype=float)
            self._rain_min = float(np.min(self._rain))
            self._rain_max = float(np.max(self._rain))
            self._tmax_min = float(np.min(self._tmax))
            self._tmax_max = float(np.max(self._tmax))

            flood = pd.read_csv(self.dataset_dir / "flood_points_clean.csv").rename(
                columns={"LAT": "latitude", "LON": "longitude"}
            )
            flood = flood.dropna(subset=["latitude", "longitude"])
            self._flood_latlon = flood[["latitude", "longitude"]].to_numpy(dtype=float)
            self._flood_ref = max(1.0, float(np.percentile(self._local_counts(self._flood_latlon, radius=1.5), 90)))

            cyclone = pd.read_csv(self.dataset_dir / "cyclone_clean.csv").rename(
                columns={"LAT": "latitude", "LON": "longitude", "WMO_WIND": "wmo_wind"}
            )
            cyclone["wmo_wind"] = pd.to_numeric(cyclone["wmo_wind"], errors="coerce").fillna(0.0)
            cyclone = cyclone.dropna(subset=["latitude", "longitude"])
            self._cyclone_latlon = cyclone[["latitude", "longitude"]].to_numpy(dtype=float)
            self._cyclone_wind = cyclone["wmo_wind"].to_numpy(dtype=float)
            self._cyclone_ref = max(
                1.0, float(np.percentile(self._local_counts(self._cyclone_latlon, radius=2.0), 90))
            )

            coast = pd.read_csv(self.dataset_dir / "coastline_points.csv").rename(
                columns={"LAT": "latitude", "LON": "longitude"}
            )
            coast = coast.dropna(subset=["latitude", "longitude"])
            self._coast_latlon = coast[["latitude", "longitude"]].to_numpy(dtype=float)
            self._loaded = True
        except Exception:
            self._loaded = False

    def _fallback_hazards(self, latitude: float, longitude: float) -> Dict[str, float]:
        abs_lat = abs(latitude)
        tropical_band = self._clamp(1.0 - (abs(abs_lat - 15.0) / 25.0))
        equatorial_heat = self._clamp(1.0 - (abs_lat / 45.0))
        coastal_proxy = max(
            self._clamp(1.0 - abs(longitude - 72.5) / 6.5),
            self._clamp(1.0 - abs(longitude - 80.5) / 6.5),
            self._clamp(1.0 - abs(longitude - 88.0) / 6.5),
        )
        monsoon_proxy = self._clamp(0.6 * tropical_band + 0.4 * coastal_proxy)
        return {
            "flood": self._clamp(0.18 + 0.52 * monsoon_proxy + 0.23 * coastal_proxy),
            "storm": self._clamp(0.10 + 0.64 * coastal_proxy + 0.18 * tropical_band),
            "heat": self._clamp(0.22 + 0.60 * equatorial_heat + 0.06 * monsoon_proxy),
            "sea_level": self._clamp(0.05 + 0.90 * coastal_proxy),
        }

    def estimate_hazards(self, latitude: float, longitude: float) -> Dict[str, float]:
        """Estimate normalized hazard projections (0-1) from local climate/event data."""
        if not self._loaded:
            return self._fallback_hazards(latitude, longitude)

        idx = self._nearest_index(self._climate_latlon, latitude, longitude)
        rainfall_mm = float(self._rain[idx])
        tmax_c = float(self._tmax[idx])

        rain_norm = self._clamp(self._min_max_norm(rainfall_mm, self._rain_min, self._rain_max))
        heat_norm = self._clamp(self._min_max_norm(tmax_c, self._tmax_min, self._tmax_max))

        flood_d2 = (self._flood_latlon[:, 0] - latitude) ** 2 + (self._flood_latlon[:, 1] - longitude) ** 2
        flood_count = float(np.sum(flood_d2 <= (1.5 * 1.5)))
        flood_norm = self._clamp(flood_count / self._flood_ref)

        cyc_d2 = (self._cyclone_latlon[:, 0] - latitude) ** 2 + (self._cyclone_latlon[:, 1] - longitude) ** 2
        cyc_mask = cyc_d2 <= (2.0 * 2.0)
        cyclone_count = float(np.sum(cyc_mask))
        cyclone_count_norm = self._clamp(cyclone_count / self._cyclone_ref)
        if cyclone_count > 0:
            wind_mean = float(np.mean(self._cyclone_wind[cyc_mask]))
            wind_norm = self._clamp((wind_mean - 20.0) / 120.0)
        else:
            wind_norm = 0.0

        dist_to_coast = self._distance_to_nearest(self._coast_latlon, latitude, longitude)
        coast_proximity = self._clamp(np.exp(-dist_to_coast / 2.0))

        flood = self._clamp(0.50 * rain_norm + 0.35 * flood_norm + 0.15 * coast_proximity)
        storm = self._clamp(0.40 * cyclone_count_norm + 0.30 * wind_norm + 0.30 * coast_proximity)
        heat = self._clamp(0.70 * heat_norm + 0.20 * (1.0 - rain_norm) + 0.10 * (1.0 - coast_proximity))
        sea_level = self._clamp(coast_proximity)

        return {
            "flood": flood,
            "storm": storm,
            "heat": heat,
            "sea_level": sea_level,
        }

    def climate_risk_index(self, hazards: Dict[str, float]) -> float:
        return float(sum(self.weights[k] * hazards[k] for k in self.weights))

    def climate_credit_score(self, property_data: PropertyInput) -> Tuple[int, Dict[str, float], float]:
        hazards = self.estimate_hazards(property_data.latitude, property_data.longitude)
        risk_index = self.climate_risk_index(hazards)
        score = int(round((1.0 - risk_index) * 100.0))
        score = max(0, min(100, score))
        return score, hazards, risk_index

    def explainability_log(self, hazards: Dict[str, float]) -> str:
        ranked = sorted(hazards.items(), key=lambda x: x[1], reverse=True)
        top = ranked[:3]
        parts = [f"{name} projection={value:.2f}" for name, value in top]
        return "Score drivers: " + ", ".join(parts) + "."
