"""
MotogpPredictor — Production inference engine.
Loads trained ensemble + calibrator + feature extractor per category.
Handles per-race win probability prediction with Harville outright estimation.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from config import R0_DIR, R1_DIR, R2_DIR, HARVILLE_TOP_N, CATEGORY_MOTOGP, CATEGORY_MOTO2, CATEGORY_MOTO3
from ml.features import FEATURES, MotogpFeatureExtractor, _normalise_category
from ml.ensemble import MotogpEnsemble
from ml.calibrator import BetaCalibrator

logger = logging.getLogger(__name__)

ENSEMBLE_PKL = "ensemble.pkl"
CALIBRATOR_PKL = "calibrator.pkl"
EXTRACTOR_PKL = "extractor.pkl"


def _harville_top_n(win_probs: np.ndarray, top_n: int) -> np.ndarray:
    """
    Harville model for P(rider finishes in top-N).
    Uses simplified Harville approximation for large fields (>15 riders).
    Exact recursive Harville for small fields with top_n <= 3.
    """
    n = len(win_probs)
    if n == 0:
        return np.array([])

    win_probs = np.clip(win_probs, 1e-9, 1.0)
    probs = win_probs / win_probs.sum()

    top_n = min(top_n, n - 1)
    result = np.zeros(n)

    if n <= 50 and top_n == 3:
        # Exact Harville for podium (top 3)
        for i in range(n):
            p_win = probs[i]
            p_2nd = 0.0
            for j in range(n):
                if j == i:
                    continue
                rest = 1.0 - probs[j]
                if rest < 1e-9:
                    continue
                p_2nd += probs[j] * (probs[i] / rest)
            p_3rd = 0.0
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    rest_jk = 1.0 - probs[j] - probs[k]
                    if rest_jk < 1e-9:
                        continue
                    rest_j = 1.0 - probs[j]
                    if rest_j < 1e-9:
                        continue
                    p_3rd += probs[j] * (probs[k] / rest_j) * (probs[i] / rest_jk)
            result[i] = p_win + p_2nd + p_3rd
    else:
        # Simplified Harville approximation for large fields or top-N > 3
        for i in range(n):
            remaining_sum = 1.0
            p_not_in_top = 1.0
            for _ in range(top_n):
                p_this_slot = probs[i] / max(remaining_sum, 1e-9)
                p_not_in_top *= (1.0 - p_this_slot)
                remaining_sum = max(remaining_sum - probs[i], 1e-9)
            result[i] = 1.0 - p_not_in_top

    return np.clip(result, 0.0, 1.0)


class _CategoryModels:
    """Holds models for a single category."""
    def __init__(self, ensemble: MotogpEnsemble, calibrator: BetaCalibrator,
                 extractor: MotogpFeatureExtractor) -> None:
        self.ensemble = ensemble
        self.calibrator = calibrator
        self.extractor = extractor


class MotogpPredictor:
    """
    Production predictor for MotoGP, Moto2, Moto3.
    Loads R0 (MotoGP), R1 (Moto2), R2 (Moto3) — R1/R2 optional.
    """

    def __init__(self) -> None:
        self._models: dict[str, _CategoryModels] = {}
        self._loaded = False

    def load(
        self,
        r0_dir: str = R0_DIR,
        r1_dir: str = R1_DIR,
        r2_dir: str = R2_DIR,
    ) -> "MotogpPredictor":
        """Load model artefacts. R0 is required; R1/R2 are optional."""
        # Load R0 (MotoGP) — mandatory
        self._models[CATEGORY_MOTOGP] = self._load_category(r0_dir, CATEGORY_MOTOGP)
        logger.info(
            "MotoGP (R0) loaded: %d riders tracked",
            self._models[CATEGORY_MOTOGP].extractor.rider_count,
        )

        # Load R1 (Moto2) — optional
        r1_ensemble = os.path.join(r1_dir, ENSEMBLE_PKL)
        if os.path.exists(r1_ensemble):
            self._models[CATEGORY_MOTO2] = self._load_category(r1_dir, CATEGORY_MOTO2)
            logger.info(
                "Moto2 (R1) loaded: %d riders tracked",
                self._models[CATEGORY_MOTO2].extractor.rider_count,
            )
        else:
            logger.info("Moto2 (R1) models not found at %s — skipping", r1_dir)

        # Load R2 (Moto3) — optional
        r2_ensemble = os.path.join(r2_dir, ENSEMBLE_PKL)
        if os.path.exists(r2_ensemble):
            self._models[CATEGORY_MOTO3] = self._load_category(r2_dir, CATEGORY_MOTO3)
            logger.info(
                "Moto3 (R2) loaded: %d riders tracked",
                self._models[CATEGORY_MOTO3].extractor.rider_count,
            )
        else:
            logger.info("Moto3 (R2) models not found at %s — skipping", r2_dir)

        self._loaded = True
        logger.info(
            "MotogpPredictor loaded. Active categories: %s",
            list(self._models.keys()),
        )
        return self

    def _load_category(self, model_dir: str, category: str) -> _CategoryModels:
        ensemble_path = os.path.join(model_dir, ENSEMBLE_PKL)
        calibrator_path = os.path.join(model_dir, CALIBRATOR_PKL)
        extractor_path = os.path.join(model_dir, EXTRACTOR_PKL)

        if not os.path.exists(ensemble_path):
            raise FileNotFoundError(
                f"[{category}] Ensemble not found: {ensemble_path}"
            )
        if not os.path.exists(calibrator_path):
            raise FileNotFoundError(
                f"[{category}] Calibrator not found: {calibrator_path}"
            )
        if not os.path.exists(extractor_path):
            raise FileNotFoundError(
                f"[{category}] Extractor not found: {extractor_path}"
            )

        ensemble = MotogpEnsemble.load(ensemble_path)
        calibrator = BetaCalibrator.load(calibrator_path)
        extractor = MotogpFeatureExtractor.load(extractor_path)
        return _CategoryModels(ensemble, calibrator, extractor)

    def predict_race(
        self,
        riders: list[dict],
        category: str,
        circuit: str = "UNKNOWN",
        weather: str = "",
        track_condition: str = "",
        season: int = 2025,
        session_type: str = "RAC",
    ) -> list[dict[str, Any]]:
        """
        Predict win + podium + top-5 probabilities for a race field.

        riders: list of dicts with keys: name (str), team (str), constructor (str)
        category: 'MotoGP', 'Moto2', or 'Moto3'
        circuit: circuit name
        weather: weather description
        track_condition: track condition
        season: race year
        session_type: 'RAC' or 'SPR'

        Returns list sorted by win_prob desc:
            [{rider_name, team, constructor, win_prob, podium_prob, top5_prob}]
        """
        if not self._loaded:
            raise RuntimeError("Predictor not loaded — call load() first")
        if not riders:
            raise ValueError("riders list cannot be empty")

        cat_norm = _normalise_category(category)

        # Fallback to MotoGP model if specific category not available
        if cat_norm not in self._models:
            if CATEGORY_MOTOGP in self._models:
                logger.warning(
                    "No model for category %s — falling back to MotoGP model",
                    cat_norm,
                )
                cat_norm = CATEGORY_MOTOGP
            else:
                raise RuntimeError(f"No model available for category {category}")

        models = self._models[cat_norm]

        # Build feature rows
        feature_dicts = models.extractor.get_features_for_race(
            riders=riders,
            category=category,
            circuit=circuit,
            weather=weather,
            track_condition=track_condition,
            season=season,
            session_type=session_type,
        )

        if not feature_dicts:
            raise RuntimeError("Feature extractor returned empty result")

        X = pd.DataFrame(
            [{k: v for k, v in fd.items() if k in FEATURES} for fd in feature_dicts]
        )

        # Ensemble predict
        raw_probs = models.ensemble.predict_proba(X)

        # Calibrate
        cal_probs = models.calibrator.calibrate(raw_probs)

        # Normalise within race to sum=1
        total = cal_probs.sum()
        if total < 1e-9:
            cal_probs = np.ones(len(cal_probs)) / len(cal_probs)
        else:
            cal_probs = cal_probs / total

        # Harville podium (top 3) and top 5
        podium_probs = _harville_top_n(cal_probs, top_n=3)
        top5_probs = _harville_top_n(cal_probs, top_n=5)

        results = []
        for i, fd in enumerate(feature_dicts):
            results.append({
                "rider_name": fd.get("rider_name", "Unknown"),
                "team": fd.get("team", "UNKNOWN"),
                "constructor": fd.get("constructor", "UNKNOWN"),
                "win_prob": float(cal_probs[i]),
                "podium_prob": float(podium_probs[i]),
                "top5_prob": float(top5_probs[i]),
            })

        results.sort(key=lambda x: x["win_prob"], reverse=True)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def rider_count(self) -> int:
        """Total unique riders across all loaded category models."""
        if not self._loaded or CATEGORY_MOTOGP not in self._models:
            return 0
        return self._models[CATEGORY_MOTOGP].extractor.rider_count

    @property
    def active_categories(self) -> list[str]:
        return list(self._models.keys())
