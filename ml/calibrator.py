"""Beta calibrator for MotoGP win probability."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class BetaCalibrator:
    """Platt-scaling (logistic) calibrator for probability outputs."""

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None
        self._isotonic: IsotonicRegression | None = None
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)
        log_odds = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
        self._lr = LogisticRegression(C=1.0, max_iter=1000)
        self._lr.fit(log_odds, y_true)
        cal_probs = self._lr.predict_proba(log_odds)[:, 1]
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(cal_probs, y_true)
        self.is_fitted = True
        logger.info("motogp_calibrator_fitted n=%d", len(y_true))

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted")
        raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)
        log_odds = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
        cal = self._lr.predict_proba(log_odds)[:, 1]
        cal = self._isotonic.predict(cal)
        return np.clip(cal, 0.02, 0.98)

    # Alias used by some callers
    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        return self.predict(raw_probs)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("motogp_calibrator_saved path=%s", path)

    @classmethod
    def load(cls, path: str) -> "BetaCalibrator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("motogp_calibrator_loaded path=%s", path)
        return obj
