"""
MotoGP Model Trainer
Temporal split: seasons 2002-2017 train, 2018-2020 val, 2021-2025 test
Trains 3 models: R0=MotoGP, R1=Moto2, R2=Moto3
Saves: ensemble.pkl, calibrator.pkl, extractor.pkl per regime dir.
Reports: AUC, Brier score on win prediction (test set)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import (
    MOTOGP_CSV,
    R0_DIR,
    R1_DIR,
    R2_DIR,
    TRAIN_SEASONS_MAX,
    VAL_SEASONS_MIN,
    VAL_SEASONS_MAX,
    TEST_SEASONS_MIN,
    CATEGORY_MOTOGP,
    CATEGORY_MOTO2,
    CATEGORY_MOTO3,
)
from ml.features import FEATURES, MotogpFeatureExtractor
from ml.ensemble import MotogpEnsemble
from ml.calibrator import BetaCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MIN_ROWS_FOR_CLASS = 3000   # Minimum train rows to train Moto2/Moto3


class MotogpTrainer:
    """Trains MotoGP race-winner prediction models per category."""

    def train(
        self,
        csv_path: str = MOTOGP_CSV,
    ) -> dict:
        """
        Full training pipeline for all 3 categories.
        Returns dict with metrics per category.
        """
        t_start = time.time()

        logger.info("=== STEP 1: Building full feature dataset ===")
        extractor = MotogpFeatureExtractor()
        full_dataset = extractor.build_dataset(csv_path)

        if full_dataset.empty:
            raise RuntimeError("Feature extraction returned empty dataset")

        logger.info(
            "Full dataset: %d rows, target balance=%.4f, categories: %s",
            len(full_dataset),
            full_dataset["target"].mean(),
            full_dataset["category"].value_counts().to_dict(),
        )

        all_metrics: dict[str, dict] = {}

        # Train per category using the shared extractor (state includes all categories)
        for cat_label, out_dir in [
            (CATEGORY_MOTOGP, R0_DIR),
            (CATEGORY_MOTO2, R1_DIR),
            (CATEGORY_MOTO3, R2_DIR),
        ]:
            cat_df = full_dataset[full_dataset["category"] == cat_label].copy()
            logger.info(
                "Category %s: %d rows, %d positive",
                cat_label, len(cat_df), cat_df["target"].sum(),
            )

            if len(cat_df) < MIN_ROWS_FOR_CLASS:
                logger.warning(
                    "Skipping %s — only %d rows (need %d)",
                    cat_label, len(cat_df), MIN_ROWS_FOR_CLASS,
                )
                continue

            metrics = self._train_category(
                dataset=cat_df,
                extractor=extractor,
                out_dir=out_dir,
                category=cat_label,
            )
            all_metrics[cat_label] = metrics

        elapsed = time.time() - t_start
        logger.info("All training complete in %.1fs", elapsed)
        logger.info("Summary: %s", all_metrics)
        return all_metrics

    def _train_category(
        self,
        dataset: pd.DataFrame,
        extractor: MotogpFeatureExtractor,
        out_dir: str,
        category: str,
    ) -> dict:
        """Train ensemble + calibrator for a single category."""
        t_start = time.time()
        os.makedirs(out_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # 2. Temporal split
        # ----------------------------------------------------------------
        logger.info("=== [%s] Temporal split ===", category)
        train_df = dataset[dataset["season"] <= TRAIN_SEASONS_MAX].copy()
        val_df = dataset[
            (dataset["season"] >= VAL_SEASONS_MIN)
            & (dataset["season"] <= VAL_SEASONS_MAX)
        ].copy()
        test_df = dataset[dataset["season"] >= TEST_SEASONS_MIN].copy()

        logger.info(
            "[%s] Split sizes — train=%d (≤%d), val=%d (%d-%d), test=%d (≥%d)",
            category,
            len(train_df), TRAIN_SEASONS_MAX,
            len(val_df), VAL_SEASONS_MIN, VAL_SEASONS_MAX,
            len(test_df), TEST_SEASONS_MIN,
        )
        logger.info(
            "[%s] Target rates — train=%.4f val=%.4f test=%.4f",
            category,
            train_df["target"].mean(),
            val_df["target"].mean(),
            test_df["target"].mean(),
        )

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise RuntimeError(
                f"[{category}] One or more splits are empty — check season filtering"
            )
        if train_df["target"].sum() == 0:
            raise RuntimeError(f"[{category}] Training set has zero positive examples")

        X_train = train_df[FEATURES]
        y_train = train_df["target"]
        groups_train = train_df["race_key"]

        X_val = val_df[FEATURES]
        y_val = val_df["target"]

        X_test = test_df[FEATURES]
        y_test = test_df["target"]

        # ----------------------------------------------------------------
        # 3. Ensemble training
        # ----------------------------------------------------------------
        logger.info("=== [%s] Ensemble training ===", category)
        ensemble = MotogpEnsemble()
        ensemble.fit(X_train, y_train, groups_train, X_val, y_val)

        # ----------------------------------------------------------------
        # 4. Calibration on validation set
        # ----------------------------------------------------------------
        logger.info("=== [%s] Calibration ===", category)
        val_raw = ensemble.predict_proba(X_val)
        calibrator = BetaCalibrator()
        calibrator.fit(val_raw, y_val.values)

        # ----------------------------------------------------------------
        # 5. Evaluation on test set
        # ----------------------------------------------------------------
        logger.info("=== [%s] Test evaluation ===", category)
        test_raw = ensemble.predict_proba(X_test)
        test_cal = calibrator.calibrate(test_raw)

        auc = roc_auc_score(y_test, test_cal)
        brier = brier_score_loss(y_test, test_cal)

        # Normalise within each race for race-level AUC
        test_df = test_df.copy()
        test_df["cal_prob"] = test_cal
        race_totals = test_df.groupby("race_key")["cal_prob"].transform("sum")
        test_df["norm_prob"] = test_df["cal_prob"] / race_totals.clip(lower=1e-9)
        norm_auc = roc_auc_score(y_test, test_df["norm_prob"])
        norm_brier = brier_score_loss(y_test, test_df["norm_prob"])

        logger.info(
            "=== [%s] RESULTS === Raw: AUC=%.4f Brier=%.4f | "
            "Normalised: AUC=%.4f Brier=%.4f",
            category, auc, brier, norm_auc, norm_brier,
        )

        # ----------------------------------------------------------------
        # 6. Save artefacts
        # ----------------------------------------------------------------
        logger.info("=== [%s] Saving artefacts to %s ===", category, out_dir)
        ensemble.save(os.path.join(out_dir, "ensemble.pkl"))
        calibrator.save(os.path.join(out_dir, "calibrator.pkl"))
        extractor.save(os.path.join(out_dir, "extractor.pkl"))

        elapsed = time.time() - t_start
        metrics = {
            "category": category,
            "auc_raw": round(auc, 4),
            "brier_raw": round(brier, 4),
            "auc_normalised": round(norm_auc, 4),
            "brier_normalised": round(norm_brier, 4),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "n_riders": extractor.rider_count,
            "elapsed_seconds": round(elapsed, 1),
        }
        logger.info("[%s] Metrics: %s", category, metrics)
        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MotoGP ML models")
    parser.add_argument("--csv", default=MOTOGP_CSV, help="Path to MotoGP CSV")
    args = parser.parse_args()

    trainer = MotogpTrainer()
    all_metrics = trainer.train(csv_path=args.csv)

    print("\n=== FINAL METRICS ===")
    for cat, m in all_metrics.items():
        print(f"\n  Category: {cat}")
        for k, v in m.items():
            print(f"    {k}: {v}")
