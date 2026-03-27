"""
Admin endpoints for MotoGP MS.
GET /api/v1/motogp/admin/status      — model status, ELO counts, model files
GET /api/v1/motogp/admin/elo-ratings — top ELO ratings per category
POST /api/v1/motogp/admin/train      — trigger model training
"""
from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from config import MODEL_DIR, R0_DIR, R1_DIR, R2_DIR, SERVICE_NAME, SERVICE_VERSION

router = APIRouter()
_START_TIME = time.time()
_training_in_progress = False


def _list_pkl_files(directory: str) -> list[str]:
    """List .pkl files in a directory, return [] if not exists."""
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".pkl")]


@router.get("/status")
async def admin_status(request: Request) -> JSONResponse:
    """
    Returns current service status including model artefacts, ELO counts,
    rider history counts, and all pkl files present on disk.
    """
    predictor = getattr(request.app.state, "predictor", None)
    optic_feed = getattr(request.app.state, "optic_feed", None)

    model_loaded = False
    rider_count = 0
    circuit_elo_count = 0
    constructor_count = 0
    active_categories: list[str] = []

    if predictor is not None and predictor.is_loaded:
        model_loaded = True
        rider_count = predictor.rider_count
        active_categories = predictor.active_categories

        # Get extractor stats from MotoGP model (R0)
        from config import CATEGORY_MOTOGP
        if CATEGORY_MOTOGP in predictor._models:
            ext = predictor._models[CATEGORY_MOTOGP].extractor
            circuit_elo_count = ext.circuit_elo_count
            constructor_count = ext.constructor_count

    return JSONResponse({
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "model_loaded": model_loaded,
        "active_categories": active_categories,
        "rider_count": rider_count,
        "circuit_elo_count": circuit_elo_count,
        "constructor_count": constructor_count,
        "optic_feed_active": optic_feed is not None,
        "training_in_progress": _training_in_progress,
        "model_files": {
            "r0_motogp": _list_pkl_files(R0_DIR),
            "r1_moto2": _list_pkl_files(R1_DIR),
            "r2_moto3": _list_pkl_files(R2_DIR),
        },
        "model_dirs": {
            "r0": R0_DIR,
            "r1": R1_DIR,
            "r2": R2_DIR,
        },
    })


@router.get("/elo-ratings")
async def get_elo_ratings(
    request: Request,
    category: str = "MotoGP",
    top_n: int = 30,
) -> JSONResponse:
    """
    Return top-N ELO ratings for riders in the specified category.
    category: MotoGP, Moto2, or Moto3
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Predictor not loaded"},
        )

    from config import CATEGORY_MOTOGP, CATEGORY_MOTO2, CATEGORY_MOTO3
    from ml.features import _normalise_category

    cat_norm = _normalise_category(category)
    if cat_norm not in predictor._models:
        return JSONResponse(
            status_code=404,
            content={"error": f"No model loaded for category: {category}"},
        )

    ext = predictor._models[cat_norm].extractor

    # Extract (rider, category) ELO
    cat_elos: list[tuple[str, float]] = [
        (rider, elo)
        for (rider, cat), elo in ext.rider_elo.items()
        if cat == cat_norm
    ]
    cat_elos.sort(key=lambda x: x[1], reverse=True)
    top_elos = cat_elos[:top_n]

    return JSONResponse({
        "category": category,
        "top_n": top_n,
        "elo_ratings": [
            {"rider": rider, "elo": round(elo, 2)}
            for rider, elo in top_elos
        ],
        "total_riders": len(cat_elos),
    })


def _run_training_background() -> None:
    """Run training in background thread."""
    global _training_in_progress
    import logging
    log = logging.getLogger(__name__)
    try:
        _training_in_progress = True
        from ml.trainer import MotogpTrainer
        trainer = MotogpTrainer()
        metrics = trainer.train()
        log.info("Background training complete: %s", metrics)
    except Exception as exc:
        log.error("Background training failed: %s", exc, exc_info=True)
    finally:
        _training_in_progress = False


@router.post("/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Trigger model training in the background.
    Training runs asynchronously — check /admin/status to monitor progress.
    """
    global _training_in_progress

    if _training_in_progress:
        return JSONResponse(
            status_code=409,
            content={"error": "Training already in progress"},
        )

    background_tasks.add_task(_run_training_background)
    return JSONResponse({
        "status": "training_started",
        "message": "Training running in background. Check /admin/status for progress.",
    })
