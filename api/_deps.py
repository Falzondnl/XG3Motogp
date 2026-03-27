"""FastAPI dependency injectors for MotoGP MS."""
from __future__ import annotations

from fastapi import HTTPException, Request, status

from ml.predictor import MotogpPredictor
from feeds.optic_odds import OpticOddsFeed


def get_predictor(request: Request) -> MotogpPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MotoGP predictor not loaded. Run: python -c \"from ml.trainer import MotogpTrainer; MotogpTrainer().train()\"",
        )
    return predictor


def get_optic_feed(request: Request) -> OpticOddsFeed:
    feed = getattr(request.app.state, "optic_feed", None)
    if feed is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Optic Odds feed not initialised",
        )
    return feed
