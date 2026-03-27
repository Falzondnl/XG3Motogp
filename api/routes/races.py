"""
Race pricing and prediction endpoints for MotoGP MS.
POST /api/v1/motogp/races/price  — price a full race with all markets
POST /api/v1/motogp/races/h2h    — H2H market between two riders
GET  /api/v1/motogp/races/upcoming — upcoming races from Optic Odds
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from api._deps import get_predictor, get_optic_feed
from ml.predictor import MotogpPredictor
from feeds.optic_odds import OpticOddsFeed
from pricing.markets import (
    build_all_markets,
    build_h2h_market,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class RiderInput(BaseModel):
    name: str = Field(..., min_length=1, description="Rider full name")
    team: str = Field(default="UNKNOWN", description="Team name")
    constructor: str = Field(default="UNKNOWN", description="Manufacturer (Ducati, Honda, Yamaha, etc.)")

    @field_validator("constructor")
    @classmethod
    def normalise_constructor(cls, v: str) -> str:
        return v.strip() if v else "UNKNOWN"


class PriceRaceRequest(BaseModel):
    race_name: str = Field(..., min_length=1, description="Race name (e.g. Qatar Grand Prix)")
    circuit: str = Field(..., min_length=1, description="Circuit name (e.g. Lusail International Circuit)")
    category: str = Field(
        default="MotoGP",
        description="Category: MotoGP, Moto2, or Moto3",
    )
    season: int = Field(default=2025, ge=2002, le=2030, description="Race season year")
    session_type: Literal["RAC", "SPR"] = Field(
        default="RAC", description="Session type: RAC=race, SPR=sprint"
    )
    weather: str = Field(default="", description="Weather conditions (e.g. Dry, Wet, Mixed)")
    track_condition: str = Field(default="", description="Track condition description")
    riders: list[RiderInput] = Field(..., min_length=2, description="Race field (min 2 riders)")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed = {"MotoGP", "Moto2", "Moto3", "500cc"}
        if v not in allowed:
            raise ValueError(f"category must be one of {sorted(allowed)}")
        return v


class H2HRequest(BaseModel):
    rider_a: RiderInput
    rider_b: RiderInput
    category: str = Field(default="MotoGP")
    circuit: str = Field(default="UNKNOWN")
    season: int = Field(default=2025, ge=2002, le=2030)
    session_type: Literal["RAC", "SPR"] = Field(default="RAC")
    weather: str = Field(default="")
    track_condition: str = Field(default="")


@router.post("/price")
async def price_race(
    body: PriceRaceRequest,
    predictor: MotogpPredictor = Depends(get_predictor),
) -> dict[str, Any]:
    """
    Price a full race with all markets:
    - Race winner (12% margin)
    - Podium top-3 Yes/No (10% margin)
    - Top-5 Yes/No (10% margin)
    - Constructor winner outright (10% margin)

    Returns predicted win_prob, podium_prob, top5_prob per rider plus all markets.
    """
    riders_input = [r.model_dump() for r in body.riders]

    try:
        predictions = predictor.predict_race(
            riders=riders_input,
            category=body.category,
            circuit=body.circuit,
            weather=body.weather,
            track_condition=body.track_condition,
            season=body.season,
            session_type=body.session_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )

    try:
        markets = build_all_markets(
            riders=predictions,
            category=body.category,
            circuit=body.circuit,
        )
    except Exception as exc:
        logger.error("Market building error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Market building failed: {exc}",
        )

    return {
        "race_name": body.race_name,
        "circuit": body.circuit,
        "category": body.category,
        "season": body.season,
        "session_type": body.session_type,
        "field_size": len(predictions),
        "predictions": predictions,
        "markets": markets["markets"],
    }


@router.post("/h2h")
async def price_h2h(
    body: H2HRequest,
    predictor: MotogpPredictor = Depends(get_predictor),
) -> dict[str, Any]:
    """
    Price an H2H market between two riders.
    Runs the full predictor on both riders then extracts their relative probabilities.
    Returns H2H market with 5% margin.
    """
    riders_input = [body.rider_a.model_dump(), body.rider_b.model_dump()]

    try:
        predictions = predictor.predict_race(
            riders=riders_input,
            category=body.category,
            circuit=body.circuit,
            weather=body.weather,
            track_condition=body.track_condition,
            season=body.season,
            session_type=body.session_type,
        )
    except Exception as exc:
        logger.error("H2H prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"H2H prediction failed: {exc}",
        )

    # predictions is sorted by win_prob — map back to input names
    pred_map = {p["rider_name"]: p for p in predictions}
    name_a = body.rider_a.name
    name_b = body.rider_b.name

    pred_a = pred_map.get(name_a)
    pred_b = pred_map.get(name_b)

    if pred_a is None or pred_b is None:
        # Try partial match
        for p in predictions:
            if pred_a is None and name_a.lower() in p["rider_name"].lower():
                pred_a = p
            if pred_b is None and name_b.lower() in p["rider_name"].lower():
                pred_b = p

    if pred_a is None or pred_b is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not match rider names to predictions",
        )

    try:
        h2h_market = build_h2h_market(rider_a=pred_a, rider_b=pred_b)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"H2H market building failed: {exc}",
        )

    return {
        "circuit": body.circuit,
        "category": body.category,
        "season": body.season,
        "h2h_market": h2h_market,
    }


@router.get("/upcoming")
async def get_upcoming_races(
    limit: int = 20,
    feed: OpticOddsFeed = Depends(get_optic_feed),
) -> dict[str, Any]:
    """Fetch upcoming motorsports races from Optic Odds."""
    try:
        races = await feed.get_upcoming_races(limit=min(limit, 100))
        return {
            "count": len(races),
            "races": races,
        }
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Failed to fetch upcoming races: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Optic Odds API error: {exc}",
        )
