"""
MotoGP prediction endpoint — GAP-A-13.

POST /api/v1/motogp/predict

Returns win / podium / top-5 probabilities for a full race field using
the trained MotogpPredictor stacking ensemble (CatBoost + LightGBM + XGBoost
+ Harville DP).  One entry per rider, sorted by win probability descending.

NEVER returns a hardcoded default probability.  Returns HTTP 503 if the
predictor has not been loaded at startup.

Standard response envelope:
    {"success": true, "data": {...}, "meta": {"request_id": "uuid", "timestamp": "ISO8601"}}
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Request, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger()
router = APIRouter(tags=["Predict"])

_VALID_CATEGORIES = {"MotoGP", "Moto2", "Moto3"}
_VALID_SESSION_TYPES = {"RAC", "SPR"}


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _meta(request_id: str) -> Dict[str, str]:
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _ok(data: Any, request_id: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "meta": _meta(request_id)}


def _error(code: str, message: str, request_id: str, http_status: int = 400) -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "success": False,
            "error": {"code": code, "message": message},
            "meta": _meta(request_id),
        },
        status_code=http_status,
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RiderInput(BaseModel):
    name: str = Field(..., min_length=1, description="Rider full name matching training data")
    team: str = Field(default="UNKNOWN", description="Team name (e.g. Ducati Lenovo Team)")
    constructor: str = Field(default="UNKNOWN", description="Manufacturer (e.g. Ducati, Honda, Yamaha)")


class PredictRequest(BaseModel):
    """Race field definition for win-probability prediction."""

    riders: List[RiderInput] = Field(
        ...,
        min_length=2,
        description="Race field. Minimum 2 riders required.",
    )
    category: str = Field(
        ...,
        description="Race category: MotoGP | Moto2 | Moto3",
    )
    circuit: str = Field(
        default="UNKNOWN",
        description="Circuit name used for feature extraction (e.g. Lusail International Circuit)",
    )
    weather: str = Field(
        default="",
        description="Weather description (e.g. 'dry', 'wet', 'mixed')",
    )
    track_condition: str = Field(
        default="",
        description="Track condition (e.g. 'dry', 'damp')",
    )
    season: int = Field(
        default=2025,
        ge=2000,
        le=2050,
        description="Race season year",
    )
    session_type: str = Field(
        default="RAC",
        description="Session type: RAC (race) or SPR (sprint)",
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        cap = v.capitalize() if v.lower() in {"motogp", "moto2", "moto3"} else v
        # normalise "motogp" → "MotoGP" etc.
        _map = {"motogp": "MotoGP", "moto2": "Moto2", "moto3": "Moto3"}
        normalised = _map.get(v.lower(), cap)
        if normalised not in _VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(_VALID_CATEGORIES)}, got {v!r}")
        return normalised

    @field_validator("session_type")
    @classmethod
    def validate_session_type(cls, v: str) -> str:
        upper = v.upper()
        if upper not in _VALID_SESSION_TYPES:
            raise ValueError(f"session_type must be one of {sorted(_VALID_SESSION_TYPES)}, got {v!r}")
        return upper


class RiderPrediction(BaseModel):
    rider_name: str
    team: str
    constructor: str
    win_prob: float = Field(description="P(rider wins), calibrated Harville-normalised")
    podium_prob: float = Field(description="P(rider finishes top 3), Harville DP")
    top5_prob: float = Field(description="P(rider finishes top 5), Harville DP")


class PredictResponseData(BaseModel):
    category: str
    circuit: str
    season: int
    session_type: str
    field_size: int
    results: List[RiderPrediction]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    summary="Predict race win / podium / top-5 probabilities (ML ensemble)",
    response_class=ORJSONResponse,
)
async def predict_race(request: Request, body: PredictRequest) -> ORJSONResponse:
    """
    Return win, podium, and top-5 probabilities for a race field.

    Uses the MotogpPredictor 3-model stacking ensemble (CatBoost + LightGBM +
    XGBoost) with BetaCalibrator and Harville DP for multi-placement markets.
    Results are sorted by win_prob descending.

    HTTP 503 is returned when:
      - The predictor has not been loaded at startup (models not trained/found)
    """
    rid = str(uuid.uuid4())
    log = logger.bind(
        request_id=rid,
        category=body.category,
        circuit=body.circuit,
        field_size=len(body.riders),
    )
    log.info("motogp_predict_requested")

    # ── Obtain predictor from app state ─────────────────────────────────────
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        log.warning("motogp_predict_predictor_not_loaded")
        return _error(
            "PREDICTOR_UNAVAILABLE",
            (
                "MotoGP predictor is not loaded. "
                "Ensure the service started successfully and models are present. "
                "Run: python -c \"from ml.trainer import MotogpTrainer; MotogpTrainer().train()\" "
                "to train models if missing."
            ),
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # ── Build rider dicts for predictor ─────────────────────────────────────
    riders_payload = [
        {
            "name": r.name,
            "team": r.team,
            "constructor": r.constructor,
        }
        for r in body.riders
    ]

    # ── Run inference ────────────────────────────────────────────────────────
    try:
        raw_results = predictor.predict_race(
            riders=riders_payload,
            category=body.category,
            circuit=body.circuit,
            weather=body.weather,
            track_condition=body.track_condition,
            season=body.season,
            session_type=body.session_type,
        )
    except ValueError as exc:
        log.warning("motogp_predict_invalid_input", error=str(exc))
        return _error("INVALID_INPUT", str(exc), rid, status.HTTP_422_UNPROCESSABLE_ENTITY)
    except RuntimeError as exc:
        log.error("motogp_predict_inference_failed", error=str(exc))
        return _error(
            "INFERENCE_FAILED",
            f"Prediction failed: {exc}",
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        log.error("motogp_predict_unexpected_error", error=str(exc))
        return _error(
            "INTERNAL_ERROR",
            f"Unexpected error during prediction: {exc}",
            rid,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # ── Build response ───────────────────────────────────────────────────────
    results = [
        RiderPrediction(
            rider_name=r["rider_name"],
            team=r.get("team", "UNKNOWN"),
            constructor=r.get("constructor", "UNKNOWN"),
            win_prob=round(float(r["win_prob"]), 6),
            podium_prob=round(float(r["podium_prob"]), 6),
            top5_prob=round(float(r["top5_prob"]), 6),
        )
        for r in raw_results
    ]

    response_data = PredictResponseData(
        category=body.category,
        circuit=body.circuit,
        season=body.season,
        session_type=body.session_type,
        field_size=len(results),
        results=results,
    )

    log.info(
        "motogp_predict_complete",
        category=body.category,
        field_size=len(results),
        top_rider=results[0].rider_name if results else None,
        top_win_prob=results[0].win_prob if results else None,
    )
    return ORJSONResponse(content=_ok(response_data.model_dump(), rid))
