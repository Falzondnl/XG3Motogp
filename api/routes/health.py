"""
Health endpoints for MotoGP MS.
GET /health        — basic health (always 200)
GET /health/ready  — readiness (predictor loaded)
GET /health/live   — liveness (always 200 if process running)
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import SERVICE_NAME, SERVICE_VERSION

router = APIRouter()
_START_TIME = time.time()


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    predictor = getattr(request.app.state, "predictor", None)
    is_ready = predictor is not None and predictor.is_loaded
    return JSONResponse({
        "status": "ok" if is_ready else "degraded",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "ready": is_ready,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    })


@router.get("/health/ready")
async def health_ready(request: Request) -> JSONResponse:
    predictor = getattr(request.app.state, "predictor", None)
    is_ready = predictor is not None and predictor.is_loaded

    if not is_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": SERVICE_NAME,
                "reason": "predictor not loaded",
            },
        )

    return JSONResponse({
        "status": "ready",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "rider_count": predictor.rider_count,
        "active_categories": predictor.active_categories,
    })


@router.get("/health/live")
async def health_live() -> JSONResponse:
    return JSONResponse({"status": "alive", "service": SERVICE_NAME})
