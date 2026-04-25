"""
XG3 MotoGP Microservice — Entry Point
FastAPI application with lifespan model loading.
Port: 8032

Markets: Race Winner, Top-3 Podium, Top-5, H2H, Constructor
ML: CatBoost + LightGBM + XGBoost stacking ensemble
Pricing: Harville DP + Shin margin normalisation
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import DEBUG, PORT, SERVICE_NAME, SERVICE_VERSION, R0_DIR, R1_DIR, R2_DIR
from ml.predictor import MotogpPredictor
from feeds.optic_odds import OpticOddsFeed
from api.routes import health, races, admin
from api.routes.outrights import router as outrights_router
from api.routes.settlement import router as settlement_router
from api.routes.predict import router as predict_router

# ---------------------------------------------------------------------------
# Sentry error monitoring — set SENTRY_DSN env var in Railway to activate
# ---------------------------------------------------------------------------
import os as _os_sentry
_SENTRY_DSN = _os_sentry.getenv("SENTRY_DSN", "")
if _SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),
            ],
            traces_sample_rate=float(_os_sentry.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.05")),
            environment=_os_sentry.getenv("ENVIRONMENT", "production"),
        )
        print(f"[Sentry] Initialized for {_os_sentry.getenv('RAILWAY_SERVICE_NAME', 'unknown')}")
    except ImportError:
        pass  # sentry-sdk not installed — non-fatal

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: load predictor models.
    Shutdown: log teardown.
    """
    logger.info("Starting %s v%s (port=%d)", SERVICE_NAME, SERVICE_VERSION, PORT)

    # Initialise Optic Odds feed
    app.state.optic_feed = OpticOddsFeed()
    logger.info("Optic Odds feed initialised")

    # Load predictor
    predictor = MotogpPredictor()
    try:
        predictor.load(r0_dir=R0_DIR, r1_dir=R1_DIR, r2_dir=R2_DIR)
        logger.info(
            "Predictor loaded: %d riders tracked, categories: %s",
            predictor.rider_count,
            predictor.active_categories,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "Model files not found (%s) — predictor in unloaded state. "
            "Run: python -c \"from ml.trainer import MotogpTrainer; MotogpTrainer().train()\"",
            exc,
        )
    except Exception as exc:
        logger.error("Predictor load failed: %s", exc, exc_info=True)

    app.state.predictor = predictor

    logger.info("%s startup complete", SERVICE_NAME)
    yield

    logger.info("%s shutting down", SERVICE_NAME)


def create_app() -> FastAPI:
    app = FastAPI(
        title="XG3 MotoGP Microservice",
        description=(
            "MotoGP, Moto2, and Moto3 race winner prediction and pricing. "
            "3-model stacking ensemble (CatBoost + LightGBM + XGBoost) "
            "with Harville DP market generation."
        ),
        version=SERVICE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health routes (no prefix — Enterprise standard)
    app.include_router(health.router, tags=["health"])

    # Domain routes
    app.include_router(
        races.router,
        prefix="/api/v1/motogp/races",
        tags=["races"],
    )
    app.include_router(
        admin.router,
        prefix="/api/v1/motogp/admin",
        tags=["admin"],
    )
    app.include_router(
        outrights_router,
        prefix="/api/v1/motogp/outrights",
        tags=["outrights"],
    )
    app.include_router(settlement_router)
    app.include_router(
        predict_router,
        prefix="/api/v1/motogp",
        tags=["predict"],
    )

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "docs": "/docs",
            "health": "/health",
        })

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
