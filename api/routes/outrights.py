"""
MotoGP Season Championship Outright Markets
=============================================

Produces season championship winner prices for MotoGP, Moto2, and Moto3
from the live ELO ratings held in app.state.elo_store.

Endpoint
--------
GET /api/v1/motogp/outrights/championship?category=MotoGP&season=2026

Returns a ranked list of riders with fair win probability and decimal odds
at the configured championship margin (10%).

Methodology
-----------
Championship probability is derived from the Harville formula applied to
season-level ELO ratings.  Each rider's ELO reflects accumulated circuit
performance across all categories.  Probabilities are normalised to sum = 1
and converted to decimal odds including margin.

Margin: 10% (configurable via CHAMPIONSHIP_MARGIN env var).

This mirrors the F1 WDC endpoint pattern:
  GET /api/v1/formula1/outrights/wdc
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Championship margin applied to all outright prices (10% = standard motorsport book)
_CHAMPIONSHIP_MARGIN: float = float(os.getenv("CHAMPIONSHIP_MARGIN", "0.10"))

# Maximum riders to include in championship market (ELO cutoff)
_TOP_N_CHAMPIONSHIP: int = int(os.getenv("CHAMPIONSHIP_TOP_N", "25"))

_VALID_CATEGORIES = {"MotoGP", "Moto2", "Moto3"}


def _harville_softmax_probs(elo_ratings: list[tuple[str, float]], temperature: float = 400.0) -> list[tuple[str, float]]:
    """
    Convert ELO ratings to championship win probabilities via the Harville
    softmax formula.

    Harville (1973): P(i wins) = exp(elo_i / T) / sum(exp(elo_j / T))

    Args:
        elo_ratings: List of (rider_name, elo_value) tuples.
        temperature:  Softmax temperature.  Higher = more uniform.
                      400 is calibrated for MotoGP/Moto2/Moto3 ELO scale.

    Returns:
        List of (rider_name, probability) tuples sorted by probability descending.
    """
    if not elo_ratings:
        return []

    # Numerical stability: subtract max ELO before exponentiation
    max_elo = max(elo for _, elo in elo_ratings)
    exp_vals = [(name, math.exp((elo - max_elo) / temperature)) for name, elo in elo_ratings]
    total = sum(v for _, v in exp_vals)
    if total == 0:
        n = len(exp_vals)
        return [(name, 1.0 / n) for name, _ in exp_vals]

    probs = [(name, v / total) for name, v in exp_vals]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs


def _probability_to_decimal(prob: float, margin: float) -> float:
    """
    Convert fair probability to decimal odds including book margin.

    odds = 1 / (prob * (1 + margin))

    Args:
        prob:   Fair win probability (0 < prob <= 1).
        margin: Book margin (e.g. 0.10 = 10%).

    Returns:
        Decimal odds rounded to 2 dp, minimum 1.01.
    """
    if prob <= 0:
        return 999.99
    raw = 1.0 / (prob * (1.0 + margin))
    return max(1.01, round(raw, 2))


@router.get("/championship")
async def get_championship_outrights(
    category: str = "MotoGP",
    season: int = 2026,
    top_n: int = _TOP_N_CHAMPIONSHIP,
    request: Request = None,
) -> JSONResponse:
    """
    Season championship outright market for MotoGP, Moto2, or Moto3.

    Returns rider win probabilities and decimal odds derived from current ELO
    ratings.  All riders in the top-N ELO bracket are included.

    Args:
        category: "MotoGP", "Moto2", or "Moto3".  Default: "MotoGP".
        season:   Championship season year.  Default: 2026.
        top_n:    Maximum number of riders in the market.  Default: 25.

    Returns:
        {
          "market": "world_championship",
          "category": "MotoGP",
          "season_year": 2026,
          "entries": [
            {"rank": 1, "name": "Marc Marquez", "probability": 0.312, "price": 2.91},
            ...
          ],
          "margin": 0.10,
          "total_probability": 1.0
        }
    """
    if category not in _VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"category must be one of {sorted(_VALID_CATEGORIES)}",
        )

    top_n = max(5, min(top_n, 50))

    # Fetch ELO ratings from app.state.elo_store
    elo_store = getattr(request.app.state, "elo_store", None) if request else None

    if elo_store is None:
        raise HTTPException(
            status_code=503,
            detail="ELO store not loaded — service may still be starting up",
        )

    try:
        # ELO store returns list of (rider_name, elo_value) for the given category
        if hasattr(elo_store, "get_top_riders"):
            raw_ratings: list[Any] = elo_store.get_top_riders(
                category=category, top_n=top_n
            )
        elif hasattr(elo_store, "get_ratings"):
            raw_ratings = elo_store.get_ratings(category=category)[:top_n]
        else:
            # Fallback: iterate dict if elo_store is a plain dict
            raw_ratings = list(elo_store.items())[:top_n]
    except Exception as exc:
        logger.error("championship: ELO store read failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"ELO store read error: {exc}",
        ) from exc

    # Normalise to list of (name, elo) tuples
    elo_pairs: list[tuple[str, float]] = []
    for item in raw_ratings:
        if isinstance(item, dict):
            name = item.get("rider") or item.get("name") or item.get("rider_name", "Unknown")
            elo = float(item.get("elo") or item.get("elo_rating") or item.get("rating", 1500.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name, elo = str(item[0]), float(item[1])
        else:
            continue
        elo_pairs.append((name, elo))

    if not elo_pairs:
        raise HTTPException(
            status_code=503,
            detail=f"No ELO data available for category={category}",
        )

    # Compute championship win probabilities via Harville softmax
    probs = _harville_softmax_probs(elo_pairs[:top_n])

    # Build market entries
    entries: list[dict[str, Any]] = []
    for rank, (name, prob) in enumerate(probs, start=1):
        entries.append({
            "rank": rank,
            "name": name,
            "probability": round(prob, 6),
            "price": _probability_to_decimal(prob, _CHAMPIONSHIP_MARGIN),
        })

    total_prob = sum(e["probability"] for e in entries)

    logger.info(
        "championship_outrights category=%s season=%d entries=%d top=%s prob=%.4f",
        category, season, len(entries),
        entries[0]["name"] if entries else "none",
        entries[0]["probability"] if entries else 0.0,
    )

    return JSONResponse({
        "market": "world_championship",
        "category": category,
        "season_year": season,
        "entries": entries,
        "margin": _CHAMPIONSHIP_MARGIN,
        "total_probability": round(total_prob, 6),
    })
