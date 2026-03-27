"""
MotoGP Market Builder
Produces race-winner, podium (top-3), top-5, H2H, and constructor championship markets.
All margins applied via Shin power-method normalisation (preserves relative probability ratios).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from config import (
    WIN_MARGIN,
    PODIUM_MARGIN,
    TOP5_MARGIN,
    H2H_MARGIN,
    CONSTRUCTOR_MARGIN,
)

logger = logging.getLogger(__name__)

MIN_SELECTIONS = 2
MIN_PROB_CLIP = 0.001   # Floor per selection (0.1%)
MAX_SELECTIONS_DISPLAY = 30  # Cap for race winner market display


def _apply_margin_shin(probs: list[float], margin: float) -> list[float]:
    """
    Shin margin application: scales to target overround = 1 + margin.
    Preserves relative probability ratios.
    """
    if not probs:
        return []
    arr = np.array(probs, dtype=float)
    arr = np.clip(arr, MIN_PROB_CLIP, 1.0)

    s = arr.sum()
    if s < 1e-9:
        arr = np.ones(len(arr)) / len(arr)
    else:
        arr = arr / s

    target_sum = 1.0 + margin
    arr = arr * target_sum
    return arr.tolist()


def _prob_to_decimal_odds(p: float) -> float:
    """Convert margined probability to decimal odds (1/p), minimum 1.0."""
    p = max(p, MIN_PROB_CLIP)
    return round(max(1.0 / p, 1.0), 3)


def _format_selection(name: str, prob: float, margined_prob: float) -> dict[str, Any]:
    return {
        "name": name,
        "probability": round(float(prob), 6),
        "margined_probability": round(float(margined_prob), 6),
        "decimal_odds": _prob_to_decimal_odds(margined_prob),
    }


def build_race_winner_market(
    riders: list[dict],
    margin: float = WIN_MARGIN,
) -> dict[str, Any]:
    """
    Race winner market: one selection per rider.
    margin applied via Shin method.

    riders: list from MotogpPredictor.predict_race() — must contain rider_name, win_prob.
    """
    if len(riders) < MIN_SELECTIONS:
        raise ValueError(f"Need at least {MIN_SELECTIONS} riders for a market")

    sorted_riders = sorted(riders, key=lambda x: x["win_prob"], reverse=True)
    display_riders = sorted_riders[:MAX_SELECTIONS_DISPLAY]

    raw_probs = [r["win_prob"] for r in display_riders]
    margined_probs = _apply_margin_shin(raw_probs, margin)
    overround = sum(margined_probs)

    selections = []
    for rider, mp in zip(display_riders, margined_probs):
        name = rider.get("rider_name", "Unknown")
        team = rider.get("team", "")
        label = f"{name} ({team})" if team and team != "UNKNOWN" else name
        selections.append(
            _format_selection(
                name=label,
                prob=rider["win_prob"],
                margined_prob=mp,
            )
        )

    return {
        "market_type": "race_winner",
        "margin": round(margin, 4),
        "overround": round(overround, 4),
        "selection_count": len(selections),
        "total_field_size": len(riders),
        "selections": selections,
    }


def build_podium_market(
    riders: list[dict],
    margin: float = PODIUM_MARGIN,
) -> dict[str, Any]:
    """
    Top-3 podium Yes/No market for each of the top riders.
    podium_prob from Harville formula in predictor.
    Returns top-10 riders most likely to podium.
    """
    if len(riders) < MIN_SELECTIONS:
        raise ValueError(f"Need at least {MIN_SELECTIONS} riders for podium market")

    sorted_riders = sorted(riders, key=lambda x: x["podium_prob"], reverse=True)
    top_riders = sorted_riders[:10]

    selections = []
    for r in top_riders:
        p_yes = max(float(r["podium_prob"]), MIN_PROB_CLIP)
        p_no = max(1.0 - p_yes, MIN_PROB_CLIP)

        yes_margined, no_margined = _apply_margin_shin([p_yes, p_no], margin)

        selections.append({
            "rider_name": r.get("rider_name", "Unknown"),
            "team": r.get("team", "UNKNOWN"),
            "constructor": r.get("constructor", "UNKNOWN"),
            "podium_yes": {
                "probability": round(p_yes, 6),
                "margined_probability": round(yes_margined, 6),
                "decimal_odds": _prob_to_decimal_odds(yes_margined),
            },
            "podium_no": {
                "probability": round(p_no, 6),
                "margined_probability": round(no_margined, 6),
                "decimal_odds": _prob_to_decimal_odds(no_margined),
            },
        })

    return {
        "market_type": "podium_finisher_top3",
        "margin": round(margin, 4),
        "selections": selections,
        "description": "Will this rider finish in the top 3?",
    }


def build_top5_market(
    riders: list[dict],
    margin: float = TOP5_MARGIN,
) -> dict[str, Any]:
    """
    Top-5 finisher Yes/No market for each of the top riders.
    """
    if len(riders) < MIN_SELECTIONS:
        raise ValueError(f"Need at least {MIN_SELECTIONS} riders for top-5 market")

    sorted_riders = sorted(riders, key=lambda x: x["top5_prob"], reverse=True)
    top_riders = sorted_riders[:12]

    selections = []
    for r in top_riders:
        p_yes = max(float(r["top5_prob"]), MIN_PROB_CLIP)
        p_no = max(1.0 - p_yes, MIN_PROB_CLIP)

        yes_margined, no_margined = _apply_margin_shin([p_yes, p_no], margin)

        selections.append({
            "rider_name": r.get("rider_name", "Unknown"),
            "team": r.get("team", "UNKNOWN"),
            "constructor": r.get("constructor", "UNKNOWN"),
            "top5_yes": {
                "probability": round(p_yes, 6),
                "margined_probability": round(yes_margined, 6),
                "decimal_odds": _prob_to_decimal_odds(yes_margined),
            },
            "top5_no": {
                "probability": round(p_no, 6),
                "margined_probability": round(no_margined, 6),
                "decimal_odds": _prob_to_decimal_odds(no_margined),
            },
        })

    return {
        "market_type": "top5_finisher",
        "margin": round(margin, 4),
        "selections": selections,
        "description": "Will this rider finish in the top 5?",
    }


def build_h2h_market(
    rider_a: dict,
    rider_b: dict,
    margin: float = H2H_MARGIN,
) -> dict[str, Any]:
    """
    H2H market between two specific riders.
    Re-normalises their win probabilities to form a binary market.
    """
    p_a = max(float(rider_a["win_prob"]), MIN_PROB_CLIP)
    p_b = max(float(rider_b["win_prob"]), MIN_PROB_CLIP)
    total = p_a + p_b
    p_a_norm = p_a / total
    p_b_norm = p_b / total

    p_a_margined, p_b_margined = _apply_margin_shin([p_a_norm, p_b_norm], margin)

    name_a = rider_a.get("rider_name", "Rider A")
    name_b = rider_b.get("rider_name", "Rider B")

    return {
        "market_type": "h2h",
        "margin": round(margin, 4),
        "overround": round(p_a_margined + p_b_margined, 4),
        "selections": [
            {
                "name": name_a,
                "team": rider_a.get("team", "UNKNOWN"),
                "probability": round(p_a_norm, 6),
                "margined_probability": round(p_a_margined, 6),
                "decimal_odds": _prob_to_decimal_odds(p_a_margined),
            },
            {
                "name": name_b,
                "team": rider_b.get("team", "UNKNOWN"),
                "probability": round(p_b_norm, 6),
                "margined_probability": round(p_b_margined, 6),
                "decimal_odds": _prob_to_decimal_odds(p_b_margined),
            },
        ],
    }


def build_constructor_market(
    riders: list[dict],
    margin: float = CONSTRUCTOR_MARGIN,
    top_n: int = 8,
) -> dict[str, Any]:
    """
    Constructor championship outright: aggregate win_prob by constructor.
    Top-N constructors by aggregated probability.
    """
    constructor_probs: dict[str, float] = {}
    for r in riders:
        cstr = r.get("constructor", "UNKNOWN")
        if not cstr or cstr == "UNKNOWN":
            cstr = "Other"
        constructor_probs[cstr] = constructor_probs.get(cstr, 0.0) + r["win_prob"]

    sorted_constructors = sorted(
        constructor_probs.items(), key=lambda x: x[1], reverse=True
    )
    top_constructors = sorted_constructors[:top_n]

    if not top_constructors:
        return {"market_type": "constructor_winner", "selections": []}

    raw_probs = [p for _, p in top_constructors]
    margined_probs = _apply_margin_shin(raw_probs, margin)
    overround = sum(margined_probs)

    selections = []
    for (cstr, prob), mp in zip(top_constructors, margined_probs):
        selections.append(
            _format_selection(
                name=cstr,
                prob=prob,
                margined_prob=mp,
            )
        )

    return {
        "market_type": "constructor_winner",
        "margin": round(margin, 4),
        "overround": round(overround, 4),
        "selection_count": len(selections),
        "selections": selections,
        "description": "Which constructor will win this race?",
    }


def build_all_markets(
    riders: list[dict],
    category: str,
    circuit: str,
    win_margin: float = WIN_MARGIN,
    podium_margin: float = PODIUM_MARGIN,
    top5_margin: float = TOP5_MARGIN,
    constructor_margin: float = CONSTRUCTOR_MARGIN,
) -> dict[str, Any]:
    """
    Builds all available markets for a race.
    Returns {race_winner, podium_finisher_top3, top5_finisher, constructor_winner}.
    H2H markets are built on-demand via /races/h2h endpoint.
    """
    if not riders:
        raise ValueError("riders cannot be empty")

    markets: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for market_name, builder, margin in [
        ("race_winner", build_race_winner_market, win_margin),
        ("podium_finisher_top3", build_podium_market, podium_margin),
        ("top5_finisher", build_top5_market, top5_margin),
        ("constructor_winner", build_constructor_market, constructor_margin),
    ]:
        try:
            if market_name == "constructor_winner":
                markets[market_name] = builder(riders, margin)
            else:
                markets[market_name] = builder(riders, margin)
        except Exception as exc:
            logger.error("Failed to build %s market: %s", market_name, exc)
            errors[market_name] = str(exc)

    result: dict[str, Any] = {
        "category": category,
        "circuit": circuit,
        "field_size": len(riders),
        "markets": markets,
    }
    if errors:
        result["errors"] = errors

    return result
