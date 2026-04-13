"""
Optic Odds feed integration for MotoGP MS.
Fetches upcoming motorsports races from the Optic Odds API v3.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from config import OPTIC_ODDS_API_KEY, OPTIC_ODDS_BASE_URL, OPTIC_ODDS_SPORT_ID

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
# VERIFIED 2026-04-13: Optic Odds v3 /leagues?sport=motorsports returns only:
#   formula_1, indycar, nascar_-_cup_series, nascar_-_truck_series, nascar_-_xfinity_series
# MotoGP is NOT carried by Optic Odds. Any attempt to query it returns 0 fixtures.
# The previous value "moto_gp" was incorrect and caused silent empty results.
# Fixture discovery falls back to the internal MotoGP calendar (see /admin/status).
_LEAGUE_ID = "moto_gp"   # NOT available on Optic — kept for reference only
_OPTIC_HAS_MOTOGP = False  # Truthful flag: prevents unnecessary API calls


class OpticOddsFeed:
    """Lightweight Optic Odds feed client for MotoGP/motorsports events."""

    def __init__(self) -> None:
        self._api_key = OPTIC_ODDS_API_KEY
        self._base_url = OPTIC_ODDS_BASE_URL
        self._sport = OPTIC_ODDS_SPORT_ID

    async def get_upcoming_races(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Fetch upcoming MotoGP races.

        NOTE: Optic Odds does NOT carry MotoGP (verified 2026-04-13).
        This method returns an empty list so callers (e.g. /races/upcoming)
        can clearly communicate that Optic is not the fixture source for MotoGP.
        The MotoGP MS uses its internal ELO/circuit calendar for fixture discovery;
        see /admin/status for rider_count and circuit_elo_count.

        Returns:
            Empty list — Optic Odds has no MotoGP fixtures.
        """
        if not _OPTIC_HAS_MOTOGP:
            logger.info(
                "motogp_optic_feed_skipped: Optic Odds does not carry MotoGP. "
                "Use the internal MotoGP calendar via /admin/status."
            )
            return []

        if not self._api_key:
            raise RuntimeError(
                "OPTIC_ODDS_API_KEY not configured — set environment variable"
            )

        url = f"{self._base_url}/fixtures/active"
        params = {
            "sport": self._sport,
            "league": _LEAGUE_ID,
            "limit": min(limit, 100),
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                url, params=params, headers={"X-Api-Key": self._api_key}
            )
            resp.raise_for_status()

        data = resp.json()
        events = data.get("data", [])

        result = []
        for ev in events[:limit]:
            result.append({
                "id": ev.get("id"),
                "name": ev.get("name", ""),
                "start_date": ev.get("start_date", ""),
                "league": ev.get("league", {}).get("name", "")
                if isinstance(ev.get("league"), dict)
                else str(ev.get("league", "")),
                "status": ev.get("status", ""),
            })

        return result

    async def get_race_odds(self, event_id: str) -> dict[str, Any]:
        """
        Fetch odds for a specific race event.
        Returns raw Optic Odds response.
        """
        if not self._api_key:
            raise RuntimeError("OPTIC_ODDS_API_KEY not configured")

        url = f"{self._base_url}/odds"
        params = {
            "key": self._api_key,
            "sportsbook": "pinnacle",
            "fixture_id": event_id,
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()

        return resp.json()
