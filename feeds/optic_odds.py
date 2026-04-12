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
_LEAGUE_ID = "moto_gp"  # Optic Odds league slug for MotoGP premier class (motorsports sport)


class OpticOddsFeed:
    """Lightweight Optic Odds feed client for MotoGP/motorsports events."""

    def __init__(self) -> None:
        self._api_key = OPTIC_ODDS_API_KEY
        self._base_url = OPTIC_ODDS_BASE_URL
        self._sport = OPTIC_ODDS_SPORT_ID

    async def get_upcoming_races(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Fetch upcoming motorsports races.
        Returns list of event dicts with id, name, start_date, league.
        """
        if not self._api_key:
            raise RuntimeError(
                "OPTIC_ODDS_API_KEY not configured — set environment variable"
            )

        url = f"{self._base_url}/fixtures"
        params = {
            "key": self._api_key,
            "sportsbook": "pinnacle",
            "sport": self._sport,
            "league": _LEAGUE_ID,
            "is_live": "false",
            "limit": min(limit, 100),
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()

        data = resp.json()
        events = data.get("data", [])

        result = []
        for ev in events[:limit]:
            result.append({
                "id": ev.get("id"),
                "name": ev.get("name", ""),
                "start_date": ev.get("start_date", ""),
                "league": ev.get("league", {}).get("name", ""),
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
