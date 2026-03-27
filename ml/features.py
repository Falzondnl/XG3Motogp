"""
MotoGP Feature Extractor
Processes race results chronologically (2002-2025).
All features use pre-race state only — strict temporal integrity.

Features extracted per rider per race (17 total):
  rider_elo           — global ELO per category (MotoGP, Moto2, Moto3 separate pools)
  circuit_elo         — circuit-specific ELO for rider
  rolling_win_3       — win rate last 3 races in same category
  rolling_win_5       — win rate last 5 races in same category
  rolling_win_10      — win rate last 10 races in same category
  rolling_avg_pos_5   — average finishing position last 5 races, same category
  rolling_avg_pos_10  — average finishing position last 10 races, same category
  rolling_avg_pts_5   — average points last 5 races, same category
  career_wins         — total career wins in this category
  career_win_rate     — wins / starts in this category
  career_podium_rate  — podiums / starts in this category
  constructor_elo     — manufacturer ELO rating (Honda, Yamaha, Ducati, etc.)
  weather_dry         — 1 if dry conditions, 0 otherwise
  weather_wet         — 1 if wet/mixed conditions
  track_condition_enc — label-encoded track condition
  season_year_log     — log(season - 2001) for recency scaling
  session_is_sprint   — 1 if session_type=SPR, 0 if RAC
  avg_speed_last3     — rolling average speed last 3 races (0 if no data)
"""
from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import (
    MOTOGP_CSV,
    ELO_K,
    ELO_DEFAULT,
    MAX_VALID_POSITION,
    MIN_RACE_SIZE,
    CATEGORY_MOTOGP,
    CATEGORY_500CC,
    CATEGORY_MOTO2,
    CATEGORY_MOTO3,
    TARGET_CATEGORIES,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Category cleaning — raw CSV uses trademark ™ (U+2122) after class names
# -------------------------------------------------------------------
def _clean_category(cat: str) -> str:
    """Strip trademark symbol and whitespace from category name."""
    return cat.replace("\u2122", "").strip()


# -------------------------------------------------------------------
# Picklable defaultdict factories
# -------------------------------------------------------------------
def _elo_default() -> float:
    return ELO_DEFAULT


def _career_default() -> list:
    """[wins, podiums, starts]"""
    return [0, 0, 0]


# Features list — order matters for DataFrame construction
FEATURES = [
    "rider_elo",
    "circuit_elo",
    "rolling_win_3",
    "rolling_win_5",
    "rolling_win_10",
    "rolling_avg_pos_5",
    "rolling_avg_pos_10",
    "rolling_avg_pts_5",
    "career_wins",
    "career_win_rate",
    "career_podium_rate",
    "constructor_elo",
    "weather_dry",
    "weather_wet",
    "track_condition_enc",
    "season_year_log",
    "session_is_sprint",
    "avg_speed_last3",
]

# Normalise category: 500cc is treated as MotoGP for model purposes
def _normalise_category(cat: str) -> str:
    if cat in (CATEGORY_MOTOGP, CATEGORY_500CC):
        return CATEGORY_MOTOGP
    if cat == CATEGORY_MOTO2:
        return CATEGORY_MOTO2
    if cat == CATEGORY_MOTO3:
        return CATEGORY_MOTO3
    return cat


def _normalise_track_condition(tc: str) -> str:
    if not tc or pd.isna(tc):
        return "UNKNOWN"
    t = str(tc).strip().upper()
    if t in ("DRY", "GREAT", "GOOD"):
        return "DRY"
    if "WET" in t or "RAIN" in t or "DAMP" in t:
        return "WET"
    if "MIX" in t or "PARTIAL" in t:
        return "MIXED"
    return "UNKNOWN"


def _weather_flags(weather: str) -> tuple[float, float]:
    """Returns (weather_dry, weather_wet)."""
    if not weather or pd.isna(weather):
        return 1.0, 0.0  # assume dry if unknown
    w = str(weather).strip().upper()
    if "WET" in w or "RAIN" in w:
        return 0.0, 1.0
    if "MIXED" in w or "PARTIALLY" in w:
        return 0.0, 1.0
    return 1.0, 0.0


def _elo_update(
    rating_a: float, rating_b: float, score_a: float, k: float
) -> tuple[float, float]:
    """Standard ELO update. score_a=1 means A won, 0 means B won."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    delta = k * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


class MotogpFeatureExtractor:
    """
    Stateful feature extractor for MotoGP/Moto2/Moto3.
    Call build_dataset() once to produce training data.
    After training, call warm() to rebuild state from CSV up to today.
    Use get_features_for_race() for live inference.
    """

    def __init__(self) -> None:
        # ELO per (rider_name, category) — separate pools per class
        self.rider_elo: dict[tuple[str, str], float] = defaultdict(_elo_default)
        # ELO per (rider_name, circuit) — circuit-specific performance
        self.circuit_elo: dict[tuple[str, str], float] = defaultdict(_elo_default)
        # Constructor ELO per (constructor, category)
        self.constructor_elo: dict[tuple[str, str], float] = defaultdict(_elo_default)
        # Race history per (rider_name, category): list of (date, position, points, speed)
        self.race_history: dict[tuple[str, str], list[tuple[datetime, int, float, float]]] = defaultdict(list)
        # Career stats per (rider_name, category): [wins, podiums, starts]
        self.career_stats: dict[tuple[str, str], list[int]] = defaultdict(_career_default)
        # Track condition encoder
        self.track_condition_encoder: dict[str, int] = {}
        # Fitted flag
        self._fitted = False

    # ------------------------------------------------------------------
    # DATASET CONSTRUCTION (training only)
    # ------------------------------------------------------------------

    def build_dataset(self, csv_path: str = MOTOGP_CSV) -> pd.DataFrame:
        """
        Reads MotoGP CSV, iterates chronologically, extracts pre-race features.
        Returns DataFrame with FEATURES + ['race_key', 'season', 'category', 'target'].
        Strict temporal integrity: features at time T use ONLY data before T.
        """
        logger.info("Loading MotoGP CSV from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("Loaded %d rows", len(df))

        # Normalise category names (strip trademark ™ symbol)
        df["category"] = df["category"].astype(str).apply(_clean_category)

        # Filter to target categories
        df = df[df["category"].isin(TARGET_CATEGORIES)].copy()
        logger.info("After category filter: %d rows", len(df))

        # Focus on RAC sessions for training (sprints added if >500 rows)
        rac_df = df[df["session_type"] == "RAC"].copy()
        spr_df = df[df["session_type"] == "SPR"].copy()
        logger.info("RAC rows: %d | SPR rows: %d", len(rac_df), len(spr_df))

        # Build combined
        work_df = pd.concat([rac_df, spr_df], ignore_index=True) if len(spr_df) >= 500 else rac_df

        # Parse date
        work_df["date_parsed"] = pd.to_datetime(
            work_df["session_date"], format="mixed", dayfirst=False, errors="coerce"
        )
        work_df = work_df.dropna(subset=["date_parsed"])

        # Filter to modern era (2002+) — MotoGP brand started 2002
        work_df = work_df[work_df["season"] >= 2002].copy()

        # Normalise category
        work_df["cat_norm"] = work_df["category"].apply(_normalise_category)

        # Normalise position — keep only valid finishers
        work_df["position"] = pd.to_numeric(work_df["position"], errors="coerce")
        work_df = work_df.dropna(subset=["position"])
        work_df = work_df[work_df["position"] <= MAX_VALID_POSITION].copy()
        work_df["position"] = work_df["position"].astype(int)

        # Points — numeric
        work_df["points"] = pd.to_numeric(work_df["points"], errors="coerce").fillna(0.0)

        # Average speed — numeric
        work_df["average_speed"] = pd.to_numeric(work_df["average_speed"], errors="coerce").fillna(0.0)

        # Build track condition encoder from full data
        all_conditions = sorted(
            set(work_df["track_condition"].dropna().apply(_normalise_track_condition).unique())
        )
        self.track_condition_encoder = {c: i for i, c in enumerate(all_conditions)}

        # Build race key: season + event_short + cat_norm + session_type
        work_df["race_key"] = (
            work_df["season"].astype(str) + "_"
            + work_df["event_short"].fillna("UNK").str.replace(" ", "_")
            + "_" + work_df["cat_norm"]
            + "_" + work_df["session_type"]
        )

        # Sort chronologically
        work_df = work_df.sort_values(
            ["date_parsed", "race_key", "position"]
        ).reset_index(drop=True)

        logger.info(
            "Building features for %d valid rider-race rows ...", len(work_df)
        )

        # Process race by race
        race_order = (
            work_df.groupby("race_key", sort=False)["date_parsed"]
            .first()
            .sort_values()
        )
        race_df_map = {rk: grp for rk, grp in work_df.groupby("race_key")}

        records: list[dict[str, Any]] = []

        for race_key, race_date in race_order.items():
            grp = race_df_map[race_key].copy()
            if len(grp) < MIN_RACE_SIZE:
                continue

            season = int(grp["season"].iloc[0])
            cat_norm = str(grp["cat_norm"].iloc[0])
            session_type = str(grp["session_type"].iloc[0])
            weather = str(grp["weather"].iloc[0]) if pd.notna(grp["weather"].iloc[0]) else ""
            track_cond = str(grp["track_condition"].iloc[0]) if pd.notna(grp["track_condition"].iloc[0]) else ""
            circuit = str(grp["circuit"].iloc[0]) if pd.notna(grp["circuit"].iloc[0]) else "UNKNOWN"

            weather_dry, weather_wet = _weather_flags(weather)
            tc_norm = _normalise_track_condition(track_cond)
            tc_enc = float(self.track_condition_encoder.get(tc_norm, 0))
            session_is_sprint = 1.0 if session_type == "SPR" else 0.0
            season_year_log = float(np.log(max(season - 2001, 1)))

            # Extract PRE-RACE features
            race_records: list[dict[str, Any]] = []
            for _, row in grp.iterrows():
                rider = str(row["rider_name"])
                constructor = str(row["constructor"]) if pd.notna(row["constructor"]) else "UNKNOWN"

                feats = self._extract_features(
                    rider=rider,
                    category=cat_norm,
                    circuit=circuit,
                    constructor=constructor,
                    weather_dry=weather_dry,
                    weather_wet=weather_wet,
                    tc_enc=tc_enc,
                    season_year_log=season_year_log,
                    session_is_sprint=session_is_sprint,
                )
                feats["race_key"] = race_key
                feats["season"] = season
                feats["category"] = cat_norm
                feats["target"] = 1 if int(row["position"]) == 1 else 0
                race_records.append(feats)

            records.extend(race_records)

            # UPDATE state AFTER extracting all features
            self._update_elo_from_race(grp, cat_norm, circuit)
            self._update_history_from_race(grp, cat_norm, race_date)

        dataset = pd.DataFrame(records)
        logger.info(
            "Dataset built: %d rows, target balance=%.4f",
            len(dataset),
            dataset["target"].mean() if len(dataset) else 0.0,
        )
        self._fitted = True
        return dataset

    def _extract_features(
        self,
        rider: str,
        category: str,
        circuit: str,
        constructor: str,
        weather_dry: float,
        weather_wet: float,
        tc_enc: float,
        season_year_log: float,
        session_is_sprint: float,
    ) -> dict[str, Any]:
        cat_key = (rider, category)
        circ_key = (rider, circuit)
        cstr_key = (constructor, category)

        # ELO
        r_elo = float(self.rider_elo[cat_key])
        c_elo = float(self.circuit_elo[circ_key])
        cstr_elo = float(self.constructor_elo[cstr_key])

        # Race history
        hist = self.race_history[cat_key]  # list of (date, pos, pts, speed)
        positions = [pos for _, pos, _, _ in hist]
        points_list = [pts for _, _, pts, _ in hist]
        speeds = [spd for _, _, _, spd in hist]

        # Rolling win rates
        def win_rate_last_n(n: int) -> float:
            last = positions[-n:] if len(positions) >= n else positions
            return float(sum(1 for p in last if p == 1) / len(last)) if last else 0.0

        rolling_win_3 = win_rate_last_n(3)
        rolling_win_5 = win_rate_last_n(5)
        rolling_win_10 = win_rate_last_n(10)

        # Rolling avg position
        last5_pos = positions[-5:] if positions else []
        last10_pos = positions[-10:] if positions else []
        rolling_avg_pos_5 = float(np.mean(last5_pos)) if last5_pos else 15.0
        rolling_avg_pos_10 = float(np.mean(last10_pos)) if last10_pos else 15.0

        # Rolling avg points
        last5_pts = points_list[-5:] if points_list else []
        rolling_avg_pts_5 = float(np.mean(last5_pts)) if last5_pts else 0.0

        # Average speed last 3
        last3_spd = [s for s in speeds[-3:] if s > 0.0] if speeds else []
        avg_speed_last3 = float(np.mean(last3_spd)) if last3_spd else 0.0

        # Career stats
        career = self.career_stats[cat_key]  # [wins, podiums, starts]
        career_wins = float(career[0])
        starts = career[2]
        career_win_rate = float(career[0] / starts) if starts > 0 else 0.0
        career_podium_rate = float(career[1] / starts) if starts > 0 else 0.0

        return {
            "rider_elo": r_elo,
            "circuit_elo": c_elo,
            "rolling_win_3": rolling_win_3,
            "rolling_win_5": rolling_win_5,
            "rolling_win_10": rolling_win_10,
            "rolling_avg_pos_5": rolling_avg_pos_5,
            "rolling_avg_pos_10": rolling_avg_pos_10,
            "rolling_avg_pts_5": rolling_avg_pts_5,
            "career_wins": career_wins,
            "career_win_rate": career_win_rate,
            "career_podium_rate": career_podium_rate,
            "constructor_elo": cstr_elo,
            "weather_dry": weather_dry,
            "weather_wet": weather_wet,
            "track_condition_enc": tc_enc,
            "season_year_log": season_year_log,
            "session_is_sprint": session_is_sprint,
            "avg_speed_last3": avg_speed_last3,
        }

    def _update_elo_from_race(
        self, grp: pd.DataFrame, cat_norm: str, circuit: str
    ) -> None:
        """
        Pairwise ELO update: higher finisher beats lower finisher.
        Scale K by 1/n to keep total ELO movement reasonable in large fields.
        """
        ranked = grp.sort_values("position").reset_index(drop=True)
        if len(ranked) < 2:
            return

        riders = [str(r["rider_name"]) for _, r in ranked.iterrows()]
        constructors = [str(r["constructor"]) if pd.notna(r["constructor"]) else "UNKNOWN"
                        for _, r in ranked.iterrows()]
        n = len(riders)
        k_scaled = ELO_K / float(n)

        for i in range(n):
            for j in range(i + 1, n):
                a_rid, b_rid = riders[i], riders[j]
                # Rider ELO (category pool)
                a_key = (a_rid, cat_norm)
                b_key = (b_rid, cat_norm)
                new_a, new_b = _elo_update(
                    self.rider_elo[a_key], self.rider_elo[b_key], 1.0, k_scaled
                )
                self.rider_elo[a_key] = new_a
                self.rider_elo[b_key] = new_b

                # Circuit ELO
                ac_key = (a_rid, circuit)
                bc_key = (b_rid, circuit)
                new_ac, new_bc = _elo_update(
                    self.circuit_elo[ac_key], self.circuit_elo[bc_key], 1.0, k_scaled
                )
                self.circuit_elo[ac_key] = new_ac
                self.circuit_elo[bc_key] = new_bc

                # Constructor ELO
                a_cstr = constructors[i]
                b_cstr = constructors[j]
                ack = (a_cstr, cat_norm)
                bck = (b_cstr, cat_norm)
                new_ac2, new_bc2 = _elo_update(
                    self.constructor_elo[ack], self.constructor_elo[bck], 1.0, k_scaled
                )
                self.constructor_elo[ack] = new_ac2
                self.constructor_elo[bck] = new_bc2

    def _update_history_from_race(
        self, grp: pd.DataFrame, cat_norm: str, race_date: datetime
    ) -> None:
        """Append race results to history and career stats."""
        for _, row in grp.iterrows():
            rider = str(row["rider_name"])
            pos = int(row["position"])
            pts = float(row["points"]) if pd.notna(row["points"]) else 0.0
            spd = float(row["average_speed"]) if pd.notna(row["average_speed"]) else 0.0
            key = (rider, cat_norm)
            self.race_history[key].append((race_date, pos, pts, spd))
            self.career_stats[key][2] += 1  # starts
            if pos == 1:
                self.career_stats[key][0] += 1  # wins
            if pos <= 3:
                self.career_stats[key][1] += 1  # podiums

    # ------------------------------------------------------------------
    # WARM: rebuild state from CSV (used at service start)
    # ------------------------------------------------------------------

    def warm(self, csv_path: str = MOTOGP_CSV) -> None:
        """
        Rebuild internal state from full CSV history.
        Same logic as build_dataset() but discards feature rows — only updates state.
        """
        logger.info("Warming MotoGP feature extractor from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        df["category"] = df["category"].astype(str).apply(_clean_category)
        df = df[df["category"].isin(TARGET_CATEGORIES)].copy()

        df["date_parsed"] = pd.to_datetime(
            df["session_date"], format="mixed", dayfirst=False, errors="coerce"
        )
        df = df.dropna(subset=["date_parsed"])
        df = df[df["season"] >= 2002].copy()
        df["cat_norm"] = df["category"].apply(_normalise_category)
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        df = df.dropna(subset=["position"])
        df = df[df["position"] <= MAX_VALID_POSITION].copy()
        df["position"] = df["position"].astype(int)
        df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
        df["average_speed"] = pd.to_numeric(df["average_speed"], errors="coerce").fillna(0.0)

        # Rebuild track condition encoder
        all_conditions = sorted(
            set(df["track_condition"].dropna().apply(_normalise_track_condition).unique())
        )
        self.track_condition_encoder = {c: i for i, c in enumerate(all_conditions)}

        df["race_key"] = (
            df["season"].astype(str) + "_"
            + df["event_short"].fillna("UNK").str.replace(" ", "_")
            + "_" + df["cat_norm"]
            + "_" + df["session_type"]
        )

        df = df.sort_values(["date_parsed", "race_key", "position"]).reset_index(drop=True)

        race_order = (
            df.groupby("race_key", sort=False)["date_parsed"]
            .first()
            .sort_values()
        )
        race_df_map = {rk: grp for rk, grp in df.groupby("race_key")}

        for race_key, race_date in race_order.items():
            grp = race_df_map[race_key].copy()
            if len(grp) < MIN_RACE_SIZE:
                continue
            cat_norm = str(grp["cat_norm"].iloc[0])
            circuit = str(grp["circuit"].iloc[0]) if pd.notna(grp["circuit"].iloc[0]) else "UNKNOWN"
            self._update_elo_from_race(grp, cat_norm, circuit)
            self._update_history_from_race(grp, cat_norm, race_date)

        self._fitted = True
        logger.info(
            "Warm complete: %d rider-category ELOs, %d circuit ELOs, %d constructor ELOs",
            len(self.rider_elo),
            len(self.circuit_elo),
            len(self.constructor_elo),
        )

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def get_features_for_race(
        self,
        riders: list[dict],
        category: str,
        circuit: str,
        weather: str = "",
        track_condition: str = "",
        season: int = 2025,
        session_type: str = "RAC",
    ) -> list[dict]:
        """
        For live inference: return feature dict per rider using current state.

        riders: list of dicts, each with keys:
            - 'name': str  (rider name)
            - 'team': str  (optional)
            - 'constructor': str  (optional)
        category: 'MotoGP', 'Moto2', or 'Moto3'
        circuit: circuit name string
        weather: weather description string
        track_condition: track condition string
        season: race season year
        session_type: 'RAC' or 'SPR'

        Returns list of dicts with FEATURES + rider name metadata.
        """
        if not self._fitted:
            raise RuntimeError(
                "Extractor not fitted — call warm() or build_dataset() first"
            )

        cat_norm = _normalise_category(category)
        weather_dry, weather_wet = _weather_flags(weather)
        tc_norm = _normalise_track_condition(track_condition)
        tc_enc = float(self.track_condition_encoder.get(tc_norm, 0))
        season_year_log = float(np.log(max(season - 2001, 1)))
        session_is_sprint = 1.0 if session_type == "SPR" else 0.0

        result = []
        for r in riders:
            rider = str(r.get("name", r.get("rider_name", "Unknown")))
            constructor = str(r.get("constructor", "UNKNOWN"))

            feats = self._extract_features(
                rider=rider,
                category=cat_norm,
                circuit=circuit,
                constructor=constructor,
                weather_dry=weather_dry,
                weather_wet=weather_wet,
                tc_enc=tc_enc,
                season_year_log=season_year_log,
                session_is_sprint=session_is_sprint,
            )
            feats["rider_name"] = rider
            feats["team"] = str(r.get("team", "UNKNOWN"))
            feats["constructor"] = constructor
            result.append(feats)

        return result

    # ------------------------------------------------------------------
    # SERIALISATION
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("MotogpFeatureExtractor saved to %s", path)

    @staticmethod
    def load(path: str) -> "MotogpFeatureExtractor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("MotogpFeatureExtractor loaded from %s", path)
        return obj

    @property
    def rider_count(self) -> int:
        return len(set(k[0] for k in self.rider_elo.keys()))

    @property
    def circuit_elo_count(self) -> int:
        return len(self.circuit_elo)

    @property
    def constructor_count(self) -> int:
        return len(set(k[0] for k in self.constructor_elo.keys()))
