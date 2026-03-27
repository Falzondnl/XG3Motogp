"""
XG3 MotoGP Microservice — Configuration
Race prediction for MotoGP, Moto2, Moto3.
"""
from __future__ import annotations

import os

SERVICE_NAME = "xg3-motogp"
SERVICE_VERSION = "1.0.0"
PORT = int(os.getenv("PORT", "8032"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Data paths
MOTOGP_CSV = os.getenv(
    "MOTOGP_CSV",
    "D:/codex/Data/motorsports/tier1/curated/motogp/api_race_results_full.csv",
)

# Model directories
MODEL_DIR = os.getenv("MODEL_DIR", "models")
R0_DIR = f"{MODEL_DIR}/r0"   # MotoGP premier class
R1_DIR = f"{MODEL_DIR}/r1"   # Moto2
R2_DIR = f"{MODEL_DIR}/r2"   # Moto3

# ELO parameters
ELO_K = 32
ELO_DEFAULT = 1500.0
HARVILLE_TOP_N = 30
MIN_RACES_FOR_ELO = 3

# Pricing margins
WIN_MARGIN = float(os.getenv("WIN_MARGIN", "0.12"))        # Race winner = 12%
PODIUM_MARGIN = float(os.getenv("PODIUM_MARGIN", "0.10"))  # Top-3 podium = 10%
TOP5_MARGIN = float(os.getenv("TOP5_MARGIN", "0.10"))      # Top-5 = 10%
H2H_MARGIN = float(os.getenv("H2H_MARGIN", "0.05"))        # H2H = 5%
CONSTRUCTOR_MARGIN = float(os.getenv("CONSTRUCTOR_MARGIN", "0.10"))  # Constructor = 10%

# Optic Odds
OPTIC_ODDS_API_KEY = os.getenv("OPTIC_ODDS_API_KEY", "")
OPTIC_ODDS_BASE_URL = "https://api.opticodds.com/api/v3"
OPTIC_ODDS_SPORT_ID = "motorsports"

# Observability
SENTRY_DSN = os.getenv("SENTRY_DSN", "")

# Feature settings
MAX_VALID_POSITION = 50    # Filter DNF/DSQ beyond position 50
MIN_RACE_SIZE = 5           # Minimum finishers to consider a race valid

# Temporal split seasons
TRAIN_SEASONS_MAX = 2017
VAL_SEASONS_MIN = 2018
VAL_SEASONS_MAX = 2020
TEST_SEASONS_MIN = 2021

# Category constants
# Note: The raw CSV uses trademark symbol ™ (U+2122) after MotoGP/Moto2/Moto3/MotoE.
# We normalise to clean names via _clean_category() in features.py.
CATEGORY_MOTOGP = "MotoGP"
CATEGORY_MOTO2 = "Moto2"
CATEGORY_MOTO3 = "Moto3"
CATEGORY_500CC = "500cc"    # Pre-2002 premier class (merged with MotoGP for model)
# These are the normalised names AFTER cleaning (strip trademark)
TARGET_CATEGORIES = {CATEGORY_MOTOGP, CATEGORY_MOTO2, CATEGORY_MOTO3, CATEGORY_500CC}
