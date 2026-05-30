"""
========================================================================
config.py — Centralized Project Configuration
========================================================================
PURPOSE:
    This module serves as the single source of truth for all paths,
    column definitions, and constants used across every step of the
    Bean Plant Classification pipeline.

WHY A SEPARATE CONFIG?
    - Modular path management: If the project directory changes,
      you only edit ONE file.
    - Consistency: Every script imports the same column lists,
      avoiding hard-coded strings scattered across files.
    - Reproducibility: A fixed RANDOM_SEED ensures all random
      operations (train/test splits, model initialization) produce
      identical results on every run.
========================================================================
"""

import os

# ──────────────────────────────────────────────────────────────────────
# 1. DIRECTORY PATHS
# ──────────────────────────────────────────────────────────────────────
# os.path.dirname(__file__) resolves to the folder containing config.py,
# making all paths relative and portable across machines.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw data files (as provided by the competition)
DATA_DIR = BASE_DIR  # CSVs live in the same directory

# Output directories — created automatically if they don't exist
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
CLEANED_DIR = os.path.join(BASE_DIR, "data_cleaned")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")

# Create output directories if they don't exist
for _dir in [FIGURES_DIR, CLEANED_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 2. DATA FILE PATHS
# ──────────────────────────────────────────────────────────────────────
TRAIN_FILE = os.path.join(DATA_DIR, "train_c.csv")
VAL_FILE   = os.path.join(DATA_DIR, "val.csv")
TEST_FILE  = os.path.join(DATA_DIR, "test_no_label.csv")

# ──────────────────────────────────────────────────────────────────────
# 3. COLUMN DEFINITIONS
# ──────────────────────────────────────────────────────────────────────
# These lists are the AUTHORITATIVE reference for column names.
# Note: The original data has a typo "AspectRation" — we will rename
# it to "AspectRatio" during preprocessing, but use the original name
# when reading raw files.

ID_COL     = "id"
TARGET_COL = "Class"

# All 16 numeric feature columns (original names from raw CSV)
FEATURE_COLS_RAW = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

# Corrected feature column names (after renaming in preprocessing)
FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

# ──────────────────────────────────────────────────────────────────────
# 4. CLASS LABELS
# ──────────────────────────────────────────────────────────────────────
# The 7 bean varieties in the dataset, ordered alphabetically
# (this is the order LabelEncoder will use)
CLASS_LABELS = [
    "BARBUNYA", "BOMBAY", "CALI", "DERMASON",
    "HOROZ", "SEKER", "SIRA"
]

# ──────────────────────────────────────────────────────────────────────
# 5. REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────
# A fixed random seed ensures that every random operation
# (shuffling, model init, cross-validation splits) produces
# identical results on every run.
RANDOM_SEED = 42
