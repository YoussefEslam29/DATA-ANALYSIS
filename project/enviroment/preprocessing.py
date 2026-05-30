"""
========================================================================
STEP 2 — Data Preprocessing & Missing/Invalid Value Imputation
========================================================================
PURPOSE:
    Clean the raw data by fixing data quality issues discovered in EDA:
    1. Negative Area values → take absolute value
    2. Zero MajorAxisLength → impute with class-conditional median
    3. Rename the typo column "AspectRation" → "AspectRatio"
    4. Detect and cap extreme outliers using IQR bounds

WHY PREPROCESSING MATTERS:
    Machine learning models learn patterns from data. If the data
    contains errors (negative physical measurements, zeros that should
    be non-zero), the model learns WRONG patterns. Preprocessing
    ensures the model sees only valid, realistic data.

CRITICAL RULE — PREVENTING TARGET LEAKAGE:
    All imputation statistics (medians, IQR bounds) are computed
    ONLY from the training set, then applied to validation and test.
    
    WHY? If we compute the median from the ENTIRE dataset (including
    val/test), we're "peeking" at data the model should never see
    during training. This inflates validation metrics and gives a
    false sense of model quality.
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    FEATURE_COLS_RAW, FEATURE_COLS, TARGET_COL, ID_COL,
    FIGURES_DIR, CLEANED_DIR, CLASS_LABELS
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_raw_data():
    """Load the raw CSV files as-is from disk."""
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VAL_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    return train_df, val_df, test_df


def fix_negative_area(train_df, val_df, test_df):
    """
    Fix negative Area values by taking the absolute value.

    WHY ABSOLUTE VALUE (not dropping rows)?
        - The magnitude of the Area values is consistent with the rest
          of the data; only the sign is wrong. This suggests a data
          entry error (e.g., a minus sign was accidentally prepended).
        - Dropping rows wastes valid data, especially if the corrupted
          rows belong to minority classes (BOMBAY).
    """
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        neg_count = (df["Area"] < 0).sum()
        if neg_count > 0:
            # np.abs() takes the absolute value of each element
            df["Area"] = np.abs(df["Area"])
            print(f"  ✓ {name}: Fixed {neg_count} negative Area values → absolute value")
        else:
            print(f"  ✓ {name}: No negative Area values found")

    return train_df, val_df, test_df


def fix_zero_major_axis(train_df, val_df, test_df):
    """
    Impute zero MajorAxisLength values with the CLASS-CONDITIONAL
    MEDIAN from the training set.

    WHY CLASS-CONDITIONAL MEDIAN (not global median)?
        Different bean varieties have different physical sizes. The
        MajorAxisLength of a BOMBAY bean (~400) is vastly different
        from a DERMASON bean (~250). Using the global median would
        be inaccurate for both.

    WHY MEDIAN (not mean)?
        The median is robust to outliers. If there are a few extreme
        MajorAxisLength values, the mean gets pulled toward them,
        but the median stays at the center of the distribution.
    """
    # Step 1: Compute class-conditional medians from TRAINING SET ONLY
    # groupby(TARGET_COL) splits data by class, .median() computes
    # the median for each group
    class_medians = train_df[train_df["MajorAxisLength"] > 0].groupby(
        TARGET_COL
    )["MajorAxisLength"].median()
    
    global_median = train_df[train_df["MajorAxisLength"] > 0]["MajorAxisLength"].median()

    print(f"\n  Class-conditional medians for MajorAxisLength (from train):")
    for cls, med in class_medians.items():
        print(f"    {cls}: {med:.2f}")

    # Step 2: Apply imputation to all datasets
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        zero_mask = df["MajorAxisLength"] == 0
        zero_count = zero_mask.sum()

        if zero_count > 0:
            if TARGET_COL in df.columns:
                # For labeled data: use class-conditional median
                for cls in df.loc[zero_mask, TARGET_COL].unique():
                    cls_mask = zero_mask & (df[TARGET_COL] == cls)
                    df.loc[cls_mask, "MajorAxisLength"] = class_medians.get(cls, global_median)
            else:
                # For test data (no labels): use global median from train
                df.loc[zero_mask, "MajorAxisLength"] = global_median
            print(f"  ✓ {name}: Imputed {zero_count} zero MajorAxisLength values")
        else:
            print(f"  ✓ {name}: No zero MajorAxisLength values found")

    return train_df, val_df, test_df


def rename_columns(train_df, val_df, test_df):
    """
    Rename the typo column 'AspectRation' → 'AspectRatio'.

    WHY BOTHER?
        Clean column names prevent confusion and bugs. Imagine
        searching your code for 'AspectRatio' and not finding it
        because it's misspelled — this is a common source of errors.
    """
    rename_map = {"AspectRation": "AspectRatio"}

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if "AspectRation" in df.columns:
            # .rename(columns=dict) renames columns in-place
            df.rename(columns=rename_map, inplace=True)
            print(f"  ✓ {name}: Renamed 'AspectRation' → 'AspectRatio'")

    return train_df, val_df, test_df


def cap_outliers_iqr(train_df, val_df, test_df):
    """
    Cap extreme outliers using the IQR (Interquartile Range) method.

    HOW IQR CAPPING WORKS:
        1. Compute Q1 (25th percentile) and Q3 (75th percentile)
        2. IQR = Q3 - Q1
        3. Lower bound = Q1 - 1.5 × IQR
        4. Upper bound = Q3 + 1.5 × IQR
        5. Any value below the lower bound → set to lower bound
           Any value above the upper bound → set to upper bound

    WHY CAP (not drop)?
        Outlier rows often contain valid information for ALL other
        features. Dropping the entire row wastes that information.
        Capping retains the row while limiting the extreme value.

    WHY 1.5 × IQR?
        This is the standard threshold (Tukey's method). It captures
        ~99.3% of data for a normal distribution. Values beyond this
        are statistically unusual enough to warrant capping.

    CRITICAL: Bounds are computed from TRAINING DATA ONLY.
    """
    bounds = {}
    total_capped = {"Train": 0, "Val": 0, "Test": 0}

    # Step 1: Compute bounds from training set
    for feature in FEATURE_COLS:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[feature] = (lower, upper)

    # Step 2: Apply bounds to ALL datasets
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        for feature in FEATURE_COLS:
            lower, upper = bounds[feature]
            # np.clip(array, min, max) constrains values to [min, max]
            original = df[feature].copy()
            df[feature] = np.clip(df[feature], lower, upper)
            capped = (original != df[feature]).sum()
            total_capped[name] += capped

    for name, count in total_capped.items():
        print(f"  ✓ {name}: Capped {count} outlier values across all features")

    return train_df, val_df, test_df, bounds


def save_cleaned_data(train_df, val_df, test_df):
    """
    Save the cleaned DataFrames to CSV files in the data_cleaned/ directory.
    """
    train_df.to_csv(f"{CLEANED_DIR}/train_cleaned.csv", index=False)
    val_df.to_csv(f"{CLEANED_DIR}/val_cleaned.csv", index=False)
    test_df.to_csv(f"{CLEANED_DIR}/test_cleaned.csv", index=False)

    print(f"\n  ✓ Saved cleaned datasets to {CLEANED_DIR}/")
    print(f"    - train_cleaned.csv ({train_df.shape[0]:,} rows)")
    print(f"    - val_cleaned.csv   ({val_df.shape[0]:,} rows)")
    print(f"    - test_cleaned.csv  ({test_df.shape[0]:,} rows)")


def visualize_before_after(raw_train, cleaned_train):
    """
    Visualize the effect of preprocessing on the Area column
    to demonstrate the impact of cleaning.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before cleaning
    axes[0].hist(raw_train["Area"], bins=50, color="salmon",
                 edgecolor="black", alpha=0.7)
    axes[0].set_title("Area Distribution — BEFORE Cleaning",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Area")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=2,
                    label="Zero line")
    axes[0].legend()

    # After cleaning
    axes[1].hist(cleaned_train["Area"], bins=50, color="lightgreen",
                 edgecolor="black", alpha=0.7)
    axes[1].set_title("Area Distribution — AFTER Cleaning",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Area")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/preprocessing_before_after.png",
                bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/preprocessing_before_after.png")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  STEP 2: Data Preprocessing & Imputation                 ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. Load raw data
    print("Loading raw datasets...")
    train_df, val_df, test_df = load_raw_data()

    # Keep a copy for before/after comparison
    raw_train = train_df.copy()

    # 2. Fix negative Area values
    print("\n[2a] Fixing negative Area values...")
    train_df, val_df, test_df = fix_negative_area(train_df, val_df, test_df)

    # 3. Impute zero MajorAxisLength
    print("\n[2b] Imputing zero MajorAxisLength values...")
    train_df, val_df, test_df = fix_zero_major_axis(train_df, val_df, test_df)

    # 4. Rename typo column
    print("\n[2c] Renaming columns...")
    train_df, val_df, test_df = rename_columns(train_df, val_df, test_df)

    # 5. Cap outliers
    print("\n[2d] Capping outliers using IQR method...")
    train_df, val_df, test_df, bounds = cap_outliers_iqr(
        train_df, val_df, test_df
    )

    # 6. Visualize before/after
    print("\n[2e] Generating before/after visualization...")
    visualize_before_after(raw_train, train_df)

    # 7. Save cleaned data
    print("\n[2f] Saving cleaned datasets...")
    save_cleaned_data(train_df, val_df, test_df)

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # VIVA PREP — Step 2
    # ──────────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  VIVA PREP — Questions Your Professor Might Ask          │")
    print("└" + "─" * 58 + "┘")
    print("""
    Q1: Why did you compute the IQR bounds from the TRAINING set
        only, and not from the entire dataset?
    A1: To prevent TARGET LEAKAGE. If we compute bounds from the
        full dataset (including validation/test), we're using
        information from data the model should never see during
        training. This would give unrealistically optimistic
        validation scores.

    Q2: Why did you use MEDIAN imputation instead of MEAN for
        MajorAxisLength?
    A2: The median is ROBUST to outliers. If there are extreme
        values in MajorAxisLength, the mean gets pulled toward
        them (e.g., mean=300 vs median=250). The median gives
        a more representative "typical" value.

    Q3: Why cap outliers instead of removing them entirely?
    A3: Each row has 16 features. Even if one feature has an
        outlier, the other 15 features in that row may be
        perfectly valid. Removing the entire row wastes that
        information. Capping preserves the row while limiting
        the extreme value to a reasonable range.
    """)
