"""
========================================================================
STEP 3 — Feature Engineering & Categorical Variable Encoding
========================================================================
PURPOSE:
    Transform the cleaned data into a format optimized for ML models:
    1. Create NEW derived features from existing ones
    2. Scale numeric features using StandardScaler
    3. Encode the target variable (Class) using LabelEncoder

WHY FEATURE ENGINEERING?
    Raw features capture individual measurements, but RELATIONSHIPS
    between features often have more predictive power. For example:
    - A bean's Area/Perimeter ratio captures its "compactness"
      differently from the existing Compactness feature.
    - Combining shape factors can reveal higher-order geometric
      properties that separate classes.

WHY STANDARD SCALING?
    - Gradient-based models (XGBoost, GBM) converge faster when
      features are on similar scales.
    - Distance-based operations (e.g., KNN, SVM) are directly
      affected by feature magnitudes — a feature with range
      [0, 100000] would dominate one with range [0, 1].
    - Tree models (RF) are NOT affected by scaling, but scaling
      doesn't hurt them, and we want a unified pipeline.

    StandardScaler: z = (x - μ) / σ
    After scaling, each feature has mean ≈ 0 and std ≈ 1.

TARGET ENCODING:
    Models need numeric inputs, not strings. LabelEncoder maps each
    class name to an integer: BARBUNYA→0, BOMBAY→1, ..., SIRA→6.
========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings

from config import (
    FEATURE_COLS, TARGET_COL, ID_COL,
    CLEANED_DIR, MODELS_DIR, RANDOM_SEED
)

warnings.filterwarnings("ignore")


def load_cleaned_data():
    """Load the cleaned datasets from Step 2."""
    train_df = pd.read_csv(f"{CLEANED_DIR}/train_cleaned.csv")
    val_df   = pd.read_csv(f"{CLEANED_DIR}/val_cleaned.csv")
    test_df  = pd.read_csv(f"{CLEANED_DIR}/test_cleaned.csv")

    print(f"  Loaded: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
    return train_df, val_df, test_df


def create_derived_features(df, dataset_name=""):
    """
    Create new features from mathematical combinations of existing ones.

    WHY THESE SPECIFIC FEATURES?
        Each derived feature captures a geometric RELATIONSHIP between
        measurements that the model cannot easily learn on its own
        (especially tree-based models that split on single features).

    IMPORTANT: We apply the SAME feature engineering to train, val,
    AND test sets to keep the feature space consistent.
    """
    # --- Feature 1: Area-to-Perimeter Ratio ---
    # WHY: A circle has the highest area/perimeter ratio for a given
    # perimeter. This ratio captures how "round" vs "elongated" the bean is.
    # Different bean types have different characteristic shapes.
    df["AreaPerimeterRatio"] = df["Area"] / df["Perimeter"]

    # --- Feature 2: Axis Ratio (alternative to AspectRatio) ---
    # WHY: MinorAxis/MajorAxis gives a value between 0 and 1.
    # Values close to 1 mean circular; close to 0 means very elongated.
    # This is the INVERSE of AspectRatio but bounded in [0,1],
    # which can be easier for some models to work with.
    df["AxisRatio"] = df["MinorAxisLength"] / df["MajorAxisLength"]

    # --- Feature 3: Roundness × Compactness ---
    # WHY: Both measure shape regularity but from different angles.
    # Their product combines both perspectives into a single feature
    # that can capture subtle shape differences between bean types.
    df["RoundnessCompactness"] = df["roundness"] * df["Compactness"]

    # --- Feature 4: Shape Index (SF1 × SF2) ---
    # WHY: ShapeFactor1 and ShapeFactor2 measure different aspects of
    # morphology. Their product creates a combined shape descriptor.
    df["ShapeIndex"] = df["ShapeFactor1"] * df["ShapeFactor2"]

    # --- Feature 5: Solidity Deficit ---
    # WHY: Solidity = Area / ConvexArea. A Solidity of 1.0 means the
    # bean has no concavities. The deficit (1 - Solidity) measures how
    # irregular the bean's boundary is.
    df["SolidityDeficit"] = 1.0 - df["Solidity"]

    # --- Feature 6: Equivalent Diameter to Major Axis ---
    # WHY: This ratio captures how much of the major axis length is
    # "filled" by the bean — another shape characterization.
    df["EquivDiameterMajorRatio"] = df["EquivDiameter"] / df["MajorAxisLength"]

    print(f"  ✓ {dataset_name}: Created 6 derived features")
    return df


def encode_target(train_df, val_df):
    """
    Encode the categorical target variable (Class) to integers
    using sklearn's LabelEncoder.

    WHY LabelEncoder (not OneHotEncoder for target)?
        LabelEncoder maps each class to a SINGLE integer:
            BARBUNYA→0, BOMBAY→1, CALI→2, DERMASON→3, etc.
        
        This is correct for multi-class classification where the
        target is a single column of class labels.
        
        OneHotEncoder would create 7 binary columns — that's for
        FEATURES, not for the TARGET variable.
    
    IMPORTANT: We fit the encoder on the TRAINING set and use the
    same mapping for validation (and later for decoding predictions).
    """
    le = LabelEncoder()

    # .fit_transform() learns the mapping AND applies it in one step
    train_df[TARGET_COL] = le.fit_transform(train_df[TARGET_COL])
    
    # .transform() applies the SAME mapping learned from train
    val_df[TARGET_COL] = le.transform(val_df[TARGET_COL])

    print(f"\n  Label Encoding Mapping:")
    for cls, idx in zip(le.classes_, range(len(le.classes_))):
        print(f"    {cls} → {idx}")

    # Save the encoder for later use (decoding predictions in Step 7)
    joblib.dump(le, f"{MODELS_DIR}/label_encoder.pkl")
    print(f"\n  ✓ Saved LabelEncoder to models/label_encoder.pkl")

    return train_df, val_df, le


def scale_features(train_df, val_df, test_df):
    """
    Standardize all numeric features using StandardScaler.

    HOW StandardScaler WORKS:
        For each feature, it computes:
            z = (x - mean) / std_dev
        After scaling:
            - Mean of each feature ≈ 0
            - Standard deviation of each feature ≈ 1

    CRITICAL — TARGET LEAKAGE PREVENTION:
        We call .fit() on TRAINING data ONLY. This computes the
        mean and std from training data.
        
        We then call .transform() on val and test using THOSE SAME
        statistics. We do NOT call .fit_transform() on val or test.
        
        If we computed separate means/stds for val/test, the scaling
        would be inconsistent across splits, AND we'd be leaking
        information from val/test into our preprocessing pipeline.
    """
    # Identify which columns to scale (all feature columns + derived ones)
    # We scale ALL numeric features including the derived ones
    feature_cols_all = [col for col in train_df.columns
                        if col not in [ID_COL, TARGET_COL]]

    scaler = StandardScaler()

    # .fit_transform() on TRAINING data: learn mean/std AND transform
    train_df[feature_cols_all] = scaler.fit_transform(
        train_df[feature_cols_all]
    )

    # .transform() on VAL/TEST: use training mean/std to transform
    val_df[feature_cols_all] = scaler.transform(val_df[feature_cols_all])
    test_df[feature_cols_all] = scaler.transform(test_df[feature_cols_all])

    print(f"\n  ✓ Scaled {len(feature_cols_all)} features using StandardScaler")
    print(f"    (fit on train, transformed on train/val/test)")

    # Save the scaler for reproducibility
    joblib.dump(scaler, f"{MODELS_DIR}/standard_scaler.pkl")
    print(f"  ✓ Saved StandardScaler to models/standard_scaler.pkl")

    return train_df, val_df, test_df, scaler, feature_cols_all


def save_engineered_data(train_df, val_df, test_df):
    """Save the feature-engineered datasets for the next step."""
    train_df.to_csv(f"{CLEANED_DIR}/train_engineered.csv", index=False)
    val_df.to_csv(f"{CLEANED_DIR}/val_engineered.csv", index=False)
    test_df.to_csv(f"{CLEANED_DIR}/test_engineered.csv", index=False)

    print(f"\n  ✓ Saved engineered datasets to {CLEANED_DIR}/")
    print(f"    - train_engineered.csv  ({train_df.shape[1]} columns)")
    print(f"    - val_engineered.csv    ({val_df.shape[1]} columns)")
    print(f"    - test_engineered.csv   ({test_df.shape[1]} columns)")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  STEP 3: Feature Engineering & Encoding                  ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. Load cleaned data from Step 2
    print("[3a] Loading cleaned datasets...")
    train_df, val_df, test_df = load_cleaned_data()

    # 2. Create derived features
    print("\n[3b] Creating derived features...")
    train_df = create_derived_features(train_df, "Train")
    val_df   = create_derived_features(val_df, "Val")
    test_df  = create_derived_features(test_df, "Test")

    # Show all columns after feature engineering
    print(f"\n  All features ({len(train_df.columns)} total):")
    for col in train_df.columns:
        print(f"    - {col}")

    # 3. Encode target variable
    print("\n[3c] Encoding target variable...")
    train_df, val_df, le = encode_target(train_df, val_df)

    # 4. Scale features
    print("\n[3d] Scaling features...")
    train_df, val_df, test_df, scaler, feature_cols_all = scale_features(
        train_df, val_df, test_df
    )

    # Verify scaling worked: check mean ≈ 0, std ≈ 1 on training data
    print(f"\n  Verification (train set after scaling):")
    print(f"    Mean range: [{train_df[feature_cols_all].mean().min():.4f}, "
          f"{train_df[feature_cols_all].mean().max():.4f}] (should be ≈ 0)")
    print(f"    Std range:  [{train_df[feature_cols_all].std().min():.4f}, "
          f"{train_df[feature_cols_all].std().max():.4f}] (should be ≈ 1)")

    # 5. Save engineered data
    print("\n[3e] Saving engineered datasets...")
    save_engineered_data(train_df, val_df, test_df)

    print()
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # VIVA PREP — Step 3
    # ──────────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  VIVA PREP — Questions Your Professor Might Ask          │")
    print("└" + "─" * 58 + "┘")
    print("""
    Q1: You called scaler.fit_transform() on training data and
        scaler.transform() on validation data. Why not call
        fit_transform() on both?
    A1: fit_transform() computes NEW statistics (mean, std) from the
        data it's given. If we did this on validation data, the
        scaling would be based on validation statistics — which is
        TARGET LEAKAGE. We must use training statistics to transform
        all datasets, so the model sees consistently scaled data.

    Q2: Why did you create AreaPerimeterRatio as a new feature
        instead of letting the model figure it out?
    A2: Tree-based models split on ONE feature at a time. They
        CANNOT learn the relationship Area/Perimeter in a single
        split — they'd need multiple cascading splits to approximate
        it. By creating the ratio explicitly, we give the model
        direct access to this geometric relationship.

    Q3: Why LabelEncoder for the target and not OneHotEncoder?
    A3: LabelEncoder maps each class to one integer (e.g., CALI→2).
        This is correct for the target variable in multi-class
        classification. OneHotEncoder would create 7 binary columns,
        which is used for FEATURE encoding, not target encoding.
        Models like RandomForest and XGBoost expect a single
        integer-encoded target column.
    """)
