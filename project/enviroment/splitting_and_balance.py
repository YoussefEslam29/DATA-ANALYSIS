"""
========================================================================
STEP 4 — Train/Validation Splitting & Addressing Class Imbalance
========================================================================
PURPOSE:
    1. Prepare final X (features) and y (target) arrays for modeling
    2. Verify the stratified distribution is maintained
    3. Apply SMOTE to the TRAINING set to handle class imbalance

WHY IS CLASS IMBALANCE A PROBLEM?
    Our dataset has 1,997 DERMASON beans but only 309 BOMBAY beans —
    a ~6.5:1 ratio. If we train a model on this imbalanced data:
    
    - The model learns to heavily favor DERMASON predictions because
      it's "right" most of the time by sheer probability.
    - Minority classes (BOMBAY, BARBUNYA) get under-represented
      in the model's decision boundaries.
    - Overall accuracy looks good (e.g., 85%) but per-class
      performance for minority classes can be terrible (e.g., 40%).

WHAT IS SMOTE?
    SMOTE = Synthetic Minority Oversampling Technique.
    
    Instead of just duplicating minority samples (which overfits),
    SMOTE creates NEW synthetic samples by:
    1. Picking a minority class sample
    2. Finding its K nearest neighbors (same class)
    3. Creating a new sample at a random point between them
    
    This gives the model more DIVERSE examples of the minority class
    to learn from, reducing overfitting compared to simple duplication.

CRITICAL: SMOTE must be applied ONLY to training data.
    Never to validation or test — those must remain untouched to
    give an honest estimate of real-world performance.
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings

from config import (
    TARGET_COL, ID_COL, CLEANED_DIR, FIGURES_DIR, RANDOM_SEED
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_engineered_data():
    """Load the feature-engineered datasets from Step 3."""
    train_df = pd.read_csv(f"{CLEANED_DIR}/train_engineered.csv")
    val_df   = pd.read_csv(f"{CLEANED_DIR}/val_engineered.csv")
    test_df  = pd.read_csv(f"{CLEANED_DIR}/test_engineered.csv")
    return train_df, val_df, test_df


def prepare_arrays(train_df, val_df, test_df):
    """
    Split DataFrames into feature arrays (X) and target arrays (y).

    WHY SEPARATE X AND y?
        - X contains ONLY the input features the model uses to make
          predictions.
        - y contains ONLY the target labels the model tries to predict.
        - The 'id' column is metadata — not a feature. Including it
          would let the model memorize row IDs instead of learning
          actual patterns.
    """
    # Feature columns = everything except 'id' and 'Class'
    feature_cols = [col for col in train_df.columns
                    if col not in [ID_COL, TARGET_COL]]

    # Extract feature arrays
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values.astype(int)

    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values.astype(int)

    # Test has no target column
    X_test = test_df[feature_cols].values
    test_ids = test_df[ID_COL].values

    print(f"  Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"    - {col}")

    print(f"\n  Array shapes:")
    print(f"    X_train: {X_train.shape}    y_train: {y_train.shape}")
    print(f"    X_val:   {X_val.shape}      y_val:   {y_val.shape}")
    print(f"    X_test:  {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, test_ids, feature_cols


def verify_stratification(y_train, y_val):
    """
    Check that the class distribution in train and val is similar.

    WHY STRATIFICATION?
        If we randomly split data, minority classes might end up
        entirely in train OR val by chance. Stratified splitting
        ensures each split has the SAME proportion of each class,
        giving a fair evaluation.
    """
    from collections import Counter

    train_dist = Counter(y_train)
    val_dist = Counter(y_val)

    print(f"\n  {'Class':<12} {'Train Count':>12} {'Train %':>10} {'Val Count':>12} {'Val %':>10}")
    print(f"  {'-'*56}")

    for cls in sorted(set(y_train)):
        train_pct = train_dist[cls] / len(y_train) * 100
        val_pct = val_dist[cls] / len(y_val) * 100
        print(f"  {cls:<12} {train_dist[cls]:>12} {train_pct:>9.1f}% {val_dist[cls]:>12} {val_pct:>9.1f}%")


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to oversample minority classes in the TRAINING set.

    SMOTE PARAMETERS:
        - random_state: Fixed seed for reproducibility
        - k_neighbors: Number of nearest neighbors to use (default=5)
          For very small minority classes, we may need to reduce this
          if a class has fewer than k_neighbors+1 samples.
    
    AFTER SMOTE:
        All classes will have the SAME number of samples (equal to
        the majority class count), giving the model equal exposure
        to every bean type.
    """
    print(f"\n  Before SMOTE: {X_train.shape[0]:,} samples")

    # Check minimum class count for k_neighbors parameter
    from collections import Counter
    class_counts = Counter(y_train)
    min_count = min(class_counts.values())

    # k_neighbors must be less than the number of samples in the
    # smallest class. Default is 5, reduce if needed.
    k = min(5, min_count - 1)

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)

    # .fit_resample() generates synthetic samples and returns
    # the augmented dataset
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:  {X_train_smote.shape[0]:,} samples")
    print(f"  k_neighbors used: {k}")

    return X_train_smote, y_train_smote


def plot_class_balance(y_before, y_after):
    """
    Visualize class distribution before and after SMOTE.
    """
    from collections import Counter

    before = Counter(y_before)
    after = Counter(y_after)
    classes = sorted(before.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("husl", n_colors=len(classes))

    # Before SMOTE
    axes[0].bar([str(c) for c in classes],
                [before[c] for c in classes],
                color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Class Distribution — BEFORE SMOTE",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Class (encoded)")
    axes[0].set_ylabel("Count")
    for i, c in enumerate(classes):
        axes[0].text(i, before[c] + 20, str(before[c]),
                     ha="center", fontweight="bold")

    # After SMOTE
    axes[1].bar([str(c) for c in classes],
                [after[c] for c in classes],
                color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Class Distribution — AFTER SMOTE",
                      fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Class (encoded)")
    axes[1].set_ylabel("Count")
    for i, c in enumerate(classes):
        axes[1].text(i, after[c] + 20, str(after[c]),
                     ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/smote_before_after.png", bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/smote_before_after.png")


def save_final_arrays(X_train, y_train, X_val, y_val, X_test, test_ids,
                      feature_cols):
    """
    Save the final arrays as .npy files for efficient loading in
    model training (Steps 5-6).

    WHY .npy FORMAT?
        - Faster to load than CSV for numeric arrays
        - Preserves exact numeric precision (no float→string→float)
        - Smaller file size for large arrays
    """
    np.save(f"{CLEANED_DIR}/X_train_smote.npy", X_train)
    np.save(f"{CLEANED_DIR}/y_train_smote.npy", y_train)
    np.save(f"{CLEANED_DIR}/X_val.npy", X_val)
    np.save(f"{CLEANED_DIR}/y_val.npy", y_val)
    np.save(f"{CLEANED_DIR}/X_test.npy", X_test)
    np.save(f"{CLEANED_DIR}/test_ids.npy", test_ids)

    # Also save feature column names for reference
    pd.Series(feature_cols).to_csv(
        f"{CLEANED_DIR}/feature_columns.csv", index=False, header=False
    )

    print(f"\n  ✓ Saved final arrays to {CLEANED_DIR}/")
    print(f"    - X_train_smote.npy: {X_train.shape}")
    print(f"    - y_train_smote.npy: {y_train.shape}")
    print(f"    - X_val.npy:         {X_val.shape}")
    print(f"    - y_val.npy:         {y_val.shape}")
    print(f"    - X_test.npy:        {X_test.shape}")
    print(f"    - test_ids.npy:      {test_ids.shape}")
    print(f"    - feature_columns.csv: {len(feature_cols)} columns")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  STEP 4: Splitting & Class Imbalance Handling            ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. Load engineered data
    print("[4a] Loading engineered datasets...")
    train_df, val_df, test_df = load_engineered_data()

    # 2. Prepare arrays
    print("\n[4b] Preparing feature and target arrays...")
    X_train, y_train, X_val, y_val, X_test, test_ids, feature_cols = \
        prepare_arrays(train_df, val_df, test_df)

    # 3. Verify stratification
    print("\n[4c] Verifying class distribution...")
    verify_stratification(y_train, y_val)

    # 4. Apply SMOTE
    print("\n[4d] Applying SMOTE to training set...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # 5. Visualize before/after
    print("\n[4e] Visualizing class balance...")
    plot_class_balance(y_train, y_train_smote)

    # 6. Save final arrays
    print("\n[4f] Saving final arrays...")
    save_final_arrays(X_train_smote, y_train_smote, X_val, y_val,
                      X_test, test_ids, feature_cols)

    print()
    print("=" * 60)
    print("SPLITTING & BALANCING COMPLETE")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # VIVA PREP — Step 4
    # ──────────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  VIVA PREP — Questions Your Professor Might Ask          │")
    print("└" + "─" * 58 + "┘")
    print("""
    Q1: Why did you apply SMOTE only to the TRAINING set and not
        to the validation set?
    A1: The validation set is supposed to simulate REAL-WORLD data,
        which is naturally imbalanced. If we SMOTE the validation
        set, our validation metrics would be artificially inflated
        and NOT representative of real performance. This is a form
        of TARGET LEAKAGE.

    Q2: How does SMOTE differ from simple random oversampling
        (duplicating minority samples)?
    A2: Random oversampling just COPIES existing minority samples,
        which means the model sees the same examples repeatedly
        and may OVERFIT to them. SMOTE creates NEW SYNTHETIC
        samples by interpolating between existing minority neighbors,
        giving the model more diverse examples to learn from.

    Q3: After SMOTE, all classes have equal counts. Could this
        ever be harmful?
    A3: Yes — if the minority class has very few UNIQUE samples,
        SMOTE might generate synthetic samples that don't represent
        real-world data well (they're interpolations, not real
        measurements). In extreme cases, this can introduce noise.
        An alternative is using class_weight='balanced' in the
        model, which adjusts loss function weights instead of
        modifying the data.
    """)
