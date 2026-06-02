"""
========================================================================
STEP 7 — Submission Extraction & Final Report Support
========================================================================
PURPOSE:
    1. Load the best tuned model from Step 6
    2. Predict class labels for the UNLABELED test set
    3. Decode integer predictions back to class names
    4. Generate the submission CSV file (id, Class)

SUBMISSION FORMAT:
    The competition expects a CSV with exactly 2 columns:
        id     — the unique identifier for each test sample
        Class  — the predicted bean class name (e.g., "DERMASON")

    The file must have exactly 3,403 rows (one per test sample)
    plus the header row.
========================================================================
"""

import numpy as np
import pandas as pd
import joblib
import warnings

from config import (
    CLEANED_DIR, MODELS_DIR, OUTPUT_DIR, CLASS_LABELS
)

warnings.filterwarnings("ignore")


def load_best_model():
    """
    Load the best tuned model identified in Step 6.

    We saved the best model's name to a text file in Step 6,
    and the actual model as a .pkl (pickle) file.
    """
    # Read which model was best
    with open(f"{MODELS_DIR}/best_model_name.txt", "r") as f:
        best_name = f.read().strip()

    model_filename = best_name.lower().replace(" ", "_")
    model_path = f"{MODELS_DIR}/{model_filename}_tuned.pkl"

    # joblib.load() deserializes the trained model from disk
    model = joblib.load(model_path)

    print(f"  Best model: {best_name}")
    print(f"  Loaded from: {model_path}")

    return model, best_name


def load_test_data():
    """Load the preprocessed test features and IDs."""
    X_test   = np.load(f"{CLEANED_DIR}/X_test.npy")
    test_ids = np.load(f"{CLEANED_DIR}/test_ids.npy")

    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features:     {X_test.shape[1]}")

    return X_test, test_ids


def generate_predictions(model, X_test):
    """
    Use the trained model to predict class labels for test data.

    .predict() returns integer-encoded labels (e.g., 0, 1, 2, ...).
    We need to decode these back to class names for submission.
    """
    # .predict() returns the most likely class for each sample
    y_pred = model.predict(X_test)

    print(f"\n  Prediction distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        print(f"    Class {cls_idx} ({CLASS_LABELS[cls_idx]}): {count}")

    return y_pred


def decode_predictions(y_pred):
    """
    Convert integer predictions back to class names using the
    LabelEncoder saved in Step 3.

    WHY USE THE SAVED ENCODER?
        We must use the EXACT SAME mapping that was used during
        training. If we manually create a new mapping, there's a
        risk of misalignment (e.g., class 0 = BARBUNYA in training
        but accidentally mapped to BOMBAY here).
    """
    le = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")

    # .inverse_transform() converts integers back to original labels
    class_names = le.inverse_transform(y_pred.astype(int))

    print(f"\n  Decoded class distribution:")
    unique, counts = np.unique(class_names, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"    {cls}: {count}")

    return class_names


def create_submission(test_ids, class_names):
    """
    Create the submission CSV file in the required format.

    FORMAT:
        id,Class
        0,DERMASON
        1,SIRA
        2,CALI
        ...
    """
    submission_df = pd.DataFrame({
        "id": test_ids.astype(int),
        "Class": class_names
    })

    # Sort by id to ensure consistent ordering
    submission_df = submission_df.sort_values("id").reset_index(drop=True)

    # Save to CSV
    submission_path = f"{OUTPUT_DIR}/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"\n  ✓ Submission saved: {submission_path}")
    print(f"    Rows: {len(submission_df)}")
    print(f"    Columns: {list(submission_df.columns)}")
    print(f"\n  Preview (first 10 rows):")
    print(submission_df.head(10).to_string(index=False))

    return submission_df


def validate_submission(submission_df):
    """
    Sanity checks on the submission file to catch obvious errors.
    """
    print(f"\n  Validation checks:")

    # Check 1: Correct number of rows
    expected_rows = 3403
    actual_rows = len(submission_df)
    status = "✓" if actual_rows == expected_rows else "✗"
    print(f"    {status} Row count: {actual_rows} (expected {expected_rows})")

    # Check 2: Correct columns
    expected_cols = ["id", "Class"]
    actual_cols = list(submission_df.columns)
    status = "✓" if actual_cols == expected_cols else "✗"
    print(f"    {status} Columns: {actual_cols}")

    # Check 3: All 7 classes present
    unique_classes = sorted(submission_df["Class"].unique())
    status = "✓" if len(unique_classes) == 7 else "✗"
    print(f"    {status} Unique classes: {len(unique_classes)}")
    for cls in unique_classes:
        print(f"      - {cls}")

    # Check 4: No missing values
    missing = submission_df.isnull().sum().sum()
    status = "✓" if missing == 0 else "✗"
    print(f"    {status} Missing values: {missing}")

    # Check 5: IDs are unique
    dupes = submission_df["id"].duplicated().sum()
    status = "✓" if dupes == 0 else "✗"
    print(f"    {status} Duplicate IDs: {dupes}")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  STEP 7: Submission Extraction                          ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. Load best model
    print("[7a] Loading best tuned model...")
    model, model_name = load_best_model()

    # 2. Load test data
    print("\n[7b] Loading test data...")
    X_test, test_ids = load_test_data()

    # 3. Generate predictions
    print("\n[7c] Generating predictions...")
    y_pred = generate_predictions(model, X_test)

    # 4. Decode to class names
    print("\n[7d] Decoding predictions to class names...")
    class_names = decode_predictions(y_pred)

    # 5. Create submission CSV
    print("\n[7e] Creating submission file...")
    submission_df = create_submission(test_ids, class_names)

    # 6. Validate
    print("\n[7f] Validating submission...")
    validate_submission(submission_df)

    print()
    print("=" * 60)
    print(f"SUBMISSION COMPLETE — Best Model: {model_name}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # VIVA PREP — Step 7
    # ──────────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  VIVA PREP — Questions Your Professor Might Ask          │")
    print("└" + "─" * 58 + "┘")
    print("""
    Q1: Why did you use inverse_transform() from the saved
        LabelEncoder instead of manually mapping integers to names?
    A1: Using the SAVED encoder guarantees the EXACT SAME mapping
        used during training. Manual mapping risks misalignment
        (e.g., accidentally mapping class 0 to BOMBAY instead of
        BARBUNYA), which would silently corrupt ALL predictions.

    Q2: You scaled the test data using the TRAINING scaler. Why
        not fit a new scaler on the test data?
    A2: The model was trained on data scaled with training
        statistics. If we scale test data with its OWN statistics,
        the features would be in a DIFFERENT scale than what the
        model learned. This would make all predictions unreliable.
        Consistency is critical: same scaler, same encoding.

    Q3: Walk me through the complete pipeline from raw test CSV
        to the final submission CSV.
    A3: 1) Load raw test_no_label.csv
        2) Fix negatives in Area (absolute value)
        3) Impute zero MajorAxisLength (using TRAINING medians)
        4) Rename AspectRation → AspectRatio
        5) Cap outliers (using TRAINING IQR bounds)
        6) Create 6 derived features
        7) Scale with StandardScaler (TRAINING mean/std)
        8) Feed into best tuned model → get integer predictions
        9) Decode integers → class names via saved LabelEncoder
        10) Save as submission.csv with columns [id, Class]
    """)
