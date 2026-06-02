"""
========================================================================
STEP 5 — Model Selection & Base Model Implementations
========================================================================
PURPOSE:
    Train and compare FOUR classification models with DEFAULT parameters
    to establish baselines before hyperparameter tuning (Step 6):
    
    1. Random Forest (RF)
    2. Gradient Boosting Machine (GBM)
    3. XGBoost
    4. LightGBM

WHY THESE FOUR MODELS?
    All are tree-based ensemble methods — the state-of-the-art for
    structured/tabular data (as opposed to deep learning, which
    excels on images/text). They work differently:

    RANDOM FOREST:
        - Builds many independent trees (parallel, "bagging")
        - Each tree sees a random subset of data AND features
        - Final prediction = majority vote of all trees
        - Strength: Robust, hard to overfit, no feature scaling needed
        - Weakness: Slower inference, less expressive than boosting

    GRADIENT BOOSTING (GBM):
        - Builds trees sequentially (serial, "boosting")
        - Each new tree corrects the ERRORS of previous trees
        - Final prediction = sum of all tree outputs
        - Strength: Very accurate, can capture complex patterns
        - Weakness: Slower to train, prone to overfitting

    XGBOOST (eXtreme Gradient Boosting):
        - Optimized GBM with regularization (L1/L2 on leaf weights)
        - Uses second-order gradients for faster convergence
        - Handles missing values natively
        - Strength: Fast, accurate, built-in regularization
        - Weakness: Many hyperparameters to tune

    LIGHTGBM:
        - Leaf-wise tree growth (vs level-wise in XGBoost)
        - Uses histogram-based splits for speed
        - Strength: Fastest training, great for large datasets
        - Weakness: Can overfit on small datasets, leaf-wise growth
          may produce deeper trees

WHY DEFAULT PARAMETERS FIRST?
    Tuning hyperparameters is expensive. We first establish a BASELINE
    with defaults to know which model(s) are worth tuning. If RF gets
    90% and GBM gets 70%, there's no point tuning GBM.
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import time
import warnings

from config import (
    CLEANED_DIR, MODELS_DIR, FIGURES_DIR, OUTPUT_DIR,
    RANDOM_SEED, CLASS_LABELS
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_arrays():
    """Load the preprocessed arrays from Step 4."""
    X_train = np.load(f"{CLEANED_DIR}/X_train_smote.npy")
    y_train = np.load(f"{CLEANED_DIR}/y_train_smote.npy")
    X_val   = np.load(f"{CLEANED_DIR}/X_val.npy")
    y_val   = np.load(f"{CLEANED_DIR}/y_val.npy")

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    return X_train, y_train, X_val, y_val


def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val):
    """
    Train a model, predict on validation set, and compute metrics.

    METRICS EXPLAINED:
        - Accuracy: % of correct predictions (overall)
        - Precision: Of all predicted class X, how many are ACTUALLY X?
          (High precision = few false positives)
        - Recall: Of all ACTUAL class X, how many did we PREDICT as X?
          (High recall = few false negatives)
        - F1-Score: Harmonic mean of precision and recall
          F1 = 2 × (precision × recall) / (precision + recall)
          Preferred over accuracy for imbalanced datasets.

    WHY macro-average for F1?
        'macro' averages the F1 score of each class EQUALLY,
        regardless of class size. This gives BOMBAY (309 samples)
        the same weight as DERMASON (1997 samples), making it a
        fairer metric for imbalanced datasets.
    """
    print(f"\n  {'='*50}")
    print(f"  Training: {model_name}")
    print(f"  {'='*50}")

    # Time the training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Compute metrics
    acc  = accuracy_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred, average="macro")
    prec = precision_score(y_val, y_pred, average="macro")
    rec  = recall_score(y_val, y_pred, average="macro")

    print(f"  Training time: {train_time:.2f}s")
    print(f"  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")

    # Full classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_val, y_pred,
                                target_names=CLASS_LABELS))

    # Save the model
    model_path = f"{MODELS_DIR}/{model_name.lower().replace(' ', '_')}_base.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ Saved model: {model_path}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "F1 (macro)": f1,
        "Precision": prec,
        "Recall": rec,
        "Train Time (s)": round(train_time, 2),
        "y_pred": y_pred,
        "model_obj": model
    }


def plot_confusion_matrices(results, y_val):
    """
    Plot confusion matrices for all models side by side.

    HOW TO READ A CONFUSION MATRIX:
        - Rows = ACTUAL class labels
        - Columns = PREDICTED class labels
        - Diagonal (top-left to bottom-right) = correct predictions
        - Off-diagonal = misclassifications
        - A perfect model has all values on the diagonal

    'normalize="true"' shows proportions instead of counts,
    making it easier to compare across classes of different sizes.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, result in enumerate(results):
        cm = confusion_matrix(y_val, result["y_pred"], normalize="true")

        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                    ax=axes[idx], cbar=True, square=True)
        axes[idx].set_title(f"{result['Model']}\nAcc: {result['Accuracy']:.4f}",
                           fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].tick_params(axis="y", rotation=0)

    fig.suptitle("Confusion Matrices — Base Models (Normalized)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrices_base.png",
                bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/confusion_matrices_base.png")


def plot_model_comparison(results):
    """
    Create a grouped bar chart comparing all models across metrics.
    """
    metrics = ["Accuracy", "F1 (macro)", "Precision", "Recall"]
    model_names = [r["Model"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.18  # Width of each bar
    colors = sns.color_palette("husl", n_colors=len(results))

    for idx, result in enumerate(results):
        values = [result[m] for m in metrics]
        offset = (idx - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=result["Model"],
                      color=colors[idx], edgecolor="black", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Base Model Comparison (Default Hyperparameters)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/model_comparison_base.png",
                bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/model_comparison_base.png")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  STEP 5: Model Selection & Base Model Training          ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. Load data
    print("[5a] Loading preprocessed arrays...")
    X_train, y_train, X_val, y_val = load_arrays()

    # 2. Define models with DEFAULT parameters
    # Each model is initialized with only random_state for reproducibility
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,       # 100 trees (sklearn default)
            random_state=RANDOM_SEED,
            n_jobs=-1               # Use all CPU cores
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,       # 100 boosting rounds
            random_state=RANDOM_SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="mlogloss",  # Multi-class log loss
            verbosity=0              # Suppress warnings
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            verbose=-1               # Suppress warnings
        )
    }

    # 3. Train and evaluate each model
    print("\n[5b] Training base models...")
    results = []
    for model_name, model in models.items():
        result = train_and_evaluate(
            model, model_name, X_train, y_train, X_val, y_val
        )
        results.append(result)

    # 4. Summary comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame([{
        k: v for k, v in r.items()
        if k not in ["y_pred", "model_obj"]
    } for r in results])
    summary_df = summary_df.sort_values("F1 (macro)", ascending=False)
    print(summary_df.to_string(index=False))

    # Save summary to CSV
    summary_df.to_csv(f"{OUTPUT_DIR}/base_model_comparison.csv", index=False)
    print(f"\n  ✓ Saved: output/base_model_comparison.csv")

    # 5. Visualizations
    print("\n[5c] Generating comparison visualizations...")
    plot_confusion_matrices(results, y_val)
    plot_model_comparison(results)

    # Identify best model
    best = max(results, key=lambda x: x["F1 (macro)"])
    print(f"\n  🏆 Best Base Model: {best['Model']}")
    print(f"     F1 (macro): {best['F1 (macro)']:.4f}")
    print(f"     Accuracy:   {best['Accuracy']:.4f}")

    print()
    print("=" * 60)
    print("BASE MODEL TRAINING COMPLETE")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # VIVA PREP — Step 5
    # ──────────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  VIVA PREP — Questions Your Professor Might Ask          │")
    print("└" + "─" * 58 + "┘")
    print("""
    Q1: What is the fundamental difference between Random Forest
        (bagging) and Gradient Boosting?
    A1: Random Forest builds trees INDEPENDENTLY in parallel —
        each tree sees a random subset of data and features,
        and the final prediction is a majority vote. This reduces
        VARIANCE (overfitting).
        
        Gradient Boosting builds trees SEQUENTIALLY — each new
        tree specifically corrects the errors of the previous
        trees. This reduces BIAS (underfitting).

    Q2: Why did you use F1 macro-average instead of accuracy
        as the primary comparison metric?
    A2: Accuracy can be misleading with imbalanced classes.
        A model that ALWAYS predicts DERMASON would get ~26%
        accuracy — that sounds bad, but for a less imbalanced
        dataset it could look good while failing on minorities.
        F1 macro treats all 7 classes EQUALLY, so poor performance
        on BOMBAY hurts the score just as much as poor performance
        on DERMASON.

    Q3: What does n_estimators=100 mean, and how would increasing
        it affect the model?
    A3: n_estimators is the number of trees in the ensemble.
        - For RF: More trees = more stable predictions (diminishing
          returns after ~100-200). No overfitting risk.
        - For GBM/XGBoost: More trees = more boosting rounds.
          Too many can OVERFIT because each tree adds complexity.
          This is controlled by learning_rate and early stopping.
    """)
