"""
========================================================================
STEP 6 — Hyperparameter Tuning & "Before vs. After" Evaluation
========================================================================
PURPOSE:
    1. Tune the top-2 best models from Step 5 using RandomizedSearchCV
    2. Compare TUNED performance against BASELINE (default) performance
    3. Compare performance on RAW data vs PREPROCESSED data
    4. Generate a comprehensive "Before vs. After" evaluation report

WHY RANDOMIZED SEARCH (not Grid Search)?
    Grid Search: Tests EVERY combination of hyperparameters.
        - For 5 params with 5 values each: 5^5 = 3,125 combinations
        - With 5-fold CV: 15,625 model fits
        - Computationally VERY expensive

    Randomized Search: Tests a RANDOM SAMPLE of combinations.
        - We specify n_iter=50 (50 random combinations)
        - With 5-fold CV: 250 model fits
        - Much faster, and research shows it finds near-optimal
          solutions because most hyperparameters DON'T interact
          strongly (Bergstra & Bengio, 2012).

HYPERPARAMETERS EXPLAINED (for each model):

    Random Forest:
        - n_estimators: Number of trees (more = better, diminishing returns)
        - max_depth: Maximum tree depth (deeper = more complex, risk overfit)
        - min_samples_split: Min samples to split a node (higher = simpler)
        - min_samples_leaf: Min samples in a leaf (higher = simpler)
        - max_features: Features considered per split (lower = more diversity)

    XGBoost:
        - n_estimators: Number of boosting rounds
        - max_depth: Tree depth (typically 3-10 for boosting)
        - learning_rate: Step size shrinkage (smaller = more rounds needed)
        - subsample: Fraction of data per tree (prevents overfitting)
        - colsample_bytree: Fraction of features per tree
        - reg_alpha: L1 regularization (encourages sparsity)
        - reg_lambda: L2 regularization (penalizes large weights)

    LightGBM:
        - n_estimators: Number of boosting rounds
        - max_depth: Tree depth (-1 = unlimited, controlled by num_leaves)
        - num_leaves: Maximum number of leaves per tree
        - learning_rate: Step size shrinkage
        - subsample: Fraction of data per iteration
        - colsample_bytree: Fraction of features per tree
        - reg_alpha: L1 regularization
        - reg_lambda: L2 regularization
========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, make_scorer
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
import joblib
import time
import warnings

from config import (
    TRAIN_FILE, VAL_FILE,
    FEATURE_COLS_RAW, FEATURE_COLS, TARGET_COL, ID_COL,
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
    return X_train, y_train, X_val, y_val


def get_param_distributions():
    """
    Define hyperparameter search spaces for each model.

    WHY THESE RANGES?
        Each range is chosen based on common best practices:
        - n_estimators: 100-500 for a good accuracy/speed tradeoff
        - max_depth: 5-30 for RF (trees can be deep), 3-10 for boosting
        - learning_rate: 0.01-0.3 for boosting (smaller = more robust)
        - subsample: 0.6-1.0 (too low = underfitting)
    """
    param_dists = {
        "Random Forest": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(5, 30),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        },
        "XGBoost": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.29),  # 0.01 to 0.30
            "subsample": uniform(0.6, 0.4),         # 0.6 to 1.0
            "colsample_bytree": uniform(0.6, 0.4),  # 0.6 to 1.0
            "reg_alpha": uniform(0, 1),              # L1 regularization
            "reg_lambda": uniform(0.5, 1.5),         # L2 regularization
        },
        "LightGBM": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 15),
            "num_leaves": randint(20, 80),
            "learning_rate": uniform(0.01, 0.29),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0.5, 1.5),
        },
        "Gradient Boosting": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.29),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "subsample": uniform(0.6, 0.4),
            "max_features": ["sqrt", "log2", None],
        }
    }
    return param_dists


def get_base_models():
    """Get model instances for tuning."""
    return {
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED
        )
    }


def tune_model(model, param_dist, model_name, X_train, y_train, n_iter=50):
    """
    Tune a model using RandomizedSearchCV with stratified k-fold.

    WHY STRATIFIED K-FOLD?
        Regular K-fold might create folds where a minority class has
        0 samples — the model can't learn it. Stratified K-fold
        ensures each fold preserves the class distribution.

    SCORING = 'f1_macro':
        We optimize for macro F1 (not accuracy) to ensure the model
        performs well on ALL classes, not just the majority.
    """
    print(f"\n  {'='*50}")
    print(f"  Tuning: {model_name}")
    print(f"  {'='*50}")

    # StratifiedKFold with 5 folds: each fold maintains class ratios
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Create scorer: macro F1 treats all classes equally
    scorer = make_scorer(f1_score, average="macro")

    # RandomizedSearchCV: randomly samples n_iter hyperparameter combinations
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,            # Parallel CV folds
        verbose=1,
        return_train_score=True
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    tune_time = time.time() - start_time

    print(f"\n  Tuning time: {tune_time:.1f}s")
    print(f"  Best CV F1 (macro): {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for param, value in search.best_params_.items():
        if isinstance(value, float):
            print(f"    {param}: {value:.4f}")
        else:
            print(f"    {param}: {value}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_on_validation(model, model_name, X_val, y_val):
    """Evaluate a model on the validation set and return metrics."""
    y_pred = model.predict(X_val)

    acc  = accuracy_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred, average="macro")
    prec = precision_score(y_val, y_pred, average="macro")
    rec  = recall_score(y_val, y_pred, average="macro")

    print(f"\n  {model_name} — Validation Metrics:")
    print(f"    Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    F1 (macro): {f1:.4f}")
    print(f"    Precision:  {prec:.4f}")
    print(f"    Recall:     {rec:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_val, y_pred,
                                target_names=CLASS_LABELS))

    return {"Model": model_name, "Accuracy": acc, "F1 (macro)": f1,
            "Precision": prec, "Recall": rec, "y_pred": y_pred}


def before_after_comparison(base_results, tuned_results):
    """
    Create a comprehensive Before vs. After comparison table.

    This is the KEY deliverable for Step 6: showing that
    hyperparameter tuning ACTUALLY improves model performance.
    """
    print("\n" + "=" * 70)
    print("BEFORE vs. AFTER HYPERPARAMETER TUNING")
    print("=" * 70)

    comparison_rows = []
    for base in base_results:
        model_name = base["Model"]
        tuned = next((t for t in tuned_results if t["Model"] == model_name), None)

        if tuned:
            comparison_rows.append({
                "Model": model_name,
                "Base Accuracy": f"{base['Accuracy']:.4f}",
                "Tuned Accuracy": f"{tuned['Accuracy']:.4f}",
                "Acc Δ": f"{(tuned['Accuracy'] - base['Accuracy'])*100:+.2f}%",
                "Base F1": f"{base['F1 (macro)']:.4f}",
                "Tuned F1": f"{tuned['F1 (macro)']:.4f}",
                "F1 Δ": f"{(tuned['F1 (macro)'] - base['F1 (macro)'])*100:+.2f}%",
            })

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(f"{OUTPUT_DIR}/before_after_tuning.csv", index=False)
    print(f"\n  ✓ Saved: output/before_after_tuning.csv")

    return comp_df


def plot_before_after(base_results, tuned_results):
    """
    Grouped bar chart: Before vs After tuning for each model.
    """
    models_to_plot = [r["Model"] for r in tuned_results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(models_to_plot))
    width = 0.35

    # --- Accuracy comparison ---
    base_acc = [next(r["Accuracy"] for r in base_results if r["Model"] == m)
                for m in models_to_plot]
    tuned_acc = [next(r["Accuracy"] for r in tuned_results if r["Model"] == m)
                 for m in models_to_plot]

    bars1 = axes[0].bar(x - width/2, base_acc, width, label="Base (Default)",
                        color="lightcoral", edgecolor="black", linewidth=0.5)
    bars2 = axes[0].bar(x + width/2, tuned_acc, width, label="Tuned",
                        color="mediumseagreen", edgecolor="black", linewidth=0.5)

    axes[0].set_title("Accuracy: Before vs After Tuning",
                      fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_to_plot, rotation=15)
    axes[0].legend()
    axes[0].set_ylim(0.8, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{bar.get_height():.3f}", ha="center", fontsize=9)

    # --- F1 comparison ---
    base_f1 = [next(r["F1 (macro)"] for r in base_results if r["Model"] == m)
               for m in models_to_plot]
    tuned_f1 = [next(r["F1 (macro)"] for r in tuned_results if r["Model"] == m)
                for m in models_to_plot]

    bars3 = axes[1].bar(x - width/2, base_f1, width, label="Base (Default)",
                        color="lightskyblue", edgecolor="black", linewidth=0.5)
    bars4 = axes[1].bar(x + width/2, tuned_f1, width, label="Tuned",
                        color="mediumpurple", edgecolor="black", linewidth=0.5)

    axes[1].set_title("F1 (Macro): Before vs After Tuning",
                      fontsize=13, fontweight="bold")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_to_plot, rotation=15)
    axes[1].legend()
    axes[1].set_ylim(0.8, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    for bar in bars3:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars4:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{bar.get_height():.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/before_after_tuning.png", bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/before_after_tuning.png")


def plot_tuned_confusion_matrices(tuned_results, y_val):
    """Plot confusion matrices for tuned models."""
    n = len(tuned_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for idx, result in enumerate(tuned_results):
        cm = confusion_matrix(y_val, result["y_pred"], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                    ax=axes[idx], cbar=True, square=True)
        axes[idx].set_title(
            f"{result['Model']} (Tuned)\nAcc: {result['Accuracy']:.4f} | "
            f"F1: {result['F1 (macro)']:.4f}",
            fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
        axes[idx].tick_params(axis="x", rotation=45)

    fig.suptitle("Confusion Matrices — Tuned Models (Normalized)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrices_tuned.png",
                bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/confusion_matrices_tuned.png")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  STEP 6: Hyperparameter Tuning & Before/After Eval      ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. Load data
    print("[6a] Loading preprocessed arrays...")
    X_train, y_train, X_val, y_val = load_arrays()

    # 2. Load base model results from Step 5
    print("\n[6b] Evaluating base (default) models...")
    base_models_dict = get_base_models()
    base_results = []

    for name, model in base_models_dict.items():
        # Train base model with defaults
        model.fit(X_train, y_train)
        result = evaluate_on_validation(model, f"{name}", X_val, y_val)
        base_results.append(result)

    # 3. Tune all models
    print("\n[6c] Hyperparameter tuning (this may take a few minutes)...")
    param_dists = get_param_distributions()
    tuned_models = {}
    tuned_results = []

    for name in base_models_dict.keys():
        model = get_base_models()[name]
        best_model, best_params, best_cv_score = tune_model(
            model, param_dists[name], name,
            X_train, y_train, n_iter=50
        )
        tuned_models[name] = best_model

        # Evaluate tuned model on validation
        result = evaluate_on_validation(best_model, name, X_val, y_val)
        tuned_results.append(result)

        # Save tuned model
        model_path = f"{MODELS_DIR}/{name.lower().replace(' ', '_')}_tuned.pkl"
        joblib.dump(best_model, model_path)
        print(f"  ✓ Saved tuned model: {model_path}")

    # 4. Before vs After comparison
    print("\n[6d] Before vs After comparison...")
    comp_df = before_after_comparison(base_results, tuned_results)

    # 5. Visualizations
    print("\n[6e] Generating comparison visualizations...")
    plot_before_after(base_results, tuned_results)
    plot_tuned_confusion_matrices(tuned_results, y_val)

    # 6. Identify the overall best model
    best_tuned = max(tuned_results, key=lambda x: x["F1 (macro)"])
    print(f"\n  🏆 BEST TUNED MODEL: {best_tuned['Model']}")
    print(f"     F1 (macro): {best_tuned['F1 (macro)']:.4f}")
    print(f"     Accuracy:   {best_tuned['Accuracy']:.4f}")

    # Save best model name for Step 7
    with open(f"{MODELS_DIR}/best_model_name.txt", "w") as f:
        f.write(best_tuned["Model"])

    print()
    print("=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # VIVA PREP — Step 6
    # ──────────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  VIVA PREP — Questions Your Professor Might Ask          │")
    print("└" + "─" * 58 + "┘")
    print("""
    Q1: Why did you use RandomizedSearchCV instead of GridSearchCV?
    A1: Grid search tests EVERY combination of hyperparameters —
        exponentially expensive (e.g., 5 params × 5 values = 3,125
        combinations). Randomized search samples N random combinations
        (we used 50), which is much faster and empirically finds
        near-optimal solutions because most hyperparameters don't
        strongly interact (Bergstra & Bengio, 2012).

    Q2: You used StratifiedKFold with 5 splits. Why 5? Why stratified?
    A2: 5-fold is a standard tradeoff between bias and variance of
        the performance estimate. 3-fold has higher variance; 10-fold
        is more compute. STRATIFIED ensures each fold preserves the
        class distribution — critical because BOMBAY has few samples
        and a random fold might have 0 BOMBAY samples.

    Q3: What does the learning_rate hyperparameter do in boosting
        models, and why would a smaller value be better?
    A3: learning_rate (also called eta or shrinkage) scales the
        contribution of each tree. A smaller value (e.g., 0.05)
        means each tree contributes less, requiring MORE trees
        (higher n_estimators) but giving a more ROBUST model that
        generalizes better. It's like taking smaller, more careful
        steps toward the optimal solution instead of large jumps
        that might overshoot.
    """)
