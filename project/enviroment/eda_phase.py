"""
========================================================================
STEP 1 — Exploratory Data Analysis (EDA) & Initial Visualizations
========================================================================
PURPOSE:
    Before building any model, we MUST understand our data. EDA reveals:
    - The shape, types, and statistical summary of each feature
    - Missing or invalid values that need imputation (Step 2)
    - Class distribution imbalance that affects model training (Step 4)
    - Feature correlations that guide feature engineering (Step 3)
    - Outliers that may distort model learning

WHY EDA FIRST?
    Garbage in = Garbage out. If we skip EDA and jump straight to
    modeling, we risk training on corrupted data (e.g., negative Area
    values) or missing crucial patterns (e.g., BOMBAY beans being
    physically much larger than all other classes).

OUTPUTS:
    - Console: .info(), .describe(), missing value counts
    - Figures saved to figures/ directory:
        1. class_distribution.png     — bar chart of target variable
        2. correlation_heatmap.png    — feature-to-feature correlations
        3. feature_boxplots.png       — outlier detection per class
        4. feature_distributions.png  — histograms + KDE per feature
        5. pairplot_top_features.png  — scatter matrix of top features
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import centralized configuration
from config import (
    TRAIN_FILE, VAL_FILE, TEST_FILE,
    FEATURE_COLS_RAW, TARGET_COL, ID_COL,
    FIGURES_DIR, CLASS_LABELS
)

# ──────────────────────────────────────────────────────────────────────
# SETUP: Suppress warnings and configure plot aesthetics
# ──────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# seaborn's "whitegrid" style adds subtle gridlines that make it
# easier to read values off plots during presentations
sns.set_style("whitegrid")
sns.set_palette("husl")  # High contrast color palette for 7 classes
plt.rcParams["figure.dpi"] = 120  # Higher resolution figures


def load_datasets():
    """
    Load the three CSV files into pandas DataFrames.

    Returns:
        tuple: (train_df, val_df, test_df) — three DataFrames
    """
    # pd.read_csv() reads comma-separated values into a DataFrame
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VAL_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    print("=" * 60)
    print("DATASET SHAPES")
    print("=" * 60)
    print(f"  Train:      {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    print(f"  Validation: {val_df.shape[0]:,} rows × {val_df.shape[1]} columns")
    print(f"  Test:       {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
    print()

    return train_df, val_df, test_df


def basic_info(df, name="DataFrame"):
    """
    Print basic information about a DataFrame: dtypes, non-null counts,
    and descriptive statistics.

    WHY .info() AND .describe()?
        - .info() shows column data types and non-null counts,
          revealing if any columns have missing values.
        - .describe() shows mean, std, min, max, and quartiles,
          helping us spot unusual values (e.g., negative Areas).
    """
    print("=" * 60)
    print(f"INFO: {name}")
    print("=" * 60)
    print(df.info())
    print()

    print(f"DESCRIPTIVE STATISTICS: {name}")
    print("-" * 60)
    # .describe() computes count, mean, std, min, 25%, 50%, 75%, max
    # .T transposes so each feature is a row (easier to read)
    print(df.describe().T.to_string())
    print()


def check_data_quality(train_df, val_df, test_df):
    """
    Inspect data quality issues: missing values, duplicates,
    negative values, and zero values in critical columns.

    WHY CHECK FOR NEGATIVES?
        Physical measurements like Area cannot be negative.
        Negative values indicate data corruption that must be
        fixed in preprocessing (Step 2).
    """
    print("=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)

    # --- Missing Values ---
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        missing = df.isnull().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            print(f"\n⚠ {name} — Missing values found:")
            # Filter to only show columns with missing values
            print(missing[missing > 0].to_string())
        else:
            print(f"  ✓ {name} — No missing values")

    # --- Duplicates ---
    print()
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        # Check duplicates excluding the 'id' column
        dupes = df.drop(columns=[ID_COL]).duplicated().sum()
        print(f"  {'⚠' if dupes > 0 else '✓'} {name} — {dupes} duplicate rows")

    # --- Negative Values in Area ---
    print()
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        neg_area = (df["Area"] < 0).sum()
        print(f"  {'⚠' if neg_area > 0 else '✓'} {name} — {neg_area} negative Area values")

    # --- Zero MajorAxisLength ---
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        zero_major = (df["MajorAxisLength"] == 0).sum()
        print(f"  {'⚠' if zero_major > 0 else '✓'} {name} — {zero_major} zero MajorAxisLength values")

    print()


def plot_class_distribution(train_df):
    """
    Visualize the distribution of the target variable (Class).

    WHY THIS MATTERS:
        If classes are severely imbalanced, the model may learn to
        always predict the majority class (DERMASON) and still achieve
        high accuracy — but fail on minority classes (BOMBAY).
        This motivates using SMOTE or class_weight in Step 4.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Bar Chart ---
    # value_counts() returns class frequencies sorted descending
    class_counts = train_df[TARGET_COL].value_counts()
    colors = sns.color_palette("husl", n_colors=len(class_counts))

    ax1 = axes[0]
    bars = ax1.bar(class_counts.index, class_counts.values, color=colors,
                   edgecolor="black", linewidth=0.5)
    ax1.set_title("Class Distribution (Training Set)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Bean Class", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)

    # Add count labels on top of each bar
    for bar, count in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 str(count), ha="center", va="bottom", fontweight="bold", fontsize=10)

    # --- Pie Chart ---
    ax2 = axes[1]
    ax2.pie(class_counts.values, labels=class_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=140,
            textprops={"fontsize": 10})
    ax2.set_title("Class Proportions", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/class_distribution.png", bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/class_distribution.png")


def plot_correlation_heatmap(train_df):
    """
    Plot a correlation matrix heatmap for all numeric features.

    WHY CORRELATIONS MATTER:
        - Highly correlated features (|r| > 0.9) are redundant; keeping
          both adds noise without information gain.
        - Moderate correlations reveal which features jointly predict
          the target, guiding feature engineering in Step 3.

    We use Pearson correlation (default), which measures LINEAR
    relationships between continuous variables.
    """
    # Select only numeric feature columns
    numeric_df = train_df[FEATURE_COLS_RAW]

    # .corr() computes pairwise Pearson correlation coefficients
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))

    # annot=True prints correlation values inside each cell
    # fmt=".2f" rounds to 2 decimal places
    # cmap="RdBu_r" uses a diverging colormap: red=positive, blue=negative
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)

    ax.set_title("Feature Correlation Heatmap (Training Set)",
                 fontsize=16, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/correlation_heatmap.png")

    # Print highly correlated pairs (|r| > 0.9)
    print("\n  Highly Correlated Feature Pairs (|r| > 0.9):")
    print("  " + "-" * 50)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.9:
                print(f"    {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: r = {r:.3f}")
    print()


def plot_feature_boxplots(train_df):
    """
    Create box plots for each feature, grouped by Class.

    WHY BOX PLOTS?
        Box plots reveal:
        - Median (center line) — typical value per class
        - IQR (box) — spread of the middle 50% of data
        - Whiskers — extent of non-outlier data
        - Dots beyond whiskers — potential outliers

        If a feature's box plots are well-separated across classes,
        that feature has strong discriminative power for classification.
    """
    features = FEATURE_COLS_RAW
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        sns.boxplot(data=train_df, x=TARGET_COL, y=feature, ax=axes[idx],
                    palette="husl", fliersize=2)
        axes[idx].set_title(feature, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].tick_params(axis="x", rotation=45)

    # Hide any unused subplot axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Box Plots by Bean Class", fontsize=18,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_boxplots.png", bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/feature_boxplots.png")


def plot_feature_distributions(train_df):
    """
    Plot histograms with KDE (Kernel Density Estimation) overlay
    for each feature, colored by Class.

    WHY KDE?
        A histogram alone can be misleading depending on bin size.
        KDE smooths the histogram into a continuous probability
        density curve, giving a cleaner picture of the distribution
        shape (normal, skewed, bimodal, etc.).

        If a feature's KDE curves for different classes overlap
        heavily, that feature is NOT useful for separating classes.
    """
    features = FEATURE_COLS_RAW
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        for cls in CLASS_LABELS:
            subset = train_df[train_df[TARGET_COL] == cls][feature]
            # Plot KDE for each class separately
            subset.plot.kde(ax=axes[idx], label=cls, alpha=0.7)

        axes[idx].set_title(feature, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel(feature)
        axes[idx].legend(fontsize=7, loc="upper right")

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Distributions by Bean Class (KDE)",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_distributions.png", bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/feature_distributions.png")


def plot_pairplot(train_df):
    """
    Create a scatter-matrix (pair plot) of the top 5 features
    that have the highest variance across class means.

    WHY NOT ALL 16?
        A 16×16 pair plot would have 256 subplots — unreadable.
        We select the top 5 features where the between-class
        variance is highest, meaning those features best separate
        the classes visually.
    """
    # Compute the variance of class means for each feature
    # Higher variance = more separation between classes
    class_means = train_df.groupby(TARGET_COL)[FEATURE_COLS_RAW].mean()
    feature_importance = class_means.var().sort_values(ascending=False)

    top_5 = feature_importance.head(5).index.tolist()
    print(f"\n  Top 5 features by between-class variance: {top_5}")

    # Create pair plot with only the top 5 features + Class
    plot_df = train_df[top_5 + [TARGET_COL]]

    # sns.pairplot creates a matrix of scatter plots (off-diagonal)
    # and histograms (diagonal), colored by class
    g = sns.pairplot(plot_df, hue=TARGET_COL, palette="husl",
                     diag_kind="kde", plot_kws={"alpha": 0.5, "s": 15},
                     height=2.5)
    g.figure.suptitle("Pair Plot — Top 5 Discriminative Features",
                     fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/pairplot_top_features.png", bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved: figures/pairplot_top_features.png")


# ======================================================================
# MAIN EXECUTION
# ======================================================================
if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  STEP 1: Exploratory Data Analysis (EDA)                 ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. Load datasets
    train_df, val_df, test_df = load_datasets()

    # 2. Basic info and statistics
    basic_info(train_df, "Training Set")
    basic_info(val_df, "Validation Set")
    basic_info(test_df, "Test Set (No Labels)")

    # 3. Data quality checks
    check_data_quality(train_df, val_df, test_df)

    # 4. Visualizations
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    print()

    plot_class_distribution(train_df)
    plot_correlation_heatmap(train_df)
    plot_feature_boxplots(train_df)
    plot_feature_distributions(train_df)
    plot_pairplot(train_df)

    print()
    print("=" * 60)
    print("EDA COMPLETE — All figures saved to figures/ directory")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # VIVA PREP — Step 1
    # ──────────────────────────────────────────────────────────────
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  VIVA PREP — Questions Your Professor Might Ask          │")
    print("└" + "─" * 58 + "┘")
    print("""
    Q1: Why did you check for negative values in the Area column
        instead of just trusting the data?
    A1: Area is a physical measurement — it CANNOT be negative.
        Negative values indicate data corruption (likely a sign
        error during collection). Blindly training on corrupted
        data would teach the model incorrect patterns.

    Q2: What does a high Pearson correlation (e.g., r = 0.95)
        between two features tell you? Should you drop one?
    A2: It means the two features carry nearly identical information
        (multicollinearity). Keeping both adds redundancy without
        information gain, and can inflate variance in linear models.
        Tree-based models are more robust, but it's still good
        practice to consider dropping one or combining them.

    Q3: The BOMBAY class has only 309 samples vs DERMASON's 1,997.
        Why is this a problem for classification?
    A3: The model may learn to always predict DERMASON (majority class)
        to maximize overall accuracy, while performing poorly on
        BOMBAY (minority class). This is called the class imbalance
        problem, and we address it in Step 4 using SMOTE or
        class_weight='balanced'.
    """)
