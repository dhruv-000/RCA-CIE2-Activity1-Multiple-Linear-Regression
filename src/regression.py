"""Multiple Linear Regression pipeline for California Housing Prices (CIE 2 Activity 1)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


FEATURES = [
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
]
TARGET = "median_house_value"
RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    """Load California Housing dataset (downloading if missing) and clean missing/duplicate rows."""
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(DATA_URL)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    # Drop rows with missing values (~1% in total_bedrooms) and exact duplicates
    df_clean = df.dropna(subset=FEATURES + [TARGET]).drop_duplicates().reset_index(drop=True)
    return df_clean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> dict:
    """Compute MAE, MSE, RMSE, R2, and Adjusted R2 on evaluation predictions."""
    n = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
    }


def manual_ols_sample(df: pd.DataFrame) -> dict:
    """Perform manual OLS calculation on 6 observations and 2 features, verified with scikit-learn."""
    sample = df[["median_income", "housing_median_age", TARGET]].iloc[:6].copy().reset_index(drop=True)
    x1 = sample["median_income"].to_numpy()
    x2 = sample["housing_median_age"].to_numpy()
    y = sample[TARGET].to_numpy()

    X_design = np.column_stack([np.ones(len(sample)), x1, x2])
    Xt = X_design.T
    XtX = Xt @ X_design
    Xty = Xt @ y
    XtX_inv = np.linalg.inv(XtX)
    b_manual = XtX_inv @ Xty

    y_pred_0 = b_manual[0] + b_manual[1] * x1[0] + b_manual[2] * x2[0]
    residual_0 = y[0] - y_pred_0

    sk_model = LinearRegression().fit(sample[["median_income", "housing_median_age"]], y)
    b_sklearn = np.array([sk_model.intercept_, sk_model.coef_[0], sk_model.coef_[1]])

    return {
        "sample": sample,
        "X_design": X_design,
        "XtX": XtX,
        "Xty": Xty,
        "XtX_inv": XtX_inv,
        "b_manual": b_manual,
        "y_pred_0": y_pred_0,
        "residual_0": residual_0,
        "b_sklearn": b_sklearn,
    }


def compute_vif(X_train: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for each feature using training data."""
    X_const = sm.add_constant(X_train)
    vif_rows = []
    for idx, col in enumerate(X_train.columns, start=1):
        vif_val = variance_inflation_factor(X_const.values, idx)
        vif_rows.append({"Feature": col, "VIF": vif_val})
    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def generate_visualizations(
    df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    out_dir: Path,
) -> None:
    """Generate and save all EDA and diagnostic plots to outputs/generated_visualizations/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET], bins=40, kde=True, color="#2563eb", ax=ax)
    ax.set_title("Distribution of Median House Value (Target Variable)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Median House Value (USD)")
    ax.set_ylabel("Frequency (Block Groups)")
    fig.tight_layout()
    fig.savefig(out_dir / "01_target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Scatter plots of each feature vs target
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, feat in zip(axes.flat, FEATURES):
        ax.scatter(df[feat], df[TARGET], alpha=0.25, s=12, color="#0f766e")
        ax.set_title(f"{feat} vs {TARGET}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Median House Value (USD)")
    fig.suptitle("Selected Features vs Median House Value", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "02_feature_scatter_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Correlation heatmap
    corr_matrix = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Features and Target", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "03_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Pair plot on a stratified/random subset for visual clarity
    pair_sample = df[["median_income", "housing_median_age", "total_rooms", TARGET]].sample(
        n=1000, random_state=RANDOM_STATE
    )
    pair_grid = sns.pairplot(pair_sample, diag_kind="kde", plot_kws={"alpha": 0.35, "s": 15})
    pair_grid.fig.suptitle("Pair Plot of Key Features and Target (n=1,000 sample)", y=1.02, fontweight="bold")
    pair_grid.savefig(out_dir / "04_pair_plot.png", dpi=130, bbox_inches="tight")
    plt.close(pair_grid.fig)

    # 5. Actual vs Predicted with ideal 45-degree line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="#1d4ed8", label="Test Predictions")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Ideal 45-degree Line (y = y_hat)")
    ax.set_title("Actual vs Predicted Median House Value (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Median House Value (USD)")
    ax.set_ylabel("Predicted Median House Value (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # 6. Residuals vs Predicted Values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=15, color="#7c3aed")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.8, label="Zero Residual Line")
    ax.set_title("Residuals vs Predicted Values (Test Set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Median House Value (USD)")
    ax.set_ylabel("Residual (Actual - Predicted, USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_residuals_vs_predicted.png", dpi=150)
    plt.close(fig)

    # 7. Residual Distribution Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=45, kde=True, color="#059669", ax=ax)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.set_title("Distribution of Test Set Residuals", fontsize=13, fontweight="bold")
    ax.set_xlabel("Residual (USD)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_residual_histogram.png", dpi=150)
    plt.close(fig)


def main() -> None:
    root = get_project_root()
    data_path = root / "data" / "housing.csv"
    out_dir = root / "outputs" / "generated_visualizations"

    df = load_and_clean_data(data_path)
    print(f"Cleaned dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    manual_res = manual_ols_sample(df)
    print("Manual OLS b0, b1, b2:", np.round(manual_res["b_manual"], 4))
    print("Sklearn OLS b0, b1, b2:", np.round(manual_res["b_sklearn"], 4))

    X = df[FEATURES]
    y = df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Mean baseline
    y_pred_base = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    base_metrics = compute_metrics(y_test, y_pred_base, n_features=1)

    # Single-feature model
    slr = LinearRegression().fit(X_train[["median_income"]], y_train)
    y_pred_slr = slr.predict(X_test[["median_income"]])
    slr_metrics = compute_metrics(y_test, y_pred_slr, n_features=1)

    # Multiple-feature model
    mlr = LinearRegression().fit(X_train, y_train)
    y_pred_mlr = mlr.predict(X_test)
    mlr_metrics = compute_metrics(y_test, y_pred_mlr, n_features=len(FEATURES))

    # Standardized coefficients (fit scaler on training data only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    mlr_scaled = LinearRegression().fit(X_train_scaled, y_train)

    print("\n--- Model Comparison (Test Set) ---")
    comp_df = pd.DataFrame(
        [
            {"Model": "Mean Baseline", **base_metrics},
            {"Model": "Single Linear Regression (median_income)", **slr_metrics},
            {"Model": "Multiple Linear Regression (6 features)", **mlr_metrics},
        ]
    )
    print(comp_df.to_string(index=False))

    print("\n--- Multiple Linear Regression Coefficients ---")
    coef_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Raw_Coefficient": mlr.coef_,
            "Standardized_Coefficient": mlr_scaled.coef_,
        }
    )
    print(f"Intercept: {mlr.intercept_:,.2f}")
    print(coef_df.to_string(index=False))

    vif_df = compute_vif(X_train)
    print("\n--- Variance Inflation Factor (VIF) ---")
    print(vif_df.to_string(index=False))

    residuals = y_test - y_pred_mlr
    generate_visualizations(df, y_test, y_pred_mlr, residuals, out_dir)
    print(f"\nVisualizations saved to: {out_dir}")


if __name__ == "__main__":
    main()
