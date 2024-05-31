# Plots for mlflow logging
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from typing import Optional, Tuple, Any


def plot_correlation(
    X: pd.DataFrame, y_df: pd.DataFrame, target_col: str = "is_attributed", save_path: Optional[str] = None, plot_size: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot correlation between features and a target variable.

    Args:
        X (pd.DataFrame): Input features DataFrame.
        y_df (pd.DataFrame): DataFrame containing the target variable.
        target_col (str, optional): Name of the target column. Defaults to 'is_attributed'.
        save_path (Optional[str], optional): Path to save the plot. Defaults to None.
        plot_size (Tuple[int, int], optional): Size of the plot. Defaults to (12, 8).

    Returns:
        plt.Figure: The matplotlib Figure object.
    """
    X_df = X.copy()
    # Combine the X and y data again
    X_df[target_col] = y_df[target_col]
    # Get corr values w/ all vars wrt the target
    correlations = X_df.corr()[target_col].drop(target_col).sort_values()

    # Make bar plot
    fig = plt.figure(figsize=plot_size)
    plt.barh(correlations.index, correlations.values)

    # Set Labels
    plt.title(f"Correlation with {target_col}")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Variable")
    plt.grid(axis="x")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    plt.close(fig)
    return fig


def plot_feature_importance(model: xgb.XGBModel, booster: str) -> plt.Figure:
    """
    Plot feature importance based on the provided XGBoost model.

    Args:
        model (xgb.XGBModel): The trained XGBoost model.
        booster (str): The booster type used in the XGBoost model.

    Returns:
        plt.Figure: The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_type = "gain"
    xgb.plot_importance(model, importance_type=importance_type, ax=ax, title=f"Feature Importance Based On {importance_type}")
    plt.tight_layout()
    plt.close(fig)
    return fig
