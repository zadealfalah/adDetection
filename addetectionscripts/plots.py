# Plots for mlflow logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb

def plot_correlation(X: pd.DataFrame, y_df: pd.DataFrame, target_col: str = 'is_attributed',
                     save_path: str = None, style: str = 'seaborn', plot_size: tuple = (12, 8)):
    X_df = X.copy()
    # Combine the X and y data again
    X_df[target_col] = y_df[target_col]
    # Get corr values w/ all vars wrt the target
    correlations = X_df.corr()[target_col].drop(target_col).sort_values()
    
    with plt.style.context(style=style):
        # Make bar plot
        fig = plt.figure(figsize=plot_size)
        plt.barh(correlations.index, correlations.values)
        
        # Set Labels
        plt.title(f'Correlation with {target_col}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Variable')
        plt.grid(axis='x')
        plt.tight_layout()
        
    if save_path:
        plt.savefig(save_path, format='png', dpi=600)
        
    plt.close(fig)
    return fig


def plot_feature_importance(model, booster):
    
    fig, ax = plt.subplots(figsize=(10,8))
    importance_type = 'gain'
    xgb.plot_importance(
        model,
        importance_type=importance_type,
        ax=ax,
        title=f"Feature Importance Based On {importance_type}"
    )
    plt.tight_layout()
    plt.close(fig)
    return fig