import mlflow 
from optuna_integration.mlflow import MLflowCallback
from config import MLFLOW_TRACKING_URI

def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get the experiment ID if it exists, otherwise create a new experiment with the given name.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        str: The ID of the existing or newly created experiment.
    """
    if experiment:= mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
