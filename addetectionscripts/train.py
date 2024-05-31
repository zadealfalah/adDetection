import mlflow
import xgboost as xgb
import optuna
from plots import *
from transforms import *
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from optuna.study import Study
from optuna.trial import FrozenTrial
from typing import Optional, Dict
from optuna.trial import Trial

from config import training_config, transformations_config  # MLFLOW_TRACKING_URI

# Set optuna to log only errors
optuna.logging.set_verbosity(optuna.logging.ERROR)
run_name = "Test 5"


class OptunaXGBoost:
    def __init__(
        self,
        run_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        feature_set_version: str,
        config: Dict,
        experiment_name: str = "Attributed Class 3.",
        dtrain: Optional[xgb.DMatrix] = None,
        dvalid: Optional[xgb.DMatrix] = None,
        num_trials: int = 25,
        tracking_uri: str = r"http://127.0.0.1:8080",
    ) -> None:
        """
        Initialize the OptunaXGBoost class.

        Args:
            run_name (str): Name of the MLflow run.
            X_train (pd.DataFrame): Training input features.
            y_train (pd.Series): Training target variable.
            X_valid (pd.DataFrame): Validation input features.
            y_valid (pd.Series): Validation target variable.
            feature_set_version (str): Version of the feature set.
            experiment_name (str, optional): Name of the MLflow experiment. Defaults to 'Attributed Class 3.'.
            dtrain (Optional[xgb.DMatrix], optional): Training DMatrix. Defaults to None.
            dvalid (Optional[xgb.DMatrix], optional): Validation DMatrix. Defaults to None.
            num_trials (int, optional): Number of optimization trials. Defaults to 25.
            tracking_uri (str, optional): URI of the MLflow tracking server. Defaults to 'http://127.0.0.1:8080'.'

        Returns:
            None
        """
        self.run_name = run_name
        self.config = config
        self.feature_set_version = feature_set_version
        self.num_trials = num_trials
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        if dtrain is None:
            self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        else:
            self.dtrain = dtrain
        if dvalid is None:
            self.dvalid = xgb.DMatrix(self.X_valid, label=self.y_valid)
        else:
            self.dvalid = dvalid

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)  ## Done in config when put to prod
        self.experiment_id = get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=self.experiment_id)

    def get_params(self, trial: Trial, param_config: Dict) -> Dict:
        """
        Get the parameters for a given XGBoost Trial using the Optuna 'suggest_*'

        Args:
            trial (Trial): The given trial of the model
            param_config (Dict): Input params to be read and put in to trial suggestions

        Returns:
            Dict: Dictonary of parameters for the given trial model utilizing Optuna's 'suggest_*' method(s)
        """
        params = {}
        for param, (ptype, *args) in param_config.items():
            if ptype == "categorical":
                params[param] = trial.suggest_categorical(param, args[0])
            elif ptype == "float":
                if len(args) == 3 and args[2] == "log":
                    params[param] = trial.suggest_float(param, args[0], args[1], log=True)
                else:
                    params[param] = trial.suggest_float(param, args[0], args[1])
            elif ptype == "int":
                params[param] = trial.suggest_int(param, args[0], args[1])
        return params

    def objective(self, trial: Trial) -> float:
        """
        Objective function for XGBoost hyperparameter optimization.

        Args:
            trial (optuna.trial.Trial): A trial object used to explore the search space.
            config (Dict): Dictonary containing the params for our trial

        Returns:
            float: The ROC AUC score on the validation set.
        """
        with mlflow.start_run(nested=True):
            params = self.get_params(trial, self.config.get("xgb_params", {}))

            bst = xgb.train(params, self.dtrain)

            # Get the training predicts/scores for graphing
            # train_preds = bst.predict(self.dtrain)
            # train_auc_score = roc_auc_score(self.y_valid, train_preds)

            # Get the validaiton predicts/scores for graphing and optimizing
            valid_preds = bst.predict(self.dvalid)
            valid_auc_score = roc_auc_score(self.y_valid, valid_preds)

            # Log to mlflow
            mlflow.log_params(params)
            mlflow.log_metric("auc", valid_auc_score)
            # mlflow.log_metric("train_auc", train_auc_score)

        return valid_auc_score

    def best_trial_callback(self, study: Study, frozen_trial: FrozenTrial) -> None:
        """
        Logging callback that reports when a new trial iteration improves upon existing
        best trial values.

        Note: This callback is not intended for use in distributed computing systems such as Spark
        or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
        workers or agents.
        The race conditions with file system state management for distributed trials will render
        inconsistent values with this callback.

        Args:
            study (optuna.study.Study): The study object.
            frozen_trial (optuna.trial.FrozenTrial): The frozen trial object.

        Returns:
            None
        """

        winner = study.user_attrs.get("winner", None)

        if study.best_value and winner != study.best_value:
            study.set_user_attr("winner", study.best_value)
            if winner:
                improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
                print(f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with " f"{improvement_percent: .4f}% improvement")
            else:
                print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

    def start_mlflow_runs(self, nested: bool = True) -> None:
        """
        Start MLflow runs for hyperparameter optimization.

        Args:
            nested (bool): Whether to have nested runs. Defaults to True.

        Returns:
            None
        """
        with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name, nested=nested):
            # Init Optuna
            study = optuna.create_study(direction="maximize")  # Want to max our AUC

            # Optimize hparams
            study.optimize(self.objective, n_trials=self.num_trials, callbacks=[self.best_trial_callback])
            # Log best params/score
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_auc", study.best_value)

            # Set log tags
            mlflow.set_tags(
                tags={
                    "project": "Ad Fraud Detection",
                    "optimizer_engine": "Optuna",
                    "model_family": "xgboost",
                    "feature_set_version": self.feature_set_version,
                }
            )

            # Log a fit model instance
            model = xgb.train(study.best_params, self.dtrain)

            # Log the correlation plot
            correlations = plot_correlation(self.X_train, self.y_train, target_col="is_attributed")
            mlflow.log_figure(figure=correlations, artifact_file="plot_correlation.png")

            # Log feature importance plot
            importances = plot_feature_importance(model, booster=study.best_params.get("booster"))
            mlflow.log_figure(figure=importances, artifact_file="feature_importances.png")

            artifact_path = "model"

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=artifact_path,
                input_example=self.X_train.iloc[[0]],
                model_format="ubj",
                metadata={"model_data_version": 1},
            )

            # Get logged model uri for loading from artifact store
            model_uri = mlflow.get_artifact_uri(artifact_path)


## For testing locally atm
if __name__ == "__main__":
    print("Starting")
    X_us, y_us, _ = init_datasets()

    X_us, y_us = apply_transformations(X_us, y_us)

    print("Splitting datasets")
    X_train, X_val, y_train, y_val, dtrain, dvalid = split_final_datasets(X_us, y_us)
    print("datasets split")

    print("Starting booster")
    booster = OptunaXGBoost(
        run_name=run_name,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_val,
        y_valid=y_val,
        feature_set_version=2,
        config=training_config,
        dtrain=dtrain,
        dvalid=dvalid,
    )
    print("set booster")
    booster.start_mlflow_runs()
