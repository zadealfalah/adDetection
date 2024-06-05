import pandas as pd
from typing import List, Dict
import typer
from urllib.parse import urlparse
import json
from numpyencoder import NumpyEncoder

from config import logger, mlflow
from transforms import CustomPreprocessor
from train import OptunaXGBoost

# Init CLI app with Typer
app = typer.Typer()


class XGBPredictor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self, batch):
        results = self.model.predict(batch)
        return {"output": results}

    def predict_proba(self, batch):
        results = self.model.predict_proba(batch)
        return {"output": results}

    def get_preprocessor(self):
        return self.preprocessor

    @classmethod
    def from_checkpoint(cls, checkpoint):
        metadata = checkpoint.get_metadata()
        preprocessor = CustomPreprocessor()
        model = OptunaXGBoost.load()
        return cls(preprocessor=preprocessor, model=model)


### Split train.py into train.py and model.py first, then revisit this
def predict_proba(df: pd.DataFrame, predictor: XGBPredictor) -> List:
    preprocessor = predictor.get_preprocessor()
    pass
    # preprocessed_df = preprocessor.transform(df)
    # outputs =

    # y_probs =
    # results =


@app.command()
def get_best_run_id(experiment_name: str = "", metric: str = "", mode: str = "") -> str:

    sorted_runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=[f"metrics.{metric} {mode}"])
    run_id = sorted_runs.iloc[0].run_id
    return run_id


def get_best_checkpoint(run_id: str):
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri)

    model = mlflow.xgboost.load_model(model_uri=artifact_dir)

    return model


@app.command()
def predict(run_id: str, ip: int, app: int, device: int, os: int, channel: int, click_time: object) -> Dict:

    best_checkpoint = get_best_checkpoint(run_id=run_id)
    predictor = XGBPredictor.from_checkpoint(best_checkpoint)

    data = {"ip": ip, "app": app, "device": device, "os": os, "channel": channel, "click_time": click_time}
    sample_df = pd.DataFrame([data])
    results = predict_proba(df=sample_df, predictor=predictor)
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__":
    app()
