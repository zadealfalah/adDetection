from fastapi import FastAPI
from typing import Dict
from http import HTTPStatus
from starlette.requests import Request
import pandas as pd
import argparse

import predict
from config import MLFLOW_TRACKING_URI, mlflow


# Define app
app = FastAPI(title="Addetection", description="Classify a given impression as fraud or not", version="0.1")


class DeployedModel:
    def __init__(self, run_id: str, threshold: int = 0.5):
        """Init the model"""
        self.threshold = threshold
        self.run_id = run_id
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        best_checkpoint = predict.get_bst_checkpoint(run_id=run_id)
        self.predictor = predict.XGBPredictor.from_checkpoint(best_checkpoint)

    # Check status
    @app.get("/")
    def _index(self) -> Dict:
        "Health check"
        response = {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK, "data": {}}
        return response

    # Get run ID for model
    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        "Get run ID"
        return {"run_id": self.run_id}

    # Predict on new data

    ### Assumes predict_proba returns preds/probs as needed, may reformat - if so, fix here!
    @app.post("/predict/")
    async def _predict(self, request: Request):
        # Request needs the following features:
        # ip,app,device,os,channel,click_time
        data = await request.json()
        sample_df = pd.DataFrame([data])
        results = predict.predict_proba(df=sample_df, predictor=self.predictor)

        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:  # Used in case we want custom threshold instead of 0.5 (default)
                results[i]["prediction"] = 0
        return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshhold", type=float, default=0.5, help="threshold value for class predictions")
    args = parser.parse_args()

    ## Below for actually running, currently have local testing elsewhere
    # ray.init()
    # serve.run(DeployedModel.bind(run_id=args.run_id, threshold=args.threshold))
