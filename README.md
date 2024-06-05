# adDetection

## Table of Contents
* [General Info](#general-info)
* [To Do](#to-do)
* [Data Source](#data-source)

## General Info
Using the TalkingData AdTracking dataset, this project attempts to create a binary classifier for fraud detection.
It is currently being tested locally with MLFlow and an XGBoost model.  The end goal is to be able to run it as if it were production code distributed via e.g. Ray.


## To Do
- Finalize splitting of train.py and model.py
- Finish predict_proba once model.py is finalized
- Update MLflow models with larger training data
- Deploy
- Create distributed version of project (Ray)
- Add CI/CD workflows, monitoring


## Data Source
Data from https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/data
