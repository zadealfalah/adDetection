# adDetection

## Table of Contents
* [General Info](#general-info)
* [To Do](#to-do)
* [Data Source](#data-source)

## General Info
Using the TalkingData AdTracking dataset, this project attempts to create a binary classifier for fraud detection.  
It is currently being tested locally with MLFlow and an XGBoost model.  The end goal is to be able to run it as if it were production code distributed via e.g. Ray.


## To Do
- Finish local MLFlow testing
    * Add tests for model
    * Add tests for training
- Convert to runnable .sh
- Create distributed version of project
- Add CI/CD workflows, monitoring


## Data Source
Data from https://www.kaggle.com/competitions/talkingdata-adtracking-fraud-detection/data