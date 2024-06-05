#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir results

# Test data
export RESULTS_FILE=results/test_data_results.txt
export X_DATASET_LOC="https://raw.githubusercontent.com/zadealfalah/adDetection/mlops-dev/datasets/X_us.csv"
export Y_DATASET_LOC="https://raw.githubusercontent.com/zadealfalah/adDetection/mlops-dev/datasets/y_us.csv"
pytest --y_dataset-loc=$Y_DATASET_LOC --x_dataset-loc=$X_DATASET_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Test code
export RESULTS_FILE=results/test_code_results.txt
python -m pytest tests/code --verbose --disable-warnings > $RESULTS_FILE
cat $RESULTS_FILE

# Train
export EXPERIMENT_NAME="xgb"
export RESULTS_FILE=results/training_results.json
export X_DATASET_LOC="https://raw.githubusercontent.com/zadealfalah/adDetection/mlops-dev/datasets/X_us.csv"
export Y_DATASET_LOC="https://raw.githubusercontent.com/zadealfalah/adDetection/mlops-dev/datasets/y_us.csv"
python addetectionscripts/train.py \
    --run_name "$EXPERIMENT_NAME" \
    --feature_set_version 2 \

# Get and save run ID
export RUN_ID=$(python -c "import os; from addetectionscripts import utils; d = utils.load_dict(os.getenv('RESULTS_FILE')); print(d['run_id'])")
export RUN_ID_FILE=results/run_id.txt
echo $RUN_ID > $RUN_ID_FILE  # used for serving later
