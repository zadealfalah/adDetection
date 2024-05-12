import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def objective(trial, train_matrix, val_matrix, y_val):
    with mlflow.start_run(nested=True):
        # Hparams
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical("booster", ["gbtree", "dart"]),
            'lambda': trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int("max_depth", 1, 9),
            'eta': trial.suggest_float("eta", 1e-8, 1.0, log=True),
            'gamma': trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        }
        bst = xgb.train(params, train_matrix)
        preds = bst.predict(val_matrix)
        auc_score = roc_auc_score(y_val, preds)
        
        
        # Log to mlflow
        mlflow.log_params(params)
        mlflow.log_metric('auc', auc_score)
        
    
    return auc_score