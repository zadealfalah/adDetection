import mlflow 

def get_or_create_experiment(experiment_name: str):
    if experiment:= mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
    
def trial_callback(study, frozen_trial):
    """
    Only log when new trial improves upon existing best trial values 
    - Do not use in distributed systems
    """
    
    winner = study.user_attrs.get('winner', None)
    
    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")