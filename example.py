import optuna
import wandb

def objective(trial):
    x = trial.suggest_float("x", -5, 0)
    return (x - 2) ** 2

config = dict(name = 'LightGBM',
            data = 'full')
study = optuna.create_study(study_name=config['name'])
study.optimize(objective, n_trials=10)

with wandb.init(config = config, 
                project="checking",
                tags = ['optimización','cuadrática']) as run:
    for step, trial in enumerate(study.trials):

        run.log(trial.params, step = step)
        run.log({"y": trial.value})

    run.summary['mae'] = study.best_value
    run.summary = study.best_params

# from catboost import CatBoostClassifier

# train_data = [[0, 3],
#               [4, 1],
#               [8, 1],
#               [9, 1]]
# train_labels = [0, 0, 1, 1]

# model = CatBoostClassifier(iterations=1000,
#                            task_type="GPU",
#                            devices='0:1')
# model.fit(train_data,
#           train_labels,
#           verbose=False)