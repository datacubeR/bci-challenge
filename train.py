import logging
from timeit import default_timer as timer

import hydra
import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

from callback import LogCallback
from models import ridge_v1

log = logging.getLogger("Training")

@hydra.main(config_path='conf', config_name='config')
def train_model(cfg: DictConfig):
    name = cfg.ds.train_name
    kfold_name = cfg.ds.kfold
    df = pd.read_csv(to_absolute_path('data/' + name))
    kfold = pd.read_csv(to_absolute_path('data/' + kfold_name))
    log.info(f"{df.shape[0]} rows imported from {name}")
    log.info(f'{kfold_name} imported')

    #======================================================
    # MERGE DE LA DATA
    #======================================================

    df = df.merge(kfold, on = ['id', 'mes'], how = 'left')
    log.info(f'Training set with {df.shape[0]} rows and {df.shape[1]} columns')


    def optimize_model(trial):
        #score_train = []
        score = []
        for val_fold in range(5):
            X_train = df.query('kfold != @val_fold').drop(columns = ['id','mes', 'kfold','target_mes'])
            y_train = df.query('kfold != @val_fold').target_mes

            X_val = df.query('kfold == @val_fold').drop(columns = ['id','mes', 'kfold','target_mes'])
            y_val = df.query('kfold == @val_fold').target_mes

            log.info(f'Training Model and Validating in Fold {val_fold}')
            
            pipe = hydra.utils.call(cfg.models.type, trial = trial)
            pipe.fit(X_train, y_train)
            
            #y_pred_train = pipe.predict(X_train)
            y_pred = pipe.predict(X_val)
            
            #mae_train = mean_absolute_error(y_train, y_pred_train)
            mae = mean_absolute_error(y_val, y_pred)
            #score_train.append(mae_train)
            score.append(mae)
            
            if cfg.ds.holdout:
                break
            
        return np.mean(score)
    
    sampler = TPESampler(seed=None)
    study = optuna.create_study(sampler = sampler, direction = 'minimize')
    study.optimize(optimize_model, n_trials = cfg.n_trials)
    
    log.info(f'Best MAE in the Test Set: {study.best_value}')
    log.info(f'Best Hyperparameters: {study.best_params}')
    
    return study
    

if __name__ == '__main__':
    tic = timer()
    train_model()
    toc = timer()
    log.info(f'Execution time: {round(toc-tic,1)/60} minutes')
    
