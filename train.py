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
from models import *

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
    target = ['target_mes']
    cat_vars = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli','month','year']
    int_variables = df.filter(like = '_trx').columns.tolist()
    float_variables = [vars for vars in df.columns if vars not in int_variables + cat_vars + target]
    
    
    df[['month', 'year']] = df[['month', 'year']].astype('int64')
    df[cat_vars] = df[cat_vars].astype('category')
    log.info(f'Training set with {df.shape[0]} rows and {df.shape[1]} columns')


    def optimize_model(trial):
        score_train = []
        score = []
        for val_fold in range(5):
            
            X = dict(train=df.query('kfold != @val_fold').drop(columns = ['id','mes', 'kfold','target_mes']),
                    test = df.query('kfold == @val_fold').drop(columns = ['id','mes', 'kfold','target_mes']))
            
            y = dict(train = df.query('kfold != @val_fold').target_mes,
                    test = df.query('kfold == @val_fold').target_mes)

            log.info(f'Training Model and Validating in Fold {val_fold}')
            
            pipe = hydra.utils.call(cfg.models.type, X = X, y = y, trial = trial)
            
            y_pred_train = pipe.predict(X['train'])
            y_pred = pipe.predict(X['test'])
            
            mae_train = mean_absolute_error(y['train'], y_pred_train)
            mae = mean_absolute_error(y['test'], y_pred)
            
            score_train.append(mae_train)
            log.info(f'Training MAE for Fold {val_fold}: {mae_train}')
            log.info(f'Validation MAE for Fold {val_fold}: {mae}')
            score.append(mae)
            
            if cfg.ds.holdout:
                break
        
        log.info(f'Training MAE: {np.mean(score_train)}')
        return np.mean(score)
    
    sampler = TPESampler(seed=None)
    study = optuna.create_study(sampler = sampler, direction = 'minimize')
    if cfg.hpo: 
        n_trials = cfg.n_trials
    else:
        n_trials = 1
    study.optimize(optimize_model, n_trials = n_trials)
    
    log.info(f'Best MAE in the Test Set: {study.best_value}')
    log.info(f'Best Hyperparameters: {study.best_params}')
    
    return {'study': study,
            'logging': True}
    

if __name__ == '__main__':
    tic = timer()
    train_model()
    toc = timer()
    log.info(f'Execution time: {round(toc-tic,1)/60} minutes')
    
