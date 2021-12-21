from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error
from src.dispatcher import *
import logging
import joblib
from box import Box

log = logging.getLogger("Training")

@hydra.main(config_path='conf',config_name='config')
def train(cfg: DictConfig):
    df = pd.read_csv(to_absolute_path(cfg.training_data))
    cat_vars = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli']
    df[cat_vars] = df[cat_vars].astype('category')
    
    score = []
    val_df = pd.DataFrame()
    for fold in range(cfg.n_splits):
        
        id_val = df.query('kfold == @fold')[['id','mes','kfold']]
        
        log.info(f'Training for Fold {fold}')
        X = Box({})
        y = Box({})
        
        X.train = df.query('kfold != @fold').drop(columns = ['id','mes','kfold','target_mes'])
        X.val = df.query('kfold == @fold')[X.train.columns]
        
        y.train = df.query('kfold != @fold').target_mes
        y.val = df.query('kfold == @fold').target_mes
        
        pipe, preds = hydra.utils.call(cfg.model.algo, X = X, y = y)
        
        val_score = mean_absolute_error(y.val, preds)
        score.append(val_score)
        
        log.info(f'Exporting Results for Fold {fold}...')
        joblib.dump(pipe, to_absolute_path(f'models/{cfg.model_name}_fold_{fold}.joblib'))
        joblib.dump(X.train.columns, to_absolute_path(f'models/{cfg.model_name}_fold_{fold}_columns.joblib'))
        
        id_val[f'{cfg.model_name}_pred'] = preds
        val_df = val_df.append(id_val)
        
        log.info(f'MAE for Fold {fold}: {val_score}')
    
    val_df.to_csv(to_absolute_path(f'preds/{cfg.model_name}_preds.csv'), index=False)
    log.info(f'Mean Score {np.mean(score)}')
    
    
    

if __name__ == '__main__':
    tic = timer()
    train()
    toc = timer()
    
    print(f'Training Time: {(toc- tic)/60} minutes')
