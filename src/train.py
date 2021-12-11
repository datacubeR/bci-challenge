from timeit import default_timer as timer

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error
from src.dispatcher import MODELS
import logging
import joblib

log = logging.getLogger("Training")

@hydra.main(config_path='conf',config_name='config')
def train(cfg: DictConfig):
    df = pd.read_csv(to_absolute_path(cfg.training_data))
    cat_vars = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli']
    
    score = []
    for fold in range(cfg.n_splits):
        train_df = df.query('kfold != @fold').drop(columns = ['id','mes','kfold','target_mes'])
        val_df = df.query('kfold == @fold').drop(columns = ['id','mes','kfold','target_mes'])
        train_columns = train_df.columns
        
        y_train = df.query('kfold != @fold').target_mes
        y_val = df.query('kfold == @fold').target_mes
        
        val_df = val_df[train_columns]
        
        pipe = MODELS[cfg.model_name]
        pipe.fit(train_df, y_train)
        preds = pipe.predict(val_df)
        val_score = mean_absolute_error(y_val, preds)
        score.append(val_score)
        joblib.dump(pipe, to_absolute_path(f'models/{cfg.model_name}_fold_{fold}.joblib'))
        joblib.dump(train_columns, to_absolute_path(f'models/{cfg.model_name}_fold_{fold}_columns.joblib'))
                
        
        log.info(f'MAE for Fold {fold}: {val_score}')
        
    log.info(f'Mean Score {np.mean(score)}')
    
    
    

if __name__ == '__main__':
    tic = timer()
    train()
    toc = timer()
    
    print(f'Training Time: {(toc- tic)/60} minutes')
