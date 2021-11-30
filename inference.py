import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from models import ridge_v1
import logging

log = logging.getLogger("Inference")

@hydra.main(config_path='conf', config_name='config')
def create_submission(cfg: DictConfig):
    train_name = cfg.ds.train_name
    test_name = cfg.ds.test_name
    df_train = pd.read_csv(to_absolute_path('data/' + train_name))
    df_test = pd.read_csv(to_absolute_path('data/' + test_name))
    log.info(f"{df_train.shape[0]} rows imported from {train_name}")
    log.info(f"{df_test.shape[0]} rows imported from {test_name}")

    X_train = df_train.drop(columns = ['id','mes','target_mes'])
    y_train = df_train.target_mes

    X_test = df_test.drop(columns = ['id','mes'])

    pipe = hydra.utils.call(cfg.models.inference)
        
    pipe.fit(X_train, y_train)

    if cfg.models.clip:
        df_test['target_mes'] = np.where(pipe.predict(X_test)<0,0, pipe.predict(X_test))
        log.info('Predictions with no clipping')
    else: 
        df_test['target_mes'] = pipe.predict(X_test)

    df_test[['id','mes','target_mes']].to_csv('submission.csv', index = False)

if __name__ == '__main__':
    tic = timer()
    create_submission()
    toc = timer()
    log.info(f'Execution time: {round(toc-tic,1)/60} minutes')