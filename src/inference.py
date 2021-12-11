import pandas as pd
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import joblib

@hydra.main(config_path='conf', config_name='config')
def predict(cfg: DictConfig):
    df_test = pd.read_csv(to_absolute_path(cfg.test_data))
    id = df_test[['id','mes']]

    preds_dict = {}
    for fold in range(cfg.n_splits):
        pipe = joblib.load(to_absolute_path(f'models/{cfg.model_name}_fold_{fold}.joblib'))
        columns = joblib.load(to_absolute_path(f'models/{cfg.model_name}_fold_{fold}_columns.joblib'))
        df_test = df_test[columns]
        preds = pipe.predict(df_test)
        
        preds_dict[fold] = preds
    
    final_preds = pd.DataFrame(preds_dict).mean(axis = 1)
    if cfg.clip:
        final_preds.where(final_preds > 0, 0)
        
    pd.concat([id, final_preds],axis = 1).to_csv(to_absolute_path(f'submissions/{cfg.model_name}.csv'), index = False)
        
if __name__ == '__main__':
    predict()
    
    
    