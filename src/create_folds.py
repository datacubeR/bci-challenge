import pandas as pd
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from sklearn.model_selection import KFold

@hydra.main(config_path='conf', config_name='config')
def create_folds(cfg: DictConfig):
    df = pd.read_csv(to_absolute_path('input/train_data.csv'))
    df['kfold'] = -1
    
    kf = KFold(n_splits=5, shuffle = True,  random_state=123)
    
    
    for fold , (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.target_mes)):
        df.loc[val_idx, 'kfold'] = fold
        
    df.to_csv(to_absolute_path('input/train_folds.csv'), index=False)

if __name__ == '__main__':
    create_folds()