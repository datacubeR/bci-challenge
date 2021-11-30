import pandas as pd
from sklearn.model_selection import KFold
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

@hydra.main(config_path='conf', config_name='config')
def create_folds(cfg: DictConfig):
    
    df = pd.read_csv(to_absolute_path('data/' + cfg.kf.base_name))
    df['kfold'] = -1

    kf = KFold(n_splits=cfg.kf.n_splits, random_state=cfg.kf.rs, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f'Train Fold Size: {len(train_idx)}, Validation Fold Size {len(val_idx)}')
        df.loc[val_idx,'kfold'] = fold
        
    df[['id','mes','kfold']].to_csv(to_absolute_path(f"data/{cfg.kf.output}"), index=False)
    
if __name__ == '__main__':
    create_folds()




