from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import MeanMedianImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor, Pool

num_vars = ['VAR5_sum',
                    'VAR5_prom',
                    'VAR5_trx',
                    'VAR11_sum',
                    'VAR12_sum',
                    'VAR13_sum',
                    'VAR14_sum',
                    'VAR15_sum',
                    'VAR18_sum',
                    'VAR19_sum',
                    'VAR20_sum',
                    'VAR21_sum',
                    'VAR22_sum',
                    'year',
                    'month']

cat_vars = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli','month','year']


#======================================================
# RIDGE
#======================================================


def ridge_v1(alpha, 
            random_state,
            trial = None):
    
    if trial is not None:
        alpha = trial.suggest_float('alpha', **alpha)
        
    pipe = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('model', Ridge(alpha = alpha, 
                            random_state=random_state))
    ])
    
    return pipe

#======================================================
# RANDOM FOREST
#======================================================

def rf(random_state,
        num_vars = num_vars,
        trial = None):
    
    if trial is not None:
        pass
    
    scaler = SklearnTransformerWrapper(transformer = StandardScaler(),
                                        variables = num_vars)
    pipe = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('imp_num', MeanMedianImputer(imputation_method='mean')),
            ('sc', scaler),            
            ('model', RandomForestRegressor(n_jobs = -1, 
                            random_state=random_state))
    ])
    
    return pipe

#======================================================
# CATBOOST
#======================================================

def catboost(X, y, 
        random_state,
        num_vars = num_vars,
        cat_vars = cat_vars,
        trial = None):
    
    if trial is not None:
        pass
    
    train_pool = Pool(X['train'], 
                    label = y['train'],
                    cat_features=cat_vars)
    validation_pool = Pool(X['test'], 
                        label = y['test'],
                        cat_features=cat_vars)
    
    model = CatBoostRegressor(loss_function='MAE',
                            #learning_rate=0.5,
                            #eval_metric = 'MAE', 
                            random_seed = random_state, 
                            task_type="GPU",
                            devices='0')
    
    model.fit(train_pool)
            #early_stopping_rounds = 10)
    
    return model