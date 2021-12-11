from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import MeanMedianImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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

num_vars_2 = ['VAR16_sum',
 'VAR24_sum',
 'VAR17_sum',
 'VAR4_sum',
 'VAR1_prom',
 'VAR4_trx',
 'VAR12_sum',
 'VAR3_trx',
 'VAR3_sum',
 'VAR5_sum',
 'VAR1_sum',
 'VAR25_sum',
 'VAR28_sum',
 'VAR23_prom',
 'VAR7_sum',
 'VAR5_prom',
 'VAR21_sum',
 'VAR27_trx',
 'VAR23_sum',
 'VAR6_sum',
 'VAR1_trx',
 'VAR29_ratio',
 'VAR2_trx',
 'VAR27_sum',
 'VAR24_trx',
 'VAR7_prom',
 'VAR11_sum',
 'VAR30_sum',
 'VAR28_prom',
 'VAR29_prom',
 'VAR7_trx',
 'VAR13_sum',
 'VAR28_trx',
 'VAR8_sum',
 'VAR3_prom',
 'VAR26_prom',
 'VAR6_trx',
 'VAR26_sum',
 'VAR20_sum',
 'VAR22_sum',
 'VAR9_prom',
 'VAR10_sum',
 'VAR9_trx',
 'VAR9_sum']

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
    
    model = CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            #learning_rate=0.5,
                            #eval_metric = 'MAE', 
                            random_seed = random_state, 
                            task_type="GPU",
                            devices='0')
    
    model.fit(train_pool)
            #early_stopping_rounds = 10)
    
    return model

#======================================================
# XGBoost
#======================================================

def xgboost(X, y, 
        random_state,
        num_vars = num_vars,
        cat_vars = cat_vars,
        trial = None):
    
    if trial is not None:
        pass
    
    model = XGBRegressor(n_estimators=1500,
                        max_depth = 3,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = random_state
                        )
    
    model.fit(X['train'], y['train'])
    
    return model


def lgbm(X, y, 
        random_state,
        num_vars = num_vars,
        cat_vars = cat_vars,
        trial = None):
    
    if trial is not None:
        pass
    
    scaler = SklearnTransformerWrapper(transformer = StandardScaler(),
                                        variables = num_vars)
    
    pipe = LGBMRegressor(n_estimators=300,
                        max_depth = 5,
                        n_jobs = -1,
                        random_state = random_state
                        )
    
    pipe.fit(X['train'], y['train'])
    
    return pipe


def cb_v2(X, y, 
        random_state,
        num_vars = num_vars_2,#eval_metric = 'MAE', 
    
    if trial is not None:
        pass
    
    train_pool = Pool(X['train'][num_vars + cat_vars], 
                    label = y['train'],
                    cat_features=cat_vars)
    validation_pool = Pool(X['test'][num_vars + cat_vars], 
                        label = y['test'],
                        cat_features=cat_vars)
    
    model = CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            #learning_rate=0.5,
                            #eval_metric = 'MAE', 
                            random_seed = random_state, 
                            task_type="GPU",
                            devices='0')
    
    model.fit(train_pool)
            #early_stopping_rounds = 10)
    
    return model