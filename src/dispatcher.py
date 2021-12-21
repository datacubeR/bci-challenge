from catboost import CatBoostRegressor
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer)
from feature_engine.wrappers import SklearnTransformerWrapper
from lightgbm.basic import LightGBMError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from xgboost import callback
from tempfile import mkdtemp
from shutil import rmtree


scaler = SklearnTransformerWrapper(StandardScaler())

def xgb_v1(X, y):
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    model.fit(X.train, y.train, 
            early_stopping_rounds=50, 
            eval_metric='mae', 
            eval_set=[(X.val, y.val)])
    
    preds = model.predict(X.val)
    
    
    return model, preds

def xgb_v2(X, y):
    
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__early_stopping_rounds=150, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def lgb_v1(X, y, cat_vars):
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = lgb.LGBMRegressor(n_estimators=1000, device="gpu")
    
    pipe = Pipeline(steps = [
        ('prep', prep), 
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__categorical_feature = cat_vars, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)],
            model__callbacks = [lgb.early_stopping(stopping_rounds=50)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def cb_v1(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 150)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds


def cb_v2(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', MeanEncoder()),
            ('mi', AddMissingIndicator()),
            #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 300)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds
    
    


# MODELS = dict(
#     cb_v1 = Pipeline(steps = [
#             ('enc', OrdinalEncoder(encoding_method='arbitrary')),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v2 = Pipeline(steps = [
#             ('enc', OneHotEncoder()),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v3 = Pipeline(steps = [
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', scaler),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v4 = Pipeline(steps = [
#         #('ord', OrdinalEncoder(encoding_method='ordered')),
#         ('model',CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             eval_metric = 'MAE', 
#                             random_seed = 123, 
#                             task_type="GPU"))]),
    
#     xgb_v1 = XGBRegressor(n_estimators=1500,
#                         objective='reg:squarederror',
#                         tree_method="gpu_hist",
#                         verbosity = 2,
#                         enable_categorical = True, 
#                         random_state = 123
#                         ),
    
#     lgb_v1 = Pipeline(steps = [
#             #('ord', OrdinalEncoder(encoding_method='ordered')),
#             #('mi', AddMissingIndicator()),
#             #('imp', Arbitfrom catboost import CatBoostRegressor
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer)
from feature_engine.wrappers import SklearnTransformerWrapper
from lightgbm.basic import LightGBMError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from xgboost import callback
from tempfile import mkdtemp
from shutil import rmtree


scaler = SklearnTransformerWrapper(StandardScaler())

def xgb_v1(X, y):
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    model.fit(X.train, y.train, 
            early_stopping_rounds=50, 
            eval_metric='mae', 
            eval_set=[(X.val, y.val)])
    
    preds = model.predict(X.val)
    
    
    return model, preds

def xgb_v2(X, y):
    
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__early_stopping_rounds=150, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def lgb_v1(X, y, cat_vars):
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = lgb.LGBMRegressor(n_estimators=1000, device="gpu")
    
    pipe = Pipeline(steps = [
        ('prep', prep), 
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__categorical_feature = cat_vars, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)],
            model__callbacks = [lgb.early_stopping(stopping_rounds=50)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def cb_v1(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 150)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds


def cb_v2(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', MeanEncoder()),
            ('mi', AddMissingIndicator()),
            #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 300)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds
    
    


# MODELS = dict(
#     cb_v1 = Pipeline(steps = [
#             ('enc', OrdinalEncoder(encoding_method='arbitrary')),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v2 = Pipeline(steps = [
#             ('enc', OneHotEncoder()),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v3 = Pipeline(steps = [
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', scaler),
#             ('model', CatBfrom catboost import CatBoostRegressor
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer)
from feature_engine.wrappers import SklearnTransformerWrapper
from lightgbm.basic import LightGBMError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from xgboost import callback
from tempfile import mkdtemp
from shutil import rmtree


scaler = SklearnTransformerWrapper(StandardScaler())

def xgb_v1(X, y):
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    model.fit(X.train, y.train, 
            early_stopping_rounds=50, 
            eval_metric='mae', 
            eval_set=[(X.val, y.val)])
    
    preds = model.predict(X.val)
    
    
    return model, preds

def xgb_v2(X, y):
    
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__early_stopping_rounds=150, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def lgb_v1(X, y, cat_vars):
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = lgb.LGBMRegressor(n_estimators=1000, device="gpu")
    
    pipe = Pipeline(steps = [
        ('prep', prep), 
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__categorical_feature = cat_vars, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)],
            model__callbacks = [lgb.early_stopping(stopping_rounds=50)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def cb_v1(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 150)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds


def cb_v2(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', MeanEncoder()),
            ('mi', AddMissingIndicator()),
            #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 300)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds
    
    


# MODELS = dict(
#     cb_v1 = Pipeline(steps = [
#             ('enc', OrdinalEncoder(encoding_method='arbitrary')),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v2 = Pipeline(steps = [
#             ('enc', OneHotEncoder()),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v3 = Pipeline(steps = [
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', scaler),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                           from catboost import CatBoostRegressor
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer)
from feature_engine.wrappers import SklearnTransformerWrapper
from lightgbm.basic import LightGBMError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from xgboost import callback
from tempfile import mkdtemp
from shutil import rmtree


scaler = SklearnTransformerWrapper(StandardScaler())

def xgb_v1(X, y):
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    model.fit(X.train, y.train, 
            early_stopping_rounds=50, 
            eval_metric='mae', 
            eval_set=[(X.val, y.val)])
    
    preds = model.predict(X.val)
    
    
    return model, preds

def xgb_v2(X, y):
    
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__early_stopping_rounds=150, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def lgb_v1(X, y, cat_vars):
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = lgb.LGBMRegressor(n_estimators=1000, device="gpu")
    
    pipe = Pipeline(steps = [
        ('prep', prep), 
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__categorical_feature = cat_vars, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)],
            model__callbacks = [lgb.early_stopping(stopping_rounds=50)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def cb_v1(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 150)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds


def cb_v2(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', MeanEncoder()),
            ('mi', AddMissingIndicator()),
            #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 300)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds
    
    


# MODELS = dict(
#     cb_v1 = Pipeline(steps = [
#             ('enc', OrdinalEncoder(encoding_method='arbitrary')),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v2 = Pipeline(steps = [
#             ('enc', OneHotEncoder()),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v3 = Pipeline(steps = [
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', scaler),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                           from catboost import CatBoostRegressor
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer)
from feature_engine.wrappers import SklearnTransformerWrapper
from lightgbm.basic import LightGBMError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from xgboost import callback
from tempfile import mkdtemp
from shutil import rmtree


scaler = SklearnTransformerWrapper(StandardScaler())

def xgb_v1(X, y):
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    model.fit(X.train, y.train, 
            early_stopping_rounds=50, 
            eval_metric='mae', 
            eval_set=[(X.val, y.val)])
    
    preds = model.predict(X.val)
    
    
    return model, preds

def xgb_v2(X, y):
    
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__early_stopping_rounds=150, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def lgb_v1(X, y, cat_vars):
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = lgb.LGBMRegressor(n_estimators=1000, device="gpu")
    
    pipe = Pipeline(steps = [
        ('prep', prep), 
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__categorical_feature = cat_vars, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)],
            model__callbacks = [lgb.early_stopping(stopping_rounds=50)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def cb_v1(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 150)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds


def cb_v2(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', MeanEncoder()),
            ('mi', AddMissingIndicator()),
            #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 300)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds
    
    


# MODELS = dict(
#     cb_v1 = Pipeline(steps = [
#             ('enc', OrdinalEncoder(encoding_method='arbitrary')),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v2 = Pipeline(steps = [
#             ('enc', OneHotEncoder()),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v3 = Pipeline(stepfrom catboost import CatBoostRegressor
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer)
from feature_engine.wrappers import SklearnTransformerWrapper
from lightgbm.basic import LightGBMError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from xgboost import callback
from tempfile import mkdtemp
from shutil import rmtree


scaler = SklearnTransformerWrapper(StandardScaler())

def xgb_v1(X, y):
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    model.fit(X.train, y.train, 
            early_stopping_rounds=50, 
            eval_metric='mae', 
            eval_set=[(X.val, y.val)])
    
    preds = model.predict(X.val)
    
    
    return model, preds

def xgb_v2(X, y):
    
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__early_stopping_rounds=150, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def lgb_v1(X, y, cat_vars):
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = lgb.LGBMRegressor(n_estimators=1000, device="gpu")
    
    pipe = Pipeline(steps = [
        ('prep', prep), 
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__categorical_feature = cat_vars, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)],
            model__callbacks = [lgb.early_stopping(stopping_rounds=50)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def cb_v1(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 150)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds


def cb_v2(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', MeanEncoder()),
            ('mi', AddMissingIndicator()),
            #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 300)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds
    
    


# MODELS = dict(
#     cb_v1 = Pipeline(steps = [
#             ('enc', OrdinalEncoder(encoding_method='arbitrary')),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v2 = Pipeline(steps = [
#             ('enc', OneHotEncoder()),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v3 = Pipeline(steps = [
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', scaler),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                          from catboost import CatBoostRegressor
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder
from feature_engine.imputation import (AddMissingIndicator,
                                       ArbitraryNumberImputer)
from feature_engine.wrappers import SklearnTransformerWrapper
from lightgbm.basic import LightGBMError
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from xgboost import callback
from tempfile import mkdtemp
from shutil import rmtree


scaler = SklearnTransformerWrapper(StandardScaler())

def xgb_v1(X, y):
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    model.fit(X.train, y.train, 
            early_stopping_rounds=50, 
            eval_metric='mae', 
            eval_set=[(X.val, y.val)])
    
    preds = model.predict(X.val)
    
    
    return model, preds

def xgb_v2(X, y):
    
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = xgb.XGBRegressor(n_estimators=1500,
                        objective='reg:squarederror',
                        tree_method="gpu_hist",
                        verbosity = 2,
                        enable_categorical = True, 
                        random_state = 123
                        )
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__early_stopping_rounds=150, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def lgb_v1(X, y, cat_vars):
    prep = Pipeline(steps = [
        ('ord', OrdinalEncoder(encoding_method='ordered')),
        ('mi', AddMissingIndicator()),
        ('imp', ArbitraryNumberImputer(arbitrary_number = 0))
    ])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = lgb.LGBMRegressor(n_estimators=1000, device="gpu")
    
    pipe = Pipeline(steps = [
        ('prep', prep), 
        ('model',model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__categorical_feature = cat_vars, 
            model__eval_metric='mae', 
            model__eval_set=[(X_val, y.val)],
            model__callbacks = [lgb.early_stopping(stopping_rounds=50)])
    
    preds = pipe.predict(X.val)
    
    return pipe, preds

def cb_v1(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 150)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds


def cb_v2(X, y, cat_vars):
    
    prep = Pipeline(steps = [
            ('ohe', MeanEncoder()),
            ('mi', AddMissingIndicator()),
            #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),])
    
    prep.fit(X.train, y.train)
    X_val = prep.transform(X.val)
    
    model = CatBoostRegressor(iterations=1500,
                            learning_rate=0.9,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0:1')
    
    pipe = Pipeline(steps = [
        ('prep', prep),
        ('model', model)
    ])
    
    pipe.fit(X.train, y.train, 
            model__eval_set = (X_val, y.val),
            #model__cat_features = cat_vars, 
            model__early_stopping_rounds = 300)
    
    preds = pipe.predict(X.val)
    
    return pipe, preds
    
    


# MODELS = dict(
#     cb_v1 = Pipeline(steps = [
#             ('enc', OrdinalEncoder(encoding_method='arbitrary')),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v2 = Pipeline(steps = [
#             ('enc', OneHotEncoder()),
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', StandardScaler()),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v3 = Pipeline(steps = [
#             ('mi', AddMissingIndicator()),
#             ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('sc', scaler),
#             ('model', CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             random_seed = 123, 
#                             task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v4 = Pipeline(steps = [
#         #('ord', OrdinalEncoder(encoding_method='ordered')),
#         ('model',CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             eval_metric = 'MAE', 
#                             random_seed = 123, 
#                             task_type="GPU"))]),
    
#     xgb_v1 = XGBRegressor(n_estimators=1500,
#                         objective='reg:squarederror',
#                         tree_method="gpu_hist",
#                         verbosity = 2,
#                         enable_categorical = True, 
#                         random_state = 123
#                         ),
    
#     lgb_v1 = Pipeline(steps = [
#             #('ord', OrdinalEncoder(encoding_method='ordered')),
#             #('mi', AddMissingIndicator()),
#             #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('model', LGBMRegressor(n_estimators=500,
#                                     device="gpu"))
#             ])
       eval_metric = 'MAE', 
#                             random_seed = 123, 
#                             task_type="GPU"))]),
    
#     xgb_v1 = XGBRegressor(n_estimators=1500,
#                         objective='reg:squarederror',
#                         tree_method="gpu_hist",
#                         verbosity = 2,
#                         enable_categorical = True, 
#                         random_state = 123
#                         ),
    
#     lgb_v1 = Pipeline(steps = [
#             #('ord', OrdinalEncoder(encoding_method='ordered')),
#             #('mi', AddMissingIndicator()),
#             #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('model', LGBMRegressor(n_estimators=500,
#                                     device="gpu"))
#             ])
       task_type="GPU",
#                             devices='0'))
#             ]),
#     cb_v4 = Pipeline(steps = [
#         #('ord', OrdinalEncoder(encoding_method='ordered')),
#         ('model',CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             eval_metric = 'MAE', 
#                             random_seed = 123, 
#                             task_type="GPU"))]),
    
#     xgb_v1 = XGBRegressor(n_estimators=1500,
#                         objective='reg:squarederror',
#                         tree_method="gpu_hist",
#                         verbosity = 2,
#                         enable_categorical = True, 
#                         random_state = 123
#                         ),
    
#     lgb_v1 = Pipeline(steps = [
#             #('ord', OrdinalEncoder(encoding_method='ordered')),
#             #('mi', AddMissingIndicator()),
#             #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('model', LGBMRegressor(n_estimators=500,
#                                     device="gpu"))
#             ])
      eval_metric = 'MAE', 
#                             random_seed = 123, 
#                             task_type="GPU"))]),
    
#     xgb_v1 = XGBRegressor(n_estimators=1500,
#                         objective='reg:squarederror',
#                         tree_method="gpu_hist",
#                         verbosity = 2,
#                         enable_categorical = True, 
#                         random_state = 123
#                         ),
    
#     lgb_v1 = Pipeline(steps = [
#             #('ord', OrdinalEncoder(encoding_method='ordered')),
#             #('mi', AddMissingIndicator()),
#             #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('model', LGBMRegressor(n_estimators=500,
#                                     device="gpu"))
#             ])
    
#     cb_v4 = Pipeline(steps = [
#         #('ord', OrdinalEncoder(encoding_method='ordered')),
#         ('model',CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             eval_metric = 'MAE', 
#                             random_seed = 123, 
#                             task_type="GPU"))]),
    
#     xgb_v1 = XGBRegressor(n_estimators=1500,
#                         objective='reg:squarederror',
#                         tree_method="gpu_hist",
#                         verbosity = 2,
#                         enable_categorical = True, 
#                         random_state = 123
#                         ),
    
#     lgb_v1 = Pipeline(steps = [
#             #('ord', OrdinalEncoder(encoding_method='ordered')),
#             #('mi', AddMissingIndicator()),
#             #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('model', LGBMRegressor(n_estimators=500,
#                                     device="gpu"))
#             ])
    coder(encoding_method='ordered')),
#         ('model',CatBoostRegressor(iterations=1500,
#                             loss_function='MAE',
#                             eval_metric = 'MAE', 
#                             random_seed = 123, 
#                             task_type="GPU"))]),
    
#     xgb_v1 = XGBRegressor(n_estimators=1500,
#                         objective='reg:squarederror',
#                         tree_method="gpu_hist",
#                         verbosity = 2,
#                         enable_categorical = True, 
#                         random_state = 123
#                         ),
    
#     lgb_v1 = Pipeline(steps = [
#             #('ord', OrdinalEncoder(encoding_method='ordered')),
#             #('mi', AddMissingIndicator()),
#             #('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
#             ('model', LGBMRegressor(n_estimators=500,
#                                     device="gpu"))
#             ])
    raryNumberImputer(arbitrary_number = 0)),
#             ('model', LGBMRegressor(n_estimators=500,
#                                     device="gpu"))
#             ])
    
#     )
