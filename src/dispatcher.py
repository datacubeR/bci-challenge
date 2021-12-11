from sklearn.pipeline import Pipeline
from feature_engine.encoding import OrdinalEncoder,OneHotEncoder
from feature_engine.imputation import ArbitraryNumberImputer, AddMissingIndicator
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from feature_engine.wrappers import SklearnTransformerWrapper

cat_vars = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli','month','year']
scaler = SklearnTransformerWrapper(StandardScaler())

MODELS = dict(
    cb_v1 = Pipeline(steps = [
            ('enc', OrdinalEncoder(encoding_method='arbitrary')),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', StandardScaler()),
            ('model', CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0'))
            ]),
    cb_v2 = Pipeline(steps = [
            ('enc', OneHotEncoder()),
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', StandardScaler()),
            ('model', CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0'))
            ]),
    cb_v3 = Pipeline(steps = [
            ('mi', AddMissingIndicator()),
            ('imp', ArbitraryNumberImputer(arbitrary_number = 0)),
            ('sc', scaler),
            ('model', CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0'))
            ]),
    cb_v4 = CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0')
    )