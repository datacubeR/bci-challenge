import pandas as pd
import lightgbm as lgbm
import xgboost as xgb
from BorutaShap import BorutaShap
from feature_engine.imputation import MeanMedianImputer
from feature_engine.creation import CyclicalTransformer
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor

df_train = pd.read_csv('data/train_full.csv')
id = df_train[['id','mes']]
target = ['target_mes']
cat_vars = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli']   
int_variables = df_train.filter(like = '_trx').columns.tolist()
float_variables = [vars for vars in df_train.columns if vars not in int_variables + cat_vars + target]

X = df_train[int_variables + float_variables].drop(columns = ['id','mes'])
y = df_train.target_mes

xgb = xgb.XGBRegressor(objective='reg:squarederror',
                        tree_method="gpu_hist")

cb = CatBoostRegressor(iterations=1500,
                            loss_function='MAE',
                            #learning_rate=0.5,
                            #eval_metric = 'MAE', 
                            random_seed = 123, 
                            task_type="GPU",
                            devices='0')

lgbm = lgbm.LGBMRegressor(n_jobs=-1, device="gpu")
pipe = Pipeline(steps=[
    ('imp_num', MeanMedianImputer(imputation_method='mean')),
    ('cyc', CyclicalTransformer(variables = ['month'])),
    ('out', Winsorizer(capping_method='quantiles', tail = 'right', fold=0.05)),
    ('fs', BorutaShap(model = xgb,
                            importance_measure='shap',
                            classification=False))
])

pipe.fit(X, y, fs__n_trials = 30, fs__sample=True,
                    fs__train_or_test='test', fs__normalize=True,
                    fs__verbose=True)

print('The Selected Features are: ' )
df_final = pipe.named_steps.fs.Subset()
print(df_final)
pd.concat([id,df_final, df_train[cat_vars], y], axis = 1).to_csv('data/Boruta_Shap_cb.csv', index = False)
