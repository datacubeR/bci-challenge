import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer


from sklearn.model_selection import KFold
from lofo import LOFOImportance, plot_importance, Dataset

df_train = pd.read_csv('data/train_full.csv')
target = ['target_mes']
cat_vars = ['tipo_ban','tipo_seg','categoria','tipo_com','tipo_cat','tipo_cli','month','year']   
int_variables = df_train.filter(like = '_trx').columns.tolist()
float_variables = [vars for vars in df_train.columns if vars not in int_variables + cat_vars + target]
float_variables.remove('mes')
float_variables.remove('id')

cv = KFold(n_splits=5, shuffle=True, random_state=123)

imputation = MeanMedianImputer(imputation_method='mean')
imp_df = pd.concat([imputation.fit_transform(
    df_train[int_variables + float_variables] ),
    df_train.target_mes], axis = 1
)

data = Dataset(df = imp_df,
            target = 'target_mes',
            features = int_variables + float_variables
)

xgb = xgb.XGBRegressor(objective='reg:squarederror',
                        tree_method="gpu_hist")

lofo_imp = LOFOImportance(data, cv = cv, 
                        scoring = 'neg_mean_absolute_error', 
                        model = xgb)

importance_df = lofo_imp.get_importance()
print(importance_df)
plot_importance(importance_df, figsize=(12, 20))
importance_df.to_csv('data/lofo_xgb.csv')
