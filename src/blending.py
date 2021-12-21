import pandas as pd
import glob 

files = glob.glob('submissions/*.csv')

df_test = pd.read_csv("input/test_data.csv")
preds_test = df_test[['id','mes']]

for f in files:
    data = pd.read_csv(f)
    preds_test = preds_test.merge(data, on = ['id','mes'], how = 'left')

id = preds_test[['id','mes']]
preds = preds_test.drop(columns = ['id','mes']).mean(axis = 1)

sub = pd.concat([id, preds], axis = 1)
sub.columns = ['id','mes','target_mes']

sub.to_csv('submissions/blending.csv', index = False)


# import pandas as pd
# import numpy as np
# import glob
# from sklearn.metrics import mean_absolute_error
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor

# files = glob.glob('preds/*.csv')
# df = pd.read_csv('input/train_folds.csv')

# for f in files:
#     data = pd.read_csv(f)
#     df = df.merge(data, on = ['id','mes','kfold'], how = 'left')
    
# columns = df.filter(like = 'pred').columns.tolist()
# final = df[['target_mes','kfold']+columns]

# score_blending = []
# score_weighted = []
# for fold in range(5):
#     X = final.query('kfold == @fold').drop(columns = ['kfold','target_mes'])
#     y = final.query('kfold == @fold').target_mes
    
#     preds_blending = X.mean(axis = 1)
#     preds_weighted = (3*X.iloc[:,0] + 1*X.iloc[:,1] + 3*X.iloc[:,2] + 3*X.iloc[:,3] + 1*X.iloc[:,4])/11
    
#     mae_blending = mean_absolute_error(preds_blending, y)
#     mae_weighted = mean_absolute_error(preds_weighted, y)
#     print(f'MAE Blending for Fold {fold}: {mae_blending}')
#     print(f'MAE Weighted for Fold {fold}: {mae_weighted}')
#     score_blending.append(mae_blending)
#     score_weighted.append(mae_weighted)
    
    
    
    
# print(f'Mean Score for Blending: {np.mean(score_blending)}')
# print(f'Mean Score for Weighted: {np.mean(score_weighted)}')
# print(files)


# df_test = pd.read_csv("input/test_data.csv")
# preds_test = df_test[['id','mes']]

# files = []