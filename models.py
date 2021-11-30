from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from feature_engine.encoding import OneHotEncoder

def ridge_v1(alpha, 
            random_state,
            trial = None):
    
    if trial is not None:
        alpha = trial.suggest_float('alpha', **alpha)
        
    pipe = Pipeline(steps = [
            ('oe', OneHotEncoder()),
            ('model', Ridge(alpha = alpha, 
                            random_state=random_state))
    ])
    
    return pipe