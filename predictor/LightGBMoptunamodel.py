import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import joblib


class LightGBM():
    def __init__(self, config=None):
        self.config = config
        self.model = None

    def fit(self, x_train, y_train):
        x_lgbtrain, x_lgbeval, y_lgbtrain, y_lgbeval = train_test_split(
            x_train, y_train, test_size=0.3, shuffle=True, random_state=314, stratify=y_train
        )
        def objective(trial):
            param = {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
              }

            scores = cross_val_score(lgb.LGBMClassifier(**param),
                                 x_train,
                                 y_train,
                                 cv=3,
                                 scoring='f1_macro')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        best_params = study.best_trial.params
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(x_train, y_train)
        
    def predict(self, x_test):
        return pd.DataFrame({'prediction': self.model.predict(x_test)}, index=x_test.index)

    def dump(self, filename="LightGBM"):
        joblib.dump(self, f"results/model/{filename}.model")