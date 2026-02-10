import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class GradientBoostPredictor:
    """Gradient boosting models for tabular feature-based prediction."""

    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        if self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                early_stopping_rounds=50,
            )
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model = lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
            )
            callbacks = [lgb.early_stopping(50)] if X_val is not None else None
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                callbacks=callbacks,
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> dict:
        if self.model_type == "xgboost":
            return dict(zip(
                self.model.get_booster().feature_names,
                self.model.feature_importances_,
            ))
        return dict(enumerate(self.model.feature_importances_))