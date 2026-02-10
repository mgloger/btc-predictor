import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit


class EnsemblePredictor:
    """
    Meta-learner that combines predictions from LSTM, XGBoost, and LightGBM
    using a stacking approach with time-series cross-validation.
    """

    def __init__(self):
        self.meta_model = Ridge(alpha=1.0)
        self.model_weights = None

    def train(self, predictions: dict[str, np.ndarray], actuals: np.ndarray):
        """
        Train the meta-learner on base model predictions.

        Args:
            predictions: {"lstm": [...], "xgboost": [...], "lightgbm": [...]}
            actuals: true price values
        """
        # Stack predictions as features for the meta-model
        X_meta = np.column_stack(list(predictions.values()))

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(X_meta):
            self.meta_model.fit(X_meta[train_idx], actuals[train_idx])
            score = self.meta_model.score(X_meta[val_idx], actuals[val_idx])
            scores.append(score)

        # Final fit on all data
        self.meta_model.fit(X_meta, actuals)
        self.model_weights = dict(zip(predictions.keys(), self.meta_model.coef_))
        print(f"Ensemble CV R² scores: {scores}")
        print(f"Model weights: {self.model_weights}")

    def predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        X_meta = np.column_stack(list(predictions.values()))
        return self.meta_model.predict(X_meta)

    def predict_with_confidence(self, predictions: dict[str, np.ndarray]) -> dict:
        """Return prediction with confidence intervals."""
        all_preds = np.column_stack(list(predictions.values()))
        ensemble_pred = self.meta_model.predict(all_preds)
        individual_preds = np.array(list(predictions.values()))

        return {
            "prediction": float(ensemble_pred[-1]),
            "confidence_interval": {
                "low": float(np.percentile(individual_preds[:, -1], 10)),
                "high": float(np.percentile(individual_preds[:, -1], 90)),
            },
            "model_agreement": float(1 - np.std(individual_preds[:, -1]) / np.mean(individual_preds[:, -1])),
            "model_weights": self.model_weights,
        }