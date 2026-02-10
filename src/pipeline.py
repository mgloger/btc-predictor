import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_collectors.market import MarketDataCollector
from src.data_collectors.onchain import OnChainCollector
from src.data_collectors.macro import MacroDataCollector
from src.data_collectors.sentiment import SentimentCollector
from src.features.engineer import FeatureEngineer
from src.models.lstm import LSTMPredictor
from src.models.xgboost_model import GradientBoostPredictor
from src.models.ensemble import EnsemblePredictor
from src.utils.config import CONFIG


class BTCPredictionPipeline:
    """End-to-end Bitcoin price prediction pipeline."""

    def __init__(self):
        self.market_collector = MarketDataCollector(CONFIG)
        self.onchain_collector = OnChainCollector(CONFIG)
        self.macro_collector = MacroDataCollector(CONFIG)
        self.sentiment_collector = SentimentCollector(CONFIG)

        self.lstm = LSTMPredictor(lookback=CONFIG["LOOKBACK_DAYS"])
        self.xgb = GradientBoostPredictor("xgboost")
        self.lgbm = GradientBoostPredictor("lightgbm")
        self.ensemble = EnsemblePredictor()

        self.scaler = StandardScaler()

    def collect_data(self) -> dict:
        """Gather data from all sources."""
        print("📊 Collecting market data...")
        market = self.market_collector.get_ohlcv(days=730)

        print("⛓️ Collecting on-chain data...")
        onchain = {
            "active_addresses": self.onchain_collector.get_active_addresses(),
            "exchange_netflow": self.onchain_collector.get_exchange_netflow(),
            "mvrv": self.onchain_collector.get_mvrv_ratio(),
            "hash_rate": self.onchain_collector.get_hash_rate(),
            "sopr": self.onchain_collector.get_sopr(),
        }

        print("🏦 Collecting macro data...")
        macro = {
            "fed_funds": self.macro_collector.get_fed_funds_rate(),
            "cpi": self.macro_collector.get_cpi(),
            "dxy": self.macro_collector.get_dxy(),
            "m2": self.macro_collector.get_m2_money_supply(),
            "sp500": self.macro_collector.get_sp500(),
            "treasury_10y": self.macro_collector.get_treasury_10y(),
        }

        print("📰 Collecting sentiment data...")
        sentiment = self.sentiment_collector.get_aggregated_sentiment()

        return {
            "market": market,
            "onchain": onchain,
            "macro": macro,
            "sentiment": sentiment,
        }

    def build_features(self, data: dict) -> pd.DataFrame:
        """Engineer features from raw data."""
        engineer = FeatureEngineer(
            market_df=data["market"],
            onchain_data=data["onchain"],
            macro_data=data["macro"],
            sentiment_score=data["sentiment"],
        )
        return engineer.build_feature_matrix()

    def train_and_predict(self, features: pd.DataFrame) -> dict:
        """Train all models and generate ensemble prediction."""
        # Prepare target: next-day close price
        target = features["close"].shift(-1).dropna()
        features = features.iloc[:-1]

        # Scale features
        feature_cols = [c for c in features.columns if c != "close"]
        X = self.scaler.fit_transform(features[feature_cols])
        y = target.values

        # Time-series train/test split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # --- Train LSTM ---
        print("🧠 Training LSTM...")
        X_seq_train, y_seq_train = self.lstm.prepare_sequences(
            np.column_stack([y_train.reshape(-1, 1), X_train])
        )
        X_seq_test, y_seq_test = self.lstm.prepare_sequences(
            np.column_stack([y_test.reshape(-1, 1), X_test])
        )
        self.lstm.train(X_seq_train, y_seq_train)
        lstm_preds = self.lstm.predict(X_seq_test)

        # --- Train XGBoost ---
        print("🌳 Training XGBoost...")
        self.xgb.train(X_train, y_train, X_test, y_test)
        xgb_preds = self.xgb.predict(X_test)

        # --- Train LightGBM ---
        print("💡 Training LightGBM...")
        self.lgbm.train(X_train, y_train, X_test, y_test)
        lgbm_preds = self.lgbm.predict(X_test)

        # Align predictions (LSTM has shorter output due to lookback)
        min_len = min(len(lstm_preds), len(xgb_preds), len(lgbm_preds))
        predictions = {
            "lstm": lstm_preds[-min_len:],
            "xgboost": xgb_preds[-min_len:],
            "lightgbm": lgbm_preds[-min_len:],
        }
        actuals = y_test[-min_len:]

        # --- Train Ensemble ---
        print("🤝 Training Ensemble...")
        self.ensemble.train(predictions, actuals)

        return self.ensemble.predict_with_confidence(predictions)

    def run(self) -> dict:
        """Execute the full prediction pipeline."""
        print("🚀 Starting BTC Prediction Pipeline...")
        data = self.collect_data()
        features = self.build_features(data)
        result = self.train_and_predict(features)
        print(f"\n✅ Prediction: ${result['prediction']:,.2f}")
        print(f"   Range: ${result['confidence_interval']['low']:,.2f} - "
              f"${result['confidence_interval']['high']:,.2f}")
        print(f"   Model Agreement: {result['model_agreement']:.2%}")
        return result