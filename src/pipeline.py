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
        self.onchain_collector = OnChainCollector()
        self.macro_collector = MacroDataCollector(CONFIG)
        self.sentiment_collector = SentimentCollector(CONFIG)

        # LSTM predictor will be initialized after we know the data size
        self.lstm = None
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
            "transaction_count": self.onchain_collector.get_transaction_count(),
            "transaction_volume": self.onchain_collector.get_transaction_volume(),
            "miner_revenue": self.onchain_collector.get_miner_revenue(),
            "hash_rate": self.onchain_collector.get_hash_rate(),
            "mempool_count": self.onchain_collector.get_mempool_count(),
            "market_cap": self.onchain_collector.get_market_cap(),
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

    def _calculate_lookback(self, total_rows: int) -> int:
        """
        Dynamically calculate LSTM lookback based on available data.
        
        Rules:
            - Need at least 20 test sequences for meaningful evaluation
            - Need at least 30 train sequences for LSTM to learn
            - Test set = 20% of data (minimum 20 + lookback rows)
            - Lookback must leave enough room for both train and test sequences
        """
        min_test_sequences = 20
        min_train_sequences = 30

        # Start with configured lookback, reduce if necessary
        lookback = CONFIG.get("LOOKBACK_DAYS", 90)

        # Max lookback that allows enough train + test sequences
        # total_rows = train_rows + test_rows
        # train_sequences = train_rows - lookback >= min_train_sequences
        # test_sequences = test_rows - lookback >= min_test_sequences
        # So: lookback <= (total_rows - min_train_sequences - min_test_sequences) / 2
        max_lookback = (total_rows - min_train_sequences - min_test_sequences) // 2

        if max_lookback < 5:
            raise ValueError(
                f"Not enough data rows ({total_rows}) to train any model. "
                f"Need at least {min_train_sequences + min_test_sequences + 10} = "
                f"{min_train_sequences + min_test_sequences + 10} rows. "
                f"Try fetching more historical data or reducing on-chain features."
            )

        lookback = min(lookback, max_lookback)
        lookback = max(lookback, 5)  # Absolute minimum

        print(f"📐 Lookback: {lookback} (configured: {CONFIG.get('LOOKBACK_DAYS', 90)}, "
              f"max possible: {max_lookback}, data rows: {total_rows})")

        return lookback

    def train_and_predict(self, features: pd.DataFrame) -> dict:
        """Train all models and generate ensemble prediction."""

        # Prepare target: next-day close price
        target = features["close"].shift(-1).dropna()
        features = features.iloc[:-1]

        # Scale features
        feature_cols = [c for c in features.columns if c != "close"]
        X = self.scaler.fit_transform(features[feature_cols])
        y = target.values

        total_rows = len(X)
        print(f"📋 Total data rows: {total_rows}, features: {X.shape[1]}")

        # ─── Dynamically calculate lookback ───
        lookback = self._calculate_lookback(total_rows)
        self.lstm = LSTMPredictor(lookback=lookback, epochs=100, lr=0.001)

        # ─── Split data ───
        # Ensure test set has enough rows for lookback + meaningful sequences
        min_test_sequences = 20
        min_test_rows = lookback + min_test_sequences

        test_rows = max(min_test_rows, int(total_rows * 0.2))
        test_rows = min(test_rows, int(total_rows * 0.4))
        split = total_rows - test_rows

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"📐 Split: {split} train rows / {test_rows} test rows")
        print(f"   Expected LSTM train sequences: {len(X_train) - lookback}")
        print(f"   Expected LSTM test sequences: {len(X_test) - lookback}")

        # ─── LSTM ───
        print("🧠 Training LSTM...")
        train_data = np.column_stack([y_train.reshape(-1, 1), X_train])
        test_data = np.column_stack([y_test.reshape(-1, 1), X_test])

        X_seq_train, y_seq_train = self.lstm.prepare_sequences(train_data)
        X_seq_test, y_seq_test = self.lstm.prepare_sequences(test_data)

        if X_seq_train.shape[0] == 0 or X_seq_test.shape[0] == 0:
            raise ValueError(
                f"Not enough sequences! Train: {X_seq_train.shape}, Test: {X_seq_test.shape}. "
                f"Data rows: {total_rows}, lookback: {lookback}"
            )

        self.lstm.train(X_seq_train, y_seq_train)
        lstm_preds = self.lstm.predict(X_seq_test)

        # ─── XGBoost ───
        print("🌳 Training XGBoost...")
        self.xgb.train(X_train, y_train, X_test, y_test)
        xgb_preds = self.xgb.predict(X_test)

        # ─── LightGBM ───
        print("💡 Training LightGBM...")
        self.lgbm.train(X_train, y_train, X_test, y_test)
        lgbm_preds = self.lgbm.predict(X_test)

        # ─── Align predictions ───
        # LSTM produces fewer predictions than XGBoost/LightGBM
        # because it needs `lookback` rows to form the first sequence
        min_len = min(len(lstm_preds), len(xgb_preds), len(lgbm_preds))
        predictions = {
            "lstm": lstm_preds[-min_len:],
            "xgboost": xgb_preds[-min_len:],
            "lightgbm": lgbm_preds[-min_len:],
        }
        actuals = y_test[-min_len:]

        print(f"🤝 Aligned {min_len} predictions across all models")

        # ─── Ensemble ───
        print("🤝 Training Ensemble...")
        self.ensemble.train(predictions, actuals)

        return self.ensemble.predict_with_confidence(predictions)

    def run(self) -> dict:
        """Execute the full prediction pipeline."""
        print("🚀 Starting BTC Prediction Pipeline...")
        data = self.collect_data()
        features = self.build_features(data)

        print(f"📋 Feature matrix: {features.shape[0]} rows × {features.shape[1]} columns")
        print(f"   Date range: {features.index.min()} → {features.index.max()}")

        result = self.train_and_predict(features)
        print(f"\n✅ Prediction: ${result['prediction']:,.2f}")
        print(f"   Range: ${result['confidence_interval']['low']:,.2f} - "
              f"${result['confidence_interval']['high']:,.2f}")
        print(f"   Model Agreement: {result['model_agreement']:.2%}")
        return result