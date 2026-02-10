import pandas as pd
import numpy as np
import ta  # Technical Analysis library


class FeatureEngineer:
    """Combines all data sources into a unified feature set for ML models."""

    def __init__(self, market_df: pd.DataFrame, onchain_data: dict,
                 macro_data: dict, sentiment_score: float):
        self.market = market_df
        self.onchain = onchain_data
        self.macro = macro_data
        self.sentiment = sentiment_score

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical analysis indicators."""
        # Trend
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
        df["sma_200"] = ta.trend.sma_indicator(df["close"], window=200)
        df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
        df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)
        df["macd"] = ta.trend.macd_diff(df["close"])

        # Momentum
        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
        df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"])

        # Volatility
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])

        # Volume
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        return df

    def add_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bitcoin halving cycle features."""
        halving_dates = [
            pd.Timestamp("2012-11-28"),
            pd.Timestamp("2016-07-09"),
            pd.Timestamp("2020-05-11"),
            pd.Timestamp("2024-04-19"),
        ]
        last_halving = max(d for d in halving_dates if d <= df.index.max())
        df["days_since_halving"] = (df.index - last_halving).days
        df["halving_cycle_phase"] = df["days_since_halving"] / (4 * 365)  # Normalized 0-1
        return df

    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns and volatility features."""
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_7d"] = df["log_return"].rolling(7).std()
        df["volatility_30d"] = df["log_return"].rolling(30).std()
        df["momentum_7d"] = df["close"].pct_change(7)
        df["momentum_30d"] = df["close"].pct_change(30)
        return df

    def build_feature_matrix(self) -> pd.DataFrame:
        """Combine all features into a single DataFrame."""
        df = self.market.copy()

        # Technical indicators
        df = self.add_technical_indicators(df)
        df = self.add_log_returns(df)
        df = self.add_cycle_features(df)

        # Merge on-chain data (resampled to daily)
        for name, onchain_df in self.onchain.items():
            onchain_daily = onchain_df.resample("1D").last().ffill()
            onchain_daily.columns = [f"onchain_{name}"]
            df = df.join(onchain_daily, how="left")

        # Merge macro data (forward-filled for missing days)
        for name, macro_df in self.macro.items():
            macro_daily = macro_df.resample("1D").last().ffill()
            macro_daily.columns = [f"macro_{name}"]
            df = df.join(macro_daily, how="left")

        # Add sentiment as a static feature (refreshed periodically)
        df["sentiment_score"] = self.sentiment

        # Forward-fill and drop initial NaNs
        df = df.ffill().dropna()

        return df