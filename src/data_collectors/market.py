import ccxt
import pandas as pd
import requests
from datetime import datetime, timedelta


class MarketDataCollector:
    """Collects price, volume, and order book data from exchanges."""

    def __init__(self, config):
        self.exchange = ccxt.binance({
            "apiKey": config["BINANCE_API_KEY"],
            "secret": config["BINANCE_SECRET"],
        })

    def get_ohlcv(self, symbol="BTC/USDT", timeframe="1d", days=365) -> pd.DataFrame:
        """Fetch historical OHLCV candles."""
        since = self.exchange.parse8601(
            (datetime.utcnow() - timedelta(days=days)).isoformat()
        )
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_order_book(self, symbol="BTC/USDT", limit=100) -> dict:
        """Fetch current order book for bid/ask depth analysis."""
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def get_funding_rate(self, symbol="BTC/USDT") -> float:
        """Fetch perpetual futures funding rate (sentiment proxy)."""
        funding = self.exchange.fetch_funding_rate(symbol)
        return funding["fundingRate"]