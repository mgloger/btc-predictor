import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # Market Data
    "COINGECKO_API_KEY": os.getenv("COINGECKO_API_KEY"),
    "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
    "BINANCE_SECRET": os.getenv("BINANCE_SECRET"),

    # On-Chain Data
    "GLASSNODE_API_KEY": os.getenv("GLASSNODE_API_KEY"),
    "CRYPTOQUANT_API_KEY": os.getenv("CRYPTOQUANT_API_KEY"),

    # Macro Data
    "FRED_API_KEY": os.getenv("FRED_API_KEY"),  # Federal Reserve Economic Data

    # Sentiment / News
    "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
    "TWITTER_BEARER_TOKEN": os.getenv("TWITTER_BEARER_TOKEN"),

    # ETF Flow Data
    "SOSOVALUE_API_KEY": os.getenv("SOSOVALUE_API_KEY"),

    # Model Settings
    "LOOKBACK_DAYS": 90,
    "PREDICTION_HORIZON_DAYS": [7, 30, 90],
    "RETRAIN_INTERVAL_HOURS": 24,
}