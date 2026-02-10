import requests
import pandas as pd
from transformers import pipeline


class SentimentCollector:
    """Collects and analyzes sentiment from news and social media."""

    def __init__(self, config):
        self.news_key = config["NEWS_API_KEY"]
        # Use a pre-trained financial sentiment model
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
        )

    def get_crypto_news(self, query="bitcoin", days=7) -> list[dict]:
        """Fetch recent crypto news articles."""
        from datetime import datetime, timedelta
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        resp = requests.get("https://newsapi.org/v2/everything", params={
            "q": query,
            "from": from_date,
            "sortBy": "relevance",
            "language": "en",
            "apiKey": self.news_key,
        })
        resp.raise_for_status()
        return resp.json().get("articles", [])

    def analyze_sentiment(self, texts: list[str]) -> pd.DataFrame:
        """Run FinBERT sentiment analysis on a batch of texts."""
        results = self.sentiment_model(texts, batch_size=16)
        df = pd.DataFrame(results)
        # Map to numeric: positive=1, neutral=0, negative=-1
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        df["sentiment_score"] = df["label"].map(label_map) * df["score"]
        return df

    def get_fear_greed_index(self) -> dict:
        """Fetch the Crypto Fear & Greed Index."""
        resp = requests.get("https://api.alternative.me/fng/?limit=30&format=json")
        resp.raise_for_status()
        return resp.json()["data"]

    def get_aggregated_sentiment(self) -> float:
        """Get a single aggregated sentiment score (-1 to +1)."""
        articles = self.get_crypto_news()
        if not articles:
            return 0.0
        titles = [a["title"] for a in articles if a.get("title")]
        sentiment_df = self.analyze_sentiment(titles)
        return sentiment_df["sentiment_score"].mean()