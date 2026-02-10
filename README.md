# 📈 BTC Price Predictor

A Bitcoin price prediction application that combines **AI/ML models**, **on-chain analytics**, **macroeconomic data**, and **sentiment analysis** to generate ensemble price forecasts with confidence intervals.

## 🏗️ Architecture

```
Data Sources → Data Pipeline → Feature Engineering → ML Models → Ensemble → API
```

### Data Sources
| Source | Provider | Signals |
|--------|----------|---------|
| Market Data | Binance (ccxt) | OHLCV, order book, funding rates |
| On-Chain | Glassnode | MVRV, SOPR, exchange flows, hash rate, active addresses |
| Macro | FRED API | Fed Funds rate, CPI, DXY, M2 supply, S&P 500, 10Y Treasury |
| Sentiment | NewsAPI + FinBERT | News sentiment scoring, Fear & Greed Index |

### ML Models
- **LSTM with Attention** (PyTorch) — deep learning for sequential patterns
- **XGBoost** — gradient boosting on tabular features
- **LightGBM** — fast gradient boosting alternative
- **Ensemble (Stacking)** — Ridge meta-learner combining all models

## 📁 Project Structure

```
btc-predictor/
├── .env                          # API keys (not committed)
├── .gitignore
├── requirements.txt
├── main.py                       # Entry point
└── src/
    ├── pipeline.py               # End-to-end prediction pipeline
    ├── data_collectors/
    │   ├── market.py             # Price, volume, order book
    │   ├── onchain.py            # Blockchain analytics
    │   ├── macro.py              # Macroeconomic indicators
    │   └── sentiment.py          # News & social sentiment
    ├── features/
    │   └── engineer.py           # Feature engineering & technical indicators
    ├── models/
    │   ├── lstm.py               # LSTM with attention
    │   ├── xgboost_model.py      # XGBoost & LightGBM
    │   └── ensemble.py           # Stacking ensemble
    ├── api/
    │   └── server.py             # FastAPI REST API
    └── utils/
        └── config.py             # Configuration management
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- API keys for the data providers (see below)

### Installation

```bash
git clone https://github.com/mgloger/btc-predictor.git
cd btc-predictor
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Key | Provider | Sign Up |
|-----|----------|---------|
| `BINANCE_API_KEY` / `BINANCE_SECRET` | Binance | [binance.com](https://www.binance.com/) |
| `GLASSNODE_API_KEY` | Glassnode | [glassnode.com](https://glassnode.com/) |
| `FRED_API_KEY` | Federal Reserve (FRED) | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `NEWS_API_KEY` | NewsAPI | [newsapi.org](https://newsapi.org/) |

### Usage

#### Run a single prediction
```bash
python main.py
```

#### Run as API server
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/predict` | Get latest prediction with confidence intervals |
| `POST` | `/retrain` | Trigger model retraining in background |
| `GET` | `/health` | Health check |

Example response from `/predict`:
```json
{
  "prediction": 98542.30,
  "confidence_interval": {
    "low": 87200.00,
    "high": 112400.00
  },
  "model_agreement": 0.87,
  "model_weights": {
    "lstm": 0.42,
    "xgboost": 0.35,
    "lightgbm": 0.23
  }
}
```

## 🔧 Features

- **50+ engineered features** — technical indicators (RSI, MACD, Bollinger Bands), on-chain metrics, macro signals, and sentiment scores
- **Bitcoin halving cycle tracking** — days since halving normalized as cycle phase
- **Automated retraining** — models retrain every 24 hours via scheduler
- **Confidence intervals** — predictions include low/high range and model agreement score
- **Feature importance** — identify which signals drive predictions the most

## 🗺️ Roadmap

- [ ] Add Bitcoin spot ETF inflow/outflow tracking
- [ ] Plotly/Dash interactive dashboard
- [ ] Walk-forward backtesting framework
- [ ] Docker containerization
- [ ] Telegram/Discord alert notifications
- [ ] Multi-timeframe predictions (7d, 30d, 90d)

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. No model can reliably predict cryptocurrency prices. Never make financial decisions based solely on algorithmic predictions. Always do your own research and consult a licensed financial advisor.

## 📄 License

MIT