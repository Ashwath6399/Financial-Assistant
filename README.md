# Trading Journal Analytics Dashboard

A comprehensive trading journal and analytics platform with machine learning-powered trade predictions. Built with Flask, SQLite, and multiple ML models including a hybrid CNN-XGBoost architecture for trade outcome prediction.

## Features

### Dashboard Tabs
- **Overview** - Portfolio summary, P&L metrics, benchmark comparison vs S&P 500
- **Account Performance** - Individual account analysis, equity curves, risk metrics
- **ML Models** - Machine learning predictions, confusion matrices, model accuracy
- **Trade Explorer** - Trade history, P&L distribution, win rate by symbol
- **Trade Screener** - Top opportunities, symbols to avoid, account rankings
- **Market Data** - Real-time price charts, historical data visualization (1-5 years)
- **Database** - ER diagram, schema browser, SQL query interface

### Analytics Features
- Win rate analysis and Sharpe ratio calculation
- Maximum drawdown tracking
- Benchmark comparison against S&P 500
- Risk-adjusted returns (Sortino ratio, Calmar ratio)
- Per-symbol and per-account performance breakdown

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-journal.git
cd trading-journal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The database will be **automatically downloaded** from Google Drive on first run (~881 MB).

Open your browser to: **http://127.0.0.1:5000**

---

## Machine Learning Models

### Model 1: XGBoost Trade Classifier
Standalone gradient boosting model for trade outcome prediction.
- **Features**: 18 engineered features (RSI, MACD, Bollinger Bands, momentum, etc.)
- **Task**: Binary classification (profitable vs losing trade)

### Model 2: 1D CNN Price Predictor
Convolutional neural network for price direction prediction.
- **Architecture**: Multi-layer 1D CNN with dilated convolutions
- **Advantage**: 10-50x faster than LSTM due to parallel processing
- **Input**: 60-day sequences with technical indicators

### Model 3: Hybrid CNN-XGBoost
Novel architecture combining deep learning with gradient boosting:
- **Stage 1**: 1D CNN extracts temporal features from price data
- **Stage 2**: XGBoost classifier uses CNN features + engineered features
- **Multi-horizon**: Predictions for 1-day, 3-day, and 7-day horizons

---

## Project Structure

```
trading-journal/
├── app.py                      # Flask web server and API endpoints
├── config.py                   # Configuration and database auto-download
├── requirements.txt            # Python dependencies
│
├── templates/
│   └── dashboard.html          # Main dashboard UI
│
├── static/
│   └── app.js                  # Frontend JavaScript
│
├── ml_models/
│   ├── xgboost_trade_classifier.py  # XGBoost model
│   ├── cnn_price_predictor.py       # 1D CNN model
│   ├── hybrid_model.py              # Hybrid CNN-XGBoost model
│   ├── lstm_price_predictor.py      # LSTM model (legacy)
│   ├── apply_predictions.py         # Apply predictions to database
│   ├── compare_models.py            # Model comparison utilities
│   └── saved_models/                # Trained model files
│       ├── hybrid_model.pkl
│       ├── xgboost_trade_classifier.pkl
│       ├── cnn_SPY_best.keras
│       └── ...
│
├── data_ingestion/
│   ├── fetch_historical_data.py   # Download market data via yfinance
│   ├── generate_realistic_trades.py
│   └── expand_database.py
│
├── analytics/
│   ├── benchmark_comparison.py    # S&P 500 comparison
│   ├── risk_analytics.py          # Risk metrics calculation
│   ├── account_reports.py         # Account-level analytics
│   └── advanced_metrics.py        # Sharpe, Sortino, etc.
│
└── Trading_Journal_ML.db          # SQLite database (auto-downloaded)
```

---

## Database Schema

The SQLite database contains 11 tables:

### Core Tables
| Table | Description |
|-------|-------------|
| `accounts` | Trading accounts |
| `brokers` | Broker information |
| `instruments` | Traded symbols (stocks, ETFs, crypto) |
| `trades` | Trade records |
| `fills` | Order fills/executions |
| `orders` | Order history |
| `prices` | OHLCV price data |

### Analytics Tables
| Table | Description |
|-------|-------------|
| `account_metrics` | Per-account performance metrics |
| `benchmark_comparison` | S&P 500 comparison results |
| `risk_analytics` | Risk metrics (VaR, drawdown, etc.) |
| `ml_predictions` | ML model predictions |

---

## API Endpoints

### Overview
- `GET /api/overview/summary` - Portfolio summary statistics
- `GET /api/overview/pnl-timeline` - Daily P&L timeline
- `GET /api/overview/asset-allocation` - Asset allocation breakdown

### Account Performance
- `GET /api/accounts` - List all accounts with metrics
- `GET /api/account/<id>/trades` - Trades for specific account
- `GET /api/account/<id>/equity-curve` - Equity curve data

### ML Predictions
- `GET /api/ml/stats` - ML model statistics
- `GET /api/ml/confusion-matrix` - Confusion matrix data
- `GET /api/ml/predictions` - All predictions with filters

### Market Data
- `GET /api/market/symbols` - Available symbols
- `GET /api/market/prices/<symbol>` - Price history

### Database
- `GET /api/database/schema` - Database schema
- `POST /api/database/query` - Execute SQL query

---

## Technologies Used

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **SQLite** - Database
- **Pandas** - Data manipulation

### Machine Learning
- **TensorFlow/Keras** - 1D CNN deep learning
- **XGBoost** - Gradient boosting
- **scikit-learn** - ML utilities

### Frontend
- **HTML/CSS/JavaScript**
- **Chart.js** - Interactive charts

### Data Sources
- **yfinance** - Yahoo Finance market data

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/trading-journal.git
   cd trading-journal
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

### For Mac M1/M2 Users
```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

---

## Troubleshooting

### Database Not Found
The database auto-downloads on first run. If it fails:
- Check internet connection
- Install gdown: `pip install gdown`

### Port 5000 in Use
```bash
lsof -ti:5000 | xargs kill -9
```

### TensorFlow Issues on Mac
```bash
pip uninstall tensorflow
pip install tensorflow-macos tensorflow-metal
```

---

## License

This project is for educational purposes.

---

*Last Updated: December 2024*
