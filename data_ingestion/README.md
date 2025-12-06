# Data Migration Guide

This directory contains scripts to migrate from synthetic data to real-world market data.

## Overview

1. **fetch_historical_data.py** - Downloads 5 years of real market data from Yahoo Finance
2. **generate_realistic_trades.py** - Creates realistic trading activity using the real price data

## Quick Start

### Step 1: Install Dependencies

```bash
pip install yfinance pandas numpy
```

### Step 2: Download Real Market Data

```bash
python3 data_ingestion/fetch_historical_data.py
```

This will:
- Download 5 years of daily OHLCV data
- For 50+ stocks, ETFs, and cryptocurrencies
- ~60,000 price records total
- Takes about 2-5 minutes

**Symbols included:**
- **Tech**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, INTC, etc.
- **Finance**: JPM, BAC, GS, MS, C, WFC, BLK, SCHW
- **Consumer**: WMT, HD, NKE, MCD, SBUX, TGT, DIS
- **Healthcare**: JNJ, UNH, PFE, ABBV, TMO, ABT
- **Energy**: XOM, CVX, COP, SLB
- **ETFs**: SPY, QQQ, DIA, IWM, VTI, VOO, GLD, SLV, TLT, HYG
- **Crypto**: BTC-USD, ETH-USD, BNB-USD, ADA-USD, SOL-USD

### Step 3: Generate Realistic Trades

```bash
python3 data_ingestion/generate_realistic_trades.py
```

This will:
- Create 500-1000 realistic fills across 12 accounts
- Using 3 trading strategies: RSI, Moving Average, Random Walk
- Based on actual market prices
- Takes about 1-2 minutes

### Step 4: Verify Data

```bash
python3 trading_analytics.py
```

You should now see analytics calculated from real market data!

## What Gets Created

### Price Data
- **60,000+** price records (OHLCV data)
- **5 years** of history (2020-2025)
- **50+** different symbols
- Source: Yahoo Finance via yfinance

### Trading Activity
- **500-1000** fills (buy/sell transactions)
- **12** simulated accounts
- **3** different strategies:
  - **RSI Strategy**: Buy when oversold (RSI < 30), sell when overbought (RSI > 70)
  - **MA Strategy**: Moving average crossover (20/50 day)
  - **Random Walk**: Simulates discretionary trading

### Database Tables Updated
- `instruments` - New instruments added
- `prices` - Real market data (source='yfinance')
- `import_batches` - Tracking of data imports
- `orders` - Buy/sell orders from strategies
- `fills` - Actual trade executions

## Data Quality Verification

The scripts include built-in data quality checks:

### Price Data Checks
- ✅ Date range validation
- ✅ Missing data gap detection
- ✅ Price sanity checks (no negatives, no extreme values)
- ✅ Volume validation

### Trade Data Checks
- ✅ All trades match real prices
- ✅ Fees included ($0.50 per trade)
- ✅ Proper FIFO accounting
- ✅ Realistic position sizing (10% of capital)

## Customization

### Add More Symbols

Edit `fetch_historical_data.py`, line 24-42:

```python
SYMBOLS = {
    'custom': ['COIN', 'SQ', 'SHOP'],  # Add your symbols here
    # ...
}
```

### Adjust Trading Parameters

Edit `generate_realistic_trades.py`, line 22-25:

```python
INITIAL_CAPITAL = 10000  # Starting capital
POSITION_SIZE_PCT = 0.1  # 10% per trade
MAX_POSITIONS = 5        # Max concurrent positions
COMMISSION = 0.50        # Fees per trade
```

### Change Strategy Mix

Edit `generate_realistic_trades.py`, line 356-360:

```python
strategies = [
    ('RSI', rsi_strategy),
    ('MA', moving_average_strategy),
    # Add your own strategy here
]
```

## Troubleshooting

### "No module named 'yfinance'"
```bash
pip install yfinance
```

### "No price data found"
Make sure you run `fetch_historical_data.py` before `generate_realistic_trades.py`

### Rate limiting / Slow downloads
yfinance includes automatic rate limiting. The script sleeps 0.5 seconds between symbols.

### Missing symbols
Some symbols may not have 5 years of data (e.g., recent IPOs, delisted stocks). The script handles this gracefully.

## Next Steps

After data migration:

1. **Verify analytics**: `python3 trading_analytics.py`
2. **Check web dashboard**: `python3 api_server.py` → Open browser to localhost:5001
3. **Start ML development**: See `MASTERS_ENHANCEMENT_PLAN.md`

## Data Sources

- **Yahoo Finance** (via yfinance library)
  - Free for personal/academic use
  - 15+ years of historical data
  - Updated daily
  - No API key required

## License & Usage

This data is for **academic/research purposes only**.

- Yahoo Finance data is for personal use
- Do not redistribute bulk data
- Cite yfinance in academic work: https://github.com/ranaroussi/yfinance

## Support

Issues? Check:
1. Python version: 3.8+ required
2. Internet connection
3. Database file exists and is writable
4. Sufficient disk space (~50 MB for data)
