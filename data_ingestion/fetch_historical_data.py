#!/usr/bin/env python3
"""
Historical Market Data Fetcher using yfinance
Downloads real market data to replace synthetic data in database

Features:
- 5 years of daily OHLCV data
- 50+ stocks, ETFs, and crypto
- Handles missing data gracefully
- Progress tracking
- Data validation
"""

import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, get_active_db_info

# Trading universe (50+ symbols for diversification)
SYMBOLS = {
    'tech': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'PYPL'
    ],
    'finance': [
        'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'SCHW'
    ],
    'consumer': [
        'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'DIS'
    ],
    'healthcare': [
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT'
    ],
    'energy': [
        'XOM', 'CVX', 'COP', 'SLB'
    ],
    'etfs': [
        'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO',
        'GLD', 'SLV', 'TLT', 'HYG'
    ],
    'crypto': [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD'
    ]
}

# Flatten symbol list
ALL_SYMBOLS = [sym for category in SYMBOLS.values() for sym in category]


def get_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_or_create_instrument(conn, symbol: str, asset_class: str) -> int:
    """Get instrument_id or create new instrument"""
    cursor = conn.cursor()

    # Check if exists
    cursor.execute("""
        SELECT instrument_id FROM instruments WHERE symbol = ?
    """, (symbol,))

    result = cursor.fetchone()
    if result:
        return result[0]

    # Create new instrument
    cursor.execute("""
        INSERT INTO instruments (symbol, symbol_raw, asset_class, exchange, currency)
        VALUES (?, ?, ?, ?, ?)
    """, (
        symbol,
        symbol,
        asset_class,
        'NASDAQ' if asset_class == 'equity' else ('CRYPTO' if 'USD' in symbol else 'NYSE'),
        'USD'
    ))
    conn.commit()

    return cursor.lastrowid


def determine_asset_class(symbol: str) -> str:
    """Determine asset class from symbol"""
    if '-USD' in symbol:
        return 'crypto'
    elif symbol in ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'GLD', 'SLV', 'TLT', 'HYG']:
        return 'etf'
    else:
        return 'equity'


def fetch_symbol_data(symbol: str, period: str = '5y') -> pd.DataFrame:
    """Fetch historical data for a symbol"""
    try:
        print(f"  Downloading {symbol}...", end=' ')
        ticker = yf.Ticker(symbol)

        # Download data with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                hist = ticker.history(period=period, interval='1d')
                if not hist.empty:
                    print(f"✓ {len(hist)} days")
                    return hist
                else:
                    print(f"⚠ No data")
                    return pd.DataFrame()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}...", end=' ')
                    time.sleep(2)
                else:
                    print(f"✗ Error: {str(e)}")
                    return pd.DataFrame()

        return pd.DataFrame()

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return pd.DataFrame()


def insert_price_data(conn, symbol: str, instrument_id: int, hist: pd.DataFrame) -> int:
    """Insert price data into database"""
    cursor = conn.cursor()
    inserted = 0

    for date, row in hist.iterrows():
        try:
            # Create unique price_id
            price_id = f"{symbol}_{date.strftime('%Y%m%d')}"

            cursor.execute("""
                INSERT OR REPLACE INTO prices
                (price_id, instrument_id, ts, px_open, px_high, px_low, px_close, px_vol, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                price_id,
                instrument_id,
                date.strftime('%Y-%m-%d'),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume']) if pd.notna(row['Volume']) else 0,
                'yfinance'
            ))
            inserted += 1

        except Exception as e:
            print(f"    Warning: Failed to insert {date}: {str(e)}")
            continue

    conn.commit()
    return inserted


def download_all_data(symbols: List[str] = None, period: str = '5y'):
    """Main function to download all historical data"""
    if symbols is None:
        symbols = ALL_SYMBOLS

    print("=" * 80)
    print("HISTORICAL DATA DOWNLOAD - yfinance")
    print("=" * 80)
    print(f"\nSymbols to download: {len(symbols)}")
    print(f"Period: {period}")
    print(f"Data source: Yahoo Finance (via yfinance)")
    print("\n" + "=" * 80 + "\n")

    conn = get_connection()

    stats = {
        'total_symbols': len(symbols),
        'successful': 0,
        'failed': 0,
        'total_records': 0,
        'start_time': time.time()
    }

    # Group by category for organized output
    for category, category_symbols in SYMBOLS.items():
        print(f"\n{category.upper()} ({len(category_symbols)} symbols)")
        print("-" * 80)

        for symbol in category_symbols:
            # Get or create instrument
            asset_class = determine_asset_class(symbol)
            instrument_id = get_or_create_instrument(conn, symbol, asset_class)

            # Fetch data
            hist = fetch_symbol_data(symbol, period)

            if not hist.empty:
                # Insert into database
                inserted = insert_price_data(conn, symbol, instrument_id, hist)
                stats['successful'] += 1
                stats['total_records'] += inserted
            else:
                stats['failed'] += 1

            # Rate limiting (be nice to Yahoo)
            time.sleep(0.5)

    conn.close()

    # Print summary
    elapsed = time.time() - stats['start_time']

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nTotal symbols attempted: {stats['total_symbols']}")
    print(f"  ✓ Successful: {stats['successful']}")
    print(f"  ✗ Failed: {stats['failed']}")
    print(f"  📊 Total price records: {stats['total_records']:,}")
    print(f"  ⏱  Time elapsed: {elapsed:.1f} seconds")
    print(f"  📈 Average: {stats['total_records']/stats['successful']:.0f} days per symbol")
    print("\n" + "=" * 80)

    return stats


def verify_data_quality():
    """Verify downloaded data quality"""
    print("\n" + "=" * 80)
    print("DATA QUALITY VERIFICATION")
    print("=" * 80 + "\n")

    conn = get_connection()

    # Check 1: Total records
    total_prices = pd.read_sql("SELECT COUNT(*) as count FROM prices WHERE source='yfinance'", conn).iloc[0]['count']
    print(f"✓ Total price records (yfinance): {total_prices:,}")

    # Check 2: Date range
    date_range = pd.read_sql("""
        SELECT
            MIN(ts) as earliest,
            MAX(ts) as latest
        FROM prices
        WHERE source='yfinance'
    """, conn)
    print(f"✓ Date range: {date_range.iloc[0]['earliest']} to {date_range.iloc[0]['latest']}")

    # Check 3: Symbols coverage
    symbols_count = pd.read_sql("""
        SELECT COUNT(DISTINCT instrument_id) as count
        FROM prices
        WHERE source='yfinance'
    """, conn).iloc[0]['count']
    print(f"✓ Unique symbols with data: {symbols_count}")

    # Check 4: Missing data gaps
    print("\nChecking for data gaps...")
    gaps = pd.read_sql("""
        SELECT
            i.symbol,
            COUNT(*) as days,
            MIN(p.ts) as first_date,
            MAX(p.ts) as last_date
        FROM prices p
        JOIN instruments i ON p.instrument_id = i.instrument_id
        WHERE p.source = 'yfinance'
        GROUP BY i.symbol
        HAVING days < 1000  -- 5 years ~= 1260 trading days
        ORDER BY days ASC
        LIMIT 10
    """, conn)

    if not gaps.empty:
        print("\n  ⚠ Symbols with potential gaps (< 1000 days):")
        for _, row in gaps.iterrows():
            print(f"    - {row['symbol']}: {row['days']} days ({row['first_date']} to {row['last_date']})")
    else:
        print("  ✓ No significant data gaps detected")

    # Check 5: Price sanity checks
    print("\nPrice sanity checks...")
    anomalies = pd.read_sql("""
        SELECT
            i.symbol,
            p.ts,
            p.px_close
        FROM prices p
        JOIN instruments i ON p.instrument_id = i.instrument_id
        WHERE p.source = 'yfinance'
          AND (p.px_close <= 0 OR p.px_close > 100000)
        LIMIT 5
    """, conn)

    if not anomalies.empty:
        print("  ⚠ Potential price anomalies:")
        for _, row in anomalies.iterrows():
            print(f"    - {row['symbol']} on {row['ts']}: ${row['px_close']}")
    else:
        print("  ✓ No obvious price anomalies")

    # Check 6: Volume checks
    zero_volume = pd.read_sql("""
        SELECT COUNT(*) as count
        FROM prices
        WHERE source = 'yfinance' AND (px_vol = 0 OR px_vol IS NULL)
    """, conn).iloc[0]['count']

    if zero_volume > 0:
        print(f"  ⚠ {zero_volume} records with zero/null volume (may be normal for crypto/after-hours)")
    else:
        print("  ✓ All records have volume data")

    conn.close()

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("TRADING JOURNAL - REAL MARKET DATA MIGRATION")
    print("=" * 80)
    print("\nThis script will download 5 years of real market data from Yahoo Finance")
    print("for 50+ stocks, ETFs, and cryptocurrencies.\n")
    print("Estimated time: 2-5 minutes")
    print("Estimated records: ~60,000 price bars")
    print("\n" + "=" * 80 + "\n")

    # Confirm
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        confirm = 'y'
    else:
        confirm = input("Continue? (y/n): ").lower()

    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Download data
    stats = download_all_data()

    # Verify quality
    verify_data_quality()

    print("\n✓ Data migration complete! You now have real market data in your database.")
    print("\nNext steps:")
    print("  1. Run: python3 data_ingestion/generate_realistic_trades.py")
    print("  2. This will create realistic trades based on the real price data")
    print("  3. Then you can start ML model training!\n")
