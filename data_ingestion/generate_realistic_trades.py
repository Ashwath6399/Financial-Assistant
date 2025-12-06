#!/usr/bin/env python3
"""
Realistic Trade Generator

Creates realistic trading activity based on real market data using simple strategies:
1. RSI Strategy (oversold/overbought)
2. Moving Average Crossover
3. Mean Reversion
4. Random walk (simulation of discretionary trading)

This generates fills/orders that match real price movements.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, get_active_db_info

# Trading parameters
INITIAL_CAPITAL = 10000  # Starting capital per account
POSITION_SIZE_PCT = 0.1  # 10% of capital per trade
MAX_POSITIONS = 5  # Maximum concurrent positions
COMMISSION = 0.50  # $0.50 per trade


def get_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_strategy(prices_df: pd.DataFrame, capital: float) -> list:
    """
    RSI Strategy: Buy when RSI < 30, Sell when RSI > 70
    Returns list of trades (date, action, qty, price)
    """
    trades = []
    prices_df['RSI'] = calculate_rsi(prices_df['px_close'])

    position = 0  # Current position size
    position_cost = 0  # Average cost basis

    for idx, row in prices_df.iterrows():
        if pd.isna(row['RSI']):
            continue

        # Buy signal: RSI < 30 and no position
        if row['RSI'] < 30 and position == 0:
            qty = int((capital * POSITION_SIZE_PCT) / row['px_close'])
            if qty > 0:
                trades.append({
                    'date': row['ts'],
                    'action': 'BUY',
                    'qty': qty,
                    'price': row['px_close'],
                    'rsi': row['RSI']
                })
                position = qty
                position_cost = row['px_close']

        # Sell signal: RSI > 70 and have position
        elif row['RSI'] > 70 and position > 0:
            trades.append({
                'date': row['ts'],
                'action': 'SELL',
                'qty': position,
                'price': row['px_close'],
                'rsi': row['RSI']
            })
            # Update capital based on P&L
            pnl = (row['px_close'] - position_cost) * position - 2 * COMMISSION
            capital += pnl
            position = 0

    # Close any open position at end
    if position > 0:
        last_row = prices_df.iloc[-1]
        trades.append({
            'date': last_row['ts'],
            'action': 'SELL',
            'qty': position,
            'price': last_row['px_close'],
            'rsi': last_row['RSI']
        })

    return trades


def moving_average_strategy(prices_df: pd.DataFrame, capital: float) -> list:
    """
    Moving Average Crossover: Buy when MA20 crosses above MA50, sell when crosses below
    """
    trades = []
    prices_df['MA20'] = prices_df['px_close'].rolling(window=20).mean()
    prices_df['MA50'] = prices_df['px_close'].rolling(window=50).mean()
    prices_df['Signal'] = 0

    # Generate signals
    prices_df.loc[prices_df['MA20'] > prices_df['MA50'], 'Signal'] = 1
    prices_df.loc[prices_df['MA20'] <= prices_df['MA50'], 'Signal'] = -1

    # Detect crossovers
    prices_df['Position'] = prices_df['Signal'].diff()

    position = 0

    for idx, row in prices_df.iterrows():
        if pd.isna(row['Position']):
            continue

        # Buy signal (crossover up)
        if row['Position'] == 2 and position == 0:
            qty = int((capital * POSITION_SIZE_PCT) / row['px_close'])
            if qty > 0:
                trades.append({
                    'date': row['ts'],
                    'action': 'BUY',
                    'qty': qty,
                    'price': row['px_close'],
                    'indicator': 'MA_CROSS_UP'
                })
                position = qty
                position_cost = row['px_close']

        # Sell signal (crossover down)
        elif row['Position'] == -2 and position > 0:
            trades.append({
                'date': row['ts'],
                'action': 'SELL',
                'qty': position,
                'price': row['px_close'],
                'indicator': 'MA_CROSS_DOWN'
            })
            pnl = (row['px_close'] - position_cost) * position - 2 * COMMISSION
            capital += pnl
            position = 0

    return trades


def random_walk_strategy(prices_df: pd.DataFrame, capital: float, num_trades: int = 20) -> list:
    """
    Random walk strategy - simulates discretionary trading
    Makes random trades with some bias toward trends
    """
    trades = []

    # Sample random dates
    sample_size = min(num_trades * 2, len(prices_df) - 1)
    sample_indices = random.sample(range(50, len(prices_df) - 1), sample_size)
    sample_indices.sort()

    position = 0
    position_cost = 0

    for idx in sample_indices:
        row = prices_df.iloc[idx]

        # Randomly decide to buy or sell
        if position == 0 and random.random() < 0.5:  # 50% chance to enter
            qty = int((capital * random.uniform(0.05, 0.15)) / row['px_close'])
            if qty > 0:
                trades.append({
                    'date': row['ts'],
                    'action': 'BUY',
                    'qty': qty,
                    'price': row['px_close'],
                    'indicator': 'DISCRETIONARY'
                })
                position = qty
                position_cost = row['px_close']

        elif position > 0 and random.random() < 0.4:  # 40% chance to exit
            trades.append({
                'date': row['ts'],
                'action': 'SELL',
                'qty': position,
                'price': row['px_close'],
                'indicator': 'DISCRETIONARY'
            })
            pnl = (row['px_close'] - position_cost) * position - 2 * COMMISSION
            capital += pnl
            position = 0

        if len([t for t in trades if t['action'] == 'BUY']) >= num_trades:
            break

    return trades


def create_import_batch(conn, account_id: str, strategy: str) -> int:
    """Create import batch record"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO import_batches (broker_id, batch_name, imported_at)
        VALUES (?, ?, ?)
    """, (
        1,  # Default broker
        f"{account_id}_{strategy}_simulated",
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    return cursor.lastrowid


def insert_trades_to_database(conn, trades: list, account_id: str, instrument_id: int, strategy: str):
    """Insert simulated trades into database"""
    if not trades:
        return 0

    cursor = conn.cursor()
    batch_id = create_import_batch(conn, account_id, strategy)

    # Get max existing IDs to avoid collisions
    cursor.execute("SELECT COALESCE(MAX(order_id), 0) FROM orders")
    order_id_counter = cursor.fetchone()[0]

    cursor.execute("SELECT COALESCE(MAX(fill_id), 0) FROM fills")
    fill_id_counter = cursor.fetchone()[0]

    inserted_fills = 0

    for i in range(0, len(trades), 2):
        # Pair BUY and SELL trades
        buy_trade = trades[i] if trades[i]['action'] == 'BUY' else None
        sell_trade = trades[i+1] if i+1 < len(trades) and trades[i+1]['action'] == 'SELL' else None

        if buy_trade:
            # Insert BUY order
            order_id_counter += 1
            cursor.execute("""
                INSERT INTO orders
                (order_id, account_id, broker_id, instrument_id, side, order_type,
                 placed_at, status, qty, source_file, import_batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id_counter,
                account_id,
                1,  # broker_id
                instrument_id,
                'BUY',
                'MKT',
                buy_trade['date'],
                'FILLED',
                buy_trade['qty'],
                f'{strategy}_simulated',
                batch_id
            ))

            # Insert BUY fill
            fill_id_counter += 1
            cursor.execute("""
                INSERT INTO fills
                (fill_id, order_id, trade_ts, qty, price, fee, fee_currency,
                 liquidity_flag, source_file, import_batch_id, instrument_id, account_id, broker_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fill_id_counter,
                order_id_counter,
                buy_trade['date'],
                buy_trade['qty'],
                buy_trade['price'],
                COMMISSION,
                'USD',
                'T',
                f'{strategy}_simulated',
                batch_id,
                instrument_id,
                account_id,
                1
            ))
            inserted_fills += 1

        if sell_trade:
            # Insert SELL order
            order_id_counter += 1
            cursor.execute("""
                INSERT INTO orders
                (order_id, account_id, broker_id, instrument_id, side, order_type,
                 placed_at, status, qty, source_file, import_batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id_counter,
                account_id,
                1,
                instrument_id,
                'SELL',
                'MKT',
                sell_trade['date'],
                'FILLED',
                sell_trade['qty'],
                f'{strategy}_simulated',
                batch_id
            ))

            # Insert SELL fill
            fill_id_counter += 1
            cursor.execute("""
                INSERT INTO fills
                (fill_id, order_id, trade_ts, qty, price, fee, fee_currency,
                 liquidity_flag, source_file, import_batch_id, instrument_id, account_id, broker_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fill_id_counter,
                order_id_counter,
                sell_trade['date'],
                sell_trade['qty'],
                sell_trade['price'],
                COMMISSION,
                'USD',
                'T',
                f'{strategy}_simulated',
                batch_id,
                instrument_id,
                account_id,
                1
            ))
            inserted_fills += 1

    conn.commit()
    return inserted_fills


def generate_all_trades():
    """Main function to generate trades for all accounts"""
    print("=" * 80)
    print("REALISTIC TRADE GENERATION")
    print("=" * 80)
    print("\nGenerating trades based on real price data using multiple strategies...")
    print()

    conn = get_connection()

    # Get available symbols with price data
    symbols_df = pd.read_sql("""
        SELECT DISTINCT i.instrument_id, i.symbol, i.asset_class
        FROM instruments i
        JOIN prices p ON i.instrument_id = p.instrument_id
        WHERE p.source = 'yfinance'
        ORDER BY i.symbol
    """, conn)

    if symbols_df.empty:
        print("❌ No price data found! Run fetch_historical_data.py first.")
        conn.close()
        return

    print(f"✓ Found {len(symbols_df)} symbols with price data\n")

    # Get or create accounts
    # Note: Accounts T001-T012 already exist from first run
    # Now creating T013-T030 for additional data
    accounts = [f'T{str(i).zfill(3)}' for i in range(13, 31)]  # 18 more accounts (total will be 30)

    strategies = [
        ('RSI', rsi_strategy),
        ('MA', moving_average_strategy),
        ('RANDOM', random_walk_strategy)
    ]

    total_fills = 0

    for account_id in accounts:
        # Ensure account exists
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO accounts (account_id, broker_id, currency, opened_at)
            VALUES (?, ?, ?, ?)
        """, (account_id, 1, 'USD', '2020-01-01'))
        conn.commit()

        # Randomly assign strategy
        strategy_name, strategy_func = random.choice(strategies)

        # Randomly pick 5-10 symbols to trade
        num_symbols = random.randint(5, 10)
        traded_symbols = symbols_df.sample(n=min(num_symbols, len(symbols_df)))

        print(f"\n{account_id} - {strategy_name} Strategy ({len(traded_symbols)} symbols)")
        print("-" * 80)

        account_fills = 0

        for _, symbol_row in traded_symbols.iterrows():
            # Get price data
            prices_df = pd.read_sql(f"""
                SELECT ts, px_open, px_high, px_low, px_close, px_vol
                FROM prices
                WHERE instrument_id = {symbol_row['instrument_id']}
                  AND source = 'yfinance'
                ORDER BY ts ASC
            """, conn)

            if len(prices_df) < 100:  # Need enough data
                continue

            # Generate trades
            if strategy_name == 'RANDOM':
                trades = strategy_func(prices_df, INITIAL_CAPITAL, num_trades=random.randint(5, 15))
            else:
                trades = strategy_func(prices_df, INITIAL_CAPITAL)

            # Insert to database
            if trades:
                fills = insert_trades_to_database(
                    conn, trades, account_id,
                    symbol_row['instrument_id'],
                    strategy_name
                )
                account_fills += fills
                print(f"  {symbol_row['symbol']:6s}: {len(trades):3d} trades → {fills} fills")

        total_fills += account_fills
        print(f"  Total fills for {account_id}: {account_fills}")

    conn.close()

    print("\n" + "=" * 80)
    print("TRADE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Generated {total_fills} fills across {len(accounts)} accounts")
    print(f"✓ Using {len(strategies)} different strategies")
    print(f"✓ Based on real market prices from yfinance")
    print("\nYour database now contains realistic trading activity!")
    print("\nNext steps:")
    print("  1. Run: python3 trading_analytics.py")
    print("  2. Verify the new trade data looks realistic")
    print("  3. Start building ML models!\n")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 80)
    print("REALISTIC TRADE GENERATOR")
    print("=" * 80)
    print("\nThis will generate simulated trades based on real price data")
    print("using RSI, Moving Average, and Random Walk strategies.\n")
    print("Estimated fills: 500-1000")
    print("Estimated time: 1-2 minutes")
    print("\n" + "=" * 80 + "\n")

    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        confirm = 'y'
    else:
        confirm = input("Continue? (y/n): ").lower()

    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    generate_all_trades()
