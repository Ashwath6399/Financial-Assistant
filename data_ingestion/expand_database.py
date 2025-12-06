#!/usr/bin/env python3
"""
Expand Database to 50 Accounts

Adds 35 new trading accounts to the existing 15 accounts.
Uses existing price data to generate realistic trades.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# Configuration
TARGET_ACCOUNTS = 50
INITIAL_CAPITAL = 10000
POSITION_SIZE_PCT = 0.1
COMMISSION = 0.50

# Trading strategies
STRATEGIES = ['RSI', 'MA_Crossover', 'Discretionary']

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ma(prices: pd.Series, short: int = 10, long: int = 30) -> tuple:
    return prices.rolling(short).mean(), prices.rolling(long).mean()


def generate_trades_rsi(prices_df: pd.DataFrame, capital: float) -> list:
    """RSI Strategy: Buy RSI<30, Sell RSI>70"""
    trades = []
    prices_df = prices_df.copy()
    prices_df['RSI'] = calculate_rsi(prices_df['px_close'])

    position = 0
    entry_price = 0

    for idx, row in prices_df.iterrows():
        if pd.isna(row['RSI']) or row['px_close'] <= 0:
            continue

        if row['RSI'] < 30 and position == 0:
            qty = max(1, int((capital * POSITION_SIZE_PCT) / row['px_close']))
            trades.append({
                'date': row['ts'], 'action': 'BUY', 'qty': qty,
                'price': row['px_close'], 'strategy': 'RSI'
            })
            position = qty
            entry_price = row['px_close']

        elif row['RSI'] > 70 and position > 0:
            trades.append({
                'date': row['ts'], 'action': 'SELL', 'qty': position,
                'price': row['px_close'], 'strategy': 'RSI'
            })
            pnl = (row['px_close'] - entry_price) * position - 2 * COMMISSION
            capital += pnl
            position = 0

    # Close open position
    if position > 0 and len(prices_df) > 0:
        last = prices_df.iloc[-1]
        trades.append({
            'date': last['ts'], 'action': 'SELL', 'qty': position,
            'price': last['px_close'], 'strategy': 'RSI'
        })

    return trades


def generate_trades_ma(prices_df: pd.DataFrame, capital: float) -> list:
    """MA Crossover: Buy when short > long, Sell when short < long"""
    trades = []
    prices_df = prices_df.copy()
    prices_df['MA_short'], prices_df['MA_long'] = calculate_ma(prices_df['px_close'])

    position = 0
    entry_price = 0
    prev_signal = None

    for idx, row in prices_df.iterrows():
        if pd.isna(row['MA_short']) or pd.isna(row['MA_long']) or row['px_close'] <= 0:
            continue

        signal = 'buy' if row['MA_short'] > row['MA_long'] else 'sell'

        if signal == 'buy' and prev_signal == 'sell' and position == 0:
            qty = max(1, int((capital * POSITION_SIZE_PCT) / row['px_close']))
            trades.append({
                'date': row['ts'], 'action': 'BUY', 'qty': qty,
                'price': row['px_close'], 'strategy': 'MA_Crossover'
            })
            position = qty
            entry_price = row['px_close']

        elif signal == 'sell' and prev_signal == 'buy' and position > 0:
            trades.append({
                'date': row['ts'], 'action': 'SELL', 'qty': position,
                'price': row['px_close'], 'strategy': 'MA_Crossover'
            })
            pnl = (row['px_close'] - entry_price) * position - 2 * COMMISSION
            capital += pnl
            position = 0

        prev_signal = signal

    # Close open position
    if position > 0 and len(prices_df) > 0:
        last = prices_df.iloc[-1]
        trades.append({
            'date': last['ts'], 'action': 'SELL', 'qty': position,
            'price': last['px_close'], 'strategy': 'MA_Crossover'
        })

    return trades


def generate_trades_discretionary(prices_df: pd.DataFrame, capital: float) -> list:
    """Random discretionary trading with mean reversion bias"""
    trades = []
    prices_df = prices_df.copy()
    prices_df['returns'] = prices_df['px_close'].pct_change()
    prices_df['MA20'] = prices_df['px_close'].rolling(20).mean()

    position = 0
    entry_price = 0

    for idx, row in prices_df.iterrows():
        if pd.isna(row['MA20']) or row['px_close'] <= 0:
            continue

        # Random entry with bias
        if position == 0 and random.random() < 0.02:  # 2% daily entry chance
            # More likely to buy when below MA
            buy_prob = 0.7 if row['px_close'] < row['MA20'] else 0.4
            if random.random() < buy_prob:
                qty = max(1, int((capital * POSITION_SIZE_PCT) / row['px_close']))
                trades.append({
                    'date': row['ts'], 'action': 'BUY', 'qty': qty,
                    'price': row['px_close'], 'strategy': 'Discretionary'
                })
                position = qty
                entry_price = row['px_close']

        # Exit conditions
        elif position > 0:
            pnl_pct = (row['px_close'] - entry_price) / entry_price

            # Take profit > 5% or stop loss < -3% or random exit
            if pnl_pct > 0.05 or pnl_pct < -0.03 or random.random() < 0.03:
                trades.append({
                    'date': row['ts'], 'action': 'SELL', 'qty': position,
                    'price': row['px_close'], 'strategy': 'Discretionary'
                })
                pnl = (row['px_close'] - entry_price) * position - 2 * COMMISSION
                capital += pnl
                position = 0

    # Close open position
    if position > 0 and len(prices_df) > 0:
        last = prices_df.iloc[-1]
        trades.append({
            'date': last['ts'], 'action': 'SELL', 'qty': position,
            'price': last['px_close'], 'strategy': 'Discretionary'
        })

    return trades


def create_new_accounts(conn, num_new_accounts: int, start_id: int):
    """Create new account records"""
    cursor = conn.cursor()

    # Get broker_id
    cursor.execute("SELECT broker_id FROM brokers LIMIT 1")
    broker_id = cursor.fetchone()[0]

    new_accounts = []
    for i in range(num_new_accounts):
        account_id = f"T{start_id + i:04d}"

        cursor.execute("""
            INSERT OR IGNORE INTO accounts (account_id, broker_id, currency, opened_at)
            VALUES (?, ?, 'USD', '2020-01-01')
        """, (account_id, broker_id))

        new_accounts.append(account_id)

    conn.commit()
    print(f"✓ Created {len(new_accounts)} new accounts")
    return new_accounts


def get_price_data(conn, symbol: str) -> pd.DataFrame:
    """Get price data for a symbol"""
    query = """
    SELECT p.ts, p.px_open, p.px_high, p.px_low, p.px_close, p.px_vol as volume
    FROM prices p
    JOIN instruments i ON p.instrument_id = i.instrument_id
    WHERE i.symbol = ?
    ORDER BY p.ts
    """
    return pd.read_sql(query, conn, params=(symbol,))


def insert_trades(conn, trades: list, account_id: str, symbol: str, batch_id: int):
    """Insert trades into orders and fills tables"""
    cursor = conn.cursor()

    # Get instrument_id
    cursor.execute("SELECT instrument_id FROM instruments WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    if not result:
        return 0
    instrument_id = result[0]

    # Get broker_id for this account
    cursor.execute("SELECT broker_id FROM accounts WHERE account_id = ?", (account_id,))
    broker_id = cursor.fetchone()[0]

    # Get next order_id and fill_id
    cursor.execute("SELECT COALESCE(MAX(order_id), 0) FROM orders")
    next_order_id = cursor.fetchone()[0] + 1
    cursor.execute("SELECT COALESCE(MAX(fill_id), 0) FROM fills")
    next_fill_id = cursor.fetchone()[0] + 1

    count = 0
    for trade in trades:
        order_id = next_order_id + count
        fill_id = next_fill_id + count
        side = 'BUY' if trade['action'] == 'BUY' else 'SELL'

        # Insert order (matching correct schema)
        cursor.execute("""
            INSERT INTO orders (order_id, account_id, broker_id, instrument_id, side, order_type, placed_at, status, qty, source_file, import_batch_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_id, account_id, broker_id, instrument_id, side, 'MKT', trade['date'], 'FILLED', trade['qty'], f"{trade['strategy']}_sim", batch_id))

        # Insert fill (matching correct schema)
        cursor.execute("""
            INSERT INTO fills (fill_id, order_id, trade_ts, qty, price, fee, fee_currency, liquidity_flag, source_file, import_batch_id, instrument_id, account_id, broker_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (fill_id, order_id, trade['date'], trade['qty'], trade['price'], COMMISSION, 'USD', 'T', f"{trade['strategy']}_sim", batch_id, instrument_id, account_id, broker_id))

        count += 1

    conn.commit()
    return count


def main():
    print("\n" + "="*80)
    print("EXPANDING DATABASE TO 50 ACCOUNTS")
    print("="*80)

    conn = get_connection()
    cursor = conn.cursor()

    # Get current accounts
    cursor.execute("SELECT COUNT(DISTINCT account_id) FROM accounts")
    current_count = cursor.fetchone()[0]
    print(f"\nCurrent accounts: {current_count}")

    # Get accounts with no fills (need to generate trades)
    cursor.execute("""
        SELECT a.account_id
        FROM accounts a
        LEFT JOIN fills f ON a.account_id = f.account_id
        GROUP BY a.account_id
        HAVING COUNT(f.fill_id) = 0
        ORDER BY a.account_id
    """)
    empty_accounts = [row[0] for row in cursor.fetchall()]

    if not empty_accounts and current_count >= TARGET_ACCOUNTS:
        print(f"All {current_count} accounts have trades already")
        conn.close()
        return

    if empty_accounts:
        print(f"Found {len(empty_accounts)} accounts with no trades - will generate trades for them")

    # Get max account number
    cursor.execute("SELECT MAX(CAST(SUBSTR(account_id, 2) AS INTEGER)) FROM accounts")
    max_num = cursor.fetchone()[0] or 0

    # Determine which accounts need trades
    if empty_accounts:
        accounts_to_populate = empty_accounts
    else:
        num_new = TARGET_ACCOUNTS - current_count
        print(f"Creating {num_new} new accounts...")
        accounts_to_populate = create_new_accounts(conn, num_new, max_num + 1)

    # Get available symbols (top traded)
    cursor.execute("""
        SELECT DISTINCT i.symbol
        FROM instruments i
        JOIN prices p ON i.instrument_id = p.instrument_id
        GROUP BY i.symbol
        HAVING COUNT(*) > 1000
        ORDER BY RANDOM()
        LIMIT 100
    """)
    available_symbols = [row[0] for row in cursor.fetchall()]
    print(f"✓ Found {len(available_symbols)} symbols with sufficient price data")

    # Generate trades for each new account
    total_trades = 0
    strategy_funcs = {
        'RSI': generate_trades_rsi,
        'MA_Crossover': generate_trades_ma,
        'Discretionary': generate_trades_discretionary
    }

    # Create import batch for expansion
    cursor.execute("""
        INSERT INTO import_batches (broker_id, batch_name, imported_at)
        VALUES (?, ?, ?)
    """, (1, 'expand_database_50_accounts', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    batch_id = cursor.lastrowid
    conn.commit()
    print(f"✓ Created import batch {batch_id}")

    for i, account_id in enumerate(accounts_to_populate):
        print(f"\n[{i+1}/{len(accounts_to_populate)}] Generating trades for {account_id}...")

        # Each account trades 3-8 symbols
        num_symbols = random.randint(3, 8)
        account_symbols = random.sample(available_symbols, min(num_symbols, len(available_symbols)))

        # Pick primary strategy for this account
        primary_strategy = random.choice(STRATEGIES)

        account_trades = 0
        for symbol in account_symbols:
            prices_df = get_price_data(conn, symbol)
            if len(prices_df) < 100:
                continue

            # Use a date range subset (random 2-4 years)
            if len(prices_df) > 500:
                start_idx = random.randint(0, max(0, len(prices_df) - 500))
                end_idx = min(start_idx + random.randint(500, 1000), len(prices_df))
                prices_df = prices_df.iloc[start_idx:end_idx].reset_index(drop=True)

            # Generate trades using the strategy
            strategy_func = strategy_funcs[primary_strategy]
            trades = strategy_func(prices_df, INITIAL_CAPITAL)

            if trades:
                count = insert_trades(conn, trades, account_id, symbol, batch_id)
                account_trades += count

        print(f"   ✓ Generated {account_trades} trades using {primary_strategy} strategy")
        total_trades += account_trades

    # Clear old predictions (they need to be regenerated)
    cursor.execute("DELETE FROM ml_predictions")
    conn.commit()

    # Final stats
    cursor.execute("SELECT COUNT(DISTINCT account_id) FROM accounts")
    final_accounts = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM fills")
    final_fills = cursor.fetchone()[0]

    print("\n" + "="*80)
    print("EXPANSION COMPLETE")
    print("="*80)
    print(f"Total accounts: {final_accounts}")
    print(f"Total fills: {final_fills:,}")
    print(f"New trades added: {total_trades:,}")
    print("\nNote: ML predictions table cleared - run apply_predictions.py to regenerate")

    conn.close()


if __name__ == "__main__":
    main()
