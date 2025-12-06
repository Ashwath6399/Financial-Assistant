"""
Advanced Trading Metrics Calculator
====================================
Calculates professional-grade trading metrics for each account:
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (worst peak-to-trough decline)
- Sortino Ratio (downside risk-adjusted returns)
- Calmar Ratio (return vs max drawdown)
- Volatility (annualized standard deviation)
- Win Rate, Average Win/Loss, Risk/Reward Ratio

These metrics are stored in the database for dynamic UI display.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

def print_section(title, char='='):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")

def create_metrics_table(conn):
    """Create table to store account metrics"""
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS account_metrics (
        account_id TEXT PRIMARY KEY,

        -- Basic Metrics
        total_trades INTEGER,
        total_pnl REAL,
        total_fees REAL,
        net_pnl REAL,

        -- Performance Metrics
        win_rate REAL,
        avg_win REAL,
        avg_loss REAL,
        largest_win REAL,
        largest_loss REAL,
        risk_reward_ratio REAL,

        -- Time Metrics
        avg_hold_days REAL,
        total_trading_days INTEGER,

        -- Risk-Adjusted Returns
        sharpe_ratio REAL,
        sortino_ratio REAL,
        calmar_ratio REAL,

        -- Risk Metrics
        max_drawdown REAL,
        max_drawdown_duration_days INTEGER,
        volatility REAL,
        downside_volatility REAL,

        -- Returns
        total_return_pct REAL,
        annualized_return_pct REAL,

        -- Trading Activity
        avg_trades_per_month REAL,
        win_streak_max INTEGER,
        loss_streak_max INTEGER,

        -- Updated timestamp
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    )
    """)

    conn.commit()
    print("✓ Created account_metrics table")

def get_account_trades(conn, account_id):
    """Get all closed trades for an account with P&L"""

    # Import from hybrid_model to get trade calculation function
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_models'))
    from hybrid_model import calculate_trade_outcomes

    # Get fills for this account
    query = """
    SELECT
        f.fill_id,
        f.account_id,
        f.instrument_id,
        i.symbol,
        f.trade_ts,
        f.qty,
        f.price,
        f.fee,
        o.side,
        f.source_file as strategy
    FROM fills f
    JOIN instruments i ON f.instrument_id = i.instrument_id
    JOIN orders o ON f.order_id = o.order_id
    WHERE f.account_id = ?
    ORDER BY f.trade_ts
    """

    fills = pd.read_sql_query(query, conn, params=(account_id,))

    if len(fills) == 0:
        return pd.DataFrame()

    # Calculate trade outcomes
    trades = calculate_trade_outcomes(fills)

    # Sort by entry timestamp
    trades = trades.sort_values('entry_ts').reset_index(drop=True)

    return trades

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe Ratio
    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    Annualized assuming 252 trading days
    """
    if len(returns) < 2:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    # Annualized
    sharpe = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)

    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sortino Ratio
    Sortino = (Mean Return - Risk Free Rate) / Downside Std Dev
    Only considers downside volatility (negative returns)
    """
    if len(returns) < 2:
        return 0.0

    mean_return = returns.mean()

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    # Annualized
    sortino = (mean_return - risk_free_rate/252) / downside_std * np.sqrt(252)

    return sortino

def calculate_max_drawdown(cumulative_pnl):
    """
    Calculate Maximum Drawdown
    Max DD = (Trough Value - Peak Value) / Peak Value
    Returns: (max_drawdown_pct, duration_days)
    """
    if len(cumulative_pnl) < 2:
        return 0.0, 0

    # Calculate running maximum
    running_max = cumulative_pnl.cummax()

    # Calculate drawdown at each point
    drawdown = (cumulative_pnl - running_max) / running_max.replace(0, 1)

    # Get maximum drawdown
    max_dd = drawdown.min()

    # Calculate drawdown duration
    # Find where max drawdown occurred
    max_dd_idx = drawdown.idxmin()

    # Find the peak before this drawdown
    peak_idx = cumulative_pnl[:max_dd_idx].idxmax() if max_dd_idx > 0 else 0

    # Duration in number of trades (proxy for days)
    duration = max_dd_idx - peak_idx if max_dd_idx > peak_idx else 0

    return max_dd * 100, duration  # Return as percentage

def calculate_calmar_ratio(annualized_return, max_drawdown):
    """
    Calculate Calmar Ratio
    Calmar = Annualized Return / |Max Drawdown|
    """
    if max_drawdown == 0:
        return 0.0

    return annualized_return / abs(max_drawdown)

def calculate_win_loss_streaks(outcomes):
    """Calculate maximum win and loss streaks"""
    if len(outcomes) == 0:
        return 0, 0

    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0

    for outcome in outcomes:
        if outcome == 1:  # Win
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        else:  # Loss
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)

    return max_win_streak, max_loss_streak

def calculate_account_metrics(conn, account_id):
    """Calculate all metrics for a single account"""

    trades = get_account_trades(conn, account_id)

    if len(trades) == 0:
        print(f"  ⚠ No trades found for {account_id}")
        return None

    # Basic metrics
    total_trades = len(trades)
    total_pnl = trades['gross_pnl'].sum()
    total_fees = trades['fees'].sum()
    net_pnl = trades['net_pnl'].sum()

    # Win/Loss analysis
    winning_trades = trades[trades['is_profitable'] == 1]
    losing_trades = trades[trades['is_profitable'] == 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

    avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0

    largest_win = trades['net_pnl'].max()
    largest_loss = trades['net_pnl'].min()

    # Risk/Reward ratio
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Time metrics - use format='mixed' to handle different date formats
    trades['entry_date'] = pd.to_datetime(trades['entry_ts'], format='mixed')
    trades['exit_date'] = pd.to_datetime(trades['exit_ts'], format='mixed')
    trades['hold_days'] = (trades['exit_date'] - trades['entry_date']).dt.days

    avg_hold_days = trades['hold_days'].mean()

    total_trading_days = (trades['exit_date'].max() - trades['entry_date'].min()).days

    # Calculate returns for each trade
    # Assume initial capital (we'll use absolute P&L for simplicity, but can be improved)
    # For proper calculation, we'd need account balance history
    trades['cumulative_pnl'] = trades['net_pnl'].cumsum()

    # Use a proxy for returns: P&L relative to cumulative position
    # This is simplified - ideally we'd have account equity curve
    initial_capital = 10000  # Assume $10k starting capital per account
    trades['equity'] = initial_capital + trades['cumulative_pnl']
    trades['return'] = trades['net_pnl'] / trades['equity'].shift(1).fillna(initial_capital)

    # Sharpe Ratio
    sharpe = calculate_sharpe_ratio(trades['return'])

    # Sortino Ratio
    sortino = calculate_sortino_ratio(trades['return'])

    # Volatility
    volatility = trades['return'].std() * np.sqrt(252)  # Annualized

    # Downside volatility
    downside_returns = trades['return'][trades['return'] < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

    # Maximum Drawdown
    max_dd, max_dd_duration = calculate_max_drawdown(trades['cumulative_pnl'])

    # Returns
    total_return_pct = (net_pnl / initial_capital) * 100

    # Annualized return
    years = total_trading_days / 365.25 if total_trading_days > 0 else 1
    annualized_return_pct = ((1 + total_return_pct/100) ** (1/years) - 1) * 100 if years > 0 else 0

    # Calmar Ratio
    calmar = calculate_calmar_ratio(annualized_return_pct, max_dd)

    # Trading activity
    months = total_trading_days / 30.44 if total_trading_days > 0 else 1
    avg_trades_per_month = total_trades / months

    # Win/Loss streaks
    max_win_streak, max_loss_streak = calculate_win_loss_streaks(trades['is_profitable'].values)

    metrics = {
        'account_id': account_id,
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'net_pnl': net_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'risk_reward_ratio': risk_reward,
        'avg_hold_days': avg_hold_days,
        'total_trading_days': total_trading_days,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'max_drawdown_duration_days': int(max_dd_duration),
        'volatility': volatility,
        'downside_volatility': downside_vol,
        'total_return_pct': total_return_pct,
        'annualized_return_pct': annualized_return_pct,
        'avg_trades_per_month': avg_trades_per_month,
        'win_streak_max': max_win_streak,
        'loss_streak_max': max_loss_streak
    }

    return metrics

def save_metrics_to_db(conn, metrics):
    """Save calculated metrics to database"""

    cursor = conn.cursor()

    # Delete existing metrics for this account
    cursor.execute("DELETE FROM account_metrics WHERE account_id = ?", (metrics['account_id'],))

    # Insert new metrics
    columns = ', '.join(metrics.keys())
    placeholders = ', '.join(['?' for _ in metrics])

    cursor.execute(f"""
        INSERT INTO account_metrics ({columns})
        VALUES ({placeholders})
    """, tuple(metrics.values()))

    conn.commit()

def display_metrics(metrics):
    """Display metrics in formatted output"""

    print(f"\n📊 ACCOUNT: {metrics['account_id']}")
    print(f"{'─' * 80}")

    print(f"\n💼 BASIC METRICS")
    print(f"  Total Trades:            {metrics['total_trades']}")
    print(f"  Gross P&L:               ${metrics['total_pnl']:,.2f}")
    print(f"  Total Fees:              ${metrics['total_fees']:,.2f}")
    print(f"  Net P&L:                 ${metrics['net_pnl']:,.2f}")
    print(f"  Trading Period:          {metrics['total_trading_days']} days")
    print(f"  Avg Trades/Month:        {metrics['avg_trades_per_month']:.1f}")

    print(f"\n📈 PERFORMANCE METRICS")
    print(f"  Win Rate:                {metrics['win_rate']*100:.2f}%")
    print(f"  Average Win:             ${metrics['avg_win']:,.2f}")
    print(f"  Average Loss:            ${metrics['avg_loss']:,.2f}")
    print(f"  Risk/Reward Ratio:       {metrics['risk_reward_ratio']:.2f}")
    print(f"  Largest Win:             ${metrics['largest_win']:,.2f}")
    print(f"  Largest Loss:            ${metrics['largest_loss']:,.2f}")
    print(f"  Avg Hold Time:           {metrics['avg_hold_days']:.1f} days")

    print(f"\n💰 RETURNS")
    print(f"  Total Return:            {metrics['total_return_pct']:.2f}%")
    print(f"  Annualized Return:       {metrics['annualized_return_pct']:.2f}%")

    print(f"\n⚠️  RISK METRICS")
    print(f"  Maximum Drawdown:        {metrics['max_drawdown']:.2f}%")
    print(f"  Max DD Duration:         {metrics['max_drawdown_duration_days']} trades")
    print(f"  Volatility (Annual):     {metrics['volatility']*100:.2f}%")
    print(f"  Downside Volatility:     {metrics['downside_volatility']*100:.2f}%")

    print(f"\n🎯 RISK-ADJUSTED RETURNS")
    print(f"  Sharpe Ratio:            {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:           {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio:            {metrics['calmar_ratio']:.3f}")

    print(f"\n🔥 STREAKS")
    print(f"  Max Win Streak:          {metrics['win_streak_max']} trades")
    print(f"  Max Loss Streak:         {metrics['loss_streak_max']} trades")

def main():
    """Main execution"""

    print_section("ADVANCED TRADING METRICS CALCULATOR", '█')

    conn = sqlite3.connect(DB_PATH)

    # Create metrics table
    print_section("STEP 1: Creating Metrics Table")
    create_metrics_table(conn)

    # Get all accounts
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT account_id FROM fills ORDER BY account_id")
    accounts = [row[0] for row in cursor.fetchall()]

    print(f"\nFound {len(accounts)} accounts to analyze")

    # Calculate metrics for each account
    print_section("STEP 2: Calculating Metrics for Each Account")

    all_metrics = []

    for account_id in accounts:
        print(f"\n{'═' * 80}")
        print(f"Processing: {account_id}")
        print(f"{'═' * 80}")

        try:
            metrics = calculate_account_metrics(conn, account_id)

            if metrics:
                # Save to database
                save_metrics_to_db(conn, metrics)

                # Display
                display_metrics(metrics)

                all_metrics.append(metrics)

                print(f"\n✓ Metrics saved to database")

        except Exception as e:
            print(f"\n✗ Error processing {account_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_section("STEP 3: Summary Statistics")

    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)

        print(f"\n📊 OVERALL STATISTICS (Across All Accounts)")
        print(f"{'─' * 80}")
        print(f"  Total Accounts:          {len(all_metrics)}")
        print(f"  Total Trades:            {df['total_trades'].sum():,.0f}")
        print(f"  Total Net P&L:           ${df['net_pnl'].sum():,.2f}")
        print(f"  Avg Win Rate:            {df['win_rate'].mean()*100:.2f}%")
        print(f"  Avg Sharpe Ratio:        {df['sharpe_ratio'].mean():.3f}")
        print(f"  Avg Max Drawdown:        {df['max_drawdown'].mean():.2f}%")

        # Best performers
        print(f"\n🏆 TOP 5 ACCOUNTS BY SHARPE RATIO")
        print(f"{'─' * 80}")
        top_sharpe = df.nlargest(5, 'sharpe_ratio')[['account_id', 'sharpe_ratio', 'total_return_pct', 'max_drawdown']]
        for idx, row in top_sharpe.iterrows():
            print(f"  {row['account_id']:8s} | Sharpe: {row['sharpe_ratio']:6.3f} | Return: {row['total_return_pct']:7.2f}% | Max DD: {row['max_drawdown']:6.2f}%")

        print(f"\n💎 TOP 5 ACCOUNTS BY TOTAL RETURN")
        print(f"{'─' * 80}")
        top_return = df.nlargest(5, 'total_return_pct')[['account_id', 'total_return_pct', 'sharpe_ratio', 'win_rate']]
        for idx, row in top_return.iterrows():
            print(f"  {row['account_id']:8s} | Return: {row['total_return_pct']:7.2f}% | Sharpe: {row['sharpe_ratio']:6.3f} | Win Rate: {row['win_rate']*100:5.2f}%")

    print_section("✓ METRICS CALCULATION COMPLETE", '█')
    print(f"\nAll metrics saved to database table: account_metrics")
    print(f"Ready for dynamic UI display!")

    conn.close()

if __name__ == '__main__':
    main()
