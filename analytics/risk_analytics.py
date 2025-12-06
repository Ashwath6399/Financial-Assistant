"""
Risk Analytics Dashboard
========================
Advanced risk metrics for each trading account:
- Drawdown periods and recovery analysis
- Value at Risk (VaR) at 95% and 99% confidence
- Enhanced win/loss streak distributions
- Risk concentration metrics

Stores results in database for dynamic UI display.
"""

import sqlite3
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

def print_section(title, char='='):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")

def create_risk_analytics_table(conn):
    """Create table to store risk analytics data"""
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS risk_analytics (
        account_id TEXT PRIMARY KEY,

        -- Drawdown Metrics
        max_drawdown_pct REAL,
        max_drawdown_duration_days INTEGER,
        avg_drawdown_pct REAL,
        avg_drawdown_duration_days REAL,
        num_drawdown_periods INTEGER,
        avg_recovery_days REAL,
        longest_recovery_days INTEGER,
        current_drawdown_pct REAL,

        -- Value at Risk (VaR)
        var_95_pct REAL,           -- 95% VaR (daily)
        var_99_pct REAL,           -- 99% VaR (daily)
        cvar_95_pct REAL,          -- Conditional VaR (Expected Shortfall)
        cvar_99_pct REAL,

        -- Streak Analysis
        max_win_streak INTEGER,
        max_loss_streak INTEGER,
        avg_win_streak REAL,
        avg_loss_streak REAL,
        current_streak INTEGER,    -- Positive = wins, Negative = losses

        -- Risk Concentration
        largest_win_pct_of_total REAL,
        largest_loss_pct_of_total REAL,
        top_5_trades_pct_of_total REAL,

        -- Trade Size Risk
        avg_position_size REAL,
        max_position_size REAL,
        position_size_std REAL,

        -- Updated timestamp
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    )
    """)

    conn.commit()
    print("✓ Created risk_analytics table")

def get_account_trades(conn, account_id):
    """Get all closed trades for an account, sorted by close date"""

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

    # Calculate trade outcomes using FIFO matching
    trades = calculate_trade_outcomes(fills)

    if len(trades) == 0:
        return pd.DataFrame()

    # All trades returned are closed by definition
    # Rename columns to match expected names
    trades = trades.rename(columns={
        'net_pnl': 'pnl',
        'exit_ts': 'close_date'
    })

    # Calculate pnl_pct if not exists
    if 'pnl_pct' not in trades.columns:
        trades['pnl_pct'] = (trades['pnl'] / (abs(trades['entry_price'] * trades['qty']))) * 100

    # Sort by close date
    if 'close_date' in trades.columns:
        trades['close_date'] = pd.to_datetime(trades['close_date'])
        trades = trades.sort_values('close_date')
    elif 'entry_ts' in trades.columns:
        trades = trades.sort_values('entry_ts')

    # Calculate cumulative P&L
    if len(trades) > 0:
        trades['cumulative_pnl'] = trades['pnl'].cumsum()

    return trades

def calculate_drawdown_metrics(trades_df, initial_capital=10000):
    """
    Calculate comprehensive drawdown metrics

    Uses initial capital to create equity curve for proper % calculations.
    Default initial_capital=10000 if not provided.
    """

    if len(trades_df) == 0:
        return {
            'max_drawdown_pct': 0,
            'max_drawdown_duration_days': 0,
            'avg_drawdown_pct': 0,
            'avg_drawdown_duration_days': 0,
            'num_drawdown_periods': 0,
            'avg_recovery_days': 0,
            'longest_recovery_days': 0,
            'current_drawdown_pct': 0
        }

    cumulative_pnl = trades_df['cumulative_pnl'].values

    # Use close_date if available, otherwise entry_ts
    if 'close_date' in trades_df.columns:
        dates = pd.to_datetime(trades_df['close_date']).values
    elif 'entry_ts' in trades_df.columns:
        dates = pd.to_datetime(trades_df['entry_ts']).values
    else:
        # No date column, return zeros
        return {
            'max_drawdown_pct': 0,
            'max_drawdown_duration_days': 0,
            'avg_drawdown_pct': 0,
            'avg_drawdown_duration_days': 0,
            'num_drawdown_periods': 0,
            'avg_recovery_days': 0,
            'longest_recovery_days': 0,
            'current_drawdown_pct': 0
        }

    # Create equity curve: initial capital + cumulative P&L
    equity_curve = initial_capital + cumulative_pnl

    # Track running max equity and drawdowns
    running_max_equity = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max_equity

    # Calculate drawdown as % of peak equity (standard finance definition)
    drawdown_pct = np.where(running_max_equity > 0, (drawdown / running_max_equity) * 100, 0)

    # Max drawdown
    max_drawdown_pct = drawdown_pct.min()
    max_dd_idx = drawdown_pct.argmin()

    # Find all drawdown periods (when not at peak)
    is_drawdown = drawdown < -0.01  # Small threshold to avoid noise

    drawdown_periods = []
    in_drawdown = False
    dd_start_idx = 0
    peak_value = 0

    for i in range(len(equity_curve)):
        if is_drawdown[i] and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            dd_start_idx = i
            peak_value = running_max_equity[i]

        elif not is_drawdown[i] and in_drawdown:
            # End of drawdown (recovery)
            in_drawdown = False
            dd_end_idx = i - 1

            # Calculate drawdown stats
            dd_depth = drawdown_pct[dd_start_idx:dd_end_idx+1].min()
            dd_duration = (dates[dd_end_idx] - dates[dd_start_idx]).astype('timedelta64[D]').astype(int)

            # Find recovery point
            recovery_idx = i
            recovery_duration = (dates[recovery_idx] - dates[dd_end_idx]).astype('timedelta64[D]').astype(int)

            drawdown_periods.append({
                'depth_pct': dd_depth,
                'duration_days': dd_duration,
                'recovery_days': recovery_duration
            })

    # If still in drawdown at end
    current_drawdown_pct = drawdown_pct[-1] if len(drawdown_pct) > 0 else 0

    # Calculate statistics
    if len(drawdown_periods) > 0:
        avg_dd_pct = np.mean([dd['depth_pct'] for dd in drawdown_periods])
        avg_dd_duration = np.mean([dd['duration_days'] for dd in drawdown_periods])
        avg_recovery = np.mean([dd['recovery_days'] for dd in drawdown_periods])
        longest_recovery = max([dd['recovery_days'] for dd in drawdown_periods])

        # Max drawdown duration
        max_dd_period = min(drawdown_periods, key=lambda x: x['depth_pct'])
        max_dd_duration = max_dd_period['duration_days']
    else:
        avg_dd_pct = 0
        avg_dd_duration = 0
        avg_recovery = 0
        longest_recovery = 0
        max_dd_duration = 0

    return {
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_duration_days': int(max_dd_duration),
        'avg_drawdown_pct': avg_dd_pct,
        'avg_drawdown_duration_days': avg_dd_duration,
        'num_drawdown_periods': len(drawdown_periods),
        'avg_recovery_days': avg_recovery,
        'longest_recovery_days': int(longest_recovery),
        'current_drawdown_pct': current_drawdown_pct
    }

def calculate_var_metrics(trades_df):
    """Calculate Value at Risk (VaR) and Conditional VaR"""

    if len(trades_df) < 2:
        return {
            'var_95_pct': 0,
            'var_99_pct': 0,
            'cvar_95_pct': 0,
            'cvar_99_pct': 0
        }

    # Use P&L percentage as returns
    returns = trades_df['pnl_pct'].values

    # Calculate VaR at 95% and 99% confidence
    var_95 = np.percentile(returns, 5)  # 5th percentile (95% confidence)
    var_99 = np.percentile(returns, 1)  # 1st percentile (99% confidence)

    # Conditional VaR (Expected Shortfall) - average of losses beyond VaR
    losses_beyond_95 = returns[returns <= var_95]
    losses_beyond_99 = returns[returns <= var_99]

    cvar_95 = losses_beyond_95.mean() if len(losses_beyond_95) > 0 else var_95
    cvar_99 = losses_beyond_99.mean() if len(losses_beyond_99) > 0 else var_99

    return {
        'var_95_pct': var_95,
        'var_99_pct': var_99,
        'cvar_95_pct': cvar_95,
        'cvar_99_pct': cvar_99
    }

def calculate_streak_metrics(trades_df):
    """Calculate enhanced win/loss streak metrics"""

    if len(trades_df) == 0:
        return {
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'avg_win_streak': 0,
            'avg_loss_streak': 0,
            'current_streak': 0
        }

    # Determine win/loss for each trade
    is_win = trades_df['pnl'] > 0

    # Track streaks
    win_streaks = []
    loss_streaks = []
    current_streak = 0
    current_streak_type = None

    for win in is_win:
        if win:
            if current_streak_type == 'win':
                current_streak += 1
            else:
                if current_streak_type == 'loss':
                    loss_streaks.append(current_streak)
                current_streak = 1
                current_streak_type = 'win'
        else:
            if current_streak_type == 'loss':
                current_streak += 1
            else:
                if current_streak_type == 'win':
                    win_streaks.append(current_streak)
                current_streak = 1
                current_streak_type = 'loss'

    # Add final streak
    if current_streak_type == 'win':
        win_streaks.append(current_streak)
    elif current_streak_type == 'loss':
        loss_streaks.append(current_streak)

    # Calculate statistics
    max_win_streak = max(win_streaks) if len(win_streaks) > 0 else 0
    max_loss_streak = max(loss_streaks) if len(loss_streaks) > 0 else 0
    avg_win_streak = np.mean(win_streaks) if len(win_streaks) > 0 else 0
    avg_loss_streak = np.mean(loss_streaks) if len(loss_streaks) > 0 else 0

    # Current streak (positive = wins, negative = losses)
    final_streak = current_streak if current_streak_type == 'win' else -current_streak

    return {
        'max_win_streak': int(max_win_streak),
        'max_loss_streak': int(max_loss_streak),
        'avg_win_streak': avg_win_streak,
        'avg_loss_streak': avg_loss_streak,
        'current_streak': int(final_streak)
    }

def calculate_concentration_metrics(trades_df):
    """Calculate risk concentration metrics"""

    if len(trades_df) == 0 or trades_df['pnl'].sum() == 0:
        return {
            'largest_win_pct_of_total': 0,
            'largest_loss_pct_of_total': 0,
            'top_5_trades_pct_of_total': 0
        }

    total_pnl = trades_df['pnl'].sum()

    # Largest win and loss as % of total P&L
    largest_win = trades_df['pnl'].max()
    largest_loss = trades_df['pnl'].min()

    largest_win_pct = (largest_win / abs(total_pnl)) * 100 if total_pnl != 0 else 0
    largest_loss_pct = (abs(largest_loss) / abs(total_pnl)) * 100 if total_pnl != 0 else 0

    # Top 5 trades contribution
    top_5_pnl = trades_df.nlargest(5, 'pnl')['pnl'].sum()
    top_5_pct = (top_5_pnl / abs(total_pnl)) * 100 if total_pnl != 0 else 0

    return {
        'largest_win_pct_of_total': largest_win_pct,
        'largest_loss_pct_of_total': largest_loss_pct,
        'top_5_trades_pct_of_total': top_5_pct
    }

def calculate_position_size_metrics(trades_df):
    """Calculate position size risk metrics"""

    if len(trades_df) == 0:
        return {
            'avg_position_size': 0,
            'max_position_size': 0,
            'position_size_std': 0
        }

    position_sizes = trades_df['qty'].abs()

    return {
        'avg_position_size': position_sizes.mean(),
        'max_position_size': position_sizes.max(),
        'position_size_std': position_sizes.std()
    }

def analyze_account_risk(conn, account_id):
    """Comprehensive risk analysis for an account"""

    # Get trades
    trades_df = get_account_trades(conn, account_id)

    if len(trades_df) == 0:
        print(f"  ⚠ No trades found for {account_id}")
        return None

    # Get initial capital from account_metrics
    # Initial capital = net_pnl / (total_return_pct / 100)
    account_metrics = pd.read_sql_query(
        "SELECT net_pnl, total_return_pct FROM account_metrics WHERE account_id = ?",
        conn, params=(account_id,)
    )

    if len(account_metrics) > 0 and account_metrics.iloc[0]['total_return_pct'] != 0:
        net_pnl = account_metrics.iloc[0]['net_pnl']
        total_return_pct = account_metrics.iloc[0]['total_return_pct']
        initial_capital = net_pnl / (total_return_pct / 100)
    else:
        # Default to $10,000 if we can't calculate
        initial_capital = 10000

    # Calculate all risk metrics
    risk_metrics = {
        'account_id': account_id
    }

    # Drawdown analysis (now using proper initial capital)
    drawdown_metrics = calculate_drawdown_metrics(trades_df, initial_capital=initial_capital)
    risk_metrics.update(drawdown_metrics)

    # VaR analysis
    var_metrics = calculate_var_metrics(trades_df)
    risk_metrics.update(var_metrics)

    # Streak analysis
    streak_metrics = calculate_streak_metrics(trades_df)
    risk_metrics.update(streak_metrics)

    # Concentration analysis
    concentration_metrics = calculate_concentration_metrics(trades_df)
    risk_metrics.update(concentration_metrics)

    # Position size analysis
    position_metrics = calculate_position_size_metrics(trades_df)
    risk_metrics.update(position_metrics)

    return risk_metrics

def save_risk_analytics_to_db(conn, risk_metrics):
    """Save risk analytics to database"""

    cursor = conn.cursor()

    # Delete existing analytics for this account
    cursor.execute("DELETE FROM risk_analytics WHERE account_id = ?", (risk_metrics['account_id'],))

    # Insert new analytics
    columns = ', '.join(risk_metrics.keys())
    placeholders = ', '.join(['?' for _ in risk_metrics])

    cursor.execute(f"""
        INSERT INTO risk_analytics ({columns})
        VALUES ({placeholders})
    """, tuple(risk_metrics.values()))

    conn.commit()

def display_risk_analytics(risk_metrics):
    """Display risk analytics in formatted output"""

    account_id = risk_metrics['account_id']

    print(f"\n📊 {account_id}")
    print(f"{'─' * 80}")

    # Drawdown Analysis
    print(f"\n  Drawdown Analysis:")
    print(f"    Max Drawdown:           {risk_metrics['max_drawdown_pct']:7.2f}%")
    print(f"    Max DD Duration:        {risk_metrics['max_drawdown_duration_days']:3d} days")
    print(f"    Avg Drawdown:           {risk_metrics['avg_drawdown_pct']:7.2f}%")
    print(f"    Drawdown Periods:       {risk_metrics['num_drawdown_periods']:3d}")
    print(f"    Avg Recovery Time:      {risk_metrics['avg_recovery_days']:7.1f} days")
    print(f"    Longest Recovery:       {risk_metrics['longest_recovery_days']:3d} days")

    current_dd = risk_metrics['current_drawdown_pct']
    dd_status = "⚠ IN DRAWDOWN" if current_dd < -1 else "✓ At Peak"
    print(f"    Current Drawdown:       {current_dd:7.2f}% ({dd_status})")

    # Value at Risk
    print(f"\n  Value at Risk (VaR):")
    print(f"    VaR 95%:                {risk_metrics['var_95_pct']:7.2f}%")
    print(f"    VaR 99%:                {risk_metrics['var_99_pct']:7.2f}%")
    print(f"    CVaR 95%:               {risk_metrics['cvar_95_pct']:7.2f}%")
    print(f"    CVaR 99%:               {risk_metrics['cvar_99_pct']:7.2f}%")

    # Streak Analysis
    print(f"\n  Win/Loss Streaks:")
    print(f"    Max Win Streak:         {risk_metrics['max_win_streak']:3d}")
    print(f"    Max Loss Streak:        {risk_metrics['max_loss_streak']:3d}")
    print(f"    Avg Win Streak:         {risk_metrics['avg_win_streak']:6.2f}")
    print(f"    Avg Loss Streak:        {risk_metrics['avg_loss_streak']:6.2f}")

    current_streak = risk_metrics['current_streak']
    streak_text = f"{abs(current_streak)} {'wins' if current_streak > 0 else 'losses'}"
    streak_icon = "🟢" if current_streak > 0 else "🔴"
    print(f"    Current Streak:         {streak_icon} {streak_text}")

    # Risk Concentration
    print(f"\n  Risk Concentration:")
    print(f"    Largest Win (% of P&L): {risk_metrics['largest_win_pct_of_total']:7.2f}%")
    print(f"    Largest Loss (% of P&L):{risk_metrics['largest_loss_pct_of_total']:7.2f}%")
    print(f"    Top 5 Trades (% of P&L):{risk_metrics['top_5_trades_pct_of_total']:7.2f}%")

    # Position Sizing
    print(f"\n  Position Sizing:")
    print(f"    Avg Position Size:      {risk_metrics['avg_position_size']:10.2f}")
    print(f"    Max Position Size:      {risk_metrics['max_position_size']:10.2f}")
    print(f"    Position Size Std Dev:  {risk_metrics['position_size_std']:10.2f}")

def main():
    """Main execution"""

    print_section("RISK ANALYTICS DASHBOARD", '█')

    conn = sqlite3.connect(DB_PATH)

    # Step 1: Create risk analytics table
    print_section("STEP 1: Creating Risk Analytics Table")
    create_risk_analytics_table(conn)

    # Step 2: Analyze each account
    print_section("STEP 2: Analyzing Risk Metrics for Each Account")

    cursor = conn.cursor()
    cursor.execute("SELECT account_id FROM account_metrics ORDER BY account_id")
    accounts = [row[0] for row in cursor.fetchall()]

    print(f"Analyzing risk for {len(accounts)} accounts...\n")

    all_risk_metrics = []

    for account_id in accounts:
        risk_metrics = analyze_account_risk(conn, account_id)

        if risk_metrics:
            # Save to database
            save_risk_analytics_to_db(conn, risk_metrics)

            # Display
            display_risk_analytics(risk_metrics)

            all_risk_metrics.append(risk_metrics)

    # Step 3: Summary Statistics
    print_section("STEP 3: Risk Summary Across All Accounts")

    if len(all_risk_metrics) > 0:
        df = pd.DataFrame(all_risk_metrics)

        print(f"\n📊 AGGREGATE RISK STATISTICS")
        print(f"{'─' * 80}")
        print(f"  Accounts Analyzed:           {len(all_risk_metrics)}")

        print(f"\n  Drawdown Statistics:")
        print(f"    Worst Max Drawdown:          {df['max_drawdown_pct'].min():7.2f}% ({df.loc[df['max_drawdown_pct'].idxmin(), 'account_id']})")
        print(f"    Best Max Drawdown:           {df['max_drawdown_pct'].max():7.2f}% ({df.loc[df['max_drawdown_pct'].idxmax(), 'account_id']})")
        print(f"    Avg Max Drawdown:            {df['max_drawdown_pct'].mean():7.2f}%")
        print(f"    Median Max Drawdown:         {df['max_drawdown_pct'].median():7.2f}%")

        print(f"\n  Value at Risk (VaR 95%):")
        print(f"    Worst VaR 95%:               {df['var_95_pct'].min():7.2f}% ({df.loc[df['var_95_pct'].idxmin(), 'account_id']})")
        print(f"    Best VaR 95%:                {df['var_95_pct'].max():7.2f}% ({df.loc[df['var_95_pct'].idxmax(), 'account_id']})")
        print(f"    Avg VaR 95%:                 {df['var_95_pct'].mean():7.2f}%")

        print(f"\n  Streak Analysis:")
        print(f"    Longest Win Streak:          {df['max_win_streak'].max():3d} trades ({df.loc[df['max_win_streak'].idxmax(), 'account_id']})")
        print(f"    Longest Loss Streak:         {df['max_loss_streak'].max():3d} trades ({df.loc[df['max_loss_streak'].idxmax(), 'account_id']})")
        print(f"    Avg Max Win Streak:          {df['max_win_streak'].mean():6.2f} trades")
        print(f"    Avg Max Loss Streak:         {df['max_loss_streak'].mean():6.2f} trades")

        # Accounts in drawdown
        in_drawdown = df[df['current_drawdown_pct'] < -1]
        print(f"\n  Current Status:")
        print(f"    Accounts in Drawdown:        {len(in_drawdown)} ({len(in_drawdown)/len(df)*100:.1f}%)")
        print(f"    Accounts at Peak:            {len(df) - len(in_drawdown)} ({(len(df) - len(in_drawdown))/len(df)*100:.1f}%)")

        # Risk concentration
        print(f"\n  Risk Concentration:")
        print(f"    Avg Top 5 Contribution:      {df['top_5_trades_pct_of_total'].mean():7.2f}%")
        print(f"    Most Concentrated:           {df['top_5_trades_pct_of_total'].max():7.2f}% ({df.loc[df['top_5_trades_pct_of_total'].idxmax(), 'account_id']})")
        print(f"    Least Concentrated:          {df['top_5_trades_pct_of_total'].min():7.2f}% ({df.loc[df['top_5_trades_pct_of_total'].idxmin(), 'account_id']})")

        # Top risk-managed accounts (smallest drawdown)
        print(f"\n🏆 TOP 5 BEST RISK-MANAGED ACCOUNTS (Smallest Max Drawdown)")
        print(f"{'─' * 80}")
        best_risk = df.nlargest(5, 'max_drawdown_pct')  # Largest because drawdown is negative

        for idx, row in best_risk.iterrows():
            print(f"  {row['account_id']:8s} | Max DD: {row['max_drawdown_pct']:7.2f}% | VaR95: {row['var_95_pct']:7.2f}% | Sharpe: Available in account_metrics")

        # Highest risk accounts
        print(f"\n⚠ TOP 5 HIGHEST RISK ACCOUNTS (Largest Max Drawdown)")
        print(f"{'─' * 80}")
        highest_risk = df.nsmallest(5, 'max_drawdown_pct')

        for idx, row in highest_risk.iterrows():
            print(f"  {row['account_id']:8s} | Max DD: {row['max_drawdown_pct']:7.2f}% | VaR95: {row['var_95_pct']:7.2f}% | Recovery: {row['longest_recovery_days']:3d} days")

    print_section("✓ RISK ANALYTICS COMPLETE", '█')
    print(f"\nAll risk analytics saved to database table: risk_analytics")
    print(f"Ready for dynamic UI display!")

    conn.close()

if __name__ == '__main__':
    main()
