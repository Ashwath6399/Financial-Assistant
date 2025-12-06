"""
Benchmark Comparison Engine
===========================
Compares each trading account's performance against S&P 500 (SPY) buy-and-hold.

Calculates:
- Buy-and-hold SPY returns for the same period
- Outperformance/underperformance vs benchmark
- Risk-adjusted comparison (Sharpe ratio)
- Whether account beat the market

Stores results in database for dynamic UI display.
"""

import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

def print_section(title, char='='):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")

def create_benchmark_table(conn):
    """Create table to store benchmark comparison data"""
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS benchmark_comparison (
        account_id TEXT PRIMARY KEY,

        -- Account Performance
        account_return_pct REAL,
        account_annualized_return_pct REAL,
        account_sharpe_ratio REAL,
        account_volatility REAL,

        -- Benchmark Performance (SPY)
        benchmark_return_pct REAL,
        benchmark_annualized_return_pct REAL,
        benchmark_sharpe_ratio REAL,
        benchmark_volatility REAL,

        -- Comparison
        outperformance_pct REAL,
        outperformance_annualized_pct REAL,
        beat_market INTEGER,  -- 1 if beat, 0 if not
        risk_adjusted_outperformance REAL,  -- Sharpe diff

        -- Period
        start_date TEXT,
        end_date TEXT,
        trading_days INTEGER,

        -- Updated timestamp
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    )
    """)

    conn.commit()
    print("✓ Created benchmark_comparison table")

def get_overall_trading_period(conn):
    """Get the overall trading period across all accounts"""

    query = """
    SELECT
        MIN(trade_ts) as earliest_trade,
        MAX(trade_ts) as latest_trade
    FROM fills
    """

    df = pd.read_sql_query(query, conn)

    start_date = pd.to_datetime(df['earliest_trade'].iloc[0]).date()
    end_date = pd.to_datetime(df['latest_trade'].iloc[0]).date()

    return start_date, end_date

def fetch_spy_data(start_date, end_date):
    """Fetch S&P 500 (SPY) data from Yahoo Finance"""

    print(f"Fetching SPY data from {start_date} to {end_date}...")

    # Add buffer to ensure we have data
    start_with_buffer = start_date - timedelta(days=30)

    try:
        spy = yf.download('SPY', start=start_with_buffer, end=end_date + timedelta(days=1), progress=False, auto_adjust=True)

        if spy.empty:
            print("⚠ No SPY data fetched")
            return None

        # Handle different column formats from yfinance
        # With auto_adjust=True, it uses 'Close' instead of 'Adj Close'
        if 'Close' in spy.columns:
            spy = spy[['Close']].copy()
            spy.columns = ['price']
        elif 'Adj Close' in spy.columns:
            spy = spy[['Adj Close']].copy()
            spy.columns = ['price']
        else:
            # Try flattening multi-index columns
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.droplevel(1)
                if 'Close' in spy.columns:
                    spy = spy[['Close']].copy()
                    spy.columns = ['price']
                elif 'Adj Close' in spy.columns:
                    spy = spy[['Adj Close']].copy()
                    spy.columns = ['price']

        print(f"✓ Fetched {len(spy)} days of SPY data")

        return spy

    except Exception as e:
        print(f"✗ Error fetching SPY data: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_spy_returns(spy_data, start_date, end_date):
    """Calculate buy-and-hold returns for SPY"""

    # Filter to exact period
    spy_period = spy_data.loc[start_date:end_date]

    if len(spy_period) == 0:
        print(f"⚠ No SPY data for period {start_date} to {end_date}")
        return None

    # Calculate returns
    initial_price = spy_period['price'].iloc[0]
    final_price = spy_period['price'].iloc[-1]

    total_return = ((final_price - initial_price) / initial_price) * 100

    # Calculate daily returns for Sharpe ratio
    spy_period['daily_return'] = spy_period['price'].pct_change()

    # Annualized return
    days = (end_date - start_date).days
    years = days / 365.25

    if years > 0:
        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    else:
        annualized_return = total_return

    # Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    daily_returns = spy_period['daily_return'].dropna()

    if len(daily_returns) > 1:
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()

        if std_return > 0:
            sharpe = (mean_return - risk_free_rate/252) / std_return * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Volatility
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0

    results = {
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'trading_days': len(spy_period)
    }

    return results

def get_account_period(conn, account_id):
    """Get trading period for specific account"""

    query = """
    SELECT
        MIN(trade_ts) as start_date,
        MAX(trade_ts) as end_date
    FROM fills
    WHERE account_id = ?
    """

    df = pd.read_sql_query(query, conn, params=(account_id,))

    start_date = pd.to_datetime(df['start_date'].iloc[0]).date()
    end_date = pd.to_datetime(df['end_date'].iloc[0]).date()

    return start_date, end_date

def compare_account_to_benchmark(conn, account_id, spy_data):
    """Compare account performance to SPY benchmark"""

    # Get account metrics
    query = """
    SELECT * FROM account_metrics WHERE account_id = ?
    """

    account_metrics = pd.read_sql_query(query, conn, params=(account_id,))

    if len(account_metrics) == 0:
        print(f"  ⚠ No metrics found for {account_id}")
        return None

    account_metrics = account_metrics.iloc[0]

    # Get account period
    start_date, end_date = get_account_period(conn, account_id)

    # Calculate SPY returns for same period
    spy_results = calculate_spy_returns(spy_data, start_date, end_date)

    if spy_results is None:
        print(f"  ⚠ Could not calculate SPY returns for {account_id}")
        return None

    # Compare
    outperformance = account_metrics['total_return_pct'] - spy_results['total_return_pct']
    outperformance_annualized = account_metrics['annualized_return_pct'] - spy_results['annualized_return_pct']

    beat_market = 1 if outperformance > 0 else 0

    # Risk-adjusted comparison
    risk_adjusted_outperformance = account_metrics['sharpe_ratio'] - spy_results['sharpe_ratio']

    comparison = {
        'account_id': account_id,
        'account_return_pct': account_metrics['total_return_pct'],
        'account_annualized_return_pct': account_metrics['annualized_return_pct'],
        'account_sharpe_ratio': account_metrics['sharpe_ratio'],
        'account_volatility': account_metrics['volatility'],
        'benchmark_return_pct': spy_results['total_return_pct'],
        'benchmark_annualized_return_pct': spy_results['annualized_return_pct'],
        'benchmark_sharpe_ratio': spy_results['sharpe_ratio'],
        'benchmark_volatility': spy_results['volatility'],
        'outperformance_pct': outperformance,
        'outperformance_annualized_pct': outperformance_annualized,
        'beat_market': beat_market,
        'risk_adjusted_outperformance': risk_adjusted_outperformance,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'trading_days': spy_results['trading_days']
    }

    return comparison

def save_comparison_to_db(conn, comparison):
    """Save benchmark comparison to database"""

    cursor = conn.cursor()

    # Delete existing comparison for this account
    cursor.execute("DELETE FROM benchmark_comparison WHERE account_id = ?", (comparison['account_id'],))

    # Insert new comparison
    columns = ', '.join(comparison.keys())
    placeholders = ', '.join(['?' for _ in comparison])

    cursor.execute(f"""
        INSERT INTO benchmark_comparison ({columns})
        VALUES ({placeholders})
    """, tuple(comparison.values()))

    conn.commit()

def display_comparison(comparison):
    """Display comparison in formatted output"""

    account_id = comparison['account_id']
    beat = "✓ BEAT" if comparison['beat_market'] else "✗ UNDERPERFORMED"
    color = "🟢" if comparison['beat_market'] else "🔴"

    print(f"\n{color} {account_id} - {beat} THE MARKET")
    print(f"{'─' * 80}")

    print(f"\n  Account Performance:")
    print(f"    Total Return:        {comparison['account_return_pct']:7.2f}%")
    print(f"    Annualized Return:   {comparison['account_annualized_return_pct']:7.2f}%")
    print(f"    Sharpe Ratio:        {comparison['account_sharpe_ratio']:7.3f}")
    print(f"    Volatility:          {comparison['account_volatility']*100:7.2f}%")

    print(f"\n  SPY Benchmark:")
    print(f"    Total Return:        {comparison['benchmark_return_pct']:7.2f}%")
    print(f"    Annualized Return:   {comparison['benchmark_annualized_return_pct']:7.2f}%")
    print(f"    Sharpe Ratio:        {comparison['benchmark_sharpe_ratio']:7.3f}")
    print(f"    Volatility:          {comparison['benchmark_volatility']*100:7.2f}%")

    print(f"\n  Outperformance:")
    print(f"    Total:               {comparison['outperformance_pct']:+7.2f}pp")
    print(f"    Annualized:          {comparison['outperformance_annualized_pct']:+7.2f}pp")
    print(f"    Risk-Adjusted:       {comparison['risk_adjusted_outperformance']:+7.3f} (Sharpe diff)")

    print(f"\n  Period: {comparison['start_date']} to {comparison['end_date']} ({comparison['trading_days']} days)")

def main():
    """Main execution"""

    print_section("BENCHMARK COMPARISON ENGINE", '█')

    conn = sqlite3.connect(DB_PATH)

    # Step 1: Create benchmark table
    print_section("STEP 1: Creating Benchmark Comparison Table")
    create_benchmark_table(conn)

    # Step 2: Get overall trading period
    print_section("STEP 2: Fetching S&P 500 (SPY) Benchmark Data")

    start_date, end_date = get_overall_trading_period(conn)
    print(f"Overall trading period: {start_date} to {end_date}")

    # Fetch SPY data
    spy_data = fetch_spy_data(start_date, end_date)

    if spy_data is None:
        print("✗ Failed to fetch SPY data. Exiting.")
        conn.close()
        return

    # Step 3: Compare each account
    print_section("STEP 3: Comparing Each Account to Benchmark")

    cursor = conn.cursor()
    cursor.execute("SELECT account_id FROM account_metrics ORDER BY account_id")
    accounts = [row[0] for row in cursor.fetchall()]

    print(f"Comparing {len(accounts)} accounts to SPY benchmark...\n")

    all_comparisons = []
    accounts_beat_market = 0

    for account_id in accounts:
        comparison = compare_account_to_benchmark(conn, account_id, spy_data)

        if comparison:
            # Save to database
            save_comparison_to_db(conn, comparison)

            # Display
            display_comparison(comparison)

            all_comparisons.append(comparison)

            if comparison['beat_market']:
                accounts_beat_market += 1

    # Step 4: Summary
    print_section("STEP 4: Summary Statistics")

    if len(all_comparisons) > 0:
        df = pd.DataFrame(all_comparisons)

        print(f"\n📊 OVERALL BENCHMARK COMPARISON")
        print(f"{'─' * 80}")
        print(f"  Accounts Analyzed:       {len(all_comparisons)}")
        print(f"  Accounts Beat Market:    {accounts_beat_market} ({accounts_beat_market/len(all_comparisons)*100:.1f}%)")
        print(f"  Accounts Underperformed: {len(all_comparisons) - accounts_beat_market} ({(len(all_comparisons) - accounts_beat_market)/len(all_comparisons)*100:.1f}%)")

        print(f"\n  Average Outperformance:  {df['outperformance_pct'].mean():+.2f}pp")
        print(f"  Median Outperformance:   {df['outperformance_pct'].median():+.2f}pp")
        print(f"  Best Outperformance:     {df['outperformance_pct'].max():+.2f}pp ({df.loc[df['outperformance_pct'].idxmax(), 'account_id']})")
        print(f"  Worst Outperformance:    {df['outperformance_pct'].min():+.2f}pp ({df.loc[df['outperformance_pct'].idxmin(), 'account_id']})")

        # Top performers vs benchmark
        print(f"\n🏆 TOP 5 ACCOUNTS VS BENCHMARK (by Outperformance)")
        print(f"{'─' * 80}")
        top_performers = df.nlargest(5, 'outperformance_pct')

        for idx, row in top_performers.iterrows():
            beat_icon = "🟢" if row['beat_market'] else "🔴"
            print(f"  {beat_icon} {row['account_id']:8s} | Account: {row['account_return_pct']:7.2f}% | SPY: {row['benchmark_return_pct']:7.2f}% | Diff: {row['outperformance_pct']:+7.2f}pp")

        # Risk-adjusted comparison
        print(f"\n⚡ TOP 5 ACCOUNTS (Risk-Adjusted Outperformance)")
        print(f"{'─' * 80}")
        top_risk_adjusted = df.nlargest(5, 'risk_adjusted_outperformance')

        for idx, row in top_risk_adjusted.iterrows():
            print(f"  {row['account_id']:8s} | Account Sharpe: {row['account_sharpe_ratio']:6.3f} | SPY Sharpe: {row['benchmark_sharpe_ratio']:6.3f} | Diff: {row['risk_adjusted_outperformance']:+7.3f}")

    print_section("✓ BENCHMARK COMPARISON COMPLETE", '█')
    print(f"\nAll comparisons saved to database table: benchmark_comparison")
    print(f"Ready for dynamic UI display!")

    conn.close()

if __name__ == '__main__':
    main()
