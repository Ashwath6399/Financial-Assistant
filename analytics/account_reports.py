"""
Account Performance Reports Generator
=====================================
Generates comprehensive performance reports for each trading account.

Features:
- Combines all metrics (performance, benchmark, risk)
- Shows ML model predictions vs actual results
- Tax reporting data (realized gains/losses by year)
- Exportable summary statistics

Ready for PDF/HTML generation or dynamic UI display.
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

def get_account_summary(conn, account_id):
    """Get comprehensive account summary from all metric tables"""

    # Get basic metrics
    account_metrics = pd.read_sql_query(
        "SELECT * FROM account_metrics WHERE account_id = ?",
        conn, params=(account_id,)
    )

    # Get benchmark comparison
    benchmark = pd.read_sql_query(
        "SELECT * FROM benchmark_comparison WHERE account_id = ?",
        conn, params=(account_id,)
    )

    # Get risk analytics
    risk = pd.read_sql_query(
        "SELECT * FROM risk_analytics WHERE account_id = ?",
        conn, params=(account_id,)
    )

    # Get ML predictions summary
    ml_predictions = pd.read_sql_query("""
        SELECT
            COUNT(*) as total_predictions,
            SUM(CASE WHEN recommendation = 'BUY' THEN 1 ELSE 0 END) as buy_recommendations,
            SUM(CASE WHEN recommendation = 'SELL' OR recommendation = 'AVOID' THEN 1 ELSE 0 END) as sell_recommendations,
            SUM(CASE WHEN recommendation = 'HOLD' THEN 1 ELSE 0 END) as hold_recommendations,
            AVG(hybrid_probability) as avg_confidence,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy
        FROM ml_predictions
        WHERE account_id = ?
    """, conn, params=(account_id,))

    summary = {
        'account_id': account_id,
        'metrics': account_metrics.iloc[0].to_dict() if len(account_metrics) > 0 else {},
        'benchmark': benchmark.iloc[0].to_dict() if len(benchmark) > 0 else {},
        'risk': risk.iloc[0].to_dict() if len(risk) > 0 else {},
        'ml_predictions': ml_predictions.iloc[0].to_dict() if len(ml_predictions) > 0 else {}
    }

    return summary

def get_tax_report(conn, account_id, tax_year=None):
    """Generate tax report showing realized gains/losses"""

    # Import trade calculation function
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

    # Calculate trades
    trades = calculate_trade_outcomes(fills)

    if len(trades) == 0:
        return pd.DataFrame()

    # Rename columns
    trades = trades.rename(columns={
        'net_pnl': 'pnl',
        'exit_ts': 'close_date',
        'entry_ts': 'open_date'
    })

    # Convert dates
    trades['close_date'] = pd.to_datetime(trades['close_date'])
    trades['open_date'] = pd.to_datetime(trades['open_date'])
    trades['tax_year'] = trades['close_date'].dt.year

    # Filter by tax year if specified
    if tax_year:
        trades = trades[trades['tax_year'] == tax_year]

    # Categorize by holding period
    trades['holding_period'] = (trades['close_date'] - trades['open_date']).dt.days
    trades['term'] = trades['holding_period'].apply(
        lambda x: 'Long-term' if x >= 365 else 'Short-term'
    )

    # Categorize gains/losses
    trades['type'] = trades['pnl'].apply(
        lambda x: 'Gain' if x > 0 else 'Loss'
    )

    return trades

def generate_tax_summary(trades_df):
    """Generate tax summary statistics"""

    if len(trades_df) == 0:
        return {
            'total_realized_pnl': 0,
            'short_term_gains': 0,
            'short_term_losses': 0,
            'long_term_gains': 0,
            'long_term_losses': 0,
            'net_short_term': 0,
            'net_long_term': 0,
            'total_trades': 0
        }

    # Short-term
    short_term = trades_df[trades_df['term'] == 'Short-term']
    st_gains = short_term[short_term['pnl'] > 0]['pnl'].sum()
    st_losses = short_term[short_term['pnl'] < 0]['pnl'].sum()

    # Long-term
    long_term = trades_df[trades_df['term'] == 'Long-term']
    lt_gains = long_term[long_term['pnl'] > 0]['pnl'].sum()
    lt_losses = long_term[long_term['pnl'] < 0]['pnl'].sum()

    return {
        'total_realized_pnl': trades_df['pnl'].sum(),
        'short_term_gains': st_gains,
        'short_term_losses': st_losses,
        'long_term_gains': lt_gains,
        'long_term_losses': lt_losses,
        'net_short_term': st_gains + st_losses,
        'net_long_term': lt_gains + lt_losses,
        'total_trades': len(trades_df),
        'short_term_trades': len(short_term),
        'long_term_trades': len(long_term)
    }

def display_account_report(conn, account_id):
    """Display comprehensive account report"""

    summary = get_account_summary(conn, account_id)
    metrics = summary['metrics']
    benchmark = summary['benchmark']
    risk = summary['risk']
    ml_pred = summary['ml_predictions']

    print(f"\n{'='*80}")
    print(f" ACCOUNT PERFORMANCE REPORT: {account_id}".center(80))
    print(f"{'='*80}")

    # Section 1: Performance Metrics
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"{'─'*80}")
    if metrics:
        print(f"  Total Trades:           {metrics.get('total_trades', 0):>10}")
        print(f"  Win Rate:               {metrics.get('win_rate', 0):>9.2f}%")
        print(f"  Total P&L:              ${metrics.get('net_pnl', 0):>10,.2f}")
        print(f"  Total Return:           {metrics.get('total_return_pct', 0):>9.2f}%")
        print(f"  Annualized Return:      {metrics.get('annualized_return_pct', 0):>9.2f}%")
        print(f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):>10.3f}")
        print(f"  Sortino Ratio:          {metrics.get('sortino_ratio', 0):>10.3f}")
        print(f"  Calmar Ratio:           {metrics.get('calmar_ratio', 0):>10.3f}")
    else:
        print("  No metrics available")

    # Section 2: Benchmark Comparison
    print(f"\n📈 BENCHMARK COMPARISON (vs S&P 500)")
    print(f"{'─'*80}")
    if benchmark:
        beat_market = "✓ BEAT MARKET" if benchmark.get('beat_market', 0) == 1 else "✗ UNDERPERFORMED"
        print(f"  Status:                 {beat_market}")
        print(f"  Account Return:         {benchmark.get('account_return_pct', 0):>9.2f}%")
        print(f"  SPY Return:             {benchmark.get('benchmark_return_pct', 0):>9.2f}%")
        print(f"  Outperformance:         {benchmark.get('outperformance_pct', 0):>+9.2f}pp")
        print(f"  Account Sharpe:         {benchmark.get('account_sharpe_ratio', 0):>10.3f}")
        print(f"  SPY Sharpe:             {benchmark.get('benchmark_sharpe_ratio', 0):>10.3f}")
        print(f"  Risk-Adj Outperf:       {benchmark.get('risk_adjusted_outperformance', 0):>+10.3f}")
    else:
        print("  No benchmark data available")

    # Section 3: Risk Analytics
    print(f"\n⚠ RISK ANALYTICS")
    print(f"{'─'*80}")
    if risk:
        dd_status = "⚠ IN DRAWDOWN" if risk.get('current_drawdown_pct', 0) < -1 else "✓ At Peak"
        current_streak = risk.get('current_streak', 0)
        streak_text = f"{abs(current_streak)} {'wins' if current_streak > 0 else 'losses'}"

        print(f"  Max Drawdown:           {risk.get('max_drawdown_pct', 0):>9.2f}%")
        print(f"  Current Drawdown:       {risk.get('current_drawdown_pct', 0):>9.2f}% ({dd_status})")
        print(f"  VaR 95%:                {risk.get('var_95_pct', 0):>9.2f}%")
        print(f"  VaR 99%:                {risk.get('var_99_pct', 0):>9.2f}%")
        print(f"  Volatility:             {metrics.get('volatility', 0)*100:>9.2f}%")
        print(f"  Max Win Streak:         {risk.get('max_win_streak', 0):>10}")
        print(f"  Max Loss Streak:        {risk.get('max_loss_streak', 0):>10}")
        print(f"  Current Streak:         {streak_text:>10}")
    else:
        print("  No risk data available")

    # Section 4: ML Predictions
    print(f"\n🤖 MACHINE LEARNING INSIGHTS")
    print(f"{'─'*80}")
    if ml_pred and ml_pred.get('total_predictions', 0) > 0:
        total = ml_pred.get('total_predictions', 1)
        buy_pct = (ml_pred.get('buy_recommendations', 0) / total) * 100
        sell_pct = (ml_pred.get('sell_recommendations', 0) / total) * 100
        hold_pct = (ml_pred.get('hold_recommendations', 0) / total) * 100

        print(f"  Total Predictions:      {ml_pred.get('total_predictions', 0):>10}")
        print(f"  Model Accuracy:         {ml_pred.get('accuracy', 0)*100:>9.2f}%")
        print(f"  Buy Recommendations:    {ml_pred.get('buy_recommendations', 0):>10} ({buy_pct:>5.1f}%)")
        print(f"  Hold Recommendations:   {ml_pred.get('hold_recommendations', 0):>10} ({hold_pct:>5.1f}%)")
        print(f"  Sell/Avoid Recomm:      {ml_pred.get('sell_recommendations', 0):>10} ({sell_pct:>5.1f}%)")
        print(f"  Avg Confidence:         {ml_pred.get('avg_confidence', 0)*100:>9.2f}%")
    else:
        print("  No ML predictions available")

    # Section 5: Tax Report (current year)
    current_year = datetime.now().year
    trades_current_year = get_tax_report(conn, account_id, current_year)
    tax_summary = generate_tax_summary(trades_current_year)

    print(f"\n💰 TAX REPORT ({current_year})")
    print(f"{'─'*80}")
    print(f"  Total Realized P&L:     ${tax_summary['total_realized_pnl']:>10,.2f}")
    print(f"  Total Trades:           {tax_summary['total_trades']:>10}")
    print(f"\n  Short-term (<1 year):")
    print(f"    Trades:               {tax_summary['short_term_trades']:>10}")
    print(f"    Gains:                ${tax_summary['short_term_gains']:>10,.2f}")
    print(f"    Losses:               ${tax_summary['short_term_losses']:>10,.2f}")
    print(f"    Net:                  ${tax_summary['net_short_term']:>10,.2f}")
    print(f"\n  Long-term (≥1 year):")
    print(f"    Trades:               {tax_summary['long_term_trades']:>10}")
    print(f"    Gains:                ${tax_summary['long_term_gains']:>10,.2f}")
    print(f"    Losses:               ${tax_summary['long_term_losses']:>10,.2f}")
    print(f"    Net:                  ${tax_summary['net_long_term']:>10,.2f}")

    print(f"\n{'='*80}\n")

def export_account_data(conn, account_id):
    """Export all account data to dictionary for JSON/CSV export"""

    summary = get_account_summary(conn, account_id)
    trades_all_years = get_tax_report(conn, account_id)
    tax_summary_all = generate_tax_summary(trades_all_years)

    export_data = {
        'account_id': account_id,
        'generated_at': datetime.now().isoformat(),
        'performance': summary['metrics'],
        'benchmark': summary['benchmark'],
        'risk': summary['risk'],
        'ml_predictions': summary['ml_predictions'],
        'tax_summary_all_time': tax_summary_all,
        'total_trades_closed': len(trades_all_years)
    }

    return export_data

def main():
    """Main execution"""

    print_section("ACCOUNT PERFORMANCE REPORTS GENERATOR", '█')

    conn = sqlite3.connect(DB_PATH)

    # Get all accounts
    cursor = conn.cursor()
    cursor.execute("SELECT account_id FROM account_metrics ORDER BY account_id")
    accounts = [row[0] for row in cursor.fetchall()]

    print(f"Generating reports for {len(accounts)} accounts...\n")

    # Generate reports for each account
    for account_id in accounts:
        display_account_report(conn, account_id)

    # Summary across all accounts
    print_section("PORTFOLIO SUMMARY (All Accounts)", '=')

    # Aggregate statistics
    all_metrics = pd.read_sql_query("SELECT * FROM account_metrics", conn)
    all_benchmark = pd.read_sql_query("SELECT * FROM benchmark_comparison", conn)
    all_risk = pd.read_sql_query("SELECT * FROM risk_analytics", conn)

    print(f"\n📊 AGGREGATE STATISTICS")
    print(f"{'─'*80}")
    print(f"  Total Accounts:         {len(accounts):>10}")
    print(f"  Total Trades:           {all_metrics['total_trades'].sum():>10,.0f}")
    print(f"  Total Net P&L:          ${all_metrics['net_pnl'].sum():>10,.2f}")
    print(f"  Avg Win Rate:           {all_metrics['win_rate'].mean():>9.2f}%")
    print(f"  Avg Sharpe Ratio:       {all_metrics['sharpe_ratio'].mean():>10.3f}")
    print(f"  Accounts Beat Market:   {all_benchmark['beat_market'].sum():>10} ({all_benchmark['beat_market'].sum()/len(all_benchmark)*100:.1f}%)")
    print(f"  Avg Max Drawdown:       {all_risk['max_drawdown_pct'].mean():>9.2f}%")
    print(f"  Accounts in Drawdown:   {len(all_risk[all_risk['current_drawdown_pct'] < -1]):>10} ({len(all_risk[all_risk['current_drawdown_pct'] < -1])/len(all_risk)*100:.1f}%)")

    print_section("✓ REPORTS GENERATION COMPLETE", '█')
    print(f"All account data available for export to PDF/HTML/CSV")
    print(f"Use export_account_data() function for programmatic access\n")

    conn.close()

if __name__ == '__main__':
    main()
