"""
Strategy Comparison Analysis
============================
Compares trading strategy performance and ML model recommendations.

Features:
- Actual performance by strategy (RSI, Moving Average, Discretionary)
- What-if analysis: "If I followed the model 100%"
- Strategy effectiveness scoring
- Recommendation vs actual outcome analysis

Helps answer: "Which strategy worked best?" and "Should I trust the ML model?"
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

def get_strategy_performance(conn):
    """Get performance by trading strategy"""

    # Import trade calculation function
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_models'))
    from hybrid_model import calculate_trade_outcomes

    # Get all fills
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
    ORDER BY f.trade_ts
    """

    fills = pd.read_sql_query(query, conn)

    if len(fills) == 0:
        return pd.DataFrame()

    # Calculate trades
    trades = calculate_trade_outcomes(fills)

    if len(trades) == 0:
        return pd.DataFrame()

    # Rename columns
    trades = trades.rename(columns={
        'net_pnl': 'pnl',
        'is_profitable': 'is_win'
    })

    # Determine strategy from source file
    def extract_strategy(row):
        strategy = str(row.get('strategy', '')).lower()
        if 'rsi' in strategy:
            return 'RSI'
        elif 'ma' in strategy or 'moving' in strategy:
            return 'Moving Average'
        elif 'disc' in strategy:
            return 'Discretionary'
        else:
            return 'Other'

    trades['strategy_name'] = trades.apply(extract_strategy, axis=1)

    return trades

def analyze_strategy_performance(trades_df):
    """Analyze performance by strategy"""

    if len(trades_df) == 0:
        return pd.DataFrame()

    # Group by strategy
    strategy_stats = []

    for strategy in trades_df['strategy_name'].unique():
        strategy_trades = trades_df[trades_df['strategy_name'] == strategy]

        if len(strategy_trades) == 0:
            continue

        total_trades = len(strategy_trades)
        wins = strategy_trades['is_win'].sum()
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100

        total_pnl = strategy_trades['pnl'].sum()
        avg_pnl = strategy_trades['pnl'].mean()
        avg_win = strategy_trades.loc[strategy_trades['is_win'] == 1, 'pnl'].mean() if wins > 0 else 0
        avg_loss = strategy_trades.loc[strategy_trades['is_win'] == 0, 'pnl'].mean() if losses > 0 else 0

        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Sharpe ratio approximation (daily returns)
        if len(strategy_trades) > 1:
            returns = strategy_trades['pnl'] / abs(strategy_trades['entry_price'] * strategy_trades['qty'])
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        strategy_stats.append({
            'strategy': strategy,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward,
            'sharpe_ratio': sharpe,
            'pnl_per_trade': total_pnl / total_trades
        })

    return pd.DataFrame(strategy_stats)

def get_ml_recommendation_analysis(conn):
    """Analyze ML predictions vs actual outcomes"""

    query = """
    SELECT
        recommendation,
        COUNT(*) as total,
        SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN actual_outcome = 0 THEN 1 ELSE 0 END) as losses,
        AVG(actual_pnl) as avg_pnl,
        SUM(actual_pnl) as total_pnl,
        AVG(hybrid_probability) as avg_confidence,
        SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy
    FROM ml_predictions
    WHERE actual_outcome IS NOT NULL
    GROUP BY recommendation
    ORDER BY total DESC
    """

    return pd.read_sql_query(query, conn)

def calculate_what_if_scenarios(conn):
    """Calculate what-if scenarios for following ML recommendations"""

    # Scenario 1: Follow all BUY recommendations
    buy_only = pd.read_sql_query("""
        SELECT
            COUNT(*) as trades,
            SUM(actual_pnl) as total_pnl,
            AVG(actual_pnl) as avg_pnl,
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
        FROM ml_predictions
        WHERE recommendation = 'BUY'
    """, conn)

    # Scenario 2: Avoid all SELL/AVOID recommendations
    avoid_sells = pd.read_sql_query("""
        SELECT
            COUNT(*) as avoided_trades,
            SUM(actual_pnl) as avoided_pnl,
            AVG(actual_pnl) as avg_avoided_pnl
        FROM ml_predictions
        WHERE recommendation IN ('SELL', 'AVOID')
    """, conn)

    # Scenario 3: Only high-confidence recommendations
    high_confidence = pd.read_sql_query("""
        SELECT
            COUNT(*) as trades,
            SUM(actual_pnl) as total_pnl,
            AVG(actual_pnl) as avg_pnl,
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate,
            AVG(hybrid_probability) as avg_confidence
        FROM ml_predictions
        WHERE confidence_level = 'HIGH' AND recommendation = 'BUY'
    """, conn)

    # Actual performance
    actual = pd.read_sql_query("""
        SELECT
            COUNT(*) as trades,
            SUM(actual_pnl) as total_pnl,
            AVG(actual_pnl) as avg_pnl,
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
        FROM ml_predictions
    """, conn)

    scenarios = {
        'actual': actual.iloc[0].to_dict() if len(actual) > 0 else {},
        'buy_only': buy_only.iloc[0].to_dict() if len(buy_only) > 0 else {},
        'avoid_sells': avoid_sells.iloc[0].to_dict() if len(avoid_sells) > 0 else {},
        'high_confidence_only': high_confidence.iloc[0].to_dict() if len(high_confidence) > 0 else {}
    }

    return scenarios

def display_strategy_analysis(strategy_stats):
    """Display strategy performance analysis"""

    if len(strategy_stats) == 0:
        print("No strategy data available")
        return

    print(f"\n📊 STRATEGY PERFORMANCE COMPARISON")
    print(f"{'─'*80}")
    print(f"{'Strategy':<20} {'Trades':>8} {'Win%':>7} {'Total P&L':>12} {'Avg/Trade':>11} {'Sharpe':>8}")
    print(f"{'─'*80}")

    for _, row in strategy_stats.iterrows():
        print(f"{row['strategy']:<20} {row['total_trades']:>8} {row['win_rate']:>6.2f}% "
              f"${row['total_pnl']:>11,.2f} ${row['avg_pnl_per_trade']:>10,.2f} {row['sharpe_ratio']:>8.3f}")

    # Best performing strategy
    best_sharpe = strategy_stats.loc[strategy_stats['sharpe_ratio'].idxmax()]
    best_pnl = strategy_stats.loc[strategy_stats['total_pnl'].idxmax()]
    best_winrate = strategy_stats.loc[strategy_stats['win_rate'].idxmax()]

    print(f"\n🏆 BEST PERFORMERS")
    print(f"{'─'*80}")
    print(f"  Best Sharpe Ratio:      {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.3f})")
    print(f"  Highest Total P&L:      {best_pnl['strategy']} (${best_pnl['total_pnl']:,.2f})")
    print(f"  Highest Win Rate:       {best_winrate['strategy']} ({best_winrate['win_rate']:.2f}%)")

def display_ml_recommendation_analysis(ml_analysis):
    """Display ML recommendation analysis"""

    if len(ml_analysis) == 0:
        print("No ML prediction data available")
        return

    print(f"\n🤖 ML RECOMMENDATION ANALYSIS")
    print(f"{'─'*80}")
    print(f"{'Recommendation':<15} {'Total':>8} {'Wins':>6} {'Win%':>7} {'Total P&L':>12} {'Accuracy':>10}")
    print(f"{'─'*80}")

    for _, row in ml_analysis.iterrows():
        win_rate = (row['wins'] / row['total']) * 100 if row['total'] > 0 else 0
        print(f"{row['recommendation']:<15} {int(row['total']):>8} {int(row['wins']):>6} "
              f"{win_rate:>6.2f}% ${row['total_pnl']:>11,.2f} {row['accuracy']*100:>9.2f}%")

    print(f"\n💡 KEY INSIGHTS")
    print(f"{'─'*80}")

    # Find best recommendation
    if len(ml_analysis) > 0:
        best_pnl_rec = ml_analysis.loc[ml_analysis['total_pnl'].idxmax()]
        best_acc_rec = ml_analysis.loc[ml_analysis['accuracy'].idxmax()]

        print(f"  Most Profitable Recommendation: {best_pnl_rec['recommendation']} (${best_pnl_rec['total_pnl']:,.2f})")
        print(f"  Most Accurate Recommendation:   {best_acc_rec['recommendation']} ({best_acc_rec['accuracy']*100:.2f}%)")

        # Overall model performance
        total_predictions = ml_analysis['total'].sum()
        overall_accuracy = (ml_analysis['wins'].sum() / total_predictions) * 100 if total_predictions > 0 else 0
        overall_pnl = ml_analysis['total_pnl'].sum()

        print(f"  Overall Model Accuracy:         {overall_accuracy:.2f}%")
        print(f"  Overall P&L (all predictions):  ${overall_pnl:,.2f}")

def display_what_if_analysis(scenarios):
    """Display what-if scenario analysis"""

    print(f"\n🔮 WHAT-IF ANALYSIS: Following ML Model 100%")
    print(f"{'─'*80}")

    actual = scenarios.get('actual', {})
    buy_only = scenarios.get('buy_only', {})
    avoid_sells = scenarios.get('avoid_sells', {})
    high_conf = scenarios.get('high_confidence_only', {})

    print(f"\nScenario 1: ACTUAL TRADING (Baseline)")
    print(f"  Trades:                 {int(actual.get('trades', 0)):>10}")
    print(f"  Win Rate:               {actual.get('win_rate', 0)*100:>9.2f}%")
    print(f"  Total P&L:              ${actual.get('total_pnl', 0):>10,.2f}")
    print(f"  Avg P&L per Trade:      ${actual.get('avg_pnl', 0):>10,.2f}")

    print(f"\nScenario 2: Follow ALL 'BUY' Recommendations")
    buy_trades = int(buy_only.get('trades', 0))
    if buy_trades > 0:
        buy_diff = buy_only.get('total_pnl', 0) - actual.get('total_pnl', 0)
        buy_diff_pct = (buy_diff / actual.get('total_pnl', 1)) * 100 if actual.get('total_pnl', 0) != 0 else 0

        print(f"  Trades:                 {buy_trades:>10}")
        print(f"  Win Rate:               {buy_only.get('win_rate', 0)*100:>9.2f}%")
        print(f"  Total P&L:              ${buy_only.get('total_pnl', 0):>10,.2f}")
        print(f"  Difference from Actual: ${buy_diff:>10,.2f} ({buy_diff_pct:+.2f}%)")
    else:
        print(f"  No BUY recommendations in dataset")

    print(f"\nScenario 3: Avoid ALL 'SELL/AVOID' Recommendations")
    avoided_trades = int(avoid_sells.get('avoided_trades', 0))
    if avoided_trades > 0:
        avoided_pnl = avoid_sells.get('avoided_pnl', 0)
        saved_pnl = -avoided_pnl if avoided_pnl < 0 else 0

        print(f"  Trades Avoided:         {avoided_trades:>10}")
        print(f"  P&L from Avoided Trades:${avoided_pnl:>10,.2f}")
        print(f"  Potential Losses Saved: ${saved_pnl:>10,.2f}")

        hypothetical_pnl = actual.get('total_pnl', 0) - avoided_pnl
        improvement = hypothetical_pnl - actual.get('total_pnl', 0)
        improvement_pct = (improvement / actual.get('total_pnl', 1)) * 100 if actual.get('total_pnl', 0) != 0 else 0

        print(f"  Hypothetical Total P&L: ${hypothetical_pnl:>10,.2f}")
        print(f"  Improvement:            ${improvement:>10,.2f} ({improvement_pct:+.2f}%)")
    else:
        print(f"  No SELL/AVOID recommendations in dataset")

    print(f"\nScenario 4: Follow ONLY High-Confidence 'BUY' Recommendations")
    high_conf_trades = int(high_conf.get('trades', 0))
    if high_conf_trades > 0:
        print(f"  Trades:                 {high_conf_trades:>10}")
        print(f"  Win Rate:               {high_conf.get('win_rate', 0)*100:>9.2f}%")
        print(f"  Avg Confidence:         {high_conf.get('avg_confidence', 0)*100:>9.2f}%")
        print(f"  Total P&L:              ${high_conf.get('total_pnl', 0):>10,.2f}")
        print(f"  Avg P&L per Trade:      ${high_conf.get('avg_pnl', 0):>10,.2f}")
    else:
        print(f"  No high-confidence BUY recommendations in dataset")

    print(f"\n{'─'*80}")
    print(f"📈 RECOMMENDATION")
    print(f"{'─'*80}")

    # Determine best strategy
    if avoided_trades > 0:
        avoided_pnl = avoid_sells.get('avoided_pnl', 0)
        if avoided_pnl < 0:
            print(f"  ✓ Avoiding SELL/AVOID recommendations could have saved ${-avoided_pnl:,.2f}")
            print(f"  ✓ Model shows value in risk avoidance")
        else:
            print(f"  ⚠ SELL/AVOID recommendations were actually profitable")
            print(f"  ⚠ Consider model may be too conservative")

    if high_conf_trades > 0 and high_conf.get('win_rate', 0) > actual.get('win_rate', 0):
        print(f"  ✓ High-confidence recommendations show better win rate")
        print(f"  ✓ Focus on high-confidence signals for better results")

def main():
    """Main execution"""

    print_section("STRATEGY COMPARISON ANALYSIS", '█')

    conn = sqlite3.connect(DB_PATH)

    # Section 1: Strategy Performance
    print_section("SECTION 1: Trading Strategy Performance")

    trades = get_strategy_performance(conn)

    if len(trades) > 0:
        strategy_stats = analyze_strategy_performance(trades)
        display_strategy_analysis(strategy_stats)
    else:
        print("No trade data available")

    # Section 2: ML Recommendation Analysis
    print_section("SECTION 2: ML Model Recommendation Analysis")

    ml_analysis = get_ml_recommendation_analysis(conn)
    display_ml_recommendation_analysis(ml_analysis)

    # Section 3: What-If Analysis
    print_section("SECTION 3: What-If Analysis")

    scenarios = calculate_what_if_scenarios(conn)
    display_what_if_analysis(scenarios)

    print_section("✓ STRATEGY COMPARISON COMPLETE", '█')
    print(f"Use these insights to optimize trading strategy and ML model usage\n")

    conn.close()

if __name__ == '__main__':
    main()
