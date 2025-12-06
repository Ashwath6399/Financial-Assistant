"""
Flask Backend for Trading Journal Dashboard
Modern API serving trading analytics data
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import sqlite3
import pandas as pd
import json
import os
from config import DB_PATH

app = Flask(__name__)
CORS(app)

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

# =============================================================================
# OVERVIEW TAB ENDPOINTS
# =============================================================================

@app.route('/api/overview/summary')
def get_overview_summary():
    """Get comprehensive portfolio overview for Tab 1"""
    conn = get_db_connection()

    # Portfolio aggregates
    portfolio = conn.execute('''
        SELECT
            COUNT(*) as total_accounts,
            SUM(total_trades) as total_trades,
            SUM(net_pnl) as total_pnl,
            AVG(sharpe_ratio) as avg_sharpe,
            AVG(win_rate) as avg_win_rate,
            MAX(largest_win) as biggest_win,
            MIN(largest_loss) as biggest_loss
        FROM account_metrics
    ''').fetchone()

    # Benchmark summary
    benchmark = conn.execute('''
        SELECT
            SUM(CASE WHEN beat_market = 1 THEN 1 ELSE 0 END) as accounts_beat_market,
            COUNT(*) as total_accounts,
            AVG(outperformance_pct) as avg_outperformance
        FROM benchmark_comparison
    ''').fetchone()

    # Best and worst accounts
    best_account = conn.execute('''
        SELECT account_id, net_pnl, win_rate
        FROM account_metrics
        ORDER BY net_pnl DESC
        LIMIT 1
    ''').fetchone()

    worst_account = conn.execute('''
        SELECT account_id, net_pnl, win_rate
        FROM account_metrics
        ORDER BY net_pnl ASC
        LIMIT 1
    ''').fetchone()

    # Biggest win/loss trades
    biggest_win_trade = conn.execute('''
        SELECT mp.symbol, mp.actual_pnl, mp.account_id
        FROM ml_predictions mp
        WHERE mp.actual_pnl = (SELECT MAX(actual_pnl) FROM ml_predictions)
        LIMIT 1
    ''').fetchone()

    biggest_loss_trade = conn.execute('''
        SELECT mp.symbol, mp.actual_pnl, mp.account_id
        FROM ml_predictions mp
        WHERE mp.actual_pnl = (SELECT MIN(actual_pnl) FROM ml_predictions)
        LIMIT 1
    ''').fetchone()

    # ML summary
    ml_stats = conn.execute('''
        SELECT
            COUNT(*) as total_predictions,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as overall_accuracy,
            SUM(CASE WHEN recommendation = 'TAKE' AND hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(SUM(CASE WHEN recommendation = 'TAKE' THEN 1 ELSE 0 END), 0) as take_accuracy,
            SUM(CASE WHEN recommendation = 'AVOID' AND hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(SUM(CASE WHEN recommendation = 'AVOID' THEN 1 ELSE 0 END), 0) as avoid_accuracy
        FROM ml_predictions
    ''').fetchone()

    # Data coverage
    data_coverage = conn.execute('''
        SELECT
            (SELECT COUNT(DISTINCT instrument_id) FROM instruments) as symbols,
            (SELECT MIN(ts) FROM prices) as min_date,
            (SELECT MAX(ts) FROM prices) as max_date,
            (SELECT COUNT(*) FROM prices) as price_count,
            (SELECT COUNT(*) FROM fills) as fills_count
    ''').fetchone()

    # All accounts for charts
    all_accounts = conn.execute('''
        SELECT account_id, net_pnl, win_rate
        FROM account_metrics
        ORDER BY net_pnl DESC
    ''').fetchall()

    conn.close()

    return jsonify({
        'portfolio': {
            'total_accounts': portfolio['total_accounts'],
            'total_trades': portfolio['total_trades'],
            'total_pnl': round(portfolio['total_pnl'] or 0, 2),
            'avg_sharpe': round(portfolio['avg_sharpe'] or 0, 2),
            'avg_win_rate': round(portfolio['avg_win_rate'] or 0, 2)
        },
        'benchmark': {
            'accounts_beat_market': benchmark['accounts_beat_market'] or 0,
            'total_accounts': benchmark['total_accounts'] or 0,
            'avg_outperformance': round(benchmark['avg_outperformance'] or 0, 2)
        },
        'best_account': {
            'account_id': best_account['account_id'] if best_account else '--',
            'net_pnl': round(best_account['net_pnl'] or 0, 2) if best_account else 0,
            'win_rate': round(best_account['win_rate'] or 0, 2) if best_account else 0
        },
        'worst_account': {
            'account_id': worst_account['account_id'] if worst_account else '--',
            'net_pnl': round(worst_account['net_pnl'] or 0, 2) if worst_account else 0,
            'win_rate': round(worst_account['win_rate'] or 0, 2) if worst_account else 0
        },
        'biggest_win': {
            'symbol': biggest_win_trade['symbol'] if biggest_win_trade else '--',
            'amount': round(biggest_win_trade['actual_pnl'] or 0, 2) if biggest_win_trade else 0,
            'account_id': biggest_win_trade['account_id'] if biggest_win_trade else '--'
        },
        'biggest_loss': {
            'symbol': biggest_loss_trade['symbol'] if biggest_loss_trade else '--',
            'amount': round(biggest_loss_trade['actual_pnl'] or 0, 2) if biggest_loss_trade else 0,
            'account_id': biggest_loss_trade['account_id'] if biggest_loss_trade else '--'
        },
        'ml_stats': {
            'total_predictions': ml_stats['total_predictions'] or 0,
            'overall_accuracy': round(ml_stats['overall_accuracy'] or 0, 2),
            'take_accuracy': round(ml_stats['take_accuracy'] or 0, 2),
            'avoid_accuracy': round(ml_stats['avoid_accuracy'] or 0, 2)
        },
        'data_coverage': {
            'symbols': data_coverage['symbols'] or 0,
            'date_range': f"{data_coverage['min_date'][:10] if data_coverage['min_date'] else '--'} to {data_coverage['max_date'][:10] if data_coverage['max_date'] else '--'}",
            'price_count': data_coverage['price_count'] or 0,
            'fills_count': data_coverage['fills_count'] or 0
        },
        'all_accounts': [
            {
                'account_id': acc['account_id'],
                'net_pnl': round(acc['net_pnl'] or 0, 2),
                'win_rate': round(acc['win_rate'] or 0, 2)
            }
            for acc in all_accounts
        ]
    })

# =============================================================================
# ACCOUNT PERFORMANCE ENDPOINTS (existing, enhanced)
# =============================================================================

@app.route('/api/accounts')
def get_accounts():
    """Get list of all accounts"""
    conn = get_db_connection()
    accounts = conn.execute('''
        SELECT DISTINCT account_id
        FROM account_metrics
        ORDER BY account_id
    ''').fetchall()
    conn.close()
    return jsonify([dict(account) for account in accounts])

@app.route('/api/account/<account_id>')
def get_account_data(account_id):
    """Get comprehensive data for a specific account"""
    conn = get_db_connection()

    # Get account metrics
    metrics = conn.execute('''
        SELECT * FROM account_metrics WHERE account_id = ?
    ''', (account_id,)).fetchone()

    # Get benchmark comparison
    benchmark = conn.execute('''
        SELECT * FROM benchmark_comparison WHERE account_id = ?
    ''', (account_id,)).fetchone()

    # Get risk analytics
    risk = conn.execute('''
        SELECT * FROM risk_analytics WHERE account_id = ?
    ''', (account_id,)).fetchone()

    # Get ML predictions summary
    ml_summary = conn.execute('''
        SELECT
            recommendation,
            COUNT(*) as count,
            SUM(actual_pnl) as total_pnl,
            AVG(hybrid_probability) as avg_probability,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy
        FROM ml_predictions
        WHERE account_id = ?
        GROUP BY recommendation
    ''', (account_id,)).fetchall()

    # Get trades over time for chart (with cumulative P&L)
    trades_timeline = conn.execute('''
        SELECT
            entry_ts as date,
            actual_pnl as pnl
        FROM ml_predictions
        WHERE account_id = ? AND actual_pnl IS NOT NULL
        ORDER BY entry_ts
    ''', (account_id,)).fetchall()

    # Get all trades for P&L distribution
    trade_pnls = conn.execute('''
        SELECT actual_pnl
        FROM ml_predictions
        WHERE account_id = ? AND actual_pnl IS NOT NULL
        ORDER BY actual_pnl
    ''', (account_id,)).fetchall()

    # Get monthly returns for heatmap
    monthly_returns = conn.execute('''
        SELECT
            strftime('%Y', entry_ts) as year,
            strftime('%m', entry_ts) as month,
            SUM(actual_pnl) as monthly_pnl,
            COUNT(*) as trade_count
        FROM ml_predictions
        WHERE account_id = ? AND actual_pnl IS NOT NULL
        GROUP BY strftime('%Y', entry_ts), strftime('%m', entry_ts)
        ORDER BY year, month
    ''', (account_id,)).fetchall()

    conn.close()

    # Calculate cumulative P&L and drawdown
    cumulative_pnl = 0
    peak_pnl = 0
    timeline_data = []
    drawdown_data = []

    for trade in trades_timeline:
        cumulative_pnl += trade['pnl']
        peak_pnl = max(peak_pnl, cumulative_pnl)
        drawdown = ((cumulative_pnl - peak_pnl) / peak_pnl * 100) if peak_pnl > 0 else 0

        timeline_data.append({
            'date': trade['date'],
            'cumulative_pnl': round(cumulative_pnl, 2)
        })
        drawdown_data.append({
            'date': trade['date'],
            'drawdown': round(drawdown, 2)
        })

    # Build P&L distribution buckets
    pnl_distribution = []
    if trade_pnls:
        pnl_values = [t['actual_pnl'] for t in trade_pnls]
        # Create buckets
        min_pnl = min(pnl_values)
        max_pnl = max(pnl_values)
        bucket_size = (max_pnl - min_pnl) / 20 if max_pnl != min_pnl else 100
        buckets = {}
        for pnl in pnl_values:
            bucket = int((pnl - min_pnl) / bucket_size) if bucket_size > 0 else 0
            bucket_key = round(min_pnl + bucket * bucket_size, 0)
            buckets[bucket_key] = buckets.get(bucket_key, 0) + 1
        pnl_distribution = [{'range': k, 'count': v} for k, v in sorted(buckets.items())]

    # Format monthly returns
    monthly_data = {}
    for row in monthly_returns:
        year = row['year']
        month = int(row['month'])
        if year not in monthly_data:
            monthly_data[year] = {}
        monthly_data[year][month] = round(row['monthly_pnl'], 2)

    return jsonify({
        'account_id': account_id,
        'metrics': dict(metrics) if metrics else {},
        'benchmark': dict(benchmark) if benchmark else {},
        'risk': dict(risk) if risk else {},
        'ml_summary': [dict(row) for row in ml_summary],
        'timeline': timeline_data,
        'drawdown': drawdown_data,
        'pnl_distribution': pnl_distribution,
        'monthly_returns': monthly_data
    })

@app.route('/api/portfolio/summary')
def get_portfolio_summary():
    """Get portfolio-level summary statistics"""
    conn = get_db_connection()

    summary = conn.execute('''
        SELECT
            COUNT(*) as total_accounts,
            SUM(total_trades) as total_trades,
            SUM(net_pnl) as total_pnl,
            AVG(sharpe_ratio) as avg_sharpe,
            AVG(win_rate) as avg_win_rate,
            SUM(CASE WHEN bc.beat_market = 1 THEN 1 ELSE 0 END) as accounts_beat_market
        FROM account_metrics am
        LEFT JOIN benchmark_comparison bc ON am.account_id = bc.account_id
    ''').fetchone()

    conn.close()
    return jsonify(dict(summary))

# =============================================================================
# METRIC FORMULAS
# =============================================================================

# =============================================================================
# ML MODELS TAB ENDPOINTS
# =============================================================================

@app.route('/api/ml/summary')
def get_ml_summary():
    """Get comprehensive ML model performance summary for Tab 3"""
    conn = get_db_connection()

    # Overall model performance
    overall = conn.execute('''
        SELECT
            COUNT(*) as total_predictions,
            SUM(CASE WHEN xgboost_correct = 1 THEN 1 ELSE 0 END) as xgboost_correct,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) as hybrid_correct,
            AVG(xgboost_probability) as avg_xgboost_prob,
            AVG(hybrid_probability) as avg_hybrid_prob,
            SUM(actual_pnl) as total_pnl
        FROM ml_predictions
    ''').fetchone()

    # Recommendation breakdown
    recommendations = conn.execute('''
        SELECT
            recommendation,
            COUNT(*) as count,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) as correct,
            SUM(actual_pnl) as total_pnl,
            AVG(hybrid_probability) as avg_probability
        FROM ml_predictions
        GROUP BY recommendation
    ''').fetchall()

    # Confidence level breakdown
    confidence_levels = conn.execute('''
        SELECT
            confidence_level,
            COUNT(*) as count,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) as correct,
            SUM(actual_pnl) as total_pnl,
            AVG(hybrid_probability) as avg_probability
        FROM ml_predictions
        GROUP BY confidence_level
    ''').fetchall()

    # Confusion matrix for hybrid model
    confusion_matrix = conn.execute('''
        SELECT
            hybrid_prediction as predicted,
            actual_outcome as actual,
            COUNT(*) as count
        FROM ml_predictions
        GROUP BY hybrid_prediction, actual_outcome
    ''').fetchall()

    # Confusion matrix for XGBoost
    xgboost_confusion = conn.execute('''
        SELECT
            xgboost_prediction as predicted,
            actual_outcome as actual,
            COUNT(*) as count
        FROM ml_predictions
        GROUP BY xgboost_prediction, actual_outcome
    ''').fetchall()

    # Performance over time (monthly)
    monthly_performance = conn.execute('''
        SELECT
            strftime('%Y-%m', entry_ts) as month,
            COUNT(*) as total,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) as correct,
            SUM(actual_pnl) as pnl
        FROM ml_predictions
        GROUP BY strftime('%Y-%m', entry_ts)
        ORDER BY month
    ''').fetchall()

    # Top symbols by prediction count
    top_symbols = conn.execute('''
        SELECT
            symbol,
            COUNT(*) as count,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) as correct,
            SUM(actual_pnl) as total_pnl
        FROM ml_predictions
        GROUP BY symbol
        ORDER BY count DESC
        LIMIT 10
    ''').fetchall()

    # Recent predictions
    recent_predictions = conn.execute('''
        SELECT
            prediction_id,
            account_id,
            symbol,
            entry_ts,
            recommendation,
            confidence_level,
            hybrid_probability,
            actual_outcome,
            actual_pnl,
            hybrid_correct
        FROM ml_predictions
        ORDER BY entry_ts DESC
        LIMIT 20
    ''').fetchall()

    conn.close()

    # Build confusion matrix dict
    cm_hybrid = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for row in confusion_matrix:
        if row['predicted'] == 1 and row['actual'] == 1:
            cm_hybrid['TP'] = row['count']
        elif row['predicted'] == 0 and row['actual'] == 0:
            cm_hybrid['TN'] = row['count']
        elif row['predicted'] == 1 and row['actual'] == 0:
            cm_hybrid['FP'] = row['count']
        elif row['predicted'] == 0 and row['actual'] == 1:
            cm_hybrid['FN'] = row['count']

    cm_xgboost = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for row in xgboost_confusion:
        if row['predicted'] == 1 and row['actual'] == 1:
            cm_xgboost['TP'] = row['count']
        elif row['predicted'] == 0 and row['actual'] == 0:
            cm_xgboost['TN'] = row['count']
        elif row['predicted'] == 1 and row['actual'] == 0:
            cm_xgboost['FP'] = row['count']
        elif row['predicted'] == 0 and row['actual'] == 1:
            cm_xgboost['FN'] = row['count']

    # Calculate metrics for both models
    def calc_metrics(cm, total):
        accuracy = (cm['TP'] + cm['TN']) / total * 100 if total > 0 else 0
        precision = cm['TP'] / (cm['TP'] + cm['FP']) * 100 if (cm['TP'] + cm['FP']) > 0 else 0
        recall = cm['TP'] / (cm['TP'] + cm['FN']) * 100 if (cm['TP'] + cm['FN']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = cm['TN'] / (cm['TN'] + cm['FP']) * 100 if (cm['TN'] + cm['FP']) > 0 else 0
        return {
            'accuracy': round(accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'f1_score': round(f1, 2),
            'specificity': round(specificity, 2)
        }

    total = overall['total_predictions']

    return jsonify({
        'overall': {
            'total_predictions': total,
            'xgboost_accuracy': round(overall['xgboost_correct'] / total * 100, 2) if total > 0 else 0,
            'hybrid_accuracy': round(overall['hybrid_correct'] / total * 100, 2) if total > 0 else 0,
            'avg_xgboost_prob': round(overall['avg_xgboost_prob'] * 100, 2) if overall['avg_xgboost_prob'] else 0,
            'avg_hybrid_prob': round(overall['avg_hybrid_prob'] * 100, 2) if overall['avg_hybrid_prob'] else 0,
            'total_pnl': round(overall['total_pnl'], 2) if overall['total_pnl'] else 0
        },
        'recommendations': [
            {
                'name': r['recommendation'],
                'count': r['count'],
                'accuracy': round(r['correct'] / r['count'] * 100, 2) if r['count'] > 0 else 0,
                'total_pnl': round(r['total_pnl'], 2) if r['total_pnl'] else 0,
                'avg_probability': round(r['avg_probability'] * 100, 2) if r['avg_probability'] else 0
            }
            for r in recommendations
        ],
        'confidence_levels': [
            {
                'name': c['confidence_level'],
                'count': c['count'],
                'accuracy': round(c['correct'] / c['count'] * 100, 2) if c['count'] > 0 else 0,
                'total_pnl': round(c['total_pnl'], 2) if c['total_pnl'] else 0,
                'avg_probability': round(c['avg_probability'] * 100, 2) if c['avg_probability'] else 0
            }
            for c in confidence_levels
        ],
        'confusion_matrix': {
            'hybrid': cm_hybrid,
            'xgboost': cm_xgboost
        },
        'model_metrics': {
            'hybrid': calc_metrics(cm_hybrid, total),
            'xgboost': calc_metrics(cm_xgboost, total)
        },
        'monthly_performance': [
            {
                'month': m['month'],
                'total': m['total'],
                'accuracy': round(m['correct'] / m['total'] * 100, 2) if m['total'] > 0 else 0,
                'pnl': round(m['pnl'], 2) if m['pnl'] else 0
            }
            for m in monthly_performance
        ],
        'top_symbols': [
            {
                'symbol': s['symbol'],
                'count': s['count'],
                'accuracy': round(s['correct'] / s['count'] * 100, 2) if s['count'] > 0 else 0,
                'total_pnl': round(s['total_pnl'], 2) if s['total_pnl'] else 0
            }
            for s in top_symbols
        ],
        'recent_predictions': [
            {
                'id': p['prediction_id'],
                'account': p['account_id'],
                'symbol': p['symbol'],
                'date': p['entry_ts'],
                'recommendation': p['recommendation'],
                'confidence': p['confidence_level'],
                'probability': round(p['hybrid_probability'] * 100, 2) if p['hybrid_probability'] else 0,
                'outcome': 'WIN' if p['actual_outcome'] == 1 else 'LOSS',
                'pnl': round(p['actual_pnl'], 2) if p['actual_pnl'] else 0,
                'correct': p['hybrid_correct'] == 1
            }
            for p in recent_predictions
        ]
    })

# =============================================================================
# TRADE EXPLORER TAB ENDPOINTS
# =============================================================================

@app.route('/api/trades')
def get_trades():
    """Get paginated list of trades with filters"""
    conn = get_db_connection()

    # Get filter parameters
    account = request.args.get('account', '')
    symbol = request.args.get('symbol', '')
    recommendation = request.args.get('recommendation', '')
    confidence = request.args.get('confidence', '')
    outcome = request.args.get('outcome', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 50))
    sort_by = request.args.get('sort_by', 'entry_ts')
    sort_order = request.args.get('sort_order', 'DESC')

    # Build query
    query = '''SELECT * FROM ml_predictions WHERE 1=1'''
    params = []

    if account:
        query += ' AND account_id = ?'
        params.append(account)
    if symbol:
        query += ' AND symbol = ?'
        params.append(symbol)
    if recommendation:
        query += ' AND recommendation = ?'
        params.append(recommendation)
    if confidence:
        query += ' AND confidence_level = ?'
        params.append(confidence)
    if outcome:
        query += ' AND actual_outcome = ?'
        params.append(1 if outcome == 'WIN' else 0)
    if date_from:
        query += ' AND entry_ts >= ?'
        params.append(date_from)
    if date_to:
        query += ' AND entry_ts <= ?'
        params.append(date_to + ' 23:59:59')

    # Count total
    count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
    total = conn.execute(count_query, params).fetchone()[0]

    # Add sorting and pagination
    valid_columns = ['entry_ts', 'symbol', 'account_id', 'actual_pnl', 'recommendation', 'confidence_level']
    if sort_by not in valid_columns:
        sort_by = 'entry_ts'
    sort_order = 'DESC' if sort_order.upper() == 'DESC' else 'ASC'

    query += f' ORDER BY {sort_by} {sort_order}'
    query += ' LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])

    trades = conn.execute(query, params).fetchall()
    conn.close()

    return jsonify({
        'trades': [
            {
                'id': t['prediction_id'],
                'account': t['account_id'],
                'symbol': t['symbol'],
                'date': t['entry_ts'],
                'recommendation': t['recommendation'],
                'confidence': t['confidence_level'],
                'xgboost_prob': round(t['xgboost_probability'] * 100, 2) if t['xgboost_probability'] else 0,
                'hybrid_prob': round(t['hybrid_probability'] * 100, 2) if t['hybrid_probability'] else 0,
                'outcome': 'WIN' if t['actual_outcome'] == 1 else 'LOSS',
                'pnl': round(t['actual_pnl'], 2) if t['actual_pnl'] else 0,
                'xgboost_correct': t['xgboost_correct'] == 1,
                'hybrid_correct': t['hybrid_correct'] == 1
            }
            for t in trades
        ],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': (total + per_page - 1) // per_page
        }
    })

@app.route('/api/trades/filters')
def get_trade_filters():
    """Get available filter options for trades"""
    conn = get_db_connection()

    accounts = conn.execute('SELECT DISTINCT account_id FROM ml_predictions ORDER BY account_id').fetchall()
    symbols = conn.execute('SELECT DISTINCT symbol FROM ml_predictions ORDER BY symbol').fetchall()

    date_range = conn.execute('''
        SELECT MIN(entry_ts) as min_date, MAX(entry_ts) as max_date
        FROM ml_predictions
    ''').fetchone()

    conn.close()

    return jsonify({
        'accounts': [a['account_id'] for a in accounts],
        'symbols': [s['symbol'] for s in symbols],
        'recommendations': ['TAKE', 'AVOID', 'UNCERTAIN'],
        'confidence_levels': ['HIGH', 'MEDIUM', 'LOW'],
        'outcomes': ['WIN', 'LOSS'],
        'date_range': {
            'min': date_range['min_date'][:10] if date_range['min_date'] else None,
            'max': date_range['max_date'][:10] if date_range['max_date'] else None
        }
    })

@app.route('/api/trades/stats')
def get_trade_stats():
    """Get trade statistics for the explorer tab - supports filtering"""
    conn = get_db_connection()

    # Get filter parameters (same as /api/trades)
    account = request.args.get('account', '')
    symbol = request.args.get('symbol', '')
    recommendation = request.args.get('recommendation', '')
    confidence = request.args.get('confidence', '')
    outcome = request.args.get('outcome', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')

    # Build WHERE clause
    where_clauses = ['1=1']
    params = []

    if account:
        where_clauses.append('account_id = ?')
        params.append(account)
    if symbol:
        where_clauses.append('symbol = ?')
        params.append(symbol)
    if recommendation:
        where_clauses.append('recommendation = ?')
        params.append(recommendation)
    if confidence:
        where_clauses.append('confidence_level = ?')
        params.append(confidence)
    if outcome:
        where_clauses.append('actual_outcome = ?')
        params.append(1 if outcome == 'WIN' else 0)
    if date_from:
        where_clauses.append('entry_ts >= ?')
        params.append(date_from)
    if date_to:
        where_clauses.append('entry_ts <= ?')
        params.append(date_to + ' 23:59:59')

    where_sql = ' AND '.join(where_clauses)

    # Overall stats with filters
    overall = conn.execute(f'''
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN actual_outcome = 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(actual_pnl) as total_pnl,
            AVG(actual_pnl) as avg_pnl,
            MAX(actual_pnl) as max_pnl,
            MIN(actual_pnl) as min_pnl,
            AVG(CASE WHEN actual_outcome = 1 THEN actual_pnl END) as avg_win,
            AVG(CASE WHEN actual_outcome = 0 THEN actual_pnl END) as avg_loss
        FROM ml_predictions
        WHERE {where_sql}
    ''', params).fetchone()

    # P&L distribution with filters
    pnl_dist = conn.execute(f'''
        SELECT
            CASE
                WHEN actual_pnl < -500 THEN '< -500'
                WHEN actual_pnl < -200 THEN '-500 to -200'
                WHEN actual_pnl < -100 THEN '-200 to -100'
                WHEN actual_pnl < 0 THEN '-100 to 0'
                WHEN actual_pnl < 100 THEN '0 to 100'
                WHEN actual_pnl < 200 THEN '100 to 200'
                WHEN actual_pnl < 500 THEN '200 to 500'
                ELSE '> 500'
            END as range,
            COUNT(*) as count
        FROM ml_predictions
        WHERE {where_sql}
        GROUP BY
            CASE
                WHEN actual_pnl < -500 THEN '< -500'
                WHEN actual_pnl < -200 THEN '-500 to -200'
                WHEN actual_pnl < -100 THEN '-200 to -100'
                WHEN actual_pnl < 0 THEN '-100 to 0'
                WHEN actual_pnl < 100 THEN '0 to 100'
                WHEN actual_pnl < 200 THEN '100 to 200'
                WHEN actual_pnl < 500 THEN '200 to 500'
                ELSE '> 500'
            END
        ORDER BY MIN(actual_pnl)
    ''', params).fetchall()

    # Win rate by symbol (top 10) with filters
    symbol_performance = conn.execute(f'''
        SELECT
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
            SUM(actual_pnl) as total_pnl
        FROM ml_predictions
        WHERE {where_sql}
        GROUP BY symbol
        ORDER BY trades DESC
        LIMIT 10
    ''', params).fetchall()

    # Win rate by account with filters
    account_performance = conn.execute(f'''
        SELECT
            account_id,
            COUNT(*) as trades,
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
            SUM(actual_pnl) as total_pnl
        FROM ml_predictions
        WHERE {where_sql}
        GROUP BY account_id
        ORDER BY total_pnl DESC
        LIMIT 10
    ''', params).fetchall()

    conn.close()

    return jsonify({
        'overall': {
            'total_trades': overall['total_trades'],
            'winning_trades': overall['winning_trades'],
            'losing_trades': overall['losing_trades'],
            'win_rate': round(overall['winning_trades'] / overall['total_trades'] * 100, 2) if overall['total_trades'] > 0 else 0,
            'total_pnl': round(overall['total_pnl'], 2) if overall['total_pnl'] else 0,
            'avg_pnl': round(overall['avg_pnl'], 2) if overall['avg_pnl'] else 0,
            'max_pnl': round(overall['max_pnl'], 2) if overall['max_pnl'] else 0,
            'min_pnl': round(overall['min_pnl'], 2) if overall['min_pnl'] else 0,
            'avg_win': round(overall['avg_win'], 2) if overall['avg_win'] else 0,
            'avg_loss': round(overall['avg_loss'], 2) if overall['avg_loss'] else 0
        },
        'pnl_distribution': [
            {'range': d['range'], 'count': d['count']}
            for d in pnl_dist
        ],
        'symbol_performance': [
            {
                'symbol': s['symbol'],
                'trades': s['trades'],
                'win_rate': round(s['win_rate'], 2),
                'total_pnl': round(s['total_pnl'], 2) if s['total_pnl'] else 0
            }
            for s in symbol_performance
        ],
        'account_performance': [
            {
                'account': a['account_id'],
                'trades': a['trades'],
                'win_rate': round(a['win_rate'], 2),
                'total_pnl': round(a['total_pnl'], 2) if a['total_pnl'] else 0
            }
            for a in account_performance
        ]
    })

# =============================================================================
# TRADE SCREENER TAB ENDPOINTS
# =============================================================================

@app.route('/api/screener/symbols')
def get_screener_symbols():
    """Get symbol performance data for trade screener"""
    conn = get_db_connection()

    # Get comprehensive symbol stats
    symbols = conn.execute('''
        SELECT
            symbol,
            COUNT(*) as total_trades,
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN actual_outcome = 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) as ml_correct,
            SUM(actual_pnl) as total_pnl,
            AVG(actual_pnl) as avg_pnl,
            AVG(CASE WHEN actual_outcome = 1 THEN actual_pnl END) as avg_win,
            AVG(CASE WHEN actual_outcome = 0 THEN actual_pnl END) as avg_loss,
            MAX(actual_pnl) as best_trade,
            MIN(actual_pnl) as worst_trade,
            SUM(CASE WHEN recommendation = 'TAKE' THEN 1 ELSE 0 END) as take_count,
            SUM(CASE WHEN recommendation = 'AVOID' THEN 1 ELSE 0 END) as avoid_count,
            SUM(CASE WHEN recommendation = 'TAKE' AND hybrid_correct = 1 THEN 1 ELSE 0 END) as take_correct,
            SUM(CASE WHEN confidence_level = 'HIGH' THEN 1 ELSE 0 END) as high_conf_count,
            SUM(CASE WHEN confidence_level = 'HIGH' AND hybrid_correct = 1 THEN 1 ELSE 0 END) as high_conf_correct
        FROM ml_predictions
        GROUP BY symbol
        HAVING total_trades >= 5
        ORDER BY total_trades DESC
    ''').fetchall()

    conn.close()

    result = []
    for s in symbols:
        total = s['total_trades']
        wins = s['wins'] or 0
        ml_correct = s['ml_correct'] or 0
        take_count = s['take_count'] or 0
        take_correct = s['take_correct'] or 0
        high_conf = s['high_conf_count'] or 0
        high_conf_correct = s['high_conf_correct'] or 0
        avg_win = s['avg_win'] or 0
        avg_loss = abs(s['avg_loss'] or 0)

        # Calculate signal strength
        ml_accuracy = (ml_correct / total * 100) if total > 0 else 0
        win_rate = (wins / total * 100) if total > 0 else 0
        take_accuracy = (take_correct / take_count * 100) if take_count > 0 else 0
        high_conf_accuracy = (high_conf_correct / high_conf * 100) if high_conf > 0 else 0
        risk_reward = (avg_win / avg_loss) if avg_loss > 0 else 0

        # Determine signal (adjusted thresholds for realistic ML model performance)
        # Model accuracy ranges from ~50% to ~72%, so thresholds are set accordingly
        if ml_accuracy >= 68 and win_rate >= 58:
            signal = 'HOT'
        elif ml_accuracy >= 65 and win_rate >= 55:
            signal = 'STRONG'
        elif ml_accuracy >= 62:
            signal = 'GOOD'
        elif ml_accuracy >= 58:
            signal = 'NEUTRAL'
        else:
            signal = 'WEAK'

        result.append({
            'symbol': s['symbol'],
            'total_trades': total,
            'wins': wins,
            'losses': s['losses'] or 0,
            'win_rate': round(win_rate, 1),
            'ml_accuracy': round(ml_accuracy, 1),
            'total_pnl': round(s['total_pnl'] or 0, 2),
            'avg_pnl': round(s['avg_pnl'] or 0, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(-abs(s['avg_loss'] or 0), 2),
            'best_trade': round(s['best_trade'] or 0, 2),
            'worst_trade': round(s['worst_trade'] or 0, 2),
            'risk_reward': round(risk_reward, 2),
            'take_count': take_count,
            'take_accuracy': round(take_accuracy, 1),
            'high_conf_count': high_conf,
            'high_conf_accuracy': round(high_conf_accuracy, 1),
            'signal': signal
        })

    return jsonify(result)

@app.route('/api/screener/accounts')
def get_screener_accounts():
    """Get account performance data for trade screener"""
    conn = get_db_connection()

    accounts = conn.execute('''
        SELECT
            mp.account_id,
            COUNT(*) as total_trades,
            SUM(CASE WHEN mp.actual_outcome = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN mp.hybrid_correct = 1 THEN 1 ELSE 0 END) as ml_correct,
            SUM(mp.actual_pnl) as total_pnl,
            AVG(mp.actual_pnl) as avg_pnl,
            am.sharpe_ratio,
            am.win_rate as overall_win_rate,
            bc.beat_market,
            bc.outperformance_pct
        FROM ml_predictions mp
        LEFT JOIN account_metrics am ON mp.account_id = am.account_id
        LEFT JOIN benchmark_comparison bc ON mp.account_id = bc.account_id
        GROUP BY mp.account_id
        ORDER BY total_pnl DESC
    ''').fetchall()

    conn.close()

    result = []
    for a in accounts:
        total = a['total_trades']
        wins = a['wins'] or 0
        ml_correct = a['ml_correct'] or 0

        ml_accuracy = (ml_correct / total * 100) if total > 0 else 0
        win_rate = (wins / total * 100) if total > 0 else 0

        result.append({
            'account_id': a['account_id'],
            'total_trades': total,
            'wins': wins,
            'win_rate': round(win_rate, 1),
            'ml_accuracy': round(ml_accuracy, 1),
            'total_pnl': round(a['total_pnl'] or 0, 2),
            'avg_pnl': round(a['avg_pnl'] or 0, 2),
            'sharpe_ratio': round(a['sharpe_ratio'] or 0, 2),
            'beat_market': a['beat_market'] == 1,
            'outperformance': round(a['outperformance_pct'] or 0, 2)
        })

    return jsonify(result)

@app.route('/api/screener/insights')
def get_screener_insights():
    """Get actionable insights for the trade screener"""
    conn = get_db_connection()

    # Best performing symbol
    best_symbol = conn.execute('''
        SELECT symbol,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy,
            COUNT(*) as trades
        FROM ml_predictions
        GROUP BY symbol
        HAVING trades >= 10
        ORDER BY accuracy DESC
        LIMIT 1
    ''').fetchone()

    # Worst performing symbol
    worst_symbol = conn.execute('''
        SELECT symbol,
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy,
            COUNT(*) as trades
        FROM ml_predictions
        GROUP BY symbol
        HAVING trades >= 10
        ORDER BY accuracy ASC
        LIMIT 1
    ''').fetchone()

    # High confidence performance
    high_conf = conn.execute('''
        SELECT
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy,
            COUNT(*) as trades,
            SUM(actual_pnl) as total_pnl
        FROM ml_predictions
        WHERE confidence_level = 'HIGH'
    ''').fetchone()

    # Low confidence performance
    low_conf = conn.execute('''
        SELECT
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy,
            COUNT(*) as trades,
            SUM(actual_pnl) as total_pnl
        FROM ml_predictions
        WHERE confidence_level = 'LOW'
    ''').fetchone()

    # TAKE vs AVOID performance
    take_perf = conn.execute('''
        SELECT
            SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
            COUNT(*) as trades,
            SUM(actual_pnl) as total_pnl,
            AVG(actual_pnl) as avg_pnl
        FROM ml_predictions
        WHERE recommendation = 'TAKE'
    ''').fetchone()

    avoid_perf = conn.execute('''
        SELECT
            SUM(CASE WHEN actual_outcome = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as correct_avoids,
            COUNT(*) as trades,
            SUM(actual_pnl) as total_pnl
        FROM ml_predictions
        WHERE recommendation = 'AVOID'
    ''').fetchone()

    # Best account
    best_account = conn.execute('''
        SELECT account_id, outperformance_pct
        FROM benchmark_comparison
        WHERE beat_market = 1
        ORDER BY outperformance_pct DESC
        LIMIT 1
    ''').fetchone()

    # Recent trend (last 100 trades)
    recent_trend = conn.execute('''
        SELECT
            SUM(CASE WHEN hybrid_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as accuracy,
            SUM(actual_pnl) as total_pnl
        FROM (
            SELECT hybrid_correct, actual_pnl
            FROM ml_predictions
            ORDER BY entry_ts DESC
            LIMIT 100
        )
    ''').fetchone()

    conn.close()

    insights = []

    # Best symbol insight
    if best_symbol:
        insights.append({
            'type': 'success',
            'icon': '🎯',
            'title': f'{best_symbol["symbol"]} is your most predictable symbol',
            'description': f'{round(best_symbol["accuracy"], 1)}% ML accuracy across {best_symbol["trades"]} trades. Consider focusing here.'
        })

    # Worst symbol insight
    if worst_symbol:
        insights.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': f'{worst_symbol["symbol"]} has low ML accuracy',
            'description': f'Only {round(worst_symbol["accuracy"], 1)}% accuracy. Model struggles with this symbol - requires extra caution.'
        })

    # Confidence insight
    if high_conf and low_conf:
        high_acc = round(high_conf['accuracy'] or 0, 1)
        low_acc = round(low_conf['accuracy'] or 0, 1)
        if high_acc > low_acc + 5:
            insights.append({
                'type': 'info',
                'icon': '📊',
                'title': 'HIGH confidence signals outperform',
                'description': f'HIGH confidence: {high_acc}% accurate. LOW confidence: {low_acc}%. Consider skipping LOW confidence signals.'
            })

    # TAKE recommendation insight
    if take_perf:
        insights.append({
            'type': 'success',
            'icon': '✅',
            'title': 'TAKE recommendations are profitable',
            'description': f'{round(take_perf["win_rate"] or 0, 1)}% win rate, ${round(take_perf["avg_pnl"] or 0, 2)} avg P&L per trade.'
        })

    # Best account insight
    if best_account:
        insights.append({
            'type': 'info',
            'icon': '🏆',
            'title': f'{best_account["account_id"]} beats the market',
            'description': f'Outperforms S&P 500 by {round(best_account["outperformance_pct"], 1)}%. Study this account\'s patterns.'
        })

    # Recent trend
    if recent_trend:
        recent_acc = round(recent_trend['accuracy'] or 0, 1)
        trend_pnl = round(recent_trend['total_pnl'] or 0, 2)
        trend_emoji = '📈' if trend_pnl > 0 else '📉'
        insights.append({
            'type': 'info' if trend_pnl > 0 else 'warning',
            'icon': trend_emoji,
            'title': 'Recent Performance (Last 100 trades)',
            'description': f'{recent_acc}% ML accuracy, ${trend_pnl:+,.2f} total P&L'
        })

    return jsonify({
        'insights': insights,
        'summary': {
            'best_symbol': best_symbol['symbol'] if best_symbol else None,
            'best_symbol_accuracy': round(best_symbol['accuracy'], 1) if best_symbol else 0,
            'worst_symbol': worst_symbol['symbol'] if worst_symbol else None,
            'high_conf_accuracy': round(high_conf['accuracy'] or 0, 1) if high_conf else 0,
            'low_conf_accuracy': round(low_conf['accuracy'] or 0, 1) if low_conf else 0,
            'take_win_rate': round(take_perf['win_rate'] or 0, 1) if take_perf else 0,
            'avoid_correct_rate': round(avoid_perf['correct_avoids'] or 0, 1) if avoid_perf else 0
        }
    })

@app.route('/api/metric/formula/<metric_name>')
def get_metric_formula(metric_name):
    """Get the formula and explanation for a specific metric"""

    formulas = {
        'net_pnl': {
            'name': 'Net Profit & Loss',
            'formula': 'Total Gains - Total Losses - Fees',
            'explanation': 'The total profit or loss after accounting for all trading fees and commissions.',
            'interpretation': 'Positive means overall profit, negative means overall loss. This is the bottom line of trading performance.'
        },
        'sharpe_ratio': {
            'name': 'Sharpe Ratio',
            'formula': '(Portfolio Return - Risk Free Rate) / Portfolio Standard Deviation',
            'explanation': 'Measures risk-adjusted return. Higher is better. Above 1 is good, above 2 is very good, above 3 is excellent.',
            'interpretation': 'Shows how much excess return you receive for the extra volatility of holding a risky asset.'
        },
        'sortino_ratio': {
            'name': 'Sortino Ratio',
            'formula': '(Portfolio Return - Risk Free Rate) / Downside Deviation',
            'explanation': 'Similar to Sharpe but only penalizes downside volatility. More accurate for asymmetric return distributions.',
            'interpretation': 'Better than Sharpe for strategies with positive skew (many small wins, few large losses).'
        },
        'calmar_ratio': {
            'name': 'Calmar Ratio',
            'formula': 'Annualized Return / Maximum Drawdown',
            'explanation': 'Measures return relative to worst peak-to-trough decline. Higher is better.',
            'interpretation': 'Shows how much return you get per unit of maximum loss risk.'
        },
        'max_drawdown': {
            'name': 'Maximum Drawdown',
            'formula': '(Trough Value - Peak Value) / Peak Value x 100%',
            'explanation': 'Largest peak-to-trough decline in portfolio value. Always negative.',
            'interpretation': 'Represents worst case loss from a previous high. Critical for understanding risk tolerance.'
        },
        'win_rate': {
            'name': 'Win Rate',
            'formula': 'Number of Winning Trades / Total Trades x 100%',
            'explanation': 'Percentage of trades that were profitable.',
            'interpretation': 'Above 50% is positive, but can be misleading if average loss > average win.'
        },
        'risk_reward_ratio': {
            'name': 'Risk/Reward Ratio',
            'formula': 'Average Win / Average Loss',
            'explanation': 'How much you make on winners vs lose on losers. Above 1 means wins are larger than losses.',
            'interpretation': 'A 2:1 ratio means you make $2 for every $1 lost on average.'
        },
        'var_95_pct': {
            'name': 'Value at Risk (95%)',
            'formula': '5th Percentile of Returns Distribution',
            'explanation': '95% confidence that daily loss will not exceed this amount.',
            'interpretation': 'On a normal day, you should not lose more than this. Useful for risk budgeting.'
        },
        'volatility': {
            'name': 'Volatility (Annualized)',
            'formula': 'Standard Deviation of Returns x sqrt(252)',
            'explanation': 'Annualized standard deviation of returns. Measures how much returns fluctuate.',
            'interpretation': 'Higher volatility = higher risk. Typical stocks: 15-25%, portfolios: 10-20%.'
        },
        'total_trades': {
            'name': 'Total Trades',
            'formula': 'Count of all executed trades',
            'explanation': 'The total number of completed trades (buy + sell pairs).',
            'interpretation': 'Higher count indicates more active trading. Consider alongside fees and win rate.'
        },
        'profit_factor': {
            'name': 'Profit Factor',
            'formula': 'Gross Profit / Gross Loss',
            'explanation': 'Ratio of total profits to total losses. Above 1 means profitable overall.',
            'interpretation': 'A profit factor of 2 means you make $2 for every $1 lost. Above 1.5 is good, above 2 is excellent.'
        },
        'expectancy': {
            'name': 'Expectancy',
            'formula': '(Win Rate x Average Win) - (Loss Rate x Average Loss)',
            'explanation': 'Expected value per trade. Positive means strategy is profitable on average.',
            'interpretation': 'Shows how much you expect to make per trade on average. Critical for position sizing.'
        },
        'largest_loss': {
            'name': 'Largest Loss',
            'formula': 'MIN(Trade P&L)',
            'explanation': 'The single largest losing trade in the account.',
            'interpretation': 'Important for understanding worst-case scenarios and setting stop losses.'
        },
        'largest_win': {
            'name': 'Largest Win',
            'formula': 'MAX(Trade P&L)',
            'explanation': 'The single largest winning trade in the account.',
            'interpretation': 'Shows potential upside. Compare with largest loss for risk assessment.'
        },
        'avg_win': {
            'name': 'Average Win',
            'formula': 'SUM(Winning Trades P&L) / Count(Winning Trades)',
            'explanation': 'Average profit on winning trades.',
            'interpretation': 'Should be compared with average loss for risk/reward assessment.'
        },
        'avg_loss': {
            'name': 'Average Loss',
            'formula': 'SUM(Losing Trades P&L) / Count(Losing Trades)',
            'explanation': 'Average loss on losing trades.',
            'interpretation': 'Compare with average win. Ideally avg win should exceed avg loss.'
        },
        'gross_profit': {
            'name': 'Gross Profit',
            'formula': 'SUM(All Winning Trades P&L)',
            'explanation': 'Total profit from all winning trades before losses.',
            'interpretation': 'Raw profit-generating capacity of the strategy.'
        },
        'gross_loss': {
            'name': 'Gross Loss',
            'formula': 'SUM(All Losing Trades P&L)',
            'explanation': 'Total loss from all losing trades.',
            'interpretation': 'Total capital lost to losing trades. Compare with gross profit for profit factor.'
        }
    }

    formula_data = formulas.get(metric_name, {
        'name': metric_name.replace('_', ' ').title(),
        'formula': 'Formula not available',
        'explanation': 'Detailed explanation not available for this metric.',
        'interpretation': 'Please refer to financial literature for more information.'
    })

    return jsonify(formula_data)

# =============================================================================
# MARKET DATA ENDPOINTS
# =============================================================================

@app.route('/api/market/symbols')
def get_market_symbols():
    """Get list of available symbols with price data"""
    conn = get_db_connection()

    symbols = conn.execute('''
        SELECT
            i.instrument_id,
            i.symbol,
            i.asset_class,
            i.exchange,
            COUNT(p.price_id) as price_count,
            MIN(p.ts) as first_date,
            MAX(p.ts) as last_date
        FROM instruments i
        LEFT JOIN prices p ON i.instrument_id = p.instrument_id
        GROUP BY i.instrument_id
        ORDER BY i.symbol
    ''').fetchall()

    conn.close()

    return jsonify([dict(row) for row in symbols])

@app.route('/api/market/prices/<symbol>')
def get_market_prices(symbol):
    """Get price history for a symbol"""
    conn = get_db_connection()

    # Get optional query parameters
    days = request.args.get('days', 365, type=int)

    # Calculate the start date based on the number of days requested
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    prices = conn.execute('''
        SELECT
            p.ts as date,
            p.px_open as open,
            p.px_high as high,
            p.px_low as low,
            p.px_close as close,
            p.px_vol as volume
        FROM prices p
        JOIN instruments i ON p.instrument_id = i.instrument_id
        WHERE i.symbol = ?
          AND date(p.ts) >= date(?)
        ORDER BY p.ts ASC
    ''', (symbol, start_date)).fetchall()

    conn.close()

    prices_list = [dict(row) for row in prices]

    return jsonify({
        'symbol': symbol,
        'prices': prices_list,
        'count': len(prices_list)
    })

@app.route('/api/market/summary')
def get_market_summary():
    """Get market summary with latest prices and changes"""
    conn = get_db_connection()

    # Get latest prices and calculate changes
    summary = conn.execute('''
        WITH latest AS (
            SELECT
                i.symbol,
                p.px_close,
                p.px_open,
                p.px_high,
                p.px_low,
                p.px_vol,
                p.ts,
                ROW_NUMBER() OVER (PARTITION BY i.instrument_id ORDER BY p.ts DESC) as rn
            FROM instruments i
            JOIN prices p ON i.instrument_id = p.instrument_id
        ),
        prev AS (
            SELECT
                i.symbol,
                p.px_close as prev_close,
                ROW_NUMBER() OVER (PARTITION BY i.instrument_id ORDER BY p.ts DESC) as rn
            FROM instruments i
            JOIN prices p ON i.instrument_id = p.instrument_id
        )
        SELECT
            l.symbol,
            l.px_close as price,
            l.px_open as open,
            l.px_high as high,
            l.px_low as low,
            l.px_vol as volume,
            l.ts as date,
            pr.prev_close,
            CASE WHEN pr.prev_close > 0
                THEN ((l.px_close - pr.prev_close) / pr.prev_close * 100)
                ELSE 0
            END as change_pct
        FROM latest l
        LEFT JOIN prev pr ON l.symbol = pr.symbol AND pr.rn = 2
        WHERE l.rn = 1
        ORDER BY l.symbol
    ''').fetchall()

    conn.close()

    return jsonify([dict(row) for row in summary])

# =============================================================================
# DATABASE EXPLORER ENDPOINTS
# =============================================================================

@app.route('/api/database/info')
def get_database_info():
    """Get database statistics"""
    conn = get_db_connection()

    # Get table count
    tables = conn.execute('''
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    ''').fetchall()
    table_count = len(tables)

    # Get total rows across all tables
    total_rows = 0
    for table in tables:
        count = conn.execute(f'SELECT COUNT(*) FROM "{table["name"]}"').fetchone()[0]
        total_rows += count

    # Get index count
    index_count = conn.execute('''
        SELECT COUNT(*) FROM sqlite_master WHERE type='index'
    ''').fetchone()[0]

    # Get database size
    db_size = os.path.getsize(DB_PATH)

    conn.close()

    return jsonify({
        'table_count': table_count,
        'total_rows': total_rows,
        'index_count': index_count,
        'db_size_bytes': db_size,
        'db_size_mb': round(db_size / (1024 * 1024), 2)
    })

@app.route('/api/database/tables')
def get_database_tables():
    """Get all tables with row counts"""
    conn = get_db_connection()

    tables = conn.execute('''
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    ''').fetchall()

    result = []
    for table in tables:
        table_name = table['name']
        count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        result.append({
            'name': table_name,
            'row_count': count
        })

    conn.close()
    return jsonify(result)

@app.route('/api/database/schema/<table_name>')
def get_table_schema(table_name):
    """Get schema for a specific table"""
    conn = get_db_connection()

    # Get table info using PRAGMA
    columns = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()

    # Get foreign keys
    foreign_keys = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()

    # Get sample data
    sample_data = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 5').fetchall()

    # Get row count
    row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

    conn.close()

    return jsonify({
        'table_name': table_name,
        'columns': [dict(col) for col in columns],
        'foreign_keys': [dict(fk) for fk in foreign_keys],
        'sample_data': [dict(row) for row in sample_data],
        'row_count': row_count
    })

@app.route('/api/database/query', methods=['POST'])
def execute_sql_query():
    """Execute a read-only SQL query"""
    data = request.get_json()
    query = data.get('query', '').strip()

    # Security: Only allow SELECT statements
    if not query.upper().startswith('SELECT'):
        return jsonify({
            'error': 'Only SELECT queries are allowed for security reasons'
        }), 400

    # Block dangerous keywords
    dangerous = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', '--', ';--']
    for keyword in dangerous:
        if keyword in query.upper():
            return jsonify({
                'error': f'Query contains forbidden keyword: {keyword}'
            }), 400

    try:
        conn = get_db_connection()
        cursor = conn.execute(query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        # Limit results to 1000 rows
        rows = rows[:1000]

        return jsonify({
            'columns': columns,
            'rows': [dict(row) for row in rows],
            'row_count': len(rows),
            'truncated': len(rows) == 1000
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
