#!/usr/bin/env python3
"""
Apply ML Model Predictions to Trading Database

This script:
1. Adds a 'predictions' table to the database
2. Runs all three models on historical trades
3. Stores predictions in the database
4. Shows which trades should have been taken/avoided
5. Calculates actual performance improvement

BUSINESS VALUE:
- Before trade: Model predicts if trade will be profitable
- Decision: Take trade if model confidence > threshold
- Result: Improved win rate and account performance
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')


def create_predictions_table(conn):
    """Add predictions table to database"""
    print("\n" + "="*80)
    print("CREATING PREDICTIONS TABLE")
    print("="*80)

    cursor = conn.cursor()

    # Create predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        fill_id INTEGER NOT NULL,
        account_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        entry_ts TIMESTAMP NOT NULL,

        -- Model predictions
        xgboost_prediction INTEGER,           -- 0=loss, 1=profit
        xgboost_probability REAL,             -- 0.0-1.0 confidence
        hybrid_prediction INTEGER,
        hybrid_probability REAL,

        -- Actual outcome
        actual_outcome INTEGER,               -- 0=loss, 1=profit
        actual_pnl REAL,

        -- Model performance
        xgboost_correct INTEGER,              -- 1 if correct, 0 if wrong
        hybrid_correct INTEGER,

        -- Recommendation
        recommendation TEXT,                  -- 'TAKE', 'AVOID', 'UNCERTAIN'
        confidence_level TEXT,                -- 'HIGH', 'MEDIUM', 'LOW'

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (fill_id) REFERENCES fills(fill_id),
        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    )
    """)

    # Create index for fast lookups
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_predictions_account
    ON ml_predictions(account_id)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_predictions_fill
    ON ml_predictions(fill_id)
    """)

    conn.commit()
    print("✓ Created ml_predictions table")


def load_models():
    """Load trained models"""
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)

    models = {}

    # Load XGBoost
    xgb_path = os.path.join(MODEL_DIR, 'xgboost_trade_classifier.pkl')
    with open(xgb_path, 'rb') as f:
        models['xgboost'] = pickle.load(f)
    print("✓ Loaded XGBoost model")

    # Load XGBoost scaler
    xgb_scaler_path = os.path.join(MODEL_DIR, 'xgboost_scaler.pkl')
    with open(xgb_scaler_path, 'rb') as f:
        models['xgboost_scaler'] = pickle.load(f)

    # Load XGBoost features
    xgb_features_path = os.path.join(MODEL_DIR, 'xgboost_features.pkl')
    with open(xgb_features_path, 'rb') as f:
        models['xgboost_features'] = pickle.load(f)

    # Load Hybrid
    hybrid_path = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
    with open(hybrid_path, 'rb') as f:
        models['hybrid'] = pickle.load(f)
    print("✓ Loaded Hybrid model")

    # Load Hybrid scaler
    hybrid_scaler_path = os.path.join(MODEL_DIR, 'hybrid_scaler.pkl')
    with open(hybrid_scaler_path, 'rb') as f:
        models['hybrid_scaler'] = pickle.load(f)

    # Load Hybrid features
    hybrid_features_path = os.path.join(MODEL_DIR, 'hybrid_features.pkl')
    with open(hybrid_features_path, 'rb') as f:
        models['hybrid_features'] = pickle.load(f)

    # Load CNN for hybrid model (replaces LSTM - 10-50x faster)
    cnn_path = os.path.join(MODEL_DIR, 'cnn_SPY_best.keras')

    if os.path.exists(cnn_path):
        models['cnn'] = keras.models.load_model(cnn_path)
        print("✓ Loaded CNN model (for hybrid) - 10-50x faster than LSTM")
    else:
        print("⚠️ CNN model not found - hybrid predictions will skip CNN features")
        models['cnn'] = None

    return models


def get_trade_outcomes(conn):
    """Get all historical trades with outcomes"""
    print("\n" + "="*80)
    print("LOADING HISTORICAL TRADES")
    print("="*80)

    # Import from hybrid_model.py
    from hybrid_model import calculate_trade_outcomes

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
    ORDER BY f.account_id, i.symbol, f.trade_ts
    """

    fills_df = pd.read_sql(query, conn)
    print(f"✓ Loaded {len(fills_df)} fills")

    trades_df = calculate_trade_outcomes(fills_df)
    print(f"✓ Calculated {len(trades_df)} closed trades with outcomes")

    return trades_df


def generate_predictions(trades_df, models, conn):
    """Generate predictions for all trades"""
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS FOR ALL TRADES")
    print("="*80)

    # Import feature engineering from xgboost_trade_classifier
    sys.path.insert(0, os.path.dirname(__file__))
    from xgboost_trade_classifier import engineer_features as engineer_xgb_features

    print("✓ Engineering XGBoost features (full feature set)...")

    # Get all XGBoost features (this calculates ALL features needed)
    trades_with_xgb_features = engineer_xgb_features(trades_df)

    # Get CNN features for hybrid model (10-50x faster than LSTM)
    print("✓ Generating CNN features (this is fast with 1D CNN)...")
    from hybrid_model import get_cnn_features_for_trades

    trades_with_all_features = get_cnn_features_for_trades(
        trades_with_xgb_features,
        models['cnn'],
        'CNN',
        60
    )

    print("✓ Features ready, making BATCH predictions (fast)...")

    # BATCH PREDICTION for speed - no more slow loops
    xgb_features = models['xgboost_features']
    hybrid_features = models['hybrid_features']

    # Prepare feature matrices
    X_xgb = trades_with_all_features[xgb_features].values
    X_hybrid = trades_with_all_features[hybrid_features].values

    # Handle NaN values
    X_xgb = np.nan_to_num(X_xgb, nan=0.0)
    X_hybrid = np.nan_to_num(X_hybrid, nan=0.0)

    # Scale features
    X_xgb_scaled = models['xgboost_scaler'].transform(X_xgb)
    X_hybrid_scaled = models['hybrid_scaler'].transform(X_hybrid)

    print("  Making XGBoost batch predictions...")
    xgb_preds = models['xgboost'].predict(X_xgb_scaled)
    xgb_probas = models['xgboost'].predict_proba(X_xgb_scaled)[:, 1]

    print("  Making Hybrid batch predictions...")
    hybrid_preds = models['hybrid'].predict(X_hybrid_scaled)
    hybrid_probas = models['hybrid'].predict_proba(X_hybrid_scaled)[:, 1]

    print("  Building predictions dataframe...")

    # Get actual outcomes
    actual_outcomes = trades_with_all_features['is_profitable'].values
    actual_pnls = trades_with_all_features['net_pnl'].values

    # Vectorized recommendation logic
    recommendations = np.where(
        hybrid_probas >= 0.65, 'TAKE',
        np.where(hybrid_probas >= 0.55, 'TAKE',
        np.where(hybrid_probas <= 0.35, 'AVOID',
        np.where(hybrid_probas <= 0.45, 'AVOID', 'UNCERTAIN')))
    )

    confidence_levels = np.where(
        (hybrid_probas >= 0.65) | (hybrid_probas <= 0.35), 'HIGH',
        np.where((hybrid_probas >= 0.55) | (hybrid_probas <= 0.45), 'MEDIUM', 'LOW')
    )

    # Build predictions dataframe directly
    predictions_df = pd.DataFrame({
        'fill_id': trades_with_all_features['fill_id'].values,
        'account_id': trades_with_all_features['account_id'].values,
        'symbol': trades_with_all_features['symbol'].values,
        'entry_ts': trades_with_all_features['entry_ts'].values,
        'xgboost_prediction': xgb_preds.astype(int),
        'xgboost_probability': xgb_probas,
        'hybrid_prediction': hybrid_preds.astype(int),
        'hybrid_probability': hybrid_probas,
        'actual_outcome': actual_outcomes.astype(int),
        'actual_pnl': actual_pnls,
        'xgboost_correct': (xgb_preds == actual_outcomes).astype(int),
        'hybrid_correct': (hybrid_preds == actual_outcomes).astype(int),
        'recommendation': recommendations,
        'confidence_level': confidence_levels
    })

    print(f"✓ Generated {len(predictions_df)} predictions (batch mode)")

    return predictions_df


def save_predictions(predictions_df, conn):
    """Save predictions to database"""
    print("\n" + "="*80)
    print("SAVING PREDICTIONS TO DATABASE")
    print("="*80)

    # Clear existing predictions
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ml_predictions")
    conn.commit()

    # Insert new predictions
    predictions_df.to_sql('ml_predictions', conn, if_exists='append', index=False)

    print(f"✓ Saved {len(predictions_df)} predictions to database")


def analyze_predictions(conn):
    """Analyze prediction performance and business impact"""
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)

    # Load predictions
    pred_df = pd.read_sql("SELECT * FROM ml_predictions", conn)

    print(f"\n📊 OVERALL STATISTICS")
    print("─" * 80)
    print(f"Total Predictions:     {len(pred_df)}")
    print(f"Actual Profitable:     {pred_df['actual_outcome'].sum()} ({pred_df['actual_outcome'].mean()*100:.1f}%)")
    print(f"Actual Losses:         {(1-pred_df['actual_outcome']).sum()} ({(1-pred_df['actual_outcome']).mean()*100:.1f}%)")

    print(f"\n🎯 MODEL ACCURACY")
    print("─" * 80)
    xgb_acc = pred_df['xgboost_correct'].mean()
    hybrid_acc = pred_df['hybrid_correct'].mean()
    print(f"XGBoost Accuracy:      {xgb_acc*100:.2f}%")
    print(f"Hybrid Accuracy:       {hybrid_acc*100:.2f}%")
    print(f"Improvement:           +{(hybrid_acc - xgb_acc)*100:.2f} percentage points")

    print(f"\n💡 RECOMMENDATIONS BREAKDOWN")
    print("─" * 80)
    recs = pred_df['recommendation'].value_counts()
    for rec, count in recs.items():
        pct = count / len(pred_df) * 100
        actual_win_rate = pred_df[pred_df['recommendation'] == rec]['actual_outcome'].mean() * 100
        print(f"{rec:12s}: {count:4d} trades ({pct:5.1f}%) - Win Rate: {actual_win_rate:.1f}%")

    print(f"\n🎲 CONFIDENCE LEVELS")
    print("─" * 80)
    conf_levels = pred_df['confidence_level'].value_counts()
    for conf, count in conf_levels.items():
        pct = count / len(pred_df) * 100
        acc = pred_df[pred_df['confidence_level'] == conf]['hybrid_correct'].mean() * 100
        print(f"{conf:12s}: {count:4d} trades ({pct:5.1f}%) - Accuracy: {acc:.1f}%")

    # Business impact simulation
    print(f"\n💰 BUSINESS IMPACT SIMULATION")
    print("─" * 80)
    print("Scenario: Follow Hybrid Model's HIGH confidence recommendations")
    print()

    # Baseline: Take all trades
    baseline_trades = len(pred_df)
    baseline_wins = pred_df['actual_outcome'].sum()
    baseline_win_rate = baseline_wins / baseline_trades * 100
    baseline_pnl = pred_df['actual_pnl'].sum()

    print(f"BASELINE (Take All Trades):")
    print(f"  Total Trades:      {baseline_trades}")
    print(f"  Winning Trades:    {baseline_wins}")
    print(f"  Win Rate:          {baseline_win_rate:.2f}%")
    print(f"  Total P&L:         ${baseline_pnl:,.2f}")
    print()

    # Strategy 1: Only take HIGH confidence TAKE recommendations
    high_take = pred_df[(pred_df['recommendation'] == 'TAKE') &
                         (pred_df['confidence_level'] == 'HIGH')]

    if len(high_take) > 0:
        strategy1_trades = len(high_take)
        strategy1_wins = high_take['actual_outcome'].sum()
        strategy1_win_rate = strategy1_wins / strategy1_trades * 100
        strategy1_pnl = high_take['actual_pnl'].sum()

        print(f"STRATEGY 1 (Only HIGH confidence TAKE):")
        print(f"  Total Trades:      {strategy1_trades} ({strategy1_trades/baseline_trades*100:.1f}% of baseline)")
        print(f"  Winning Trades:    {strategy1_wins}")
        print(f"  Win Rate:          {strategy1_win_rate:.2f}% (+{strategy1_win_rate - baseline_win_rate:.2f}pp)")
        print(f"  Total P&L:         ${strategy1_pnl:,.2f}")
        print(f"  P&L Change:        ${strategy1_pnl - baseline_pnl:,.2f}")
        print()

    # Strategy 2: Take TAKE recommendations (HIGH + MEDIUM)
    take_all = pred_df[pred_df['recommendation'] == 'TAKE']

    if len(take_all) > 0:
        strategy2_trades = len(take_all)
        strategy2_wins = take_all['actual_outcome'].sum()
        strategy2_win_rate = strategy2_wins / strategy2_trades * 100
        strategy2_pnl = take_all['actual_pnl'].sum()

        print(f"STRATEGY 2 (All TAKE recommendations):")
        print(f"  Total Trades:      {strategy2_trades} ({strategy2_trades/baseline_trades*100:.1f}% of baseline)")
        print(f"  Winning Trades:    {strategy2_wins}")
        print(f"  Win Rate:          {strategy2_win_rate:.2f}% (+{strategy2_win_rate - baseline_win_rate:.2f}pp)")
        print(f"  Total P&L:         ${strategy2_pnl:,.2f}")
        print(f"  P&L Change:        ${strategy2_pnl - baseline_pnl:,.2f}")

    return pred_df


def show_example_predictions(conn):
    """Show specific example predictions"""
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    query = """
    SELECT
        p.account_id,
        p.symbol,
        p.entry_ts,
        p.hybrid_probability,
        p.recommendation,
        p.confidence_level,
        p.actual_outcome,
        p.actual_pnl,
        CASE WHEN p.hybrid_correct = 1 THEN '✓ CORRECT' ELSE '✗ WRONG' END as result
    FROM ml_predictions p
    WHERE p.confidence_level = 'HIGH'
    ORDER BY p.hybrid_probability DESC
    LIMIT 10
    """

    df = pd.read_sql(query, conn)

    print("\n🔝 TOP 10 HIGH CONFIDENCE PREDICTIONS:")
    print(df.to_string(index=False))

    # Show some that were wrong
    query_wrong = """
    SELECT
        p.account_id,
        p.symbol,
        p.entry_ts,
        p.hybrid_probability,
        p.recommendation,
        p.actual_outcome,
        p.actual_pnl
    FROM ml_predictions p
    WHERE p.hybrid_correct = 0
      AND p.confidence_level = 'HIGH'
    ORDER BY ABS(p.hybrid_probability - 0.5) DESC
    LIMIT 5
    """

    df_wrong = pd.read_sql(query_wrong, conn)

    if len(df_wrong) > 0:
        print("\n\n❌ EXAMPLES WHERE MODEL WAS WRONG (HIGH CONFIDENCE):")
        print(df_wrong.to_string(index=False))


def generate_account_reports(conn):
    """Generate per-account impact reports"""
    print("\n" + "="*80)
    print("ACCOUNT-LEVEL IMPACT")
    print("="*80)

    query = """
    SELECT
        p.account_id,
        COUNT(*) as total_trades,
        SUM(p.actual_outcome) as actual_wins,
        ROUND(AVG(p.actual_outcome) * 100, 2) as actual_win_rate,
        SUM(CASE WHEN p.recommendation = 'TAKE' THEN 1 ELSE 0 END) as recommended_trades,
        SUM(CASE WHEN p.recommendation = 'TAKE' THEN p.actual_outcome ELSE 0 END) as recommended_wins,
        ROUND(
            CASE
                WHEN SUM(CASE WHEN p.recommendation = 'TAKE' THEN 1 ELSE 0 END) > 0
                THEN AVG(CASE WHEN p.recommendation = 'TAKE' THEN p.actual_outcome ELSE NULL END) * 100
                ELSE 0
            END,
            2
        ) as recommended_win_rate,
        ROUND(SUM(p.actual_pnl), 2) as actual_pnl,
        ROUND(SUM(CASE WHEN p.recommendation = 'TAKE' THEN p.actual_pnl ELSE 0 END), 2) as recommended_pnl
    FROM ml_predictions p
    GROUP BY p.account_id
    ORDER BY actual_pnl DESC
    LIMIT 15
    """

    df = pd.read_sql(query, conn)

    print("\n📈 TOP 15 ACCOUNTS BY ACTUAL P&L:")
    print(df.to_string(index=False))

    print("\n\n💡 INSIGHTS:")

    # Calculate overall improvement
    total_actual_wr = df['actual_wins'].sum() / df['total_trades'].sum() * 100
    total_recommended_wr = df['recommended_wins'].sum() / df['recommended_trades'].sum() * 100

    print(f"  Overall Win Rate:           {total_actual_wr:.2f}%")
    print(f"  Win Rate (Following Model): {total_recommended_wr:.2f}%")
    print(f"  Improvement:                +{total_recommended_wr - total_actual_wr:.2f} percentage points")

    total_actual_pnl = df['actual_pnl'].sum()
    total_recommended_pnl = df['recommended_pnl'].sum()

    print(f"\n  Total P&L (All Trades):     ${total_actual_pnl:,.2f}")
    print(f"  Total P&L (Following Model):${total_recommended_pnl:,.2f}")
    print(f"  Difference:                 ${total_recommended_pnl - total_actual_pnl:,.2f}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("APPLYING ML PREDICTIONS TO TRADING DATABASE")
    print("="*80)
    print(f"\nDatabase: {DB_PATH}")
    print("This will:")
    print("  1. Add predictions table to database")
    print("  2. Load trained ML models")
    print("  3. Generate predictions for all historical trades")
    print("  4. Store predictions in database")
    print("  5. Analyze business impact")
    print("="*80)

    conn = sqlite3.connect(DB_PATH)

    try:
        # 1. Create predictions table
        create_predictions_table(conn)

        # 2. Load models
        models = load_models()

        # 3. Get trade outcomes
        trades_df = get_trade_outcomes(conn)

        # 4. Generate predictions
        predictions_df = generate_predictions(trades_df, models, conn)

        # 5. Save to database
        save_predictions(predictions_df, conn)

        # 6. Analyze results
        analyze_predictions(conn)

        # 7. Show examples
        show_example_predictions(conn)

        # 8. Account reports
        generate_account_reports(conn)

        print("\n" + "="*80)
        print("✓ PREDICTIONS APPLIED SUCCESSFULLY!")
        print("="*80)
        print("\nYou can now query predictions from the database:")
        print("  SELECT * FROM ml_predictions WHERE recommendation = 'TAKE';")
        print("  SELECT * FROM ml_predictions WHERE confidence_level = 'HIGH';")
        print("\nOr view in the web interface (if running api_server.py)")
        print()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
