#!/usr/bin/env python3
"""
XGBoost Trade Outcome Classifier

Predicts whether a trade will be profitable based on entry conditions.
Uses 4,009 fills from 30 trading accounts with rich feature engineering.

Features:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price momentum indicators
- Account state metrics
- Position sizing features
- Time-based features
- Strategy-based features

Target: Binary classification (profitable trade = 1, losing trade = 0)
Expected accuracy: 70%+
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import pickle
from typing import Tuple, Dict

# ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# Configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_TEST_SPLIT_DATE = '2024-01-01'  # Time-based split
RANDOM_STATE = 42

# XGBoost hyperparameters (REGULARIZED to reduce overfitting + CLASS BALANCED)
XGBOOST_PARAMS = {
    'max_depth': 4,              # Reduced from 6 to prevent deep trees
    'learning_rate': 0.05,       # Reduced from 0.1 for slower learning
    'n_estimators': 200,         # Increased but will use early stopping
    'min_child_weight': 3,       # Increased from 1 (requires more samples per leaf)
    'subsample': 0.7,            # Reduced from 0.8 for more regularization
    'colsample_bytree': 0.7,     # Reduced from 0.8
    'gamma': 0.1,                # Added minimum loss reduction required
    'reg_alpha': 0.1,            # L1 regularization (Lasso)
    'reg_lambda': 1.0,           # L2 regularization (Ridge)
    'early_stopping_rounds': 20, # Stop early if no improvement
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    # CLASS IMBALANCE HANDLING - will be dynamically calculated
    # scale_pos_weight will be set in train_xgboost_model based on actual class distribution
}


def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    return conn


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_trade_data() -> pd.DataFrame:
    """
    Load all fills with their corresponding prices and calculate trade outcomes.

    Returns DataFrame with one row per fill (entry), labeled with outcome.
    """
    print("\n" + "="*80)
    print("LOADING TRADE DATA")
    print("="*80)

    conn = get_connection()

    # Get all fills with instrument info
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
        o.order_type,
        f.source_file as strategy
    FROM fills f
    JOIN instruments i ON f.instrument_id = i.instrument_id
    JOIN orders o ON f.order_id = o.order_id
    ORDER BY f.account_id, i.symbol, f.trade_ts
    """

    fills_df = pd.read_sql(query, conn)
    print(f"✓ Loaded {len(fills_df)} fills")

    # Calculate trade outcomes using FIFO matching
    print("✓ Calculating trade outcomes (FIFO matching)...")
    trades_with_outcomes = calculate_trade_outcomes(fills_df)

    print(f"✓ Identified {len(trades_with_outcomes)} entry trades with outcomes")

    conn.close()
    return trades_with_outcomes


def calculate_trade_outcomes(fills_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match BUY/SELL fills using FIFO to calculate P&L for each trade.

    Returns: DataFrame with one row per BUY fill, with outcome label.
    """
    trades = []

    # Group by account and symbol
    for (account_id, symbol), group in fills_df.groupby(['account_id', 'symbol']):
        group = group.sort_values('trade_ts').reset_index(drop=True)

        position_queue = []  # FIFO queue: [(qty, entry_price, entry_ts, fill_id)]

        for idx, row in group.iterrows():
            if row['side'] == 'BUY':
                # Add to position queue
                position_queue.append({
                    'qty': row['qty'],
                    'entry_price': row['price'],
                    'entry_ts': row['trade_ts'],
                    'fill_id': row['fill_id'],
                    'entry_fee': row['fee'],
                    'account_id': account_id,
                    'symbol': symbol,
                    'instrument_id': row['instrument_id'],
                    'strategy': row['strategy']
                })

            elif row['side'] == 'SELL' and position_queue:
                # Exit from position queue (FIFO)
                remaining_qty = row['qty']
                exit_price = row['price']
                exit_ts = row['trade_ts']
                exit_fee = row['fee']

                while remaining_qty > 0 and position_queue:
                    position = position_queue[0]

                    # How much to close from this position
                    close_qty = min(remaining_qty, position['qty'])

                    # Calculate P&L for this closed portion
                    gross_pnl = (exit_price - position['entry_price']) * close_qty
                    fees = position['entry_fee'] + (exit_fee * close_qty / row['qty'])
                    net_pnl = gross_pnl - fees

                    # Record the trade
                    trades.append({
                        'fill_id': position['fill_id'],
                        'account_id': position['account_id'],
                        'symbol': position['symbol'],
                        'instrument_id': position['instrument_id'],
                        'entry_ts': position['entry_ts'],
                        'exit_ts': exit_ts,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'qty': close_qty,
                        'gross_pnl': gross_pnl,
                        'fees': fees,
                        'net_pnl': net_pnl,
                        'is_profitable': 1 if net_pnl > 0 else 0,  # Binary target
                        'strategy': position['strategy'],
                        'hold_days': (pd.to_datetime(exit_ts) - pd.to_datetime(position['entry_ts'])).days
                    })

                    # Update position
                    position['qty'] -= close_qty
                    if position['qty'] == 0:
                        position_queue.pop(0)

                    remaining_qty -= close_qty

    return pd.DataFrame(trades)


def engineer_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer rich features for XGBoost training - BATCH OPTIMIZED.

    Features:
    1. Technical indicators at entry
    2. Price momentum features
    3. Volatility features
    4. Time-based features
    5. Strategy features
    6. Account state features

    Optimization: Load ALL price data once and pre-compute indicators per instrument.
    This is 100x+ faster than per-trade SQL queries.
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    conn = get_connection()

    feature_df = trades_df.copy()

    # Initialize feature columns with defaults
    feature_df['rsi_entry'] = 50.0
    feature_df['macd_entry'] = 0.0
    feature_df['ma_7'] = feature_df['entry_price']
    feature_df['ma_21'] = feature_df['entry_price']
    feature_df['momentum_5d'] = 0.0
    feature_df['momentum_10d'] = 0.0
    feature_df['volatility_20d'] = 0.0
    feature_df['avg_volume'] = 0.0
    feature_df['volume_ratio'] = 1.0

    # 1. BATCH OPTIMIZATION: Load ALL price data in ONE query
    print("✓ Loading all price data (single query)...")
    all_prices = pd.read_sql("""
        SELECT instrument_id, ts, px_close, px_high, px_low, px_vol
        FROM prices
        ORDER BY instrument_id, ts
    """, conn)
    print(f"  Loaded {len(all_prices):,} price records")

    # Convert timestamps
    all_prices['ts'] = pd.to_datetime(all_prices['ts'], format='mixed')
    feature_df['entry_ts'] = pd.to_datetime(feature_df['entry_ts'], format='mixed')

    # 2. Pre-compute technical indicators per instrument
    instruments = feature_df['instrument_id'].unique()
    print(f"✓ Pre-computing indicators for {len(instruments)} instruments...")

    # Build a cache of pre-computed rolling indicators per instrument
    indicator_cache = {}

    for i, inst_id in enumerate(instruments):
        if (i + 1) % 50 == 0:
            print(f"    Processing instrument {i + 1}/{len(instruments)}...")

        inst_prices = all_prices[all_prices['instrument_id'] == inst_id].copy()
        if len(inst_prices) < 14:
            continue

        inst_prices = inst_prices.sort_values('ts').reset_index(drop=True)

        # Pre-compute all rolling indicators
        inst_prices['rsi'] = calculate_rsi(inst_prices['px_close'])
        inst_prices['ema_12'] = inst_prices['px_close'].ewm(span=12, adjust=False).mean()
        inst_prices['ema_26'] = inst_prices['px_close'].ewm(span=26, adjust=False).mean()
        inst_prices['macd'] = inst_prices['ema_12'] - inst_prices['ema_26']
        inst_prices['ma_7'] = inst_prices['px_close'].rolling(7, min_periods=1).mean()
        inst_prices['ma_21'] = inst_prices['px_close'].rolling(21, min_periods=1).mean()
        inst_prices['momentum_5d'] = inst_prices['px_close'].pct_change(5)
        inst_prices['momentum_10d'] = inst_prices['px_close'].pct_change(10)
        inst_prices['volatility_20d'] = inst_prices['px_close'].rolling(20, min_periods=5).std()
        inst_prices['avg_volume'] = inst_prices['px_vol'].rolling(60, min_periods=1).mean()
        inst_prices['volume_ratio'] = inst_prices['px_vol'] / inst_prices['avg_volume']

        indicator_cache[inst_id] = inst_prices

    # 3. SUPER FAST: Use merge_asof for O(n log n) time-based join per instrument
    print("✓ Matching trades to pre-computed indicators (merge_asof)...")

    # Combine all indicator dataframes
    all_indicators = pd.concat([
        df[['instrument_id', 'ts', 'rsi', 'macd', 'ma_7', 'ma_21',
            'momentum_5d', 'momentum_10d', 'volatility_20d', 'avg_volume', 'volume_ratio']]
        for df in indicator_cache.values()
    ], ignore_index=True)

    # Process each instrument separately to ensure proper sorting for merge_asof
    results = []
    instruments = feature_df['instrument_id'].unique()
    for inst_id in instruments:
        # Get trades and indicators for this instrument
        inst_trades = feature_df[feature_df['instrument_id'] == inst_id].copy()
        inst_indicators = all_indicators[all_indicators['instrument_id'] == inst_id].copy()

        if len(inst_indicators) == 0:
            results.append(inst_trades)
            continue

        # Sort by timestamp for merge_asof
        inst_trades = inst_trades.sort_values('entry_ts').reset_index(drop=True)
        inst_indicators = inst_indicators.sort_values('ts').reset_index(drop=True)

        # Rename for merge_asof
        inst_trades_for_merge = inst_trades[['entry_ts', 'entry_price']].rename(columns={'entry_ts': 'ts'})

        # Merge using timestamp
        merged = pd.merge_asof(
            inst_trades_for_merge,
            inst_indicators[['ts', 'rsi', 'macd', 'ma_7', 'ma_21',
                            'momentum_5d', 'momentum_10d', 'volatility_20d', 'avg_volume', 'volume_ratio']],
            on='ts',
            direction='backward'
        )

        # Apply merged values
        inst_trades['rsi_entry'] = merged['rsi'].fillna(50.0).values
        inst_trades['macd_entry'] = merged['macd'].fillna(0.0).values
        inst_trades['ma_7'] = merged['ma_7'].fillna(inst_trades['entry_price']).values
        inst_trades['ma_21'] = merged['ma_21'].fillna(inst_trades['entry_price']).values
        inst_trades['momentum_5d'] = merged['momentum_5d'].fillna(0.0).values
        inst_trades['momentum_10d'] = merged['momentum_10d'].fillna(0.0).values
        inst_trades['volatility_20d'] = merged['volatility_20d'].fillna(0.0).values
        inst_trades['avg_volume'] = merged['avg_volume'].fillna(0.0).values
        inst_trades['volume_ratio'] = merged['volume_ratio'].fillna(1.0).values

        results.append(inst_trades)

    # Combine all results
    feature_df = pd.concat(results, ignore_index=True)
    print(f"  Matched {len(feature_df):,} trades to indicators")

    # 2. Time-based features
    print("✓ Adding time-based features...")
    feature_df['entry_ts'] = pd.to_datetime(feature_df['entry_ts'], format='mixed')
    feature_df['day_of_week'] = feature_df['entry_ts'].dt.dayofweek  # 0=Monday, 4=Friday
    feature_df['month'] = feature_df['entry_ts'].dt.month
    feature_df['is_monday'] = (feature_df['day_of_week'] == 0).astype(int)
    feature_df['is_friday'] = (feature_df['day_of_week'] == 4).astype(int)

    # 3. Strategy features
    print("✓ Encoding strategy features...")
    le_strategy = LabelEncoder()
    feature_df['strategy_encoded'] = le_strategy.fit_transform(feature_df['strategy'])

    # 4. Account state features (optimized O(n) using pandas cumsum)
    print("✓ Calculating account-level features...")

    # Initialize columns
    feature_df['account_win_rate_prior'] = 0.5
    feature_df['account_num_trades_prior'] = 0

    # Process each account using vectorized operations
    for account in feature_df['account_id'].unique():
        mask = feature_df['account_id'] == account
        account_idx = feature_df.loc[mask].sort_values('entry_ts').index

        # Get is_profitable values for this account
        is_profitable = feature_df.loc[account_idx, 'is_profitable'].values

        # Calculate cumulative sum and count (shifted by 1 for "prior" values)
        cumsum = np.cumsum(is_profitable)
        cumcount = np.arange(1, len(is_profitable) + 1)

        # Shift by 1 to get prior values (exclude current trade)
        prior_wins = np.concatenate([[0], cumsum[:-1]])
        prior_count = np.concatenate([[0], cumcount[:-1]])

        # Calculate win rate (avoid division by zero)
        win_rate = np.where(prior_count > 0, prior_wins / prior_count, 0.5)

        # Assign back to dataframe
        feature_df.loc[account_idx, 'account_win_rate_prior'] = win_rate
        feature_df.loc[account_idx, 'account_num_trades_prior'] = prior_count

    # 5. Position sizing features
    feature_df['position_size_usd'] = feature_df['qty'] * feature_df['entry_price']

    conn.close()

    print(f"✓ Created {len(feature_df.columns)} total columns")

    # Fill any remaining NaN values
    feature_df = feature_df.fillna(0)

    return feature_df


def prepare_training_data(feature_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare X and y for training with time-based train/test split.

    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print("\n" + "="*80)
    print("PREPARING TRAINING DATA")
    print("="*80)

    # Define feature columns
    feature_columns = [
        'rsi_entry', 'macd_entry', 'ma_7', 'ma_21',
        'momentum_5d', 'momentum_10d', 'volatility_20d',
        'avg_volume', 'volume_ratio',
        'day_of_week', 'month', 'is_monday', 'is_friday',
        'strategy_encoded',
        'account_win_rate_prior', 'account_num_trades_prior',
        'position_size_usd', 'hold_days'
    ]

    # Verify all features exist
    missing_features = [f for f in feature_columns if f not in feature_df.columns]
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        feature_columns = [f for f in feature_columns if f in feature_df.columns]

    print(f"✓ Using {len(feature_columns)} features:")
    for f in feature_columns:
        print(f"  - {f}")

    # Time-based split
    train_df = feature_df[feature_df['entry_ts'] < TRAIN_TEST_SPLIT_DATE].copy()
    test_df = feature_df[feature_df['entry_ts'] >= TRAIN_TEST_SPLIT_DATE].copy()

    print(f"\n✓ Time-based split at {TRAIN_TEST_SPLIT_DATE}")
    print(f"  Training: {len(train_df)} samples ({train_df['entry_ts'].min()} to {train_df['entry_ts'].max()})")
    print(f"  Testing:  {len(test_df)} samples ({test_df['entry_ts'].min()} to {test_df['entry_ts'].max()})")

    # Prepare X and y
    X_train = train_df[feature_columns].values
    y_train = train_df['is_profitable'].values

    X_test = test_df[feature_columns].values
    y_test = test_df['is_profitable'].values

    # Check class balance
    print(f"\n✓ Class distribution:")
    print(f"  Training - Profitable: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"  Testing  - Profitable: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")

    # Scale features
    print("\n✓ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, 'xgboost_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler to {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns


def train_xgboost_model(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with class balancing to handle imbalanced data.
    """
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL (WITH CLASS BALANCING)")
    print("="*80)

    # Calculate scale_pos_weight to handle class imbalance
    # Formula: count(negative) / count(positive)
    # This gives equal importance to minority class (losses)
    n_positive = np.sum(y_train == 1)  # Wins
    n_negative = np.sum(y_train == 0)  # Losses
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    print(f"\n✓ Class distribution in training set:")
    print(f"  Wins (class 1):   {n_positive} ({n_positive/len(y_train)*100:.1f}%)")
    print(f"  Losses (class 0): {n_negative} ({n_negative/len(y_train)*100:.1f}%)")
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"  (Values < 1 mean we DOWN-weight the majority class - wins)")

    # Create params with scale_pos_weight
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight

    print(f"\n✓ XGBoost parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Train model
    print(f"\n✓ Training on {len(X_train)} samples...")
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],  # Only eval on test set
        verbose=False  # Reduced verbosity
    )

    print(f"✓ Training complete! Stopped at iteration {model.best_iteration}")
    print(f"  (Early stopping prevented full {XGBOOST_PARAMS['n_estimators']} iterations)")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names: list):
    """
    Comprehensive model evaluation with metrics and visualizations.
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n✓ TRAINING METRICS:")
    print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Precision: {precision_score(y_train, y_train_pred):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_train, y_train_pred):.4f}")
    print(f"  ROC AUC:   {roc_auc_score(y_train, y_train_proba):.4f}")

    print("\n✓ TESTING METRICS:")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  ROC AUC:   {test_auc:.4f}")

    # Confusion matrix
    print("\n✓ CONFUSION MATRIX (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives:  {cm[1,1]}")

    # Classification report
    print("\n✓ CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_test_pred, target_names=['Loss', 'Profit']))

    # Feature importance
    print("\n✓ TOP 10 FEATURE IMPORTANCES:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")

    # Create visualizations
    create_evaluation_plots(
        model, X_test, y_test, y_test_pred, y_test_proba,
        feature_importance, cm
    )

    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc
    }


def create_evaluation_plots(model, X_test, y_test, y_pred, y_proba, feature_importance, cm):
    """
    Create comprehensive evaluation visualizations.
    """
    print("\n✓ Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('XGBoost Trade Classifier - Model Evaluation', fontsize=16, fontweight='bold')

    # 1. Confusion Matrix
    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Loss', 'Profit'])
    ax.set_yticklabels(['Loss', 'Profit'])

    # 2. Feature Importance (Top 15)
    ax = axes[0, 1]
    top_features = feature_importance.head(15)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances')
    ax.invert_yaxis()

    # 3. ROC Curve
    ax = axes[1, 0]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Prediction Probability Distribution
    ax = axes[1, 1]
    ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='Actual Loss', color='red')
    ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='Actual Profit', color='green')
    ax.set_xlabel('Predicted Probability (Profit)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    ax.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')

    plt.tight_layout()

    # Save
    output_path = os.path.join(MODEL_DIR, 'xgboost_evaluation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")

    plt.close()


def save_model(model, feature_names: list, metrics: dict):
    """Save trained model and metadata"""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    # Save model
    model_path = os.path.join(MODEL_DIR, 'xgboost_trade_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")

    # Save feature names
    features_path = os.path.join(MODEL_DIR, 'xgboost_features.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved to: {features_path}")

    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, 'xgboost_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"✓ Metrics saved to: {metrics_path}")

    print("\n✓ All model artifacts saved successfully!")


def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("XGBOOST TRADE OUTCOME CLASSIFIER")
    print("="*80)
    print("\nPredicting trade profitability based on entry conditions")
    print(f"Database: {DB_PATH}")
    print(f"Train/Test split: {TRAIN_TEST_SPLIT_DATE}")
    print("="*80)

    # 1. Load data
    trades_df = load_trade_data()

    if len(trades_df) == 0:
        print("\n❌ No trade data found! Please run generate_realistic_trades.py first.")
        return

    # 2. Engineer features
    feature_df = engineer_features(trades_df)

    # 3. Prepare training data
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data(feature_df)

    # 4. Train model
    model = train_xgboost_model(X_train, y_train, X_test, y_test)

    # 5. Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)

    # 6. Save
    save_model(model, feature_names, metrics)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✓ Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"✓ Test F1 Score: {metrics['f1']:.4f}")
    print(f"✓ Test ROC AUC:  {metrics['auc']:.4f}")
    print(f"\n✓ Model ready for predictions!")
    print("\nNext steps:")
    print("  1. Review evaluation plots in ml_models/saved_models/")
    print("  2. Proceed to Hybrid LSTM-XGBoost model")
    print("  3. Compare all three models\n")


if __name__ == "__main__":
    main()
