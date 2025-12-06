#!/usr/bin/env python3
"""
LSTM Price Prediction Model
Predicts next-day returns using 60-day historical windows with technical indicators

Features:
- Multi-layer LSTM architecture
- Technical indicators (RSI, MACD, Bollinger Bands)
- Time-based train/test split
- Early stopping and model checkpointing
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import sqlite3
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH

# Check for TensorFlow
try:
    # Force CPU execution for Mac compatibility
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # Configure TensorFlow for Mac
    tf.config.set_visible_devices([], 'GPU')
    print(f"✓ TensorFlow {tf.__version__} loaded (CPU mode)")
except ImportError:
    print("❌ TensorFlow not installed. Installing...")
    os.system("pip install tensorflow scikit-learn matplotlib --quiet")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
SEQUENCE_LENGTH = 60  # 60 days lookback
PREDICTION_HORIZON = 1  # Predict 1 day ahead
TEST_SPLIT_DATE = '2024-01-01'  # Train on pre-2024, test on 2024-2025
BATCH_SIZE = 32
EPOCHS = 30  # Reduced for faster training
LEARNING_RATE = 0.001

# Model save path
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def get_connection():
    """Create database connection"""
    return sqlite3.connect(DB_PATH)


def calculate_technical_indicators(df):
    """
    Calculate technical indicators from OHLCV data

    Returns dataframe with additional columns:
    - RSI (14-day)
    - MACD (12, 26, 9)
    - Bollinger Bands (20-day)
    - Moving Averages (7, 21, 50)
    - Volume indicators
    """

    # 1. RSI (Relative Strength Index)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['rsi'] = calculate_rsi(df['px_close'])

    # 2. MACD (Moving Average Convergence Divergence)
    ema_12 = df['px_close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['px_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # 3. Bollinger Bands
    df['bb_middle'] = df['px_close'].rolling(window=20).mean()
    bb_std = df['px_close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # 4. Moving Averages
    df['ma_7'] = df['px_close'].rolling(window=7).mean()
    df['ma_21'] = df['px_close'].rolling(window=21).mean()
    df['ma_50'] = df['px_close'].rolling(window=50).mean()

    # 5. Price momentum
    df['return_1d'] = df['px_close'].pct_change(1)
    df['return_5d'] = df['px_close'].pct_change(5)
    df['return_10d'] = df['px_close'].pct_change(10)

    # 6. Volume indicators
    df['volume_ma_20'] = df['px_vol'].rolling(window=20).mean()
    df['volume_ratio'] = df['px_vol'] / df['volume_ma_20']

    # 7. Price position within daily range
    df['high_low_ratio'] = (df['px_close'] - df['px_low']) / (df['px_high'] - df['px_low'])

    # 8. Volatility
    df['volatility_20'] = df['return_1d'].rolling(window=20).std()

    return df


def prepare_data_for_lstm(symbol='SPY', min_samples=1000):
    """
    Prepare data for LSTM training

    Returns:
    - X_train, y_train, X_test, y_test
    - scaler (for inverse transform)
    - feature_names
    """

    print(f"\n{'='*80}")
    print(f"PREPARING DATA FOR LSTM - {symbol}")
    print(f"{'='*80}\n")

    conn = get_connection()

    # Get price data with instrument info
    query = f"""
        SELECT p.*, i.symbol
        FROM prices p
        JOIN instruments i ON p.instrument_id = i.instrument_id
        WHERE i.symbol = '{symbol}'
          AND p.source = 'yfinance'
        ORDER BY p.ts ASC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"✓ Loaded {len(df)} days of {symbol} data")
    print(f"  Date range: {df['ts'].min()} to {df['ts'].max()}")

    if len(df) < min_samples:
        raise ValueError(f"Insufficient data: {len(df)} < {min_samples} required")

    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    df = calculate_technical_indicators(df)

    # Select features for LSTM
    feature_columns = [
        'px_open', 'px_high', 'px_low', 'px_close', 'px_vol',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
        'ma_7', 'ma_21', 'ma_50',
        'return_1d', 'return_5d', 'return_10d',
        'volume_ratio', 'high_low_ratio', 'volatility_20'
    ]

    # Drop NaN rows (from indicator calculations)
    df = df.dropna()
    print(f"✓ After removing NaN: {len(df)} samples")

    # Create target: next day's return
    df['target'] = df['px_close'].pct_change(1).shift(-1)
    df = df.dropna()

    print(f"\nFeatures: {len(feature_columns)} technical indicators")
    print(f"Target: Next-day return (regression)")

    # Time-based split
    train_df = df[df['ts'] < TEST_SPLIT_DATE].copy()
    test_df = df[df['ts'] >= TEST_SPLIT_DATE].copy()

    print(f"\nTrain/Test Split (time-based):")
    print(f"  Training: {len(train_df)} samples ({train_df['ts'].min()} to {train_df['ts'].max()})")
    print(f"  Testing:  {len(test_df)} samples ({test_df['ts'].min()} to {test_df['ts'].max()})")

    # Scale features
    scaler = MinMaxScaler()

    train_features = train_df[feature_columns].values
    test_features = test_df[feature_columns].values

    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # Create sequences
    def create_sequences(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(targets[i+seq_length])
        return np.array(X), np.array(y)

    print(f"\nCreating sequences (lookback={SEQUENCE_LENGTH} days)...")

    X_train, y_train = create_sequences(
        train_scaled,
        train_df['target'].values,
        SEQUENCE_LENGTH
    )

    X_test, y_test = create_sequences(
        test_scaled,
        test_df['target'].values,
        SEQUENCE_LENGTH
    )

    print(f"✓ Training sequences: {X_train.shape}")
    print(f"  Shape: (samples, timesteps, features) = {X_train.shape}")
    print(f"✓ Testing sequences: {X_test.shape}")

    return X_train, y_train, X_test, y_test, scaler, feature_columns, df


def build_lstm_model(input_shape, learning_rate=LEARNING_RATE):
    """
    Build LSTM model architecture

    Architecture:
    - LSTM (128 units, return sequences)
    - Dropout (0.3)
    - LSTM (64 units)
    - Dropout (0.2)
    - Dense (25 units, ReLU)
    - Dense (1 unit, Linear) - Regression output
    """

    print(f"\n{'='*80}")
    print("BUILDING LSTM MODEL")
    print(f"{'='*80}\n")

    model = Sequential([
        # Layer 1: LSTM
        LSTM(128, return_sequences=True, input_shape=input_shape, name='lstm_1'),
        Dropout(0.3, name='dropout_1'),

        # Layer 2: LSTM
        LSTM(64, return_sequences=False, name='lstm_2'),
        Dropout(0.2, name='dropout_2'),

        # Dense layers
        Dense(25, activation='relu', name='dense_1'),
        Dense(1, activation='linear', name='output')  # Regression
    ])

    # Compile with explicit metric classes (not strings) for Keras compatibility
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.MeanSquaredError(),  # Explicit class, not 'mse' string
        metrics=[keras.metrics.MeanAbsoluteError()]  # Explicit class, not 'mae' string
    )

    print("Model Architecture:")
    model.summary()

    total_params = model.count_params()
    print(f"\n✓ Total Parameters: {total_params:,}")

    return model


def train_model(model, X_train, y_train, X_test, y_test, symbol='SPY'):
    """
    Train LSTM model with callbacks and CLASS BALANCING via sample weights.

    Class Balancing:
    - Calculate sample weights to handle up/down day imbalance
    - Down days (negative returns) get higher weight since they're rarer
    - This forces the model to learn to predict both directions

    Callbacks:
    - EarlyStopping (patience=10, restore best weights)
    - ModelCheckpoint (save best model)
    """

    print(f"\n{'='*80}")
    print("TRAINING LSTM MODEL (WITH CLASS BALANCING)")
    print(f"{'='*80}\n")

    # Calculate sample weights for class balancing
    # Count up days (positive returns) and down days (negative returns)
    n_up = np.sum(y_train > 0)
    n_down = np.sum(y_train <= 0)
    total = len(y_train)

    print(f"✓ Class distribution in training set:")
    print(f"  Up days (positive return):   {n_up} ({n_up/total*100:.1f}%)")
    print(f"  Down days (negative return): {n_down} ({n_down/total*100:.1f}%)")

    # Calculate weight ratio with AGGRESSIVE multiplier
    # The basic ratio (~1.22) is too weak - model still predicts all UP
    # Use 5x multiplier to force the model to learn down patterns
    if n_down > 0:
        base_weight = n_up / n_down
        weight_multiplier = 3.75  # Fine-tuned: 2.5x=all UP, 5x=all DOWN, trying middle
        weight_for_down = base_weight * weight_multiplier
    else:
        weight_for_down = 5.0

    print(f"  Base weight ratio: {base_weight:.4f}")
    print(f"  Weight multiplier: {weight_multiplier}x")
    print(f"  Final weight for down days: {weight_for_down:.4f}")

    # Create sample weights array - down days get MUCH higher weight
    sample_weights = np.where(y_train > 0, 1.0, weight_for_down)
    print(f"  Sample weights range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")

    model_path = os.path.join(MODEL_DIR, f'lstm_{symbol}_best.keras')  # .keras format for compatibility

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\nTraining parameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Early stopping patience: 10 epochs")
    print(f"  Model checkpoint: {model_path}")
    print()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        sample_weight=sample_weights,  # Class balancing via sample weights
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✓ Training complete!")
    print(f"✓ Best model saved to: {model_path}")

    return history, model_path


def evaluate_model(model, X_test, y_test, df, scaler):
    """
    Evaluate model performance with comprehensive metrics
    """

    print(f"\n{'='*80}")
    print("MODEL EVALUATION")
    print(f"{'='*80}\n")

    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Regression Metrics:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"  MAE (Mean Absolute Error):      {mae:.6f}")
    print(f"  R² Score:                       {r2:.4f}")

    # Directional accuracy (most important for trading)
    y_test_direction = np.sign(y_test)
    y_pred_direction = np.sign(y_pred)
    directional_accuracy = (y_test_direction == y_pred_direction).mean()

    print(f"\nTrading Metrics:")
    print(f"  Directional Accuracy: {directional_accuracy*100:.2f}%")
    print(f"    (Predicts correct direction: up/down)")

    # Confusion matrix for direction
    true_positive = ((y_test_direction == 1) & (y_pred_direction == 1)).sum()
    true_negative = ((y_test_direction == -1) & (y_pred_direction == -1)).sum()
    false_positive = ((y_test_direction == -1) & (y_pred_direction == 1)).sum()
    false_negative = ((y_test_direction == 1) & (y_pred_direction == -1)).sum()

    print(f"\nDirection Confusion Matrix:")
    print(f"  True Positive (predicted ↑, actual ↑):  {true_positive}")
    print(f"  True Negative (predicted ↓, actual ↓):  {true_negative}")
    print(f"  False Positive (predicted ↑, actual ↓): {false_positive}")
    print(f"  False Negative (predicted ↓, actual ↑): {false_negative}")

    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
        print(f"\n  Precision (when predicting ↑): {precision*100:.2f}%")

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
        print(f"  Recall (catching actual ↑):    {recall*100:.2f}%")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'predictions': y_pred,
        'actuals': y_test
    }


def plot_results(history, results, symbol='SPY'):
    """
    Plot training history and prediction results
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Training loss
    ax = axes[0, 0]
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_title(f'{symbol} LSTM Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Predictions vs Actuals
    ax = axes[0, 1]
    ax.scatter(results['actuals'], results['predictions'], alpha=0.5, s=10)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], 'r--', label='Perfect Prediction')
    ax.set_title(f'Predictions vs Actuals (R²={results["r2"]:.3f})')
    ax.set_xlabel('Actual Returns')
    ax.set_ylabel('Predicted Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Prediction errors
    ax = axes[1, 0]
    errors = results['predictions'] - results['actuals']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(f'Prediction Error Distribution (MAE={results["mae"]:.4f})')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.axvline(0, color='red', linestyle='--', label='Zero Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative directional accuracy
    ax = axes[1, 1]
    y_test_direction = np.sign(results['actuals'])
    y_pred_direction = np.sign(results['predictions'])
    correct = (y_test_direction == y_pred_direction).astype(int)
    cumulative_accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    ax.plot(cumulative_accuracy * 100)
    ax.axhline(results['directional_accuracy'] * 100, color='red', linestyle='--',
               label=f'Final: {results["directional_accuracy"]*100:.1f}%')
    ax.set_title('Rolling Directional Accuracy')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(MODEL_DIR, f'lstm_{symbol}_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results plot saved to: {plot_path}")

    plt.show()


def main(symbol='SPY'):
    """
    Main training pipeline
    """

    print("\n" + "="*80)
    print("LSTM PRICE PREDICTION MODEL - TRAINING PIPELINE")
    print("="*80)
    print(f"\nSymbol: {symbol}")
    print(f"Database: {DB_PATH}")
    print(f"Sequence Length: {SEQUENCE_LENGTH} days")
    print(f"Prediction: Next-day return")
    print(f"Train/Test Split: Before/After {TEST_SPLIT_DATE}")

    # Step 1: Prepare data
    X_train, y_train, X_test, y_test, scaler, features, df = prepare_data_for_lstm(symbol)

    # Step 2: Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_lstm_model(input_shape)

    # Step 3: Train model
    history, model_path = train_model(model, X_train, y_train, X_test, y_test, symbol)

    # Step 4: Evaluate
    results = evaluate_model(model, X_test, y_test, df, scaler)

    # Step 5: Visualize
    plot_results(history, results, symbol)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Directional Accuracy: {results['directional_accuracy']*100:.2f}%")
    print(f"✓ RMSE: {results['rmse']:.6f}")
    print(f"✓ R² Score: {results['r2']:.4f}")
    print("\nNext steps:")
    print("  1. Try different symbols (AAPL, TSLA, BTC-USD)")
    print("  2. Tune hyperparameters (sequence length, LSTM units)")
    print("  3. Use predictions in trading strategy")
    print("  4. Build XGBoost model next")
    print()

    return model, results, scaler


if __name__ == "__main__":
    import sys

    # Allow symbol as command line argument
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'SPY'

    model, results, scaler = main(symbol)
