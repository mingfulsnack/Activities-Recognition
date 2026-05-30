"""
Model Comparison - 13-Feature Stress Prediction
=================================================
Compare 5 architectures on same data pipeline:
1. MLP (Dense only) - simplest baseline
2. Simple LSTM (1 layer, unidirectional)
3. Stacked Bi-LSTM (original baseline)
4. Bi-GRU (GRU alternative)
5. Stacked Bi-LSTM Tuned (best from HP tuning)

"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import json
import time
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Bidirectional, Dense, Dropout, Input, Flatten,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'optimized_health_data_13features.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'model_comparison')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

SEQ_LENGTH = 60
N_FEATURES = 13
EPOCHS = 80
BATCH_SIZE = 32
PATIENCE = 15
CATEGORICAL_FEATURES = ['Activity', 'Location']

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# Data Pipeline (identical for all models)
# ============================================================
# DA: MODEL_COMPARISON_PIPELINE
# Shared 13-feature split/encode/scale/sequence pipeline for all benchmark models.
def prepare_data():
    """Load, split, encode, normalize, create sequences."""
    print("[DATA] Loading and preparing data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} samples, {len(df.columns)} columns")
    
    X = df.drop('Stress_Level', axis=1)
    y = df['Stress_Level']
    
    # Split 70/15/15
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, X_val, X_test = X.iloc[:train_end].copy(), X.iloc[train_end:val_end].copy(), X.iloc[val_end:].copy()
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    
    # Encode categorical (fit train only)
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        enc = LabelEncoder()
        X_train[col] = enc.fit_transform(X_train[col].astype(str))
        X_val[col] = enc.transform(X_val[col].astype(str))
        X_test[col] = enc.transform(X_test[col].astype(str))
        encoders[col] = enc
    
    # Normalize (fit train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)
    X_test_s = scaler.transform(X_test.values)
    
    # Create sequences
    def make_seq(X, y, seq_len):
        Xs, ys = [], []
        yv = y.values if hasattr(y, 'values') else y
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(yv[i+seq_len])
        return np.array(Xs), np.array(ys)
    
    X_train_seq, y_train_seq = make_seq(X_train_s, y_train, SEQ_LENGTH)
    X_val_seq, y_val_seq = make_seq(X_val_s, y_val, SEQ_LENGTH)
    X_test_seq, y_test_seq = make_seq(X_test_s, y_test, SEQ_LENGTH)
    
    print(f"  Train: {X_train_seq.shape} | Val: {X_val_seq.shape} | Test: {X_test_seq.shape}")
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq


# ============================================================
# Model Definitions
# ============================================================

# DA: BUILD_MLP
# Dense-only baseline for comparison with recurrent models.
def build_mlp():
    """Model 1: MLP (Dense only) - simplest baseline."""
    model = Sequential([
        Input(shape=(SEQ_LENGTH, N_FEATURES)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# DA: BUILD_SIMPLE_LSTM
# Single-layer unidirectional LSTM baseline.
def build_simple_lstm():
    """Model 2: Simple LSTM (1 layer, unidirectional)."""
    model = Sequential([
        Input(shape=(SEQ_LENGTH, N_FEATURES)),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# DA: BUILD_STACKED_BILSTM
# Original stacked Bidirectional LSTM baseline architecture.
def build_stacked_bilstm_baseline():
    """Model 3: Stacked Bi-LSTM (original baseline: 128->64, dropout=0.3)."""
    model = Sequential([
        Input(shape=(SEQ_LENGTH, N_FEATURES)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# DA: BUILD_BIGRU
# Bidirectional GRU alternative benchmark.
def build_bigru():
    """Model 4: Stacked Bi-GRU (GRU alternative)."""
    model = Sequential([
        Input(shape=(SEQ_LENGTH, N_FEATURES)),
        Bidirectional(GRU(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(GRU(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# DA: BUILD_TUNED_BILSTM
# Tuned stacked Bi-LSTM architecture used in final model comparison.
def build_stacked_bilstm_tuned():
    """Model 5: Stacked Bi-LSTM Tuned (best HP: 64->64, dropout=0.1, dense=128, lr=0.01)."""
    model = Sequential([
        Input(shape=(SEQ_LENGTH, N_FEATURES)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(LSTM(64)),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
    return model


# ============================================================
# Training & Evaluation
# ============================================================

# DA: MODEL_COMPARISON_TRAIN_EVAL
# Trains one benchmark model on the shared pipeline and returns MAE/RMSE/R2.
def train_and_evaluate(model, name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train model and return metrics + history."""
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Parameters: {model.count_params():,}")
    print(f"{'='*60}")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
    ]
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0  # Silent training - show summary after
    )
    
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    epochs_trained = len(history.history['loss'])
    best_val_mae = min(history.history['val_mae'])
    
    print(f"  Epochs: {epochs_trained}/{EPOCHS}")
    print(f"  Time: {train_time:.1f}s ({train_time/epochs_trained:.1f}s/epoch)")
    print(f"  Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
    
    result = {
        'name': name,
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'params': int(model.count_params()),
        'epochs_trained': epochs_trained,
        'train_time_sec': float(train_time),
        'best_val_mae': float(best_val_mae),
        'history': {
            'loss': [float(v) for v in history.history['loss']],
            'val_loss': [float(v) for v in history.history['val_loss']],
            'mae': [float(v) for v in history.history['mae']],
            'val_mae': [float(v) for v in history.history['val_mae']],
        }
    }
    
    return result


# ============================================================
# Visualization
# ============================================================

def create_comparison_plots(results, results_dir):
    """Create comprehensive comparison visualizations."""
    print("\n[PLOTS] Creating comparison visualizations...")
    
    names = [r['name'] for r in results]
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    r2s = [r['r2'] for r in results]
    params = [r['params'] for r in results]
    times = [r['train_time_sec'] for r in results]
    
    short_names = ['MLP', 'LSTM', 'Bi-LSTM', 'Bi-GRU', 'Bi-LSTM\n(Tuned)']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Model Comparison - 13-Feature Stress Prediction', fontsize=16, fontweight='bold')
    
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#e67e22', '#e74c3c']
    
    # --- 1. MAE Comparison ---
    ax = axes[0, 0]
    bars = ax.bar(short_names, maes, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('MAE (lower is better)')
    ax.set_title('Mean Absolute Error', fontweight='bold')
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    best_idx = np.argmin(maes)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # --- 2. RMSE Comparison ---
    ax = axes[0, 1]
    bars = ax.bar(short_names, rmses, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Root Mean Squared Error', fontweight='bold')
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    best_idx = np.argmin(rmses)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # --- 3. R2 Comparison ---
    ax = axes[0, 2]
    bars = ax.bar(short_names, r2s, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('R2 Score (higher is better)')
    ax.set_title('R-Squared', fontweight='bold')
    ax.set_ylim(min(r2s) - 0.05, 1.0)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    best_idx = np.argmax(r2s)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    # --- 4. Training Time ---
    ax = axes[1, 0]
    bars = ax.bar(short_names, times, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time', fontweight='bold')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=9)
    
    # --- 5. Parameters ---
    ax = axes[1, 1]
    bars = ax.bar(short_names, [p/1000 for p in params], color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Parameters (thousands)')
    ax.set_title('Model Complexity', fontweight='bold')
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/1000 + 1,
                f'{val/1000:.1f}K', ha='center', va='bottom', fontsize=9)
    
    # --- 6. Learning Curves (val_mae) ---
    ax = axes[1, 2]
    for i, r in enumerate(results):
        ax.plot(r['history']['val_mae'], color=colors[i], label=short_names[i].replace('\n', ' '),
                linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE')
    ax.set_title('Learning Curves', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: model_comparison.png")
    
    # --- Radar Chart ---
    create_radar_chart(results, results_dir, short_names, colors)
    
    # --- Summary Table Plot ---
    create_summary_table_plot(results, results_dir, short_names, colors)


def create_radar_chart(results, results_dir, short_names, colors):
    """Create radar chart comparing models on multiple metrics."""
    categories = ['MAE\n(inv)', 'RMSE\n(inv)', 'R2', 'Speed\n(inv)', 'Efficiency\n(R2/params)']
    N = len(categories)
    
    # Normalize metrics (0-1 scale, higher is better)
    maes = [r['mae'] for r in results]
    rmses = [r['rmse'] for r in results]
    r2s = [r['r2'] for r in results]
    times = [r['train_time_sec'] for r in results]
    params = [r['params'] for r in results]
    
    def normalize_inv(vals):
        """Inverse normalize (lower is better -> higher score)."""
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [1.0] * len(vals)
        return [(mx - v) / (mx - mn) for v in vals]
    
    def normalize(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [1.0] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]
    
    norm_mae = normalize_inv(maes)
    norm_rmse = normalize_inv(rmses)
    norm_r2 = normalize(r2s)
    norm_speed = normalize_inv(times)
    efficiency = [r2s[i] / (params[i] / 100000) for i in range(len(results))]
    norm_eff = normalize(efficiency)
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = [n * 2 * np.pi / N for n in range(N)]
    angles += angles[:1]
    
    for i, r in enumerate(results):
        values = [norm_mae[i], norm_rmse[i], norm_r2[i], norm_speed[i], norm_eff[i]]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i],
                label=short_names[i].replace('\n', ' '), markersize=5)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Comparison - Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: model_comparison_radar.png")


def create_summary_table_plot(results, results_dir, short_names, colors):
    """Create a visual summary table."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    
    headers = ['Model', 'MAE', 'RMSE', 'R2', 'Params', 'Time (s)', 'Epochs']
    rows = []
    for r in results:
        rows.append([
            r['name'],
            f"{r['mae']:.4f}",
            f"{r['rmse']:.4f}",
            f"{r['r2']:.4f}",
            f"{r['params']:,}",
            f"{r['train_time_sec']:.0f}",
            str(r['epochs_trained'])
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#ecf0f1']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Highlight best model row
    best_idx = np.argmin([r['mae'] for r in results])
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor('#d5f5e3')
    
    ax.set_title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] Saved: model_comparison_table.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  MODEL COMPARISON - 13-FEATURE STRESS PREDICTION")
    print("  5 Architectures on Same Data Pipeline")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Prepare data (shared)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    
    # Define models
    models = [
        ('1. MLP (Dense)', build_mlp),
        ('2. Simple LSTM', build_simple_lstm),
        ('3. Stacked Bi-LSTM (Baseline)', build_stacked_bilstm_baseline),
        ('4. Stacked Bi-GRU', build_bigru),
        ('5. Stacked Bi-LSTM (Tuned)', build_stacked_bilstm_tuned),
    ]
    
    # Train and evaluate all models
    all_results = []
    total_start = time.time()
    
    for name, build_fn in models:
        model = build_fn()
        result = train_and_evaluate(model, name, X_train, y_train, X_val, y_val, X_test, y_test)
        all_results.append(result)
        
        # Clear session to free memory
        keras.backend.clear_session()
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    
    fmt = "  {:<35} {:>8} {:>8} {:>8} {:>10} {:>8}"
    print(fmt.format('Model', 'MAE', 'RMSE', 'R2', 'Params', 'Time'))
    print(f"  {'-'*80}")
    
    for r in all_results:
        print(fmt.format(
            r['name'], f"{r['mae']:.4f}", f"{r['rmse']:.4f}",
            f"{r['r2']:.4f}", f"{r['params']:,}", f"{r['train_time_sec']:.0f}s"
        ))
    
    # Find best
    best = min(all_results, key=lambda x: x['mae'])
    print(f"\n  [BEST] {best['name']} - MAE: {best['mae']:.4f}, R2: {best['r2']:.4f}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    
    # Create visualizations
    create_comparison_plots(all_results, RESULTS_DIR)
    
    # Save results JSON
    save_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_config': {
            'dataset': 'optimized_health_data_13features.csv',
            'samples': 54448,
            'features': 13,
            'seq_length': SEQ_LENGTH,
            'split': '70/15/15',
            'pipeline': 'Split -> Encode(fit train) -> Normalize -> Sequences'
        },
        'training_config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'early_stopping_patience': PATIENCE,
            'optimizer': 'Adam',
            'loss': 'MSE'
        },
        'results': [{
            'name': r['name'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'r2': r['r2'],
            'params': r['params'],
            'epochs_trained': r['epochs_trained'],
            'train_time_sec': r['train_time_sec'],
            'best_val_mae': r['best_val_mae']
        } for r in all_results],
        'best_model': best['name'],
        'total_time_sec': total_time
    }
    
    with open(os.path.join(RESULTS_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\n  [OK] Saved: comparison_results.json")
    
    # Save training histories
    histories = {r['name']: r['history'] for r in all_results}
    with open(os.path.join(RESULTS_DIR, 'training_histories.json'), 'w') as f:
        json.dump(histories, f, indent=2)
    print("  [OK] Saved: training_histories.json")
    
    # Save summary CSV
    summary_df = pd.DataFrame([{
        'Model': r['name'],
        'MAE': r['mae'],
        'RMSE': r['rmse'],
        'R2': r['r2'],
        'Parameters': r['params'],
        'Training_Time_sec': r['train_time_sec'],
        'Epochs_Trained': r['epochs_trained'],
        'Best_Val_MAE': r['best_val_mae']
    } for r in all_results])
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_summary.csv'), index=False)
    print("  [OK] Saved: comparison_summary.csv")
    
    print("\n" + "=" * 70)
    print("  [DONE] MODEL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
