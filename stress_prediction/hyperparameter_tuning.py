"""
Hyperparameter Tuning - 13-Feature LSTM Model
================================================
Bayesian Optimization using Keras Tuner

Hyperparameters tuned:
- LSTM Layer 1 units: [64, 128, 256]
- LSTM Layer 2 units: [32, 64, 128]
- Dropout rate: 0.1 - 0.5
- Dense units: [32, 64, 128]
- Learning rate: 1e-4 → 1e-2
- Batch size: [16, 32, 64]

Pipeline: Split → Encode (fit train) → Normalize → Sequences (no data leakage)

Author: [Your Name]
Date: February 19, 2026
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import keras_tuner as kt

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Data Preprocessing (reuse from train_lstm_13features.py)
# ============================================================

class DataPreprocessor:
    """Handle data loading, encoding, and normalization - FIXED PIPELINE."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_features = ['Activity', 'Location']
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f" Loaded {len(self.df):,} samples, {len(self.df.columns)} columns")
        return self
        
    def split_data(self):
        """Split RAW data: 70/15/15 (BEFORE encoding)."""
        X = self.df.drop('Stress_Level', axis=1)
        y = self.df['Stress_Level']
        
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def encode_categorical_features(self, X_train, X_val, X_test):
        """Encode AFTER split (fit train only)."""
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        
        for col in self.categorical_features:
            if col in X_train.columns:
                encoder = LabelEncoder()
                X_train[col] = encoder.fit_transform(X_train[col].astype(str))
                X_val[col] = encoder.transform(X_val[col].astype(str))
                X_test[col] = encoder.transform(X_test[col].astype(str))
                self.label_encoders[col] = encoder
        
        return X_train, X_val, X_test
        
    def normalize_features(self, X_train, X_val, X_test):
        """Normalize (fit train only)."""
        if hasattr(X_train, 'values'):
            X_train = X_train.values
            X_val = X_val.values
            X_test = X_test.values
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    def create_sequences(self, X, y, seq_length=60):
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        if hasattr(y, 'values'):
            y = y.values
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)


# ============================================================
# Hyperparameter Tuning
# ============================================================

def build_model(hp):
    """Build model with tunable hyperparameters."""
    
    # Tunable hyperparameters
    lstm_units_1 = hp.Choice('lstm_units_1', values=[64, 128, 256])
    lstm_units_2 = hp.Choice('lstm_units_2', values=[32, 64, 128])
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    dense_units = hp.Choice('dense_units', values=[32, 64, 128])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    # Build model
    model = Sequential([
        Input(shape=(60, 13)),  # (seq_length, n_features)
        
        # Layer 1: Bi-LSTM
        Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
        Dropout(dropout_rate),
        
        # Layer 2: Bi-LSTM
        Bidirectional(LSTM(lstm_units_2)),
        Dropout(dropout_rate),
        
        # Dense layers
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(dense_units // 2, activation='relu'),
        Dense(1)  # Regression output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def prepare_data(seq_length=60):
    """Prepare data with correct pipeline."""
    print("\n Preparing data...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'optimized_health_data_13features.csv')
    
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    
    # Split → Encode → Normalize → Sequences
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    X_train, X_val, X_test = preprocessor.encode_categorical_features(X_train, X_val, X_test)
    X_train_s, X_val_s, X_test_s = preprocessor.normalize_features(X_train, X_val, X_test)
    
    X_train_seq, y_train_seq = preprocessor.create_sequences(X_train_s, y_train, seq_length)
    X_val_seq, y_val_seq = preprocessor.create_sequences(X_val_s, y_val, seq_length)
    X_test_seq, y_test_seq = preprocessor.create_sequences(X_test_s, y_test, seq_length)
    
    print(f"  Train: {X_train_seq.shape}")
    print(f"  Val:   {X_val_seq.shape}")
    print(f"  Test:  {X_test_seq.shape}")
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, preprocessor


def run_tuning(X_train, y_train, X_val, y_val, max_trials=20, epochs_per_trial=30):
    """Run Bayesian Optimization."""
    
    print("\n" + "=" * 70)
    print("  BAYESIAN OPTIMIZATION - HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # Setup tuner
    tuner_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'hp_tuning'
    )
    
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_mae',
        max_trials=max_trials,
        num_initial_points=5,  # Random exploration before Bayesian
        directory=tuner_dir,
        project_name='lstm_13features',
        overwrite=True
    )
    
    print(f"\n Search space summary:")
    tuner.search_space_summary()
    
    print(f"\n Starting search ({max_trials} trials, {epochs_per_trial} epochs each)...")
    print(f"   Estimated time: ~{max_trials * 3:.0f} minutes\n")
    
    # Callbacks for each trial
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    start_time = time.time()
    
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_per_trial,
        batch_size=32,  # Will also tune via separate runs 
        callbacks=callbacks,
        verbose=1
    )
    
    total_time = time.time() - start_time
    
    print(f"\n Tuning completed in {total_time/60:.1f} minutes")
    
    return tuner


def evaluate_best_model(tuner, X_test, y_test, X_train, y_train, X_val, y_val, preprocessor):
    """Evaluate best model from tuning."""
    
    print("\n" + "=" * 70)
    print("  BEST MODEL EVALUATION")
    print("=" * 70)
    
    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n Best Hyperparameters:")
    best_params = {}
    for param in ['lstm_units_1', 'lstm_units_2', 'dropout_rate', 'dense_units', 'learning_rate']:
        val = best_hp.get(param)
        best_params[param] = val
        print(f"  {param}: {val}")
    
    # Retrain best model with more epochs
    print("\n Retraining best model with full epochs...")
    
    best_model = tuner.hypermodel.build(best_hp)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models', 'lstm_13features_tuned.keras'
            ),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    y_pred = best_model.predict(X_test, verbose=0).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'best_params': best_params,
        'predictions': y_pred.tolist(),
        'actuals': y_test.tolist()
    }
    
    print(f"\n TUNED MODEL - Test Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Compare with baseline
    print(f"\n Comparison with Baseline:")
    print(f"  {'Metric':<8} {'Baseline':>10} {'Tuned':>10} {'Change':>10}")
    print(f"  {'-'*40}")
    
    baseline = {'mae': 0.6757, 'rmse': 0.8571, 'r2': 0.9271}
    
    for metric in ['mae', 'rmse', 'r2']:
        base_val = baseline[metric]
        tuned_val = results[metric]
        change = tuned_val - base_val
        pct = (change / base_val) * 100
        
        if metric == 'r2':
            indicator = '✅' if change > 0 else '❌'
        else:
            indicator = '✅' if change < 0 else '❌'
        
        print(f"  {metric.upper():<8} {base_val:>10.4f} {tuned_val:>10.4f} {change:>+10.4f} ({pct:>+.2f}%) {indicator}")
    
    return results, best_model, history


def save_results(results, history):
    """Save tuning results."""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'hp_tuning')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(results_dir, 'tuning_results.json')
    save_data = {
        'best_params': results['best_params'],
        'test_metrics': {
            'mae': results['mae'],
            'rmse': results['rmse'],
            'r2': results['r2']
        },
        'baseline_metrics': {
            'mae': 0.6757,
            'rmse': 0.8571,
            'r2': 0.9271
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n Results saved: {metrics_path}")
    
    # Save training history
    history_path = os.path.join(results_dir, 'tuned_training_history.json')
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f" History saved: {history_path}")
    
    # Save metrics text file
    metrics_txt_path = os.path.join(base_dir, 'results', 'metrics_13features_tuned.txt')
    with open(metrics_txt_path, 'w') as f:
        f.write("13-FEATURE LSTM MODEL (TUNED) - TEST METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"MAE:  {results['mae']:.4f}\n")
        f.write(f"RMSE: {results['rmse']:.4f}\n")
        f.write(f"R²:   {results['r2']:.4f}\n")
        f.write(f"\nBest Hyperparameters:\n")
        for k, v in results['best_params'].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f" Metrics saved: {metrics_txt_path}")
    
    # Save preprocessor artifacts
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    return results_dir


def main():
    """Main hyperparameter tuning pipeline."""
    
    print("=" * 70)
    print("  HYPERPARAMETER TUNING - 13-FEATURE LSTM MODEL")
    print("  Method: Bayesian Optimization (Keras Tuner)")
    print("=" * 70)
    
    # Configuration
    MAX_TRIALS = 20       # Number of HP combinations to try
    EPOCHS_PER_TRIAL = 30 # Epochs per trial (early stopping still applies)
    SEQ_LENGTH = 60
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # 1. Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = prepare_data(SEQ_LENGTH)
        
        # 2. Run tuning
        tuner = run_tuning(X_train, y_train, X_val, y_val, MAX_TRIALS, EPOCHS_PER_TRIAL)
        
        # 3. Get top 3 results
        print("\n Top 3 Trials:")
        top_hps = tuner.get_best_hyperparameters(num_trials=3)
        for i, hp in enumerate(top_hps):
            print(f"\n  #{i+1}:")
            for param in ['lstm_units_1', 'lstm_units_2', 'dropout_rate', 'dense_units', 'learning_rate']:
                print(f"    {param}: {hp.get(param)}")
        
        # 4. Evaluate best model (retrain with more epochs)
        results, best_model, history = evaluate_best_model(
            tuner, X_test, y_test, X_train, y_train, X_val, y_val, preprocessor
        )
        
        # 5. Save everything
        save_results(results, history)
        
        # Save preprocessor
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'models')
        
        with open(os.path.join(model_dir, 'scaler_13features_tuned.pkl'), 'wb') as f:
            pickle.dump(preprocessor.scaler, f)
        for col, enc in preprocessor.label_encoders.items():
            with open(os.path.join(model_dir, f'label_encoder_13features_tuned_{col}.pkl'), 'wb') as f:
                pickle.dump(enc, f)
        
        print("\n" + "=" * 70)
        print("   HYPERPARAMETER TUNING COMPLETE!")
        print("=" * 70)
        print(f"\n  Best params: {results['best_params']}")
        print(f"  R²:   {results['r2']:.4f} (baseline: 0.9271)")
        print(f"  MAE:  {results['mae']:.4f} (baseline: 0.6757)")
        print(f"  RMSE: {results['rmse']:.4f} (baseline: 0.8571)")
        print(f"\n  Model: models/lstm_13features_tuned.keras")
        print(f"  Results: results/hp_tuning/")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
