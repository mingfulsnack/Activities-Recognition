"""
Train LSTM Model with 13 Simplified Features
=============================================
Purpose: Train stress prediction model with core + high-importance features (no complex engineering)

Feature Set (13):
- Core (7): Hour, Day_of_Week, Activity, Acc_X/Y/Z, Heart_Rate
- High-Importance (6): Location, Screen_Usage, Phone_Events, Mood, Energy, Sleep_Duration

Advantages over 17-feature version:
- No rolling windows → No data leakage
- Simpler features → More robust training
- Evidence-based core + ML-selected importance

PIPELINE FIX (Feb 13, 2026):
   Encoding now happens AFTER train/test split
   Correct order: Load → Split RAW → Encode (fit train, transform val/test) → Normalize → Sequences
   Previous bug: Load → Encode ALL → Split (leaked test info into train encoders)

Author: [Your Name]
Date: February 2026
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handle data loading, encoding, and normalization - FIXED PIPELINE."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_features = ['Activity', 'Location']
        
    def load_data(self):
        """Load 13-feature dataset."""
        print(f"📂 Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df):,} samples with {len(self.df.columns)} columns")
        print(f"\nColumns: {list(self.df.columns)}")
        return self
        
    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """Split RAW data into train/val/test sets (BEFORE encoding)."""
        print(f"\n Splitting RAW data (before encoding)...")
        
        # Separate features and target
        X = self.df.drop('Stress_Level', axis=1)
        y = self.df['Stress_Level']
        
        # Train + (Val+Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, shuffle=False
        )
        
        # Val + Test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, shuffle=False
        )
        
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def encode_categorical_features(self, X_train, X_val, X_test):
        """Encode categorical features AFTER split (fit on train, transform on val/test)."""
        print("\n Encoding categorical features (AFTER split)...")
        
        # Make copies to avoid modifying originals
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()
        
        for col in self.categorical_features:
            if col in X_train.columns:
                print(f"\n  Encoding: {col}")
                
                encoder = LabelEncoder()
                
                # FIT on train set only
                X_train[col] = encoder.fit_transform(X_train[col].astype(str))
                
                # TRANSFORM val and test sets
                X_val[col] = encoder.transform(X_val[col].astype(str))
                X_test[col] = encoder.transform(X_test[col].astype(str))
                
                self.label_encoders[col] = encoder
                
                # Show mapping
                mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                print(f"    {col} mapping: {mapping}")
        
        print(f"\n Encoded {len(self.categorical_features)} categorical features")
        print("✓ NO DATA LEAKAGE: Fitted on train only, transformed val/test")
        
        return X_train, X_val, X_test
        
    def normalize_features(self, X_train, X_val, X_test):
        """Normalize features using StandardScaler (fit on train only)."""
        print("\n Normalizing features...")
        
        # Convert DataFrames to numpy arrays if needed
        if hasattr(X_train, 'values'):
            X_train = X_train.values
            X_val = X_val.values
            X_test = X_test.values
        
        # FIT on train, TRANSFORM on val/test
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Normalized {X_train_scaled.shape[1]} features")
        print(f"  Mean: {self.scaler.mean_[:3]} ...")
        print(f"  Std:  {self.scaler.scale_[:3]} ...")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    def create_sequences(self, X, y, seq_length=60):
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        
        # Convert to numpy if pandas Series
        if hasattr(y, 'values'):
            y = y.values
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
        
    def save_preprocessor(self, save_dir='models'):
        """Save scaler and label encoders."""
        os.makedirs(save_dir, exist_ok=True)
        
        scaler_path = os.path.join(save_dir, 'scaler_13features.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n Saved scaler to: {scaler_path}")
        
        for col, encoder in self.label_encoders.items():
            encoder_path = os.path.join(save_dir, f'label_encoder_13features_{col}.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder, f)
            print(f" Saved {col} encoder to: {encoder_path}")


class LSTMModel:
    """Build and train LSTM model."""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[128, 64], dropout=0.3):
        """Build stacked Bi-LSTM model (same architecture as baseline for fair comparison)."""
        print("\n  Building model...")
        
        self.model = Sequential([
            Input(shape=self.input_shape),
            
            # First Bi-LSTM layer
            Bidirectional(LSTM(lstm_units[0], return_sequences=True)),
            Dropout(dropout),
            
            # Second Bi-LSTM layer
            Bidirectional(LSTM(lstm_units[1])),
            Dropout(dropout),
            
            # Dense output
            Dense(64, activation='relu'),
            Dropout(dropout),
            Dense(32, activation='relu'),
            Dense(1)  # Regression output
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print(self.model.summary())
        
        # Count params
        total_params = self.model.count_params()
        print(f"\n Model built:")
        print(f"  Architecture: Stacked Bi-LSTM ({lstm_units[0]} → {lstm_units[1]})")
        print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Model size: ~{total_params * 4 / 1024**2:.2f} MB")
        
        return self
        
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, patience=10):
        """Train model with early stopping."""
        print("\n Starting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Patience: {patience}")
        print("")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/lstm_13features_best.keras',
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
        
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
        
        return self
        
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        print("\n Evaluating on test set...")
        
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Test Performance:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Validation verdict
        if r2 >= 0.93 and mae <= 0.55:
            print(f"\n VALIDATION SUCCESS: 17 features achieve excellent performance!")
            print(f"   Expected improvements in temporal/activity context modeling")
        elif r2 >= 0.90:
            print(f"\n  GOOD PERFORMANCE: R²={r2:.4f}, but room for improvement")
        else:
            print(f"\n PERFORMANCE CONCERN: R²={r2:.4f} below expectations")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actuals': y_test
        }
        
    def save_metrics(self, metrics, save_dir='results'):
        """Save metrics to file."""
        os.makedirs(save_dir, exist_ok=True)
        
        metrics_path = os.path.join(save_dir, 'metrics_13features.txt')
        with open(metrics_path, 'w') as f:
            f.write("13-FEATURE LSTM MODEL - TEST METRICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"MAE:  {metrics['mae']:.4f}\n")
            f.write(f"RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"R²:   {metrics['r2']:.4f}\n")
            f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\n Metrics saved to: {metrics_path}")
        
        return self


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("TRAINING LSTM WITH 13 SIMPLIFIED FEATURES")
    print("=" * 80)
    print("")
    
    # Configuration
    data_path = 'data/optimized_health_data_13features.csv'
    seq_length = 60  # 60 minutes window
    
    # Check data
    if not os.path.exists(data_path):
        print(f" Error: Dataset not found: {data_path}")
        print(f"   Run feature_engineering.py first!")
        return
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Pipeline (FIXED: Split → Encode → Normalize)
    try:
        # 1. Load data
        preprocessor = DataPreprocessor(data_path)
        preprocessor.load_data()
        
        # 2. Split RAW data FIRST (no encoding yet)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
        
        # 3. Encode categorical features AFTER split (fit on train, transform val/test)
        X_train, X_val, X_test = preprocessor.encode_categorical_features(
            X_train, X_val, X_test
        )
        
        # 4. Normalize features (fit on train, transform val/test)
        X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
            X_train, X_val, X_test
        )
        
        # 5. Create sequences
        print(f"\n Creating sequences (length={seq_length})...")
        X_train_seq, y_train_seq = preprocessor.create_sequences(
            X_train_scaled, y_train, seq_length
        )
        X_val_seq, y_val_seq = preprocessor.create_sequences(
            X_val_scaled, y_val, seq_length
        )
        X_test_seq, y_test_seq = preprocessor.create_sequences(
            X_test_scaled, y_test, seq_length
        )
        
        print(f"✓ Sequences created:")
        print(f"  Train: {X_train_seq.shape} → {y_train_seq.shape}")
        print(f"  Val:   {X_val_seq.shape} → {y_val_seq.shape}")
        print(f"  Test:  {X_test_seq.shape} → {y_test_seq.shape}")
        
        # 6. Build and train model
        n_features = X_train_seq.shape[2]  # Get feature count from sequences
        print(f"\n Number of features: {n_features}")
        
        model = LSTMModel(input_shape=(seq_length, n_features))
        model.build_model(lstm_units=[128, 64], dropout=0.3)
        model.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=50,
            batch_size=32,
            patience=10
        )
        
        # 7. Evaluate
        metrics = model.evaluate(X_test_seq, y_test_seq)
        
        # 8. Save
        preprocessor.save_preprocessor()
        model.save_metrics(metrics)
        
        print("\n" + "=" * 80)
        print("✓ TRAINING PIPELINE COMPLETED - NO DATA LEAKAGE!")
        print("=" * 80)
        print(f"\nModel saved to: models/lstm_13features_best.keras")
        print(f"Metrics saved to: results/metrics_13features.txt")
        print(f"\nTest Performance:")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"\nNext step: python stress_prediction/comparison_analysis_13.py")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
