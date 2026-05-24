"""
LSTM Training Script for 10-Feature Model
==========================================
Purpose: Train Stacked Bidirectional LSTM with reduced feature set (10 features)
         and compare with baseline model (23 features -> 21 after encoding)

Architecture: Same as baseline
- 2-layer Stacked Bidirectional LSTM (128 + 64 units)
- Dropout 0.3
- Dense(32) + Output(1)

Expected: Performance should be similar to baseline (R² ~ 0.93) 
          since top 10 features cover 98% importance
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'data_file': 'data/optimized_health_data_10features.csv',
    'sequence_length': 60,  # Same as baseline
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.15,
    'test_split': 0.15,
    'model_dir': 'models/',
    'results_dir': 'results/feature_comparison/',
    'random_state': 42
}

# Create directories
os.makedirs(CONFIG['model_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)


class DataPreprocessor:
    """Handle data loading and preprocessing for 10-feature model."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = 'Stress_Level'
        self.categorical_features = ['Location']  # Location is categorical
        
    def load_data(self):
        """Load 10-feature dataset."""
        print(f" Loading dataset: {self.config['data_file']}")
        df = pd.read_csv(self.config['data_file'])
        print(f"✓ Loaded {len(df):,} samples with {len(df.columns)} columns")
        print(f"  Features: {list(df.columns)}")
        return df
        
    def prepare_features(self, df):
        """Prepare features for training."""
        print("\n🔧 Preparing features...")
        
        # Separate features and target
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        print(f"  Features ({len(self.feature_columns)}): {self.feature_columns}")
        print(f"  Target: {self.target_column}")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Encode categorical features (Location)
        if 'Location' in df_processed.columns:
            print(f"\n  Encoding categorical feature: Location")
            df_processed['Location'] = self.label_encoder.fit_transform(df_processed['Location'])
            print(f"  Location mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        X = df_processed[self.feature_columns].values
        y = df_processed[self.target_column].values
        
        print(f"\n✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        
        return X, y
        
    def normalize_features(self, X_train, X_val, X_test):
        """Normalize features using StandardScaler."""
        print("\n Normalizing features...")
        
        # Flatten for scaling
        n_train, n_features = X_train.shape
        n_val = X_val.shape[0]
        n_test = X_test.shape[0]
        
        # Fit on train, transform all
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Scaled train: {X_train_scaled.shape}")
        print(f"✓ Scaled val: {X_val_scaled.shape}")
        print(f"✓ Scaled test: {X_test_scaled.shape}")
        
        # Save scaler
        scaler_path = os.path.join(self.config['model_dir'], 'scaler_10features.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to: {scaler_path}")
        
        # Save label encoder
        encoder_path = os.path.join(self.config['model_dir'], 'label_encoder_10features.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✓ Label encoder saved to: {encoder_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    def create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM."""
        print(f"\n Creating sequences (length={sequence_length})...")
        
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"✓ Sequences shape: {X_seq.shape}")
        print(f"✓ Targets shape: {y_seq.shape}")
        
        return X_seq, y_seq


class LSTMModel:
    """LSTM model builder and trainer."""
    
    def __init__(self, config, n_features):
        self.config = config
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build Stacked Bidirectional LSTM (same architecture as baseline)."""
        print("\n  Building LSTM model...")
        
        model = Sequential([
            # Layer 1: Bidirectional LSTM
            Bidirectional(LSTM(
                units=128,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.3
            ), input_shape=(self.config['sequence_length'], self.n_features)),
            
            # Layer 2: Bidirectional LSTM
            Bidirectional(LSTM(
                units=64,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.3
            )),
            
            # Dense layer
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            # Output layer (regression)
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        print("✓ Model built successfully!")
        print(f"\n Model Architecture:")
        model.summary()
        
        return self
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model."""
        print("\n Training LSTM model...")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.config['model_dir'], 'lstm_10features_best.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        return self
        
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        print("\n Evaluating model on test set...")
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Test Set Performance:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Compare with baseline
        baseline_r2 = 0.9343
        baseline_mae = 0.5095
        
        print(f"\n Comparison with Baseline (23 features):")
        print(f"  R² Change:   {baseline_r2:.4f} → {r2:.4f} ({(r2-baseline_r2):.4f}, {(r2-baseline_r2)/baseline_r2*100:+.2f}%)")
        print(f"  MAE Change:  {baseline_mae:.4f} → {mae:.4f} ({(mae-baseline_mae):.4f}, {(mae-baseline_mae)/baseline_mae*100:+.2f}%)")
        
        if r2 >= 0.92:  # Within 1.5% of baseline
            print(f"\n VALIDATION SUCCESS: 10 features achieve comparable performance!")
        elif r2 >= 0.90:
            print(f"\n  ACCEPTABLE: Performance slightly lower but still good (R² ≥ 0.90)")
        else:
            print(f"\n PERFORMANCE DROP: 10 features may not be sufficient (R² < 0.90)")
        
        # Save metrics
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_features': self.n_features,
            'baseline_r2': baseline_r2,
            'baseline_mae': baseline_mae
        }
        
        metrics_path = os.path.join(self.config['results_dir'], 'metrics_10features.txt')
        with open(metrics_path, 'w') as f:
            f.write("10-Feature Model Performance\n")
            f.write("=" * 50 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✓ Metrics saved to: {metrics_path}")
        
        return metrics, y_pred
        
    def plot_training_history(self):
        """Plot training history."""
        print("\n Plotting training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training & Validation Loss (10 Features)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Training & Validation MAE (10 Features)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['results_dir'], 'training_history_10features.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to: {plot_path}")
        
        plt.close()
        
    def plot_predictions(self, y_test, y_pred):
        """Plot predictions vs actual."""
        print("\n Plotting predictions...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
        axes[0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Stress Level', fontsize=12)
        axes[0].set_ylabel('Predicted Stress Level', fontsize=12)
        axes[0].set_title('Predictions vs Actual (10 Features)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = y_pred - y_test
        axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[1].set_xlabel('Prediction Error', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Error Distribution (10 Features)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['results_dir'], 'predictions_10features.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Predictions plot saved to: {plot_path}")
        
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("LSTM TRAINING WITH 10 FEATURES")
    print("=" * 80)
    print("")
    
    # Check if data file exists
    if not os.path.exists(CONFIG['data_file']):
        print(f" Error: Data file not found: {CONFIG['data_file']}")
        print(f"  Please run feature_selection.py first to create the 10-feature dataset.")
        return
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(CONFIG)
    
    # Load data
    df = preprocessor.load_data()
    
    # Prepare features
    X, y = preprocessor.prepare_features(df)
    
    # Split data (same as baseline)
    print(f"\n  Splitting data...")
    print(f"  Test split: {CONFIG['test_split']*100:.0f}%")
    print(f"  Validation split: {CONFIG['validation_split']*100:.0f}%")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_split'], 
        random_state=CONFIG['random_state'],
        shuffle=False  # Keep temporal order
    )
    
    # Second split: train vs val
    val_size = CONFIG['validation_split'] / (1 - CONFIG['test_split'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=CONFIG['random_state'],
        shuffle=False
    )
    
    print(f"✓ Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✓ Val set:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"✓ Test set:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Normalize
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.normalize_features(
        X_train, X_val, X_test
    )
    
    # Create sequences
    X_train_seq, y_train_seq = preprocessor.create_sequences(
        X_train_scaled, y_train, CONFIG['sequence_length']
    )
    X_val_seq, y_val_seq = preprocessor.create_sequences(
        X_val_scaled, y_val, CONFIG['sequence_length']
    )
    X_test_seq, y_test_seq = preprocessor.create_sequences(
        X_test_scaled, y_test, CONFIG['sequence_length']
    )
    
    print(f"\n Final sequence shapes:")
    print(f"  Train: {X_train_seq.shape} → {y_train_seq.shape}")
    print(f"  Val:   {X_val_seq.shape} → {y_val_seq.shape}")
    print(f"  Test:  {X_test_seq.shape} → {y_test_seq.shape}")
    
    # Build and train model
    n_features = X_train_seq.shape[2]
    print(f"\n Number of features: {n_features}")
    
    model = LSTMModel(CONFIG, n_features)
    model.build_model()
    model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    
    # Evaluate
    metrics, y_pred = model.evaluate(X_test_seq, y_test_seq)
    
    # Visualize
    model.plot_training_history()
    model.plot_predictions(y_test_seq, y_pred)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel saved to: {os.path.join(CONFIG['model_dir'], 'lstm_10features_best.keras')}")
    print(f"Results saved to: {CONFIG['results_dir']}")
    print("\nNext step: Run comparison_analysis.py to compare 10-feature vs 23-feature models")


if __name__ == '__main__':
    main()
