"""
LSTM Baseline Model for Stress Prediction
1. LSTM architecture for stress level prediction
2. Training loop with callbacks
3. Model saving and loading
4. Evaluation on test set
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import config
from data_pipeline import StressDataPipeline

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


class LSTMStressPredictor:
    """
    LSTM-based stress level predictor
    """
    
    def __init__(self, sequence_length, num_features, lstm_units=128, dropout_rate=0.3):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Length of input sequences
            num_features: Number of input features
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build LSTM architecture
        
        Architecture:
        - Input: (sequence_length, num_features)
        - LSTM layer with dropout
        - Dense layer with ReLU
        - Output layer with linear activation (regression)
        """
        print("Building LSTM model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.num_features)),
            
            # LSTM layer with return_sequences=False (only last output)
            layers.LSTM(self.lstm_units, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            
            # Dense layer
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.dropout_rate),
            
            # Output layer (regression - predict stress level 1-9)
            layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='mse',  # Mean Squared Error for regression
            metrics=[
                'mae',  # Mean Absolute Error
                keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        
        self.model = model
        
        print("\nModel Architecture:")
        print("="*60)
        model.summary()
        print("="*60)
        
        return model
    
    def create_callbacks(self, model_dir):
        """
        Create training callbacks
        
        Args:
            model_dir: Directory to save model checkpoints
            
        Returns:
            List of callbacks
        """
        os.makedirs(model_dir, exist_ok=True)
        
        callback_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=os.path.join(model_dir, 'lstm_baseline_best.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=os.path.join(model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        return callback_list
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=64, epochs=100, model_dir=None):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            model_dir: Directory to save models
            
        Returns:
            Training history
        """
        if model_dir is None:
            model_dir = config.MODEL_DIR
        
        if self.model is None:
            self.build_model()
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {epochs}")
        print(f"Early stopping patience: {config.PATIENCE}")
        print("="*60 + "\n")
        
        # Create callbacks
        callback_list = self.create_callbacks(model_dir)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        print("\n" + "="*60)
        print("Training Completed!")
        print("="*60)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("Evaluating on Test Set")
        print("="*60)
        
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        print(f"\nTest Set Results:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print("="*60)
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # RMSE
        axes[2].plot(history['rmse'], label='Train RMSE', linewidth=2)
        axes[2].plot(history['val_rmse'], label='Val RMSE', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('RMSE', fontsize=12)
        axes[2].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            print("No model to save")
            return
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """
    Main training script
    """
    print("\n" + "="*60)
    print("LSTM BASELINE FOR STRESS PREDICTION")
    print("Phase 2 - Step 1")
    print("="*60 + "\n")
    
    # Step 1: Prepare data
    print("Step 1: Data Preparation")
    print("-"*60)
    pipeline = StressDataPipeline(
        sequence_length=config.SEQUENCE_LENGTH,
        prediction_horizon=config.PREDICTION_HORIZON
    )
    data = pipeline.prepare_data()
    
    # Step 2: Build model
    print("\nStep 2: Model Building")
    print("-"*60)
    model = LSTMStressPredictor(
        sequence_length=data['sequence_length'],
        num_features=data['num_features'],
        lstm_units=config.LSTM_UNITS,
        dropout_rate=config.DROPOUT_RATE
    )
    model.build_model()
    
    # Step 3: Train model
    print("\nStep 3: Model Training")
    print("-"*60)
    history = model.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        model_dir=config.MODEL_DIR
    )
    
    # Step 4: Evaluate model
    print("\nStep 4: Model Evaluation")
    print("-"*60)
    results = model.evaluate(data['X_test'], data['y_test'])
    
    # Step 5: Save results
    print("\nStep 5: Saving Results")
    print("-"*60)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, 'lstm_baseline_final.keras')
    model.save_model(final_model_path)
    
    # Plot and save training history
    plot_path = os.path.join(config.RESULTS_DIR, 'lstm_baseline_training_history.png')
    model.plot_training_history(save_path=plot_path)
    
    # Save results to file
    results_file = os.path.join(config.RESULTS_DIR, 'lstm_baseline_results.txt')
    with open(results_file, 'w') as f:
        f.write("LSTM Baseline - Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"MSE:  {results['mse']:.4f}\n")
        f.write(f"MAE:  {results['mae']:.4f}\n")
        f.write(f"RMSE: {results['rmse']:.4f}\n")
        f.write(f"R²:   {results['r2']:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"Results saved to {results_file}")
    
    print("\n" + "="*60)
    print("LSTM BASELINE TRAINING COMPLETED!")
    print("="*60)
    
    return model, results


if __name__ == '__main__':
    model, results = main()
