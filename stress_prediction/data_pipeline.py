"""
Data Pipeline for Stress Prediction
1. Loading data from CSV
2. Preprocessing features
3. Creating sequences for time-series prediction
4. Train/Val/Test splitting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import config

class StressDataPipeline:
    """
    Data pipeline for stress prediction using time-series sequences
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=1):
        """
        Initialize the data pipeline
        
        Args:
            sequence_length: Number of time steps to use for prediction (default: 60 minutes)
            prediction_horizon: Number of steps ahead to predict (default: 1)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = config.FEATURE_COLUMNS
        self.target_column = config.TARGET_COLUMN
        
    def load_data(self, data_file):
        """
        Load data from CSV file
        
        Args:
            data_file: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {data_file}...")
        self.df = pd.read_csv(data_file)
        print(f"Loaded {len(self.df)} samples with {len(self.df.columns)} columns")
        
        # Verify all required columns exist
        missing_cols = set(self.feature_columns + [self.target_column]) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        return self.df
    
    # Legacy 23-feature preprocessing pipeline: encode categorical columns and scale features.
    def preprocess_features(self, df, fit=True):
        """
        Preprocess features:
        1. Encode categorical features
        2. Scale numerical features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders/scalers (True for training data)
            
        Returns:
            Preprocessed features as numpy array
        """
        print("Preprocessing features...")
        df_processed = df[self.feature_columns].copy()
        
        # Identify categorical columns
        categorical_cols = ['Activity', 'Location', 'Sleep_Quality', 'Weather_Condition']
        
        # Encode categorical features
        for col in categorical_cols:
            if col in df_processed.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Scale all features
        if fit:
            scaled_features = self.scaler.fit_transform(df_processed)
        else:
            scaled_features = self.scaler.transform(df_processed)
        
        print(f"Preprocessed features shape: {scaled_features.shape}")
        return scaled_features
    
    # Legacy sequence creation for LSTM stress prediction.
    def create_sequences(self, features, targets):
        """
        Create sequences for time-series prediction
        
        Args:
            features: Preprocessed features (N, num_features)
            targets: Target stress levels (N,)
            
        Returns:
            X: Sequences (num_sequences, sequence_length, num_features)
            y: Target values (num_sequences,)
        """
        print(f"Creating sequences with length {self.sequence_length}...")
        
        X, y = [], []
        
        # Create sliding windows
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(features[i:i + self.sequence_length])
            # Target: stress level after prediction_horizon steps
            y.append(targets[i + self.sequence_length + self.prediction_horizon - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    # Legacy chronological split used before the final 13-feature pipeline.
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into train/val/test sets
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        print(f"Splitting data: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=config.RANDOM_SEED, shuffle=False
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=config.RANDOM_SEED, shuffle=False
        )
        
        print(f"Train: {len(X_train)} sequences")
        print(f"Val: {len(X_val)} sequences")
        print(f"Test: {len(X_test)} sequences")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data(self, data_file=None):
        """
        Complete data preparation pipeline
        
        Args:
            data_file: Path to CSV file (optional, uses config default)
            
        Returns:
            Dictionary with train/val/test data
        """
        if data_file is None:
            data_file = config.DATA_FILE
        
        # Load data
        df = self.load_data(data_file)
        
        # Extract features and targets
        features = self.preprocess_features(df, fit=True)
        targets = df[self.target_column].values
        
        print(f"\nTarget statistics:")
        print(f"  Min: {targets.min():.2f}")
        print(f"  Max: {targets.max():.2f}")
        print(f"  Mean: {targets.mean():.2f}")
        print(f"  Std: {targets.std():.2f}")
        
        # Create sequences
        X, y = self.create_sequences(features, targets)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, 
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'num_features': X_train.shape[2],
            'sequence_length': X_train.shape[1]
        }


if __name__ == '__main__':
    # Test the data pipeline
    print("="*50)
    print("Testing Data Pipeline")
    print("="*50)
    
    pipeline = StressDataPipeline(
        sequence_length=config.SEQUENCE_LENGTH,
        prediction_horizon=config.PREDICTION_HORIZON
    )
    
    data = pipeline.prepare_data()
    
    print("\n" + "="*50)
    print("Data Pipeline Summary")
    print("="*50)
    print(f"Sequence Length: {data['sequence_length']}")
    print(f"Number of Features: {data['num_features']}")
    print(f"\nTraining Set: {data['X_train'].shape}")
    print(f"Validation Set: {data['X_val'].shape}")
    print(f"Test Set: {data['X_test'].shape}")
    print(f"\nTarget Range: [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")
    print("="*50)
