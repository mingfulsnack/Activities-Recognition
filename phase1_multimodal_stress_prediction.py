"""
PHASE 1: MULTI-MODAL STRESS PREDICTION - CORE RESEARCH
=====================================================

Research Question: "Can we predict stress levels by combining human activity recognition 
with real-time physiological and behavioral data?"

This module implements the core research for multi-modal stress prediction using:
1. Bidirectional LSTM for activity recognition (proven 95% accuracy)
2. Multi-modal fusion for physiological and behavioral data
3. Enhanced dataset with realistic variations
4. Comprehensive evaluation and analysis

Author: Research Team
Date: July 2025
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Input, Concatenate, 
    Attention, LayerNormalization, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiModalStressPrediction:
    """
    Multi-Modal Deep Learning Framework for Real-Time Stress Prediction
    Combining HAR (Human Activity Recognition) with Physiological Monitoring
    """
    
    def __init__(self, config=None):
        """Initialize the multi-modal stress prediction framework"""
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        self.feature_importance = {}
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['random_seed'])
        tf.random.set_seed(self.config['random_seed'])
        
    def _get_default_config(self):
        """Default configuration for the research"""
        return {
            'random_seed': 42,
            'test_size': 0.2,
            'validation_size': 0.2,
            'sequence_length': 180,  # From proven WISDM research
            'batch_size': 64,        # Increased for faster training  
            'epochs': 20,            # Reduced for research feasibility
            'early_stopping_patience': 5,  # Reduced patience
            'reduce_lr_patience': 3,        # Reduced patience  
            'learning_rate': 0.001,
            'hidden_units': 30,      # From proven WISDM research
            'dropout_rate': 0.3,     # Reduced dropout for faster convergence
            'l2_regularization': 0.01,
            'stress_threshold_low': 3.0,
            'stress_threshold_high': 6.0
        }
    
    def load_and_preprocess_data(self, data_path, sample_fraction=1.0):
        """
        Load sequential behavioral health dataset and prepare for multi-modal analysis
        sample_fraction: Use subset of data for faster research (1.0 = full data)
        """
        print("🔄 Loading and preprocessing sequential behavioral health data...")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Use subset for faster research
        if sample_fraction < 1.0:
            n_samples = int(len(df) * sample_fraction)
            df = df.sample(n=n_samples, random_state=42).sort_values('Timestamp').reset_index(drop=True)
            print(f"📊 Using {sample_fraction*100:.0f}% of data: {len(df):,} records")
        else:
            print(f"📊 Using full dataset: {len(df):,} records")
        
        print(f"📊 Loaded {len(df):,} records with {df.shape[1]} features")
        
        # Convert timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Define feature groups for multi-modal approach (updated for actual dataset columns)
        self.feature_groups = {
            'accelerometer': ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z'],
            'physiological': ['Heart_Rate', 'Sleep_Duration', 'Sleep_Quality', 'Energy_Level', 'Reaction_Time', 'Mood_Score'],
            'behavioral': ['Screen_Time', 'Step_Count', 'Calories', 'Exercise_Minutes', 'Social_Interaction', 'Screen_Usage_Current'],
            'environmental': ['Ambient_Light', 'Noise_Level', 'Weather_Condition', 'Location'],
            'temporal': ['Age'],  # Will add time-based features
            'categorical': ['Gender', 'Activity']
        }
        
        # Target variable
        self.target = 'Stress_Level'
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Handle categorical variables
        df = self._encode_categorical_features(df)
        
        # Create stress level categories for classification
        df['Stress_Category'] = self._categorize_stress(df[self.target])
        
        # Store processed data
        self.data = df
        
        print(f"✅ Preprocessing completed. Data shape: {df.shape}")
        print(f"📈 Stress Level Range: {df[self.target].min():.1f} - {df[self.target].max():.1f}")
        print(f"📊 Stress Categories Distribution:")
        print(df['Stress_Category'].value_counts().sort_index())
        
        return df
    
    def _add_temporal_features(self, df):
        """Add time-based features for temporal modeling"""
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['TimeOfDay'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], 
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Circadian rhythm features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Update feature groups
        self.feature_groups['temporal'].extend([
            'Hour', 'DayOfWeek', 'IsWeekend', 'Hour_Sin', 'Hour_Cos', 
            'DayOfWeek_Sin', 'DayOfWeek_Cos'
        ])
        self.feature_groups['categorical'].append('TimeOfDay')
        
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_features = self.feature_groups['categorical']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.encoders[feature] = le
        
        return df
    
    def _categorize_stress(self, stress_levels):
        """Convert continuous stress to categories"""
        categories = pd.cut(stress_levels, 
                          bins=[0, self.config['stress_threshold_low'], 
                               self.config['stress_threshold_high'], 10],
                          labels=['Low', 'Medium', 'High'],
                          include_lowest=True)
        return categories
    
    def create_sequences(self, df):
        """
        Create sequences for time series modeling with proper behavioral sequences
        Based on proven WISDM research parameters
        """
        print(f"🔄 Creating sequences with length {self.config['sequence_length']}...")
        
        sequence_length = self.config['sequence_length']
        
        # =====================================
        # Accelerometer sequences (time series)
        # =====================================
        accel_features = self.feature_groups['accelerometer']
        accel_data = df[accel_features].values
        
        # =====================================
        # Behavioral sequences (time series patterns)
        # =====================================
        behavioral_features = self.feature_groups['behavioral']
        behavioral_data = df[behavioral_features].values
        
        # =====================================
        # Other features (point-in-time)
        # =====================================
        other_features = []
        for group in ['physiological', 'environmental', 'temporal']:
            other_features.extend(self.feature_groups[group])
        
        # Add encoded categorical features
        encoded_features = []
        for feature in self.feature_groups['categorical']:
            if f'{feature}_encoded' in df.columns:
                encoded_features.append(f'{feature}_encoded')
        
        # Only include numeric features in other_features
        other_features = []
        for group in ['physiological', 'environmental', 'temporal']:
            for feature in self.feature_groups[group]:
                if feature in df.columns:
                    # Check if feature is numeric
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        other_features.append(feature)
                    else:
                        # Encode non-numeric environmental features
                        if feature not in self.encoders:
                            le = LabelEncoder()
                            df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                            self.encoders[feature] = le
                            encoded_features.append(f'{feature}_encoded')
        
        # Add all encoded categorical features
        other_features.extend(encoded_features)
        
        other_data = df[other_features].values
        target_data = df[self.target].values
        target_cat_data = df['Stress_Category'].values
        
        # Create sequences
        X_accel, X_behavioral, X_other, y_reg, y_cat = [], [], [], [], []
        
        for i in range(sequence_length, len(df)):
            # Accelerometer sequence (180 timesteps × 3 features)
            X_accel.append(accel_data[i-sequence_length:i])
            
            # Behavioral sequence (180 timesteps × 6 features)
            X_behavioral.append(behavioral_data[i-sequence_length:i])
            
            # Other features (current timestep values)
            X_other.append(other_data[i])
            
            # Targets
            y_reg.append(target_data[i])
            y_cat.append(target_cat_data[i])
        
        # Convert to numpy arrays
        X_accel = np.array(X_accel)
        X_behavioral = np.array(X_behavioral)
        X_other = np.array(X_other)
        y_reg = np.array(y_reg)
        
        # Encode categorical target
        le_target = LabelEncoder()
        y_cat_encoded = le_target.fit_transform(y_cat)
        self.encoders['stress_category'] = le_target
        
        print(f"✅ Created {len(X_accel)} sequences")
        print(f"📊 Accelerometer shape: {X_accel.shape}")
        print(f"📊 Behavioral shape: {X_behavioral.shape}")
        print(f"📊 Other features shape: {X_other.shape}")
        
        return X_accel, X_behavioral, X_other, y_reg, y_cat_encoded, other_features
    
    def split_data_temporal(self, X_accel, X_behavioral, X_other, y_reg, y_cat):
        """
        Temporal split to preserve time series nature
        """
        print("🔄 Performing temporal data split...")
        
        total_samples = len(X_accel)
        
        # Calculate split indices
        train_end = int(total_samples * (1 - self.config['test_size'] - self.config['validation_size']))
        val_end = int(total_samples * (1 - self.config['test_size']))
        
        # Split data
        X_accel_train = X_accel[:train_end]
        X_accel_val = X_accel[train_end:val_end]
        X_accel_test = X_accel[val_end:]
        
        X_behavioral_train = X_behavioral[:train_end]
        X_behavioral_val = X_behavioral[train_end:val_end]
        X_behavioral_test = X_behavioral[val_end:]
        
        X_other_train = X_other[:train_end]
        X_other_val = X_other[train_end:val_end]
        X_other_test = X_other[val_end:]
        
        y_reg_train = y_reg[:train_end]
        y_reg_val = y_reg[train_end:val_end]
        y_reg_test = y_reg[val_end:]
        
        y_cat_train = y_cat[:train_end]
        y_cat_val = y_cat[train_end:val_end]
        y_cat_test = y_cat[val_end:]
        
        print(f"📊 Train: {len(X_accel_train)} samples")
        print(f"📊 Validation: {len(X_accel_val)} samples")
        print(f"📊 Test: {len(X_accel_test)} samples")
        
        return (X_accel_train, X_accel_val, X_accel_test,
                X_behavioral_train, X_behavioral_val, X_behavioral_test,
                X_other_train, X_other_val, X_other_test,
                y_reg_train, y_reg_val, y_reg_test,
                y_cat_train, y_cat_val, y_cat_test)
    
    def scale_features(self, X_other_train, X_other_val, X_other_test):
        """Scale non-temporal features (Physiological + Environmental)"""
        scaler = StandardScaler()
        
        X_other_train_scaled = scaler.fit_transform(X_other_train)
        X_other_val_scaled = scaler.transform(X_other_val)
        X_other_test_scaled = scaler.transform(X_other_test)
        
        self.scalers['other_features'] = scaler
        
        print("✅ Scaled physiological and environmental features")
        print("ℹ️  Note: Accelerometer and behavioral sequences not scaled (preserve temporal patterns)")
        
        return X_other_train_scaled, X_other_val_scaled, X_other_test_scaled
    
    def build_multimodal_model(self, accel_shape, behavioral_shape, other_shape, num_classes=3):
        """
        Build true multi-modal architecture as per original research plan:
        - HAR Branch: Bidirectional LSTM (proven 95% accuracy)
        - Physiological Branch: Dense layers cho continuous features  
        - Behavioral Branch: Embedding + LSTM cho sequential patterns
        - Environmental Branch: Categorical encoding + Dense
        - Fusion Layer: Attention mechanism để combine features
        """
        print("🏗️ Building true multi-modal stress prediction model...")
        
        # =====================================
        # INPUT LAYERS
        # =====================================
        accel_input = Input(shape=accel_shape, name='accelerometer_input')
        behavioral_input = Input(shape=behavioral_shape, name='behavioral_input')
        other_input = Input(shape=(other_shape[1],), name='other_features_input')
        
        # =====================================
        # HAR BRANCH: Bidirectional LSTM (proven 95% accuracy)
        # =====================================
        print("🔧 Building HAR Branch (Bidirectional LSTM)...")
        har_branch = Bidirectional(
            LSTM(self.config['hidden_units'], 
                 return_sequences=True,
                 dropout=self.config['dropout_rate'],
                 recurrent_dropout=self.config['dropout_rate']),
            name='har_lstm_1'
        )(accel_input)
        
        har_branch = BatchNormalization()(har_branch)
        
        har_branch = Bidirectional(
            LSTM(self.config['hidden_units'] // 2,
                 return_sequences=False,
                 dropout=self.config['dropout_rate'],
                 recurrent_dropout=self.config['dropout_rate']),
            name='har_lstm_2'
        )(har_branch)
        
        har_features = Dense(64, activation='relu', name='har_dense')(har_branch)
        har_features = Dropout(self.config['dropout_rate'])(har_features)
        
        # =====================================
        # BEHAVIORAL BRANCH: Embedding + LSTM cho sequential patterns
        # =====================================
        print("🔧 Building Behavioral Branch (LSTM for sequential patterns)...")
        behavioral_branch = LSTM(
            self.config['hidden_units'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate'],
            name='behavioral_lstm_1'
        )(behavioral_input)
        
        behavioral_branch = BatchNormalization()(behavioral_branch)
        
        behavioral_branch = LSTM(
            self.config['hidden_units'] // 2,
            return_sequences=False,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate'],
            name='behavioral_lstm_2'
        )(behavioral_branch)
        
        behavioral_features = Dense(64, activation='relu', name='behavioral_dense')(behavioral_branch)
        behavioral_features = Dropout(self.config['dropout_rate'])(behavioral_features)
        
        # =====================================
        # PHYSIOLOGICAL & ENVIRONMENTAL BRANCH: Dense layers cho continuous features
        # =====================================
        print("🔧 Building Physiological & Environmental Branch (Dense layers)...")
        physio_env_branch = Dense(128, activation='relu', name='physio_env_dense_1')(other_input)
        physio_env_branch = BatchNormalization()(physio_env_branch)
        physio_env_branch = Dropout(self.config['dropout_rate'])(physio_env_branch)
        
        physio_env_branch = Dense(64, activation='relu', name='physio_env_dense_2')(physio_env_branch)
        physio_env_branch = Dropout(self.config['dropout_rate'])(physio_env_branch)
        
        physio_env_features = Dense(32, activation='relu', name='physio_env_features')(physio_env_branch)
        
        # =====================================
        # MULTI-MODAL FUSION WITH ATTENTION
        # =====================================
        print("🔧 Building Attention-Based Fusion Layer...")
        # Concatenate all branch features
        combined_features = Concatenate(name='multi_modal_fusion')([
            har_features,           # 64 features from HAR
            behavioral_features,    # 64 features from Behavioral
            physio_env_features    # 32 features from Physio+Env
        ])  # Total: 160 features
        
        # Advanced attention mechanism for feature weighting
        attention_weights = Dense(combined_features.shape[-1], activation='softmax', name='attention_weights')(combined_features)
        
        # Element-wise multiplication for attention
        from tensorflow.keras.layers import Multiply
        attended_features = Multiply(name='attended_features')([combined_features, attention_weights])
        
        # Final fusion layers
        fusion = Dense(128, activation='relu', name='fusion_dense_1')(attended_features)
        fusion = LayerNormalization()(fusion)
        fusion = Dropout(self.config['dropout_rate'])(fusion)
        
        fusion = Dense(64, activation='relu', name='fusion_dense_2')(fusion)
        fusion = Dropout(self.config['dropout_rate'])(fusion)
        
        fusion = Dense(32, activation='relu', name='fusion_dense_3')(fusion)
        fusion = Dropout(self.config['dropout_rate'])(fusion)
        
        # =====================================
        # OUTPUT HEADS (Multi-task Learning)
        # =====================================
        # Regression output (continuous stress)
        stress_regression = Dense(16, activation='relu', name='stress_reg_dense')(fusion)
        stress_regression = Dense(1, activation='linear', name='stress_regression')(stress_regression)
        
        # Classification output (stress categories)
        stress_classification = Dense(16, activation='relu', name='stress_cls_dense')(fusion)
        stress_classification = Dense(num_classes, activation='softmax', name='stress_classification')(stress_classification)
        
        # =====================================
        # CREATE MODEL
        # =====================================
        model = Model(
            inputs=[accel_input, behavioral_input, other_input],
            outputs=[stress_regression, stress_classification],
            name='TrueMultiModal_Stress_Predictor'
        )
        
        # Compile with multi-task losses
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss={
                'stress_regression': 'mse',
                'stress_classification': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'stress_regression': 0.6,  # Primary focus on regression
                'stress_classification': 0.4
            },
            metrics={
                'stress_regression': ['mae', 'mse'],
                'stress_classification': ['accuracy', 'sparse_categorical_crossentropy']
            }
        )
        
        print("✅ True multi-modal model built successfully!")
        print(f"📊 Model parameters: {model.count_params():,}")
        print("🏗️ Architecture Summary:")
        print("   ├── HAR Branch: Bidirectional LSTM (64 features)")
        print("   ├── Behavioral Branch: LSTM sequences (64 features)")  
        print("   ├── Physio+Env Branch: Dense layers (32 features)")
        print("   ├── Fusion: Attention mechanism (160 → 32 features)")
        print("   └── Outputs: Regression + Classification")
        
        return model
    
    def train_model(self, model, X_accel_train, X_behavioral_train, X_other_train, y_reg_train, y_cat_train,
                   X_accel_val, X_behavioral_val, X_other_val, y_reg_val, y_cat_val):
        """Train the true multi-modal stress prediction model"""
        print("🚀 Training true multi-modal stress prediction model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_true_multimodal_stress_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model with all three input streams
        history = model.fit(
            [X_accel_train, X_behavioral_train, X_other_train],
            [y_reg_train, y_cat_train],
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=([X_accel_val, X_behavioral_val, X_other_val], [y_reg_val, y_cat_val]),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        
        # Store training history
        self.training_history = history
        
        return history
    
    def evaluate_model(self, model, X_accel_test, X_behavioral_test, X_other_test, y_reg_test, y_cat_test):
        """Comprehensive evaluation of the true multi-modal model"""
        print("📊 Evaluating true multi-modal stress prediction model...")
        
        # Make predictions with all three input streams
        predictions = model.predict([X_accel_test, X_behavioral_test, X_other_test])
        y_reg_pred = predictions[0].flatten()
        y_cat_pred = predictions[1]
        y_cat_pred_classes = np.argmax(y_cat_pred, axis=1)
        
        # =====================================
        # Regression Evaluation
        # =====================================
        reg_results = {
            'mae': mean_absolute_error(y_reg_test, y_reg_pred),
            'mse': mean_squared_error(y_reg_test, y_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)),
            'r2': r2_score(y_reg_test, y_reg_pred),
            'mape': np.mean(np.abs((y_reg_test - y_reg_pred) / y_reg_test)) * 100
        }
        
        # =====================================
        # Classification Evaluation
        # =====================================
        cls_results = {
            'accuracy': np.mean(y_cat_pred_classes == y_cat_test),
            'classification_report': classification_report(y_cat_test, y_cat_pred_classes),
            'confusion_matrix': confusion_matrix(y_cat_test, y_cat_pred_classes)
        }
        
        # Store results
        self.results = {
            'regression': reg_results,
            'classification': cls_results,
            'predictions': {
                'y_reg_true': y_reg_test,
                'y_reg_pred': y_reg_pred,
                'y_cat_true': y_cat_test,
                'y_cat_pred': y_cat_pred_classes
            }
        }
        
        # Print results
        print("\n" + "="*50)
        print("📈 REGRESSION RESULTS (Continuous Stress Prediction)")
        print("="*50)
        print(f"MAE (Mean Absolute Error): {reg_results['mae']:.4f}")
        print(f"MSE (Mean Squared Error): {reg_results['mse']:.4f}")
        print(f"RMSE (Root Mean Squared Error): {reg_results['rmse']:.4f}")
        print(f"R² Score: {reg_results['r2']:.4f}")
        print(f"MAPE (Mean Absolute Percentage Error): {reg_results['mape']:.2f}%")
        
        print("\n" + "="*50)
        print("📊 CLASSIFICATION RESULTS (Stress Categories)")
        print("="*50)
        print(f"Accuracy: {cls_results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(cls_results['classification_report'])
        
        return self.results
    
    def plot_results(self):
        """Create comprehensive result visualizations"""
        print("📊 Creating result visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Modal Stress Prediction Results', fontsize=16, fontweight='bold')
        
        # 1. Training History
        if hasattr(self, 'training_history'):
            history = self.training_history.history
            
            # Loss curves
            axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Model Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Regression MAE
            axes[0, 1].plot(history['stress_regression_mae'], label='Training MAE', linewidth=2)
            axes[0, 1].plot(history['val_stress_regression_mae'], label='Validation MAE', linewidth=2)
            axes[0, 1].set_title('Regression MAE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Classification Accuracy
            axes[0, 2].plot(history['stress_classification_accuracy'], label='Training Accuracy', linewidth=2)
            axes[0, 2].plot(history['val_stress_classification_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[0, 2].set_title('Classification Accuracy')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 2. Prediction vs Actual (Regression)
        if 'predictions' in self.results:
            y_true = self.results['predictions']['y_reg_true']
            y_pred = self.results['predictions']['y_reg_pred']
            
            axes[1, 0].scatter(y_true, y_pred, alpha=0.6, c='blue', s=20)
            axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual Stress Level')
            axes[1, 0].set_ylabel('Predicted Stress Level')
            axes[1, 0].set_title(f'Regression: R² = {self.results["regression"]["r2"]:.4f}')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 3. Confusion Matrix
        if 'classification' in self.results:
            cm = self.results['classification']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
        
        # 4. Residuals Plot
        if 'predictions' in self.results:
            y_true = self.results['predictions']['y_reg_true']
            y_pred = self.results['predictions']['y_reg_pred']
            residuals = y_true - y_pred
            
            axes[1, 2].scatter(y_pred, residuals, alpha=0.6, c='green', s=20)
            axes[1, 2].axhline(y=0, color='r', linestyle='--')
            axes[1, 2].set_xlabel('Predicted Stress Level')
            axes[1, 2].set_ylabel('Residuals')
            axes[1, 2].set_title('Residuals Plot')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('true_multimodal_stress_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'true_multimodal_stress_prediction_results.png'")

def main():
    """Main execution function for Phase 1 research with TRUE multi-modal architecture"""
    print("🚀 Starting Phase 1: TRUE Multi-Modal Stress Prediction Research")
    print("=" * 60)
    print("🏗️ Architecture: HAR + Behavioral + Physiological + Environmental")
    print("=" * 60)
    
    # Initialize research framework
    research = MultiModalStressPrediction()
    
    # Step 1: Load and preprocess data (using full dataset)
    data_path = 'data/sequential_behavioral_health_data_30days.csv'
    df = research.load_and_preprocess_data(data_path, sample_fraction=1.0)
    
    # Step 2: Create sequences for true multi-modal modeling
    X_accel, X_behavioral, X_other, y_reg, y_cat, feature_names = research.create_sequences(df)
    
    # Step 3: Temporal data split for all modalities
    (X_accel_train, X_accel_val, X_accel_test,
     X_behavioral_train, X_behavioral_val, X_behavioral_test,
     X_other_train, X_other_val, X_other_test,
     y_reg_train, y_reg_val, y_reg_test,
     y_cat_train, y_cat_val, y_cat_test) = research.split_data_temporal(
        X_accel, X_behavioral, X_other, y_reg, y_cat)
    
    # Step 4: Scale features (only non-temporal features)
    X_other_train_scaled, X_other_val_scaled, X_other_test_scaled = research.scale_features(
        X_other_train, X_other_val, X_other_test)
    
    # Step 5: Build TRUE multi-modal model
    model = research.build_multimodal_model(
        accel_shape=X_accel_train.shape[1:],
        behavioral_shape=X_behavioral_train.shape[1:],
        other_shape=X_other_train_scaled.shape,
        num_classes=3
    )
    
    # Step 6: Train model with all three input streams
    history = research.train_model(
        model, 
        X_accel_train, X_behavioral_train, X_other_train_scaled, 
        y_reg_train, y_cat_train,
        X_accel_val, X_behavioral_val, X_other_val_scaled, 
        y_reg_val, y_cat_val
    )
    
    # Step 7: Evaluate model with all three input streams
    results = research.evaluate_model(
        model, 
        X_accel_test, X_behavioral_test, X_other_test_scaled, 
        y_reg_test, y_cat_test
    )
    
    # Step 8: Create visualizations
    research.plot_results()
    
    print("\n🎉 Phase 1 TRUE Multi-Modal Research Completed Successfully!")
    print("🏗️ Architecture Validated:")
    print("   ├── HAR Branch: Bidirectional LSTM ✅")
    print("   ├── Behavioral Branch: LSTM sequences ✅") 
    print("   ├── Physio+Env Branch: Dense layers ✅")
    print("   └── Fusion: Attention mechanism ✅")
    print("\n📊 Key Findings:")
    print(f"   • Regression R² Score: {results['regression']['r2']:.4f}")
    print(f"   • Regression RMSE: {results['regression']['rmse']:.4f}")
    print(f"   • Classification Accuracy: {results['classification']['accuracy']:.4f}")
    
    return research, results

if __name__ == "__main__":
    research_framework, final_results = main()
