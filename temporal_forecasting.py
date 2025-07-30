"""
PHASE 2: TEMPORAL STRESS FORECASTING - MODULE
===========================================

Research Focus: "Can AI predict stress patterns 24-48 hours in advance?"

This module implements:
1. Multi-step temporal prediction (Sequence-to-Sequence models)
2. Stress pattern forecasting with uncertainty quantification
3. Early warning system for stress episodes
4. Time-series analysis with seasonal decomposition

Author: Research Team
Date: July 30, 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, 
    LayerNormalization, TimeDistributed, RepeatVector,
    Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten,
    Concatenate, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series specific imports
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

class TemporalStressForecasting:
    """
    Advanced Temporal Stress Forecasting System
    Multi-step prediction with uncertainty quantification
    """
    
    def __init__(self, lookback_hours=24, forecast_hours=48):
        """Initialize the forecasting system"""
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.forecast_history = []
        
    def prepare_temporal_data(self, df):
        """
        Prepare data for temporal forecasting
        """
        print(f"📊 Preparing temporal data for forecasting...")
        print(f"   • Lookback window: {self.lookback_hours} hours")
        print(f"   • Forecast horizon: {self.forecast_hours} hours")
        
        # Sort by timestamp
        df_sorted = df.sort_values('Timestamp').copy()
        
        # Create temporal features
        df_sorted = self._create_temporal_features(df_sorted)
        
        # Create lag features
        df_sorted = self._create_lag_features(df_sorted)
        
        # Create rolling statistics
        df_sorted = self._create_rolling_features(df_sorted)
        
        # Handle missing values
        df_sorted = df_sorted.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Temporal data prepared: {len(df_sorted)} samples, {len(df_sorted.columns)} features")
        
        return df_sorted
    
    def _create_temporal_features(self, df):
        """Create comprehensive temporal features"""
        df = df.copy()
        
        # Time components
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['DayOfMonth'] = df['Timestamp'].dt.day
        df['Month'] = df['Timestamp'].dt.month
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Time-based patterns
        df['WorkingHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17) & (df['DayOfWeek'] < 5)).astype(int)
        df['EveningHours'] = ((df['Hour'] >= 18) & (df['Hour'] <= 22)).astype(int)
        df['NightHours'] = ((df['Hour'] >= 23) | (df['Hour'] <= 6)).astype(int)
        
        return df
    
    def _create_lag_features(self, df):
        """Create lag features for temporal dependencies"""
        df = df.copy()
        
        # Stress level lags
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'Stress_lag_{lag}h'] = df['Stress_Level'].shift(lag)
        
        # Activity lags
        for feature in ['Step_Count', 'Heart_Rate', 'Sleep_Quality', 'Screen_Time']:
            if feature in df.columns:
                for lag in [1, 3, 6, 12]:
                    df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df):
        """Create rolling window statistics"""
        df = df.copy()
        
        # Rolling statistics for stress
        for window in [6, 12, 24]:
            df[f'Stress_rolling_mean_{window}h'] = df['Stress_Level'].rolling(window=window).mean()
            df[f'Stress_rolling_std_{window}h'] = df['Stress_Level'].rolling(window=window).std()
            df[f'Stress_rolling_min_{window}h'] = df['Stress_Level'].rolling(window=window).min()
            df[f'Stress_rolling_max_{window}h'] = df['Stress_Level'].rolling(window=window).max()
        
        # Rolling statistics for key features
        for feature in ['Step_Count', 'Heart_Rate', 'Sleep_Quality']:
            if feature in df.columns:
                for window in [6, 12, 24]:
                    df[f'{feature}_rolling_mean_{window}h'] = df[feature].rolling(window=window).mean()
                    df[f'{feature}_rolling_trend_{window}h'] = df[feature].rolling(window=window).apply(
                        lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
                    )
        
        return df
    
    def create_sequences(self, df, target_col='Stress_Level'):
        """
        Create sequences for multi-step prediction
        """
        print(f"🔄 Creating sequences for forecasting...")
        
        # Select features for modeling
        feature_cols = self._select_forecasting_features(df)
        
        # Prepare data
        data = df[feature_cols + [target_col]].values
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        self.scalers['target'] = MinMaxScaler()
        
        scaled_features = self.scalers['features'].fit_transform(data[:, :-1])
        scaled_target = self.scalers['target'].fit_transform(data[:, -1].reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.lookback_hours, len(data) - self.forecast_hours):
            # Input sequence (lookback_hours of features)
            X.append(scaled_features[i-self.lookback_hours:i])
            
            # Target sequence (forecast_hours of stress levels)
            y.append(scaled_target[i:i+self.forecast_hours])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Sequences created:")
        print(f"   • Input shape: {X.shape}")
        print(f"   • Target shape: {y.shape}")
        print(f"   • Features used: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def _select_forecasting_features(self, df):
        """Select relevant features for forecasting"""
        base_features = [
            'Step_Count', 'Heart_Rate', 'Sleep_Duration', 'Sleep_Quality',
            'Exercise_Minutes', 'Screen_Time', 'Social_Interaction', 'Energy_Level'
        ]
        
        # Add temporal features
        temporal_features = [
            'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'WorkingHours', 'EveningHours', 'NightHours', 'IsWeekend'
        ]
        
        # Add lag features
        lag_features = [col for col in df.columns if 'lag_' in col]
        
        # Add rolling features
        rolling_features = [col for col in df.columns if 'rolling_' in col]
        
        # Combine all features
        all_features = base_features + temporal_features + lag_features + rolling_features
        
        # Filter existing columns
        available_features = [f for f in all_features if f in df.columns]
        
        return available_features[:30]  # Limit to prevent overfitting
    
    def build_seq2seq_model(self, input_shape, output_length):
        """
        Build Sequence-to-Sequence model for multi-step forecasting
        """
        print(f"🏗️ Building Seq2Seq model...")
        print(f"   • Input shape: {input_shape}")
        print(f"   • Output length: {output_length}")
        
        # Encoder
        encoder_inputs = Input(shape=input_shape, name='encoder_input')
        
        # Multi-scale feature extraction
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(encoder_inputs)
        conv2 = Conv1D(64, 5, activation='relu', padding='same')(encoder_inputs)
        conv3 = Conv1D(64, 7, activation='relu', padding='same')(encoder_inputs)
        
        # Combine multi-scale features
        conv_combined = Concatenate()([conv1, conv2, conv3])
        conv_combined = Dropout(0.2)(conv_combined)
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(conv_combined)
        lstm1 = LayerNormalization()(lstm1)
        
        lstm2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(lstm2, lstm2)
        attention = Dropout(0.2)(attention)
        attention = LayerNormalization()(attention + lstm2)
        
        # Encoder output
        encoder_output = GlobalAveragePooling1D()(attention)
        encoder_output = Dense(256, activation='relu')(encoder_output)
        encoder_output = Dropout(0.3)(encoder_output)
        
        # Decoder
        # Repeat encoder output for each forecast step
        decoder_input = RepeatVector(output_length)(encoder_output)
        
        # Decoder LSTM
        decoder_lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(decoder_input)
        decoder_lstm1 = LayerNormalization()(decoder_lstm1)
        
        decoder_lstm2 = LSTM(128, return_sequences=True, dropout=0.2)(decoder_lstm1)
        decoder_lstm2 = LayerNormalization()(decoder_lstm2)
        
        # Output layer
        outputs = TimeDistributed(Dense(64, activation='relu'))(decoder_lstm2)
        outputs = TimeDistributed(Dropout(0.2))(outputs)
        outputs = TimeDistributed(Dense(1, activation='sigmoid'))(outputs)
        
        # Flatten for loss calculation
        outputs = tf.squeeze(outputs, axis=-1)
        
        model = Model(inputs=encoder_inputs, outputs=outputs, name='StressForecaster')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("✅ Seq2Seq model built successfully")
        print(f"   • Total parameters: {model.count_params():,}")
        
        return model
    
    def build_transformer_forecaster(self, input_shape, output_length):
        """
        Build Transformer-based forecasting model
        """
        print(f"🔮 Building Transformer forecaster...")
        
        inputs = Input(shape=input_shape, name='transformer_input')
        
        # Positional encoding
        x = inputs
        
        # Multi-head attention blocks
        for i in range(3):  # 3 transformer blocks
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=8, 
                key_dim=64,
                dropout=0.1
            )(x, x)
            attn_output = Dropout(0.1)(attn_output)
            x = LayerNormalization()(x + attn_output)
            
            # Feed forward
            ffn_output = Dense(256, activation='relu')(x)
            ffn_output = Dropout(0.1)(ffn_output)
            ffn_output = Dense(input_shape[-1])(ffn_output)
            x = LayerNormalization()(x + ffn_output)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Forecasting head
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(output_length, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='TransformerForecaster')
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        print("✅ Transformer forecaster built successfully")
        return model
    
    def train_forecasting_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple forecasting models
        """
        print(f"🚀 Training forecasting models...")
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7, monitor='val_loss')
        ]
        
        # Train Seq2Seq model
        print("\n📈 Training Seq2Seq model...")
        seq2seq_model = self.build_seq2seq_model(X_train.shape[1:], y_train.shape[1])
        
        seq2seq_history = seq2seq_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['seq2seq'] = seq2seq_model
        
        # Train Transformer model
        print("\n🔮 Training Transformer model...")
        transformer_model = self.build_transformer_forecaster(X_train.shape[1:], y_train.shape[1])
        
        transformer_history = transformer_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['transformer'] = transformer_model
        
        print("✅ All models trained successfully")
        
        return {
            'seq2seq_history': seq2seq_history,
            'transformer_history': transformer_history
        }
    
    def forecast_stress_patterns(self, X_input, model_name='ensemble'):
        """
        Generate stress forecasts with uncertainty quantification
        """
        print(f"🔮 Generating stress forecasts using {model_name}...")
        
        if model_name == 'ensemble':
            # Ensemble prediction
            predictions = []
            
            for name, model in self.models.items():
                pred = model.predict(X_input, verbose=0)
                # Inverse transform
                pred_rescaled = self.scalers['target'].inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
                predictions.append(pred_rescaled)
            
            # Average ensemble
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Calculate uncertainty (standard deviation across models)
            uncertainty = np.std(predictions, axis=0)
            
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            pred = model.predict(X_input, verbose=0)
            ensemble_pred = self.scalers['target'].inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
            uncertainty = np.zeros_like(ensemble_pred)  # No uncertainty for single model
        
        # Generate forecast metadata
        forecast_metadata = {
            'model_used': model_name,
            'forecast_horizon': self.forecast_hours,
            'confidence_intervals': self._calculate_confidence_intervals(ensemble_pred, uncertainty),
            'risk_alerts': self._generate_risk_alerts(ensemble_pred),
            'peak_stress_times': self._identify_peak_stress_times(ensemble_pred)
        }
        
        print(f"✅ Forecasts generated:")
        print(f"   • Samples forecasted: {len(ensemble_pred)}")
        print(f"   • Forecast horizon: {self.forecast_hours} hours")
        print(f"   • Risk alerts: {len(forecast_metadata['risk_alerts'])} detected")
        
        return ensemble_pred, uncertainty, forecast_metadata
    
    def _calculate_confidence_intervals(self, predictions, uncertainty, confidence_level=0.95):
        """Calculate confidence intervals for predictions"""
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        confidence_intervals = []
        for i in range(len(predictions)):
            lower_bound = predictions[i] - z_score * uncertainty[i]
            upper_bound = predictions[i] + z_score * uncertainty[i]
            
            confidence_intervals.append({
                'sample': i,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'width': upper_bound - lower_bound
            })
        
        return confidence_intervals
    
    def _generate_risk_alerts(self, predictions, high_stress_threshold=7.0):
        """Generate risk alerts for high stress periods"""
        alerts = []
        
        for i, forecast in enumerate(predictions):
            # Find peaks in stress forecast
            high_stress_hours = np.where(forecast > high_stress_threshold)[0]
            
            if len(high_stress_hours) > 0:
                # Group consecutive hours
                groups = []
                current_group = [high_stress_hours[0]]
                
                for hour in high_stress_hours[1:]:
                    if hour == current_group[-1] + 1:
                        current_group.append(hour)
                    else:
                        groups.append(current_group)
                        current_group = [hour]
                groups.append(current_group)
                
                # Create alerts for each group
                for group in groups:
                    alert = {
                        'sample': i,
                        'start_hour': group[0],
                        'end_hour': group[-1],
                        'duration': len(group),
                        'max_stress': forecast[group].max(),
                        'avg_stress': forecast[group].mean(),
                        'severity': 'high' if forecast[group].max() > 8.5 else 'moderate'
                    }
                    alerts.append(alert)
        
        return alerts
    
    def _identify_peak_stress_times(self, predictions):
        """Identify peak stress times in forecasts"""
        peak_times = []
        
        for i, forecast in enumerate(predictions):
            # Find peaks
            peaks, properties = find_peaks(forecast, height=6.0, distance=3)
            
            for peak in peaks:
                peak_times.append({
                    'sample': i,
                    'hour': peak,
                    'stress_level': forecast[peak],
                    'prominence': properties['peak_heights'][list(peaks).index(peak)] if len(properties['peak_heights']) > list(peaks).index(peak) else 0
                })
        
        return peak_times
    
    def evaluate_forecasts(self, y_true, y_pred, model_name):
        """
        Comprehensive forecast evaluation
        """
        print(f"📊 Evaluating {model_name} forecasts...")
        
        # Reshape for evaluation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        # Forecast-specific metrics
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        
        # Directional accuracy (for trend prediction)
        y_true_diff = np.diff(y_true, axis=1)
        y_pred_diff = np.diff(y_pred, axis=1)
        
        directional_accuracy = np.mean(
            np.sign(y_true_diff.flatten()) == np.sign(y_pred_diff.flatten())
        )
        
        # Peak prediction accuracy
        peak_accuracy = self._evaluate_peak_prediction(y_true, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'peak_accuracy': peak_accuracy
        }
        
        print(f"✅ Evaluation completed:")
        print(f"   • MAE: {mae:.4f}")
        print(f"   • RMSE: {rmse:.4f}")
        print(f"   • R²: {r2:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        print(f"   • Directional Accuracy: {directional_accuracy:.4f}")
        print(f"   • Peak Accuracy: {peak_accuracy:.4f}")
        
        return metrics
    
    def _evaluate_peak_prediction(self, y_true, y_pred, threshold=7.0):
        """Evaluate accuracy of peak stress prediction"""
        peak_accuracies = []
        
        for i in range(len(y_true)):
            true_peaks = find_peaks(y_true[i], height=threshold)[0]
            pred_peaks = find_peaks(y_pred[i], height=threshold)[0]
            
            if len(true_peaks) == 0 and len(pred_peaks) == 0:
                peak_accuracies.append(1.0)  # Perfect prediction of no peaks
            elif len(true_peaks) == 0:
                peak_accuracies.append(0.0)  # False positive peaks
            else:
                # Calculate how well predicted peaks match true peaks
                matches = 0
                for true_peak in true_peaks:
                    # Allow ±2 hour tolerance
                    if any(abs(pred_peak - true_peak) <= 2 for pred_peak in pred_peaks):
                        matches += 1
                
                peak_accuracies.append(matches / len(true_peaks))
        
        return np.mean(peak_accuracies)
    
    def visualize_forecasts(self, y_true, y_pred, uncertainty=None, sample_idx=0):
        """
        Create comprehensive forecast visualizations
        """
        print(f"📊 Creating forecast visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stress Forecasting Results', fontsize=16, fontweight='bold')
        
        # 1. Sample forecast with uncertainty
        hours = np.arange(self.forecast_hours)
        
        axes[0, 0].plot(hours, y_true[sample_idx], 'b-', label='True Stress', linewidth=2)
        axes[0, 0].plot(hours, y_pred[sample_idx], 'r--', label='Predicted Stress', linewidth=2)
        
        if uncertainty is not None:
            axes[0, 0].fill_between(
                hours,
                y_pred[sample_idx] - uncertainty[sample_idx],
                y_pred[sample_idx] + uncertainty[sample_idx],
                alpha=0.3, color='red', label='Uncertainty'
            )
        
        axes[0, 0].axhline(y=7, color='orange', linestyle=':', label='High Stress Threshold')
        axes[0, 0].set_title('Sample Forecast with Uncertainty')
        axes[0, 0].set_xlabel('Hours Ahead')
        axes[0, 0].set_ylabel('Stress Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Prediction vs True scatter plot
        axes[0, 1].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 1].set_title('Predicted vs True Stress')
        axes[0, 1].set_xlabel('True Stress Level')
        axes[0, 1].set_ylabel('Predicted Stress Level')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Forecast horizon accuracy
        horizon_mae = []
        for h in range(self.forecast_hours):
            mae_h = mean_absolute_error(y_true[:, h], y_pred[:, h])
            horizon_mae.append(mae_h)
        
        axes[1, 0].plot(hours, horizon_mae, 'g-', marker='o', linewidth=2)
        axes[1, 0].set_title('Prediction Accuracy by Forecast Horizon')
        axes[1, 0].set_xlabel('Hours Ahead')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Stress level distribution comparison
        axes[1, 1].hist(y_true.flatten(), bins=30, alpha=0.7, label='True', density=True)
        axes[1, 1].hist(y_pred.flatten(), bins=30, alpha=0.7, label='Predicted', density=True)
        axes[1, 1].set_title('Stress Level Distribution')
        axes[1, 1].set_xlabel('Stress Level')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_stress_forecasting.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'temporal_stress_forecasting.png'")

def main():
    """Demonstration of temporal stress forecasting"""
    print("🚀 Starting Temporal Stress Forecasting Demo")
    print("=" * 60)
    
    # Initialize forecasting system
    forecaster = TemporalStressForecasting(lookback_hours=24, forecast_hours=48)
    
    # Load sample data
    try:
        df = pd.read_csv('data/sequential_behavioral_health_data_30days.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        print(f"📊 Loaded sample data: {len(df)} records")
        
        # Prepare temporal data
        df_temporal = forecaster.prepare_temporal_data(df)
        
        # Create sequences
        X, y, feature_cols = forecaster.create_sequences(df_temporal)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"📦 Data split:")
        print(f"   • Training: {len(X_train)} sequences")
        print(f"   • Validation: {len(X_val)} sequences")
        
        # Train models
        training_history = forecaster.train_forecasting_models(X_train, y_train, X_val, y_val)
        
        # Generate forecasts
        predictions, uncertainty, metadata = forecaster.forecast_stress_patterns(X_val[:50])  # Demo subset
        
        # Evaluate forecasts
        for model_name in forecaster.models.keys():
            model_predictions, _, _ = forecaster.forecast_stress_patterns(X_val[:50], model_name)
            metrics = forecaster.evaluate_forecasts(y_val[:50], model_predictions, model_name)
        
        # Visualize results
        forecaster.visualize_forecasts(y_val[:50], predictions, uncertainty)
        
        # Print risk alerts
        if metadata['risk_alerts']:
            print(f"\n⚠️ Risk Alerts Generated:")
            for alert in metadata['risk_alerts'][:5]:  # Show first 5
                print(f"   • Sample {alert['sample']}: High stress from hour {alert['start_hour']} to {alert['end_hour']}")
                print(f"     Duration: {alert['duration']} hours, Max stress: {alert['max_stress']:.1f}")
        
        print("\n🎉 Temporal Forecasting Demo Completed!")
        print("💡 Key Features Demonstrated:")
        print("   ├── Multi-step stress prediction (48-hour horizon)")
        print("   ├── Uncertainty quantification")
        print("   ├── Risk alert generation")
        print("   ├── Peak stress identification")
        print("   └── Comprehensive evaluation metrics")
        
    except FileNotFoundError:
        print("❌ Sample data file not found. Please ensure data file exists.")
        print("💡 Demo would work with temporal health data.")

if __name__ == "__main__":
    main()
