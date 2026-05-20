"""
VALIDATE ACCELEROMETER DATA với HAR MODEL
Kiểm tra độ chính xác của dữ liệu accelerometer đã generate
với HAR model classificator_model.keras (96% accuracy)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os
from sklearn.preprocessing import StandardScaler

# Add HAR directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'HAR'))

from config import LABELS_NAMES, SEGMENT_TIME_SIZE, TIME_STEP, N_FEATURES
from preprocessing import one_hot_encode, label_position

class AccelerometerValidator:
    """Validate accelerometer data với HAR model"""
    
    def __init__(self):
        # Load trained HAR model
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'HAR', 'model', 'classificator_model.keras')
        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler = StandardScaler()
        
        print(f" Loaded HAR model from: {self.model_path}")
        print(f" Model input shape: {self.model.input_shape}")
        print(f" Model classes: {LABELS_NAMES}")

    def prepare_data_for_har(self, df):
        """
        Chuẩn bị data theo định dạng HAR model expects
        """
        print(f"\n === PREPARING DATA FOR HAR VALIDATION ===")
        
        # Sort by timestamp to ensure proper sequence
        df_sorted = df.sort_values('Timestamp').copy()
        print(f" Total samples: {len(df_sorted)}")
        
        # Create HAR format dataframe
        har_data = pd.DataFrame({
            'user': [1] * len(df_sorted),  # Dummy user ID
            'activity': df_sorted['Activity'],
            'timestamp': range(len(df_sorted)),
            'x-axis': df_sorted['Accelerometer_X'],
            'y-axis': df_sorted['Accelerometer_Y'], 
            'z-axis': df_sorted['Accelerometer_Z']
        })
        
        print(f" Created HAR format data")
        return har_data

    def create_segments(self, data):
        """
        Tạo segments theo sliding window như HAR model training
        """
        print(f"\n === CREATING HAR SEGMENTS ===")
        print(f" SEGMENT_TIME_SIZE: {SEGMENT_TIME_SIZE}")
        print(f" TIME_STEP: {TIME_STEP}")
        
        data_segments = []
        labels = []
        segment_info = []
        
        # Group by activity để tạo continuous segments
        activity_groups = data.groupby('activity')
        
        for activity, group in activity_groups:
            if len(group) < SEGMENT_TIME_SIZE:
                print(f"  Skipping {activity}: only {len(group)} samples (need {SEGMENT_TIME_SIZE})")
                continue
                
            # Create segments within this activity group
            for i in range(0, len(group) - SEGMENT_TIME_SIZE + 1, TIME_STEP):
                x = group['x-axis'].iloc[i: i + SEGMENT_TIME_SIZE].values
                y = group['y-axis'].iloc[i: i + SEGMENT_TIME_SIZE].values
                z = group['z-axis'].iloc[i: i + SEGMENT_TIME_SIZE].values
                
                # Stack and transpose to match HAR model format
                segment = np.array([x, y, z]).T  # Shape: (180, 3)
                data_segments.append(segment)
                
                # Label is the activity for this segment
                labels.append(activity)
                segment_info.append({
                    'activity': activity,
                    'start_idx': i,
                    'samples': len(group),
                    'segment_mean_x': np.mean(x),
                    'segment_mean_y': np.mean(y),
                    'segment_mean_z': np.mean(z)
                })
        
        # Convert to numpy arrays
        X = np.array(data_segments, dtype=np.float32)  # Shape: (n_segments, 180, 3)
        
        # One-hot encode labels
        y = one_hot_encode(labels)
        
        print(f" Created {len(X)} segments")
        print(f" Segment shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        
        # Show segment distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n Segment distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count} segments")
            
        return X, y, labels, segment_info

    def normalize_data(self, X):
        """Normalize data như training process"""
        print(f"\n === NORMALIZING DATA ===")
        
        # Reshape to 2D for StandardScaler
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])  # (n_segments * 180, 3)
        
        # Fit and transform
        X_normalized = self.scaler.fit_transform(X_reshaped)
        
        # Reshape back
        X_normalized = X_normalized.reshape(original_shape)
        
        print(f" Normalized data shape: {X_normalized.shape}")
        print(f" Mean after normalization: {np.mean(X_normalized, axis=(0,1))}")
        print(f" Std after normalization: {np.std(X_normalized, axis=(0,1))}")
        
        return X_normalized

    def validate_with_har_model(self, X, y, labels, segment_info):
        """
        Validate với HAR model và tính accuracy
        """
        print(f"\n === HAR MODEL VALIDATION ===")
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Convert predictions to labels
        predicted_labels = [LABELS_NAMES[np.argmax(pred)] for pred in predictions]
        actual_labels = labels
        
        # Calculate overall accuracy
        correct_predictions = sum(1 for actual, pred in zip(actual_labels, predicted_labels) if actual == pred)
        overall_accuracy = correct_predictions / len(actual_labels)
        
        print(f" OVERALL ACCURACY: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
        
        # Per-activity accuracy
        print(f"\n === PER-ACTIVITY ACCURACY ===")
        activity_stats = {}
        
        for activity in LABELS_NAMES:
            activity_indices = [i for i, label in enumerate(actual_labels) if label == activity]
            if not activity_indices:
                continue
                
            activity_correct = sum(1 for i in activity_indices if predicted_labels[i] == activity)
            activity_accuracy = activity_correct / len(activity_indices)
            
            activity_stats[activity] = {
                'total_segments': len(activity_indices),
                'correct_predictions': activity_correct,
                'accuracy': activity_accuracy
            }
            
            print(f"  {activity:12}: {activity_correct:3d}/{len(activity_indices):3d} = {activity_accuracy:.3f} ({activity_accuracy*100:.1f}%)")
        
        # Confusion matrix analysis
        print(f"\n === DETAILED PREDICTION ANALYSIS ===")
        
        # Show some examples
        print(f"\nSample predictions (first 10):")
        for i in range(min(10, len(actual_labels))):
            actual = actual_labels[i]
            predicted = predicted_labels[i]
            confidence = np.max(predictions[i])
            status = "Correct" if actual == predicted else "Incorrect"
            
            print(f"  {status} Actual: {actual:12} | Predicted: {predicted:12} | Confidence: {confidence:.3f}")
        
        # Find most problematic cases
        print(f"\n === MOST PROBLEMATIC CASES ===")
        wrong_predictions = []
        for i, (actual, predicted) in enumerate(zip(actual_labels, predicted_labels)):
            if actual != predicted:
                confidence = np.max(predictions[i])
                wrong_predictions.append({
                    'index': i,
                    'actual': actual,
                    'predicted': predicted,
                    'confidence': confidence,
                    'info': segment_info[i]
                })
        
        # Sort by confidence (most confident wrong predictions)
        wrong_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Found {len(wrong_predictions)} wrong predictions")
        print(f"Top 5 most confident wrong predictions:")
        for i, wp in enumerate(wrong_predictions[:5]):
            print(f"  {i+1}. {wp['actual']} → {wp['predicted']} (conf: {wp['confidence']:.3f})")
            info = wp['info']
            print(f"      Segment means: X={info['segment_mean_x']:.3f}, Y={info['segment_mean_y']:.3f}, Z={info['segment_mean_z']:.3f}")
        
        return overall_accuracy, activity_stats, predictions

    def analyze_accelerometer_quality(self, df):
        """
        Phân tích chất lượng data accelerometer
        """
        print(f"\n === ACCELEROMETER DATA QUALITY ANALYSIS ===")
        
        # Basic statistics
        for axis in ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']:
            values = df[axis]
            print(f"{axis}:")
            print(f"  Range: {values.min():.3f} to {values.max():.3f}")
            print(f"  Mean: {values.mean():.3f} ± {values.std():.3f}")
            print(f"  Outliers (>3σ): {len(values[abs(values - values.mean()) > 3 * values.std()])}")
        
        # Per-activity statistics
        print(f"\n === PER-ACTIVITY ACCELEROMETER STATS ===")
        for activity in df['Activity'].unique():
            activity_data = df[df['Activity'] == activity]
            magnitude = np.sqrt(
                activity_data['Accelerometer_X']**2 + 
                activity_data['Accelerometer_Y']**2 + 
                activity_data['Accelerometer_Z']**2
            )
            print(f"{activity:12}: Magnitude {magnitude.min():.1f}-{magnitude.max():.1f} (avg: {magnitude.mean():.1f})")

def main():
    """Main validation function"""
    print(" === HAR MODEL VALIDATION FOR GENERATED ACCELEROMETER DATA ===")
    
    # Load generated dataset - USE 23-FIELD VERSION (has accelerometer + optimized features)
    data_path = 'data/optimized_health_data_23features.csv'
    print(f" Loading data from: {data_path}")
    print(f"  Using 23-field dataset (Accelerometer X,Y,Z + 20 optimized features)")
    print(f"  This is the RECOMMENDED dataset for research")
    
    try:
        df = pd.read_csv(data_path)
        print(f" Loaded {len(df)} samples")
        print(f" Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f" Activities: {df['Activity'].unique()}")
        
    except Exception as e:
        print(f" Error loading data: {e}")
        return
    
    # Initialize validator
    try:
        validator = AccelerometerValidator()
    except Exception as e:
        print(f" Error loading HAR model: {e}")
        return
    
    # Analyze data quality
    validator.analyze_accelerometer_quality(df)
    
    # Prepare data for HAR
    har_data = validator.prepare_data_for_har(df)
    
    # Create segments
    try:
        X, y, labels, segment_info = validator.create_segments(har_data)
        
        if len(X) == 0:
            print(" No valid segments created. Check data continuity.")
            return
            
    except Exception as e:
        print(f" Error creating segments: {e}")
        return
    
    # Normalize data
    X_normalized = validator.normalize_data(X)
    
    # Validate with HAR model
    try:
        accuracy, activity_stats, predictions = validator.validate_with_har_model(
            X_normalized, y, labels, segment_info
        )
        
        print(f"\n === FINAL ASSESSMENT ===")
        if accuracy >= 0.85:
            print(f" EXCELLENT: {accuracy:.1%} accuracy - Generated data is HAR-compatible!")
        elif accuracy >= 0.70:
            print(f" GOOD: {accuracy:.1%} accuracy - Generated data is acceptable for HAR")
        elif accuracy >= 0.50:
            print(f"  FAIR: {accuracy:.1%} accuracy - Generated data needs improvement")
        else:
            print(f" POOR: {accuracy:.1%} accuracy - Generated data has significant issues")
            
        print(f"\n === RECOMMENDATIONS ===")
        if accuracy >= 0.85:
            print(" Data is ready for advanced stress prediction modeling")
            print(" Accelerometer patterns are realistic and HAR-compatible")
        else:
            print(" Consider improving accelerometer generation logic")
            print(" Check for activity-specific acceleration patterns")
            
    except Exception as e:
        print(f" Error during validation: {e}")
        return

if __name__ == "__main__":
    main()
