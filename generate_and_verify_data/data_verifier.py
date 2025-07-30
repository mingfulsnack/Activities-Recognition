"""
DATA VERIFICATION TOOL
======================

Verify lại data sau khi đã fix để đảm bảo chất lượng.

Author: Research Team  
Date: July 30, 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import config từ HAR model gốc
from config import LABELS_NAMES, SEGMENT_TIME_SIZE, N_FEATURES

class DataVerifier:
    """
    Verify data quality sau khi đã fix
    """
    
    def __init__(self):
        """Initialize với HAR model"""
        print("🔧 Initializing Data Verifier...")
        
        # Load HAR model
        try:
            self.har_model = tf.keras.models.load_model('classificator_model.keras')
            print("✅ Loaded HAR model successfully!")
        except Exception as e:
            print(f"❌ Error loading HAR model: {e}")
            raise
        
        self.scaler = StandardScaler()

    def prepare_sequence_for_prediction(self, accelerometer_sequence):
        """Prepare sequence for HAR model prediction"""
        if len(accelerometer_sequence) < SEGMENT_TIME_SIZE:
            padding = SEGMENT_TIME_SIZE - len(accelerometer_sequence)
            accelerometer_sequence = np.pad(
                accelerometer_sequence, 
                ((0, padding), (0, 0)), 
                mode='edge'
            )
        elif len(accelerometer_sequence) > SEGMENT_TIME_SIZE:
            accelerometer_sequence = accelerometer_sequence[-SEGMENT_TIME_SIZE:]
        
        # Normalize
        original_shape = accelerometer_sequence.shape
        sequence_reshaped = accelerometer_sequence.reshape(-1, N_FEATURES)
        sequence_normalized = self.scaler.fit_transform(sequence_reshaped)
        sequence_final = sequence_normalized.reshape(original_shape)
        
        return sequence_final.reshape(1, SEGMENT_TIME_SIZE, N_FEATURES)

    def predict_activity_from_sequence(self, accelerometer_sequence):
        """Predict activity using HAR model"""
        input_sequence = self.prepare_sequence_for_prediction(accelerometer_sequence)
        
        predictions = self.har_model.predict(input_sequence, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_activity = LABELS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_activity, confidence, predictions[0]

    def verify_data_comprehensive(self, data_path, sample_size=200):
        """
        Comprehensive verification của data
        """
        print(f"\n🔍 COMPREHENSIVE DATA VERIFICATION")
        print("=" * 50)
        
        # Load data
        print(f"📂 Loading data: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df):,} records")
        
        # Basic info
        print(f"\n📊 BASIC DATA INFO:")
        print(f"   Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Data types: {df.dtypes.to_dict()}")
        
        # Activity distribution
        print(f"\n🏷️ ACTIVITY DISTRIBUTION:")
        activity_counts = df['Activity'].value_counts()
        for activity, count in activity_counts.items():
            percentage = count / len(df) * 100
            print(f"   {activity}: {count:,} ({percentage:.1f}%)")
        
        # HAR Confidence distribution
        if 'HAR_Confidence' in df.columns:
            print(f"\n🎯 HAR CONFIDENCE DISTRIBUTION:")
            confidence_stats = df['HAR_Confidence'].describe()
            print(f"   Mean: {confidence_stats['mean']:.3f}")
            print(f"   Std: {confidence_stats['std']:.3f}")
            print(f"   Min: {confidence_stats['min']:.3f}")
            print(f"   Max: {confidence_stats['max']:.3f}")
            
            # Confidence by activity
            print(f"\n📈 CONFIDENCE BY ACTIVITY:")
            for activity in LABELS_NAMES:
                if activity in activity_counts.index:
                    activity_data = df[df['Activity'] == activity]
                    avg_conf = activity_data['HAR_Confidence'].mean()
                    print(f"   {activity}: {avg_conf:.3f}")
        
        # Sample verification với HAR model
        print(f"\n🧪 SAMPLE VERIFICATION WITH HAR MODEL:")
        print(f"Testing {sample_size} random sequences...")
        
        # Random sampling
        total_sequences = len(df) // SEGMENT_TIME_SIZE
        sample_indices = np.random.choice(total_sequences, min(sample_size, total_sequences), replace=False)
        
        verification_results = {
            'correct_predictions': 0,
            'total_tested': 0,
            'confidence_scores': [],
            'activity_accuracy': {activity: {'correct': 0, 'total': 0} for activity in LABELS_NAMES}
        }
        
        for seq_idx in sample_indices:
            start_idx = seq_idx * SEGMENT_TIME_SIZE
            end_idx = start_idx + SEGMENT_TIME_SIZE
            
            if end_idx > len(df):
                continue
                
            # Extract sequence
            sequence_data = df.iloc[start_idx:end_idx]
            accelerometer_seq = sequence_data[['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']].values
            current_activity = sequence_data['Activity'].iloc[0]  # Assume consistent activity in sequence
            
            try:
                # Predict với HAR model
                predicted_activity, confidence, all_probs = self.predict_activity_from_sequence(accelerometer_seq)
                
                verification_results['total_tested'] += 1
                verification_results['confidence_scores'].append(confidence)
                
                # Check accuracy
                is_correct = predicted_activity == current_activity
                if is_correct:
                    verification_results['correct_predictions'] += 1
                    
                # Track by activity
                if current_activity in verification_results['activity_accuracy']:
                    verification_results['activity_accuracy'][current_activity]['total'] += 1
                    if is_correct:
                        verification_results['activity_accuracy'][current_activity]['correct'] += 1
                        
            except Exception as e:
                print(f"⚠️ Error testing sequence {start_idx}-{end_idx}: {e}")
                continue
        
        # Calculate overall accuracy
        if verification_results['total_tested'] > 0:
            overall_accuracy = verification_results['correct_predictions'] / verification_results['total_tested']
            avg_confidence = np.mean(verification_results['confidence_scores'])
            
            print(f"\n📈 VERIFICATION RESULTS:")
            print(f"✅ Sequences tested: {verification_results['total_tested']}")
            print(f"✓ Correct predictions: {verification_results['correct_predictions']}")
            print(f"🎯 Overall accuracy: {overall_accuracy:.2%}")
            print(f"🔮 Average confidence: {avg_confidence:.3f}")
            
            # Activity-wise accuracy
            print(f"\n📊 ACCURACY BY ACTIVITY:")
            for activity, stats in verification_results['activity_accuracy'].items():
                if stats['total'] > 0:
                    activity_acc = stats['correct'] / stats['total']
                    print(f"   {activity}: {activity_acc:.2%} ({stats['correct']}/{stats['total']})")
                    
        else:
            print("❌ No sequences could be verified!")
            
        return verification_results

    def check_data_consistency(self, data_path):
        """
        Check data consistency và quality
        """
        print(f"\n🔧 DATA CONSISTENCY CHECK")
        print("=" * 40)
        
        df = pd.read_csv(data_path)
        
        consistency_issues = []
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"⚠️ Missing values found:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"   {col}: {count} missing")
                consistency_issues.append(f"Missing values in {col}")
        else:
            print(f"✅ No missing values")
            
        # Check accelerometer ranges
        print(f"\n📊 ACCELEROMETER DATA RANGES:")
        for axis in ['X', 'Y', 'Z']:
            col = f'Accelerometer_{axis}'
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"   {axis}: [{min_val:.3f}, {max_val:.3f}]")
                
                # Check for unrealistic values
                if abs(min_val) > 50 or abs(max_val) > 50:
                    print(f"   ⚠️ Unrealistic {axis} values detected!")
                    consistency_issues.append(f"Unrealistic {axis} values")
        
        # Check timestamp consistency
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            time_diffs = df['Timestamp'].diff().dropna()
            
            print(f"\n⏰ TIMESTAMP ANALYSIS:")
            print(f"   Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
            print(f"   Total duration: {df['Timestamp'].max() - df['Timestamp'].min()}")
            
            # Check for gaps
            large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
            if len(large_gaps) > 0:
                print(f"   ⚠️ Found {len(large_gaps)} time gaps > 2 hours")
                consistency_issues.append("Large time gaps detected")
            else:
                print(f"   ✅ No large time gaps detected")
        
        # Check activity labels
        print(f"\n🏷️ ACTIVITY LABEL CHECK:")
        unique_activities = df['Activity'].unique()
        invalid_activities = [act for act in unique_activities if act not in LABELS_NAMES]
        
        if invalid_activities:
            print(f"   ⚠️ Invalid activities found: {invalid_activities}")
            consistency_issues.append("Invalid activity labels")
        else:
            print(f"   ✅ All activities are valid HAR labels")
            
        # Summary
        print(f"\n📋 CONSISTENCY SUMMARY:")
        if consistency_issues:
            print(f"   ⚠️ Issues found: {len(consistency_issues)}")
            for issue in consistency_issues:
                print(f"      - {issue}")
        else:
            print(f"   ✅ No consistency issues detected!")
            
        return consistency_issues

    def compare_with_backup(self, current_path, backup_path):
        """
        So sánh data hiện tại với backup
        """
        print(f"\n🔄 COMPARISON WITH BACKUP")
        print("=" * 40)
        
        try:
            df_current = pd.read_csv(current_path)
            df_backup = pd.read_csv(backup_path)
            
            print(f"📊 BASIC COMPARISON:")
            print(f"   Current records: {len(df_current):,}")
            print(f"   Backup records: {len(df_backup):,}")
            print(f"   Record difference: {len(df_current) - len(df_backup):,}")
            
            # Activity distribution comparison
            print(f"\n🏷️ ACTIVITY DISTRIBUTION CHANGES:")
            current_counts = df_current['Activity'].value_counts()
            backup_counts = df_backup['Activity'].value_counts()
            
            all_activities = set(current_counts.index) | set(backup_counts.index)
            
            for activity in sorted(all_activities):
                current_count = current_counts.get(activity, 0)
                backup_count = backup_counts.get(activity, 0)
                change = current_count - backup_count
                
                current_pct = current_count / len(df_current) * 100
                backup_pct = backup_count / len(df_backup) * 100
                
                change_sign = "+" if change > 0 else ""
                print(f"   {activity}:")
                print(f"      Before: {backup_count:,} ({backup_pct:.1f}%)")
                print(f"      After:  {current_count:,} ({current_pct:.1f}%)")
                print(f"      Change: {change_sign}{change:,}")
            
            # New columns
            new_columns = set(df_current.columns) - set(df_backup.columns)
            if new_columns:
                print(f"\n📋 NEW COLUMNS ADDED:")
                for col in new_columns:
                    print(f"   + {col}")
            
        except Exception as e:
            print(f"❌ Error comparing with backup: {e}")

def main():
    """Main verification process"""
    print("🚀 Starting Data Verification Process")
    print("=" * 60)
    
    # Initialize verifier
    verifier = DataVerifier()
    
    # Paths
    current_data_path = 'data/sequential_behavioral_health_data_30days.csv'
    backup_data_path = 'data/sequential_behavioral_health_data_30days_backup.csv'
    
    # 1. Comprehensive verification
    verification_results = verifier.verify_data_comprehensive(current_data_path, sample_size=300)
    
    # 2. Data consistency check
    consistency_issues = verifier.check_data_consistency(current_data_path)
    
    # 3. Compare with backup
    verifier.compare_with_backup(current_data_path, backup_data_path)
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 30)
    
    if verification_results['total_tested'] > 0:
        accuracy = verification_results['correct_predictions'] / verification_results['total_tested']
        
        if accuracy >= 0.95:
            quality_score = "🟢 EXCELLENT"
        elif accuracy >= 0.85:
            quality_score = "🟡 GOOD"
        elif accuracy >= 0.70:
            quality_score = "🟠 ACCEPTABLE"
        else:
            quality_score = "🔴 NEEDS IMPROVEMENT"
            
        print(f"📊 Data Quality: {quality_score}")
        print(f"🎯 HAR Accuracy: {accuracy:.2%}")
        print(f"🔧 Consistency Issues: {len(consistency_issues)}")
        
        if accuracy >= 0.85 and len(consistency_issues) == 0:
            print(f"\n✅ DATA VERIFICATION PASSED!")
            print(f"📁 Data is ready for Phase 1.5 implementation")
        else:
            print(f"\n⚠️ DATA NEEDS ATTENTION")
            print(f"📝 Consider additional fixes before proceeding")
    
    return verifier, verification_results

if __name__ == "__main__":
    verifier_instance, results = main()
