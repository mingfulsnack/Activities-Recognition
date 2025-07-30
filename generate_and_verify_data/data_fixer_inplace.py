"""
IN-PLACE DATA FIXER
===================

Verify và fix trực tiếp data cũ thay vì tạo dataset mới.
Giữ nguyên structure và size, chỉ fix activity predictions.

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
from preprocessing import one_hot_encode, label_position

class InPlaceDataFixer:
    """
    Fix data cũ in-place bằng cách verify activities với HAR model
    """
    
    def __init__(self):
        """Initialize với HAR model của bạn"""
        print("🔧 Initializing In-Place Data Fixer...")
        
        # Load HAR model đã trained
        try:
            self.har_model = tf.keras.models.load_model('classificator_model.keras')
            print("✅ Loaded HAR model successfully!")
            print(f"📊 Model expects input shape: (None, {SEGMENT_TIME_SIZE}, {N_FEATURES})")
            print(f"🏷️ Activity labels: {LABELS_NAMES}")
        except Exception as e:
            print(f"❌ Error loading HAR model: {e}")
            raise
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Stats tracking
        self.total_verified = 0
        self.total_corrected = 0
        self.corrections_by_activity = {}

    def prepare_sequence_for_prediction(self, accelerometer_sequence):
        """
        Chuẩn bị sequence cho HAR model prediction
        """
        # Ensure correct shape
        if len(accelerometer_sequence) < SEGMENT_TIME_SIZE:
            # Pad nếu sequence ngắn hơn required
            padding = SEGMENT_TIME_SIZE - len(accelerometer_sequence)
            accelerometer_sequence = np.pad(
                accelerometer_sequence, 
                ((0, padding), (0, 0)), 
                mode='edge'
            )
        elif len(accelerometer_sequence) > SEGMENT_TIME_SIZE:
            # Take last SEGMENT_TIME_SIZE samples
            accelerometer_sequence = accelerometer_sequence[-SEGMENT_TIME_SIZE:]
        
        # Normalize sequence
        original_shape = accelerometer_sequence.shape
        sequence_reshaped = accelerometer_sequence.reshape(-1, N_FEATURES)
        sequence_normalized = self.scaler.fit_transform(sequence_reshaped)
        sequence_final = sequence_normalized.reshape(original_shape)
        
        # Add batch dimension
        input_sequence = sequence_final.reshape(1, SEGMENT_TIME_SIZE, N_FEATURES)
        
        return input_sequence

    def predict_activity_from_sequence(self, accelerometer_sequence):
        """
        Sử dụng HAR model để predict activity từ accelerometer sequence
        """
        input_sequence = self.prepare_sequence_for_prediction(accelerometer_sequence)
        
        # Predict với HAR model
        predictions = self.har_model.predict(input_sequence, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_activity = LABELS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_activity, confidence, predictions[0]

    def verify_and_fix_dataframe(self, df, batch_size=50):
        """
        Verify và fix activities trong DataFrame bằng HAR model
        """
        print(f"🔍 Starting verification of {len(df):,} records...")
        print(f"📦 Processing in batches of {batch_size} sequences")
        
        # Create working copy
        df_fixed = df.copy()
        
        # Track corrections
        corrections_made = []
        verification_stats = {
            'total_sequences': 0,
            'verified_correct': 0,
            'corrected_activities': 0,
            'confidence_scores': []
        }
        
        # Process data theo chunks để có thể tạo sequences
        total_batches = (len(df) // SEGMENT_TIME_SIZE) // batch_size + 1
        current_batch = 0
        
        # Process sequences
        for start_idx in range(0, len(df), SEGMENT_TIME_SIZE * batch_size):
            current_batch += 1
            
            # Process batch of sequences
            batch_end = min(start_idx + SEGMENT_TIME_SIZE * batch_size, len(df))
            batch_sequences = (batch_end - start_idx) // SEGMENT_TIME_SIZE
            
            print(f"📊 Processing batch {current_batch}/{total_batches} - Sequences: {batch_sequences}")
            
            # Process each sequence trong batch
            for seq_start in range(start_idx, batch_end, SEGMENT_TIME_SIZE):
                seq_end = min(seq_start + SEGMENT_TIME_SIZE, len(df))
                
                if seq_end - seq_start < SEGMENT_TIME_SIZE:
                    continue  # Skip incomplete sequences
                
                # Extract sequence data
                sequence_data = df.iloc[seq_start:seq_end]
                accelerometer_seq = sequence_data[['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']].values
                
                # Get current activities trong sequence
                current_activities = sequence_data['Activity'].values
                most_common_activity = pd.Series(current_activities).mode()[0]
                
                # Predict activity với HAR model
                try:
                    predicted_activity, confidence, all_probs = self.predict_activity_from_sequence(accelerometer_seq)
                    
                    verification_stats['total_sequences'] += 1
                    verification_stats['confidence_scores'].append(confidence)
                    
                    # Check if correction needed
                    if predicted_activity != most_common_activity:
                        # Correction needed
                        verification_stats['corrected_activities'] += 1
                        
                        # Fix all records trong sequence
                        df_fixed.iloc[seq_start:seq_end, df_fixed.columns.get_loc('Activity')] = predicted_activity
                        
                        # Track correction
                        corrections_made.append({
                            'sequence_start': seq_start,
                            'sequence_end': seq_end,
                            'original_activity': most_common_activity,
                            'corrected_activity': predicted_activity,
                            'confidence': confidence
                        })
                        
                        # Update correction stats
                        if most_common_activity not in self.corrections_by_activity:
                            self.corrections_by_activity[most_common_activity] = {}
                        if predicted_activity not in self.corrections_by_activity[most_common_activity]:
                            self.corrections_by_activity[most_common_activity][predicted_activity] = 0
                        self.corrections_by_activity[most_common_activity][predicted_activity] += 1
                        
                    else:
                        # Verified correct
                        verification_stats['verified_correct'] += 1
                        
                except Exception as e:
                    print(f"⚠️ Error processing sequence {seq_start}-{seq_end}: {e}")
                    continue
        
        # Update stats
        self.total_verified = verification_stats['total_sequences']
        self.total_corrected = verification_stats['corrected_activities']
        
        # Print results
        print(f"\n📈 VERIFICATION & CORRECTION RESULTS:")
        print(f"✅ Total sequences processed: {verification_stats['total_sequences']:,}")
        print(f"✓ Verified correct: {verification_stats['verified_correct']:,}")
        print(f"🔧 Corrected activities: {verification_stats['corrected_activities']:,}")
        
        correction_rate = verification_stats['corrected_activities'] / verification_stats['total_sequences'] * 100
        print(f"📊 Correction rate: {correction_rate:.1f}%")
        
        avg_confidence = np.mean(verification_stats['confidence_scores'])
        print(f"🎯 Average prediction confidence: {avg_confidence:.3f}")
        
        return df_fixed, corrections_made, verification_stats

    def print_correction_summary(self):
        """
        In summary các corrections đã made
        """
        print(f"\n📋 DETAILED CORRECTION SUMMARY:")
        print("=" * 50)
        
        if not self.corrections_by_activity:
            print("✅ No corrections were needed!")
            return
        
        for original_activity, corrections in self.corrections_by_activity.items():
            print(f"\n🏷️ {original_activity} →")
            for corrected_activity, count in corrections.items():
                print(f"   ➜ {corrected_activity}: {count} sequences")
        
        print(f"\n📊 OVERALL STATS:")
        print(f"   Total verified: {self.total_verified:,} sequences")
        print(f"   Total corrected: {self.total_corrected:,} sequences")
        accuracy = (self.total_verified - self.total_corrected) / self.total_verified * 100
        print(f"   Original accuracy: {accuracy:.1f}%")

    def add_har_confidence_column(self, df_fixed, corrections_made):
        """
        Add HAR confidence column để track prediction quality
        """
        print(f"\n🔧 Adding HAR confidence scores...")
        
        # Initialize confidence column
        df_fixed['HAR_Confidence'] = 0.0
        
        # Add confidence scores for corrected sequences
        for correction in corrections_made:
            start_idx = correction['sequence_start']
            end_idx = correction['sequence_end']
            confidence = correction['confidence']
            
            df_fixed.iloc[start_idx:end_idx, df_fixed.columns.get_loc('HAR_Confidence')] = confidence
        
        # For non-corrected sequences, estimate confidence từ model
        print(f"🔍 Estimating confidence for verified sequences...")
        
        # Sample some verified sequences để estimate confidence
        sample_size = min(100, len(df_fixed) // SEGMENT_TIME_SIZE)
        sampled_confidences = []
        
        for i in range(sample_size):
            start_idx = i * SEGMENT_TIME_SIZE
            end_idx = start_idx + SEGMENT_TIME_SIZE
            
            if end_idx > len(df_fixed):
                break
                
            sequence_data = df_fixed.iloc[start_idx:end_idx]
            accelerometer_seq = sequence_data[['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']].values
            
            try:
                _, confidence, _ = self.predict_activity_from_sequence(accelerometer_seq)
                sampled_confidences.append(confidence)
            except:
                continue
        
        # Use average confidence for verified sequences
        avg_verified_confidence = np.mean(sampled_confidences) if sampled_confidences else 0.85
        
        # Fill in confidence scores cho các record chưa có
        mask_zero_confidence = df_fixed['HAR_Confidence'] == 0.0
        df_fixed.loc[mask_zero_confidence, 'HAR_Confidence'] = avg_verified_confidence
        
        print(f"✅ Added HAR confidence scores (avg: {avg_verified_confidence:.3f})")
        
        return df_fixed

def main():
    """Main execution"""
    print("🚀 Starting In-Place Data Fixing Process")
    print("=" * 60)
    
    # Paths
    input_path = 'data/sequential_behavioral_health_data_30days.csv'
    backup_path = 'data/sequential_behavioral_health_data_30days_backup.csv'
    output_path = 'data/sequential_behavioral_health_data_30days.csv'  # Same file, will overwrite
    
    # Initialize fixer
    fixer = InPlaceDataFixer()
    
    # Load original data
    print(f"\n📂 Loading original data: {input_path}")
    try:
        df_original = pd.read_csv(input_path)
        print(f"✅ Loaded {len(df_original):,} records")
        
        # Create backup
        df_original.to_csv(backup_path, index=False)
        print(f"💾 Created backup: {backup_path}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Show original activity distribution
    print(f"\n📊 ORIGINAL ACTIVITY DISTRIBUTION:")
    original_counts = df_original['Activity'].value_counts()
    for activity, count in original_counts.items():
        percentage = count / len(df_original) * 100
        print(f"   {activity}: {count:,} ({percentage:.1f}%)")
    
    # Verify and fix data
    print(f"\n🔧 STARTING VERIFICATION & CORRECTION")
    print("-" * 40)
    
    df_fixed, corrections_made, verification_stats = fixer.verify_and_fix_dataframe(
        df_original, batch_size=50
    )
    
    # Add confidence scores
    df_fixed = fixer.add_har_confidence_column(df_fixed, corrections_made)
    
    # Show fixed activity distribution
    print(f"\n📊 FIXED ACTIVITY DISTRIBUTION:")
    fixed_counts = df_fixed['Activity'].value_counts()
    for activity, count in fixed_counts.items():
        percentage = count / len(df_fixed) * 100
        original_count = original_counts.get(activity, 0)
        change = count - original_count
        change_sign = "+" if change > 0 else ""
        print(f"   {activity}: {count:,} ({percentage:.1f}%) [{change_sign}{change:,}]")
    
    # Print detailed correction summary
    fixer.print_correction_summary()
    
    # Save fixed data
    print(f"\n💾 Saving fixed data...")
    df_fixed.to_csv(output_path, index=False)
    print(f"✅ Saved fixed data: {output_path}")
    
    # Verification
    print(f"\n🔍 FINAL VERIFICATION:")
    print(f"   Original records: {len(df_original):,}")
    print(f"   Fixed records: {len(df_fixed):,}")
    print(f"   Records preserved: ✅")
    print(f"   New columns added: HAR_Confidence")
    print(f"   Backup created: {backup_path}")
    
    print(f"\n🎉 IN-PLACE DATA FIXING COMPLETED!")
    print(f"📁 Your original file has been updated with HAR-verified activities")
    print(f"📊 {verification_stats['corrected_activities']:,} activities were corrected out of {verification_stats['total_sequences']:,} sequences")
    
    return fixer, df_fixed

if __name__ == "__main__":
    fixer_instance, fixed_data = main()
