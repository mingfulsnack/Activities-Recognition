"""
Simplified Feature Selection - 13 Core + High-Importance Features
==================================================================
Purpose: Use only essential features without complex feature engineering

Feature Set (13):
- CORE (7): Hour, Day_of_Week, Activity, Accelerometer_X/Y/Z, Heart_Rate
- HIGH-IMPORTANCE (6): Location, Screen_Usage_Current, Phone_Event_Frequency,
                       Mood_Score, Energy_Level, Sleep_Duration

Why simplify:
- Engineered rolling features cause data leakage in sequences
- Simpler features = more interpretable + more robust
- Evidence-based: Core features backed by literature, rest by RF importance

Author: [Your Name]
Date: February 2026
"""

import pandas as pd
import numpy as np
import os

def create_simplified_dataset():
    """Create 13-feature dataset without complex feature engineering."""
    
    print("=" * 80)
    print("SIMPLIFIED FEATURE SELECTION - 13 FEATURES")
    print("=" * 80)
    print("")
    
    # Load original 23-feature dataset
    input_file = 'generate_and_verify_data/Data generator/data/optimized_health_data_23features.csv'
    output_file = 'data/optimized_health_data_13features.csv'
    
    print(f" Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f" Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    # Extract Hour and Day_of_Week from Timestamp if not present
    if 'Timestamp' in df.columns and 'Hour' not in df.columns:
        print(f"\n  Extracting Hour and Day_of_Week from Timestamp...")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
        print(f"   Extracted Hour (range: {df['Hour'].min()}-{df['Hour'].max()})")
        print(f"   Extracted Day_of_Week (range: {df['Day_of_Week'].min()}-{df['Day_of_Week'].max()})")
    
    # Select 13 features
    SELECTED_FEATURES = [
        # === CORE (7) - Evidence-based ===
        'Hour',                # [Schlotz 2004] Circadian rhythm
        'Day_of_Week',         # Temporal pattern
        'Activity',            # [Garcia-Ceja 2018] Activity-stress link
        'Accelerometer_X',     # [Kusserow 2013] Movement patterns
        'Accelerometer_Y',
        'Accelerometer_Z',
        'Heart_Rate',          # [Hovsepian 2015] Context-dependent stress
        
        # === HIGH-IMPORTANCE (6) - RF-based ===
        'Location',            # 64.98% importance
        'Screen_Usage_Current',# 7.46%
        'Phone_Event_Frequency',# 3.35%
        'Mood_Score',          # 2.55%
        'Energy_Level',        # 1.99%
        'Sleep_Duration',      # 1.06%
        
        # === TARGET ===
        'Stress_Level'
    ]
    
    # Check availability
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    
    if missing:
        print(f"\n  Missing features: {missing}")
        return
    
    # Create dataset
    df_final = df[available].copy()
    
    print(f"\n✓ Selected 13-feature dataset:")
    print(f"  Features: {len(df_final.columns) - 1}")
    print(f"  Samples: {len(df_final):,}")
    print(f"\n  Feature list:")
    for i, feat in enumerate(available[:-1], 1):
        print(f"    {i:2d}. {feat}")
    
    # Check data quality
    print(f"\n Data Quality Check:")
    nan_count = df_final.isnull().sum().sum()
    inf_count = np.isinf(df_final.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    if nan_count == 0 and inf_count == 0:
        print(f"   Data quality: GOOD")
    else:
        print(f"    Data quality issues detected")
    
    # Target statistics
    print(f"\n Target Statistics:")
    print(f"  Range: [{df_final['Stress_Level'].min():.2f}, {df_final['Stress_Level'].max():.2f}]")
    print(f"  Mean: {df_final['Stress_Level'].mean():.4f}")
    print(f"  Std:  {df_final['Stress_Level'].std():.4f}")
    
    # Save
    print(f"\n Saving to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_csv(output_file, index=False)
    
    file_size = os.path.getsize(output_file) / 1024**2
    print(f" Saved successfully!")
    print(f"  File size: {file_size:.2f} MB")
    
    # Create report
    report_path = os.path.join(os.path.dirname(output_file), 'feature_selection_13features_report.txt')
    with open(report_path, 'w') as f:
        f.write("13-FEATURE SELECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. RATIONALE\n")
        f.write("-" * 80 + "\n")
        f.write("After testing 17 features with complex feature engineering:\n")
        f.write("    Problem: Rolling features caused data leakage + scaling issues\n")
        f.write("    Result: Training loss ~10.23 (vs ~0.92 expected)\n")
        f.write("\n")
        f.write("Simplified approach:\n")
        f.write("   Use only essential features (no rolling windows)\n")
        f.write("   Combine domain knowledge (core 7) + ML insights (high-importance 6)\n")
        f.write("   Simpler = more interpretable + more robust\n")
        f.write("\n")
        
        f.write("2. FEATURE CATEGORIES\n")
        f.write("-" * 80 + "\n")
        f.write("CORE FEATURES (7) - Evidence-based\n")
        f.write("  1. Hour                - [Schlotz 2004] Circadian rhythm\n")
        f.write("  2. Day_of_Week        - Temporal weekly pattern\n")
        f.write("  3. Activity           - [Garcia-Ceja 2018] Activity-stress link\n")
        f.write("  4-6. Accelerometer_XYZ - [Kusserow 2013] Movement patterns\n")
        f.write("  7. Heart_Rate         - [Hovsepian 2015] Context-dependent stress\n")
        f.write("\n")
        f.write("HIGH-IMPORTANCE FEATURES (6) - RF importance\n")
        f.write("  8. Location            - 64.98% importance\n")
        f.write("  9. Screen_Usage_Current- 7.46%\n")
        f.write(" 10. Phone_Event_Frequency- 3.35%\n")
        f.write(" 11. Mood_Score          - 2.55%\n")
        f.write(" 12. Energy_Level        - 1.99%\n")
        f.write(" 13. Sleep_Duration      - 1.06%\n")
        f.write("\n")
        
        f.write("3. EXPECTED PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write("Compared to previous models:\n")
        f.write("  Baseline (21 features): R²=0.9343, MAE=0.5095\n")
        f.write("  Reduced (10 features):  R²=0.9431, MAE=0.5218 (missing core features)\n")
        f.write("  Engineering (17 features): Training failed (data leakage)\n")
        f.write("\n")
        f.write("Expected for 13 features:\n")
        f.write("  R² = 0.940 - 0.950 (has all core features + top RF features)\n")
        f.write("  MAE = 0.50 - 0.53\n")
        f.write("  Training: Stable, no data leakage\n")
        f.write("\n")
        
        f.write("4. DEFENSIBILITY\n")
        f.write("-" * 80 + "\n")
        f.write("For medical presentation:\n")
        f.write("   Core features: All backed by published papers\n")
        f.write("   Additional features: Selected by ML importance (interpretable)\n")
        f.write("   No complex engineering: Reduces overfitting risk\n")
        f.write("   HAR integration: Full accelerometer + activity preserved\n")
        f.write("\n")
    
    print(f" Report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("✓ 13-FEATURE DATASET CREATED!")
    print("=" * 80)
    print(f"\nNext step: python stress_prediction/train_lstm_13features.py")


if __name__ == '__main__':
    create_simplified_dataset()
