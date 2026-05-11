"""
Create 23-feature dataset (20 + Accelerometer X,Y,Z)
For end-to-end learning: Sensor → Stress
"""

import pandas as pd
from pathlib import Path

def create_23feature_dataset():
    """Add accelerometer back to 20-feature dataset"""
    
    print("="*70)
    print("Creating 23-Feature Dataset (20 + Accelerometer X,Y,Z)")
    print("="*70)
    
    # Load 44-field dataset (has everything)
    df_full = pd.read_csv('data/quota_balanced_health_data_30days_v2.csv')
    print(f"\n Loaded full dataset: {df_full.shape}")
    
    # 23 Selected Features
    SELECTED_23_FEATURES = [
        # Sensor Data (3)
        'Accelerometer_X',
        'Accelerometer_Y',
        'Accelerometer_Z',
        
        # Core Features (9)
        'Timestamp',
        'Activity',
        'Location',
        'Heart_Rate',
        'Sleep_Duration',
        'Sleep_Quality',
        'Energy_Level',
        'Mood_Score',
        'Stress_Level',  # Target
        
        # Behavioral Sequences (7)
        'Screen_Usage_Current',
        'Screen_Usage_15min_Avg',
        'Screen_Usage_Trend',
        'Phone_Usage_Intensity',
        'Phone_Event_Frequency',
        'Social_Current_Level',
        'Social_1hour_Avg',
        
        # Environmental Context (4)
        'Ambient_Light',
        'Noise_Level',
        'Weather_Condition',
        'Exercise_Minutes'
    ]
    
    # Select features
    df_23 = df_full[SELECTED_23_FEATURES].copy()
    
    print(f"\n Created 23-feature dataset: {df_23.shape}")
    print(f" Reduction: 44 → 23 features (47.7% reduction)")
    
    # Save
    output_path = 'data/optimized_health_data_23features.csv'
    df_23.to_csv(output_path, index=False)
    
    print(f"\n Saved to: {output_path}")
    
    # Show stats
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    
    print(f"\nTotal samples: {len(df_23):,}")
    print(f"\nActivity distribution:")
    print(df_23['Activity'].value_counts())
    
    print(f"\nStress statistics:")
    print(df_23['Stress_Level'].describe())
    
    print(f"\nAccelerometer ranges:")
    print(f"  X: [{df_23['Accelerometer_X'].min():.2f}, {df_23['Accelerometer_X'].max():.2f}]")
    print(f"  Y: [{df_23['Accelerometer_Y'].min():.2f}, {df_23['Accelerometer_Y'].max():.2f}]")
    print(f"  Z: [{df_23['Accelerometer_Z'].min():.2f}, {df_23['Accelerometer_Z'].max():.2f}]")
    
    # Feature groups
    print("\n" + "="*70)
    print("Feature Groups")
    print("="*70)
    print(" Sensor Data: 3 features")
    print(" Core Features: 9 features")
    print(" Behavioral: 7 features")
    print(" Environmental: 4 features")
    print("="*70)
    print("TOTAL: 23 features")
    
    return df_23


if __name__ == "__main__":
    df = create_23feature_dataset()
    
    print("\n" + "="*70)
    print(" 23-Feature Dataset Ready!")
    print("="*70)
    print("\n Use cases:")
    print("  1. End-to-end learning: X,Y,Z → Stress")
    print("  2. Multi-task learning: X,Y,Z → Activity + Stress")
    print("  3. Two-stage: X,Y,Z → Activity → Stress")
    print("  4. Flexible: Can use with or without Activity label")
    print("\n This is the RECOMMENDED dataset for research!")
