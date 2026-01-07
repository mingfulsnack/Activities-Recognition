"""
Feature Selection Script
Converts 44-field dataset to 20-field optimized dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Selected 20 features (excluding Stress_Level which is target)
SELECTED_FEATURES = [
    # Core Features (8)
    'Timestamp',
    'Activity',
    'Location',
    'Heart_Rate',
    'Sleep_Duration',
    'Sleep_Quality',
    'Energy_Level',
    'Mood_Score',
    
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
    'Exercise_Minutes',
    
    # Target
    'Stress_Level'
]

def select_features(input_csv_path, output_csv_path=None):
    """
    Select 20 important features from 44-field dataset
    
    Args:
        input_csv_path: Path to original 44-field CSV
        output_csv_path: Path to save reduced dataset (optional)
    
    Returns:
        DataFrame with selected features
    """
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Check if all selected features exist
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        # Use only available features
        available_features = [f for f in SELECTED_FEATURES if f in df.columns]
        df_reduced = df[available_features].copy()
    else:
        df_reduced = df[SELECTED_FEATURES].copy()
    
    print(f"Reduced shape: {df_reduced.shape}")
    print(f"Reduced columns: {len(df_reduced.columns)}")
    print(f"Reduction: {len(df.columns)} → {len(df_reduced.columns)} ({100*(1-len(df_reduced.columns)/len(df.columns)):.1f}% reduction)")
    
    # Basic statistics
    print("\n=== Data Quality Check ===")
    print(f"Missing values per column:")
    missing = df_reduced.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values")
    else:
        print(missing[missing > 0])
    
    print(f"\nActivity distribution:")
    print(df_reduced['Activity'].value_counts())
    
    print(f"\nLocation distribution:")
    print(df_reduced['Location'].value_counts())
    
    print(f"\nStress statistics:")
    print(df_reduced['Stress_Level'].describe())
    
    # Save if output path provided
    if output_csv_path:
        df_reduced.to_csv(output_csv_path, index=False)
        print(f"\n✓ Saved reduced dataset to {output_csv_path}")
    
    return df_reduced


def analyze_removed_features(input_csv_path):
    """
    Analyze correlation of removed features to validate removal
    """
    df = pd.read_csv(input_csv_path)
    
    removed_features = [col for col in df.columns if col not in SELECTED_FEATURES]
    
    print("\n=== Removed Features Analysis ===")
    print(f"Total removed: {len(removed_features)} features")
    print("\nRemoved features:")
    for i, feat in enumerate(removed_features, 1):
        print(f"  {i}. {feat}")
    
    # Correlation with Stress_Level
    if 'Stress_Level' in df.columns:
        print("\n=== Correlation with Stress_Level ===")
        numeric_cols = df[removed_features].select_dtypes(include=[np.number]).columns
        correlations = df[list(numeric_cols) + ['Stress_Level']].corr()['Stress_Level'].drop('Stress_Level')
        correlations = correlations.abs().sort_values(ascending=False)
        
        print("\nTop 10 removed features by correlation with Stress:")
        for feat, corr in correlations.head(10).items():
            print(f"  {feat}: {corr:.3f}")
        
        # Check if any important features were removed
        high_corr = correlations[correlations > 0.3]
        if len(high_corr) > 0:
            print("\n⚠️  WARNING: High correlation features removed:")
            for feat, corr in high_corr.items():
                print(f"  {feat}: {corr:.3f}")
        else:
            print("\n✓ All removed features have low correlation (< 0.3)")


def compare_datasets(original_path, reduced_path):
    """
    Compare original and reduced datasets
    """
    df_orig = pd.read_csv(original_path)
    df_red = pd.read_csv(reduced_path)
    
    print("\n=== Dataset Comparison ===")
    print(f"Original: {df_orig.shape[0]} rows × {df_orig.shape[1]} cols")
    print(f"Reduced:  {df_red.shape[0]} rows × {df_red.shape[1]} cols")
    print(f"Size reduction: {df_orig.memory_usage(deep=True).sum() / 1024**2:.2f} MB → {df_red.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Reduction: {100*(1 - df_red.memory_usage(deep=True).sum() / df_orig.memory_usage(deep=True).sum()):.1f}%")


if __name__ == "__main__":
    # Paths
    input_path = Path(__file__).parent / "data" / "quota_balanced_health_data_30days.csv"
    output_path = Path(__file__).parent / "data" / "quota_balanced_health_data_20features.csv"
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Please generate the original dataset first.")
        exit(1)
    
    # Select features
    df_reduced = select_features(input_path, output_path)
    
    # Analyze removed features
    analyze_removed_features(input_path)
    
    # Compare datasets
    if output_path.exists():
        compare_datasets(input_path, output_path)
    
    print("\n" + "="*60)
    print("✓ Feature selection completed!")
    print("="*60)
