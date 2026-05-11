"""
Feature Selection Script
========================
Purpose: Create reduced dataset with top 10 features based on Random Forest importance analysis.

Top 10 Features (98.07% cumulative importance):
1. Location (64.98%)
2. Heart_Rate (13.93%)
3. Screen_Usage_Current (7.46%)
4. Phone_Event_Frequency (3.35%)
5. Mood_Score (2.55%)
6. Energy_Level (1.99%)
7. Exercise_Minutes (1.09%)
8. Sleep_Duration (1.06%)
9. Screen_Usage_15min_Avg (1.05%)
10. Sleep_Quality (0.63%)

Author: [Your Name]
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Constants
TOP_10_FEATURES = [
    'Location',
    'Heart_Rate',
    'Screen_Usage_Current',
    'Phone_Event_Frequency',
    'Mood_Score',
    'Energy_Level',
    'Exercise_Minutes',
    'Sleep_Duration',
    'Screen_Usage_15min_Avg',
    'Sleep_Quality'
]

TARGET = 'Stress_Level'


class FeatureSelector:
    """Class to handle feature selection and dataset creation."""
    
    def __init__(self, input_file, output_dir='data/'):
        """
        Initialize FeatureSelector.
        
        Parameters:
        -----------
        input_file : str
            Path to original dataset (23 features)
        output_dir : str
            Directory to save reduced dataset
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.df_original = None
        self.df_reduced = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load original dataset."""
        print(f" Loading original dataset from: {self.input_file}")
        self.df_original = pd.read_csv(self.input_file)
        print(f"✓ Loaded {len(self.df_original):,} samples with {len(self.df_original.columns)} features")
        print(f"  Original features: {list(self.df_original.columns)}")
        return self
        
    def select_features(self):
        """Select top 10 features + target."""
        print(f"\n Selecting top 10 features...")
        
        # Check if all features exist
        missing_features = [f for f in TOP_10_FEATURES if f not in self.df_original.columns]
        if missing_features:
            print(f"  Missing features: {missing_features}")
            print(f"  Available features: {list(self.df_original.columns)}")
            raise ValueError(f"Some features are missing in dataset: {missing_features}")
        
        # Select features + target
        selected_columns = TOP_10_FEATURES + [TARGET]
        self.df_reduced = self.df_original[selected_columns].copy()
        
        print(f"✓ Selected {len(TOP_10_FEATURES)} features + target")
        print(f"  Selected features: {TOP_10_FEATURES}")
        print(f"  Reduced dataset shape: {self.df_reduced.shape}")
        
        return self
        
    def analyze_reduction(self):
        """Analyze data reduction statistics."""
        print(f"\n Reduction Analysis:")
        print(f"  Original features: {len(self.df_original.columns) - 1}")  # -1 for target
        print(f"  Reduced features: {len(TOP_10_FEATURES)}")
        print(f"  Reduction: {len(self.df_original.columns) - 1} → {len(TOP_10_FEATURES)} "
              f"({(1 - len(TOP_10_FEATURES)/(len(self.df_original.columns)-1))*100:.1f}% reduction)")
        
        # Memory usage
        original_memory = self.df_original.memory_usage(deep=True).sum() / 1024**2
        reduced_memory = self.df_reduced.memory_usage(deep=True).sum() / 1024**2
        print(f"  Memory usage: {original_memory:.2f} MB → {reduced_memory:.2f} MB "
              f"({(1 - reduced_memory/original_memory)*100:.1f}% reduction)")
        
        # Statistical summary
        print(f"\n Feature Statistics (Reduced Dataset):")
        print(self.df_reduced.describe())
        
        return self
        
    def save_reduced_dataset(self, filename='optimized_health_data_10features.csv'):
        """Save reduced dataset to CSV."""
        output_path = os.path.join(self.output_dir, filename)
        print(f"\n Saving reduced dataset to: {output_path}")
        self.df_reduced.to_csv(output_path, index=False)
        print(f"  Saved successfully!")
        print(f"  Samples: {len(self.df_reduced):,}")
        print(f"  Features: {len(self.df_reduced.columns) - 1}")
        print(f"  File size: {os.path.getsize(output_path) / 1024**2:.2f} MB")
        
        return self
        
    def create_comparison_report(self):
        """Create comparison report between original and reduced dataset."""
        print(f"\n Creating Comparison Report...")
        
        report = []
        report.append("=" * 80)
        report.append("FEATURE SELECTION COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset info
        report.append("1. DATASET INFORMATION")
        report.append("-" * 80)
        report.append(f"Original Dataset:")
        report.append(f"  - Features: {len(self.df_original.columns) - 1}")
        report.append(f"  - Samples: {len(self.df_original):,}")
        report.append(f"  - Memory: {self.df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append("")
        report.append(f"Reduced Dataset:")
        report.append(f"  - Features: {len(TOP_10_FEATURES)}")
        report.append(f"  - Samples: {len(self.df_reduced):,}")
        report.append(f"  - Memory: {self.df_reduced.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append("")
        
        # Selected features
        report.append("2. SELECTED FEATURES (Top 10)")
        report.append("-" * 80)
        for i, feature in enumerate(TOP_10_FEATURES, 1):
            report.append(f"  {i:2d}. {feature}")
        report.append("")
        
        # Feature importance (from previous analysis)
        report.append("3. FEATURE IMPORTANCE (Random Forest)")
        report.append("-" * 80)
        importance_data = [
            ("Location", 64.98),
            ("Heart_Rate", 13.93),
            ("Screen_Usage_Current", 7.46),
            ("Phone_Event_Frequency", 3.35),
            ("Mood_Score", 2.55),
            ("Energy_Level", 1.99),
            ("Exercise_Minutes", 1.09),
            ("Sleep_Duration", 1.06),
            ("Screen_Usage_15min_Avg", 1.05),
            ("Sleep_Quality", 0.63)
        ]
        
        cumulative = 0
        for feature, importance in importance_data:
            cumulative += importance
            report.append(f"  {feature:30s} {importance:6.2f}%  (Cumulative: {cumulative:6.2f}%)")
        report.append("")
        
        # Expected impact
        report.append("4. EXPECTED IMPACT")
        report.append("-" * 80)
        report.append("  ✓ Performance retention: ~98% (based on cumulative importance)")
        report.append("  ✓ Training speed: Faster (43% fewer features)")
        report.append("  ✓ Model complexity: Reduced (lower overfitting risk)")
        report.append("  ✓ Interpretability: Improved (fewer features to explain)")
        report.append("  ✓ Memory usage: Reduced by ~43%")
        report.append("")
        
        # Next steps
        report.append("5. NEXT STEPS")
        report.append("-" * 80)
        report.append("  1. Create sequences with 10 features (60 timesteps)")
        report.append("  2. Train LSTM with same architecture as baseline")
        report.append("  3. Compare performance:")
        report.append("     - Baseline (23 features): R² = 0.9343, MAE = 0.5095")
        report.append("     - Reduced (10 features): R² = ?, MAE = ?")
        report.append("  4. Validate that 10 features are sufficient")
        report.append("")
        
        report.append("=" * 80)
        
        # Print and save report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file
        report_path = os.path.join(self.output_dir, 'feature_selection_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n Report saved to: {report_path}")
        
        return self


def main():
    """Main execution function."""
    print("=" * 80)
    print("FEATURE SELECTION: Reducing from 23 to 10 Features")
    print("=" * 80)
    print("")
    
    # Configuration
    input_file = 'generate_and_verify_data/Data generator/data/optimized_health_data_23features.csv'
    output_dir = 'data/'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f" Error: Input file not found: {input_file}")
        print(f"  Please ensure the 23-feature dataset exists.")
        return
    
    # Create FeatureSelector and run pipeline
    selector = FeatureSelector(input_file, output_dir)
    
    try:
        (selector
         .load_data()
         .select_features()
         .analyze_reduction()
         .save_reduced_dataset()
         .create_comparison_report())
        
        print("\n" + "=" * 80)
        print(" FEATURE SELECTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext step: Run train_lstm_10features.py to train model with reduced dataset")
        
    except Exception as e:
        print(f"\n Error during feature selection: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
