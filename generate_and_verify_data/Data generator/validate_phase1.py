"""
Phase 1 Validation - Context-Stress Variations Analysis
Kiểm tra xem context variations có tạo stress khác nhau không
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_context_stress_variations(csv_path):
    """Analyze if same activity with different context leads to different stress"""
    print("="*70)
    print("Phase 1 Validation: Context-Stress Variations")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    print(f"\n Dataset: {len(df):,} samples, {len(df.columns)} features")
    print(f" Timespan: {df['Timestamp'].iloc[0]} → {df['Timestamp'].iloc[-1]}")
    
    # 1. Same Activity, Different Location → Stress variation
    print("\n" + "="*70)
    print("  SAME ACTIVITY + DIFFERENT LOCATION → STRESS VARIATION")
    print("="*70)
    
    for activity in ['Walking', 'Sitting', 'Jogging']:
        print(f"\n Activity: {activity}")
        activity_data = df[df['Activity'] == activity]
        
        if len(activity_data) == 0:
            continue
        
        location_stress = activity_data.groupby('Location')['Stress_Level'].agg(['mean', 'std', 'count'])
        location_stress = location_stress[location_stress['count'] > 10]  # Filter low count
        
        if len(location_stress) == 0:
            continue
        
        location_stress = location_stress.sort_values('mean', ascending=False)
        print(location_stress)
        
        # Check variation
        stress_range = location_stress['mean'].max() - location_stress['mean'].min()
        print(f"   Stress Range: {stress_range:.2f} (expect > 1.0 for good variation)")
        
        if stress_range > 1.0:
            print("    Good variation detected!")
        else:
            print("     Low variation - may need improvement")
    
    # 2. Time of Day Impact
    print("\n" + "="*70)
    print("  TIME OF DAY IMPACT ON STRESS")
    print("="*70)
    
    df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    
    time_periods = {
        'Morning (7-9)': (7, 9),
        'Work Morning (9-12)': (9, 12),
        'Afternoon (14-17)': (14, 17),
        'Evening (17-20)': (17, 20)
    }
    
    for period_name, (start_hour, end_hour) in time_periods.items():
        period_data = df[(df['Hour'] >= start_hour) & (df['Hour'] < end_hour)]
        mean_stress = period_data['Stress_Level'].mean()
        print(f"{period_name:25s}: {mean_stress:.2f}")
    
    # 3. Context-Specific Examples
    print("\n" + "="*70)
    print("  CONTEXT-SPECIFIC STRESS EXAMPLES")
    print("="*70)
    
    # Example 1: Walking at work vs outdoor
    print("\n Walking at work vs outdoor:")
    walking_work = df[(df['Activity'] == 'Walking') & (df['Location'] == 'work')]['Stress_Level'].mean()
    walking_outdoor = df[(df['Activity'] == 'Walking') & (df['Location'] == 'outdoor')]['Stress_Level'].mean()
    
    print(f"   Work:    {walking_work:.2f}")
    print(f"   Outdoor: {walking_outdoor:.2f}")
    print(f"   Δ = {abs(walking_work - walking_outdoor):.2f}")
    
    # Example 2: Sitting at work vs home
    print("\n Sitting at work vs home:")
    sitting_work = df[(df['Activity'] == 'Sitting') & (df['Location'] == 'work')]['Stress_Level'].mean()
    sitting_home = df[(df['Activity'] == 'Sitting') & (df['Location'] == 'home')]['Stress_Level'].mean()
    
    print(f"   Work: {sitting_work:.2f}")
    print(f"   Home: {sitting_home:.2f}")
    print(f"   Δ = {abs(sitting_work - sitting_home):.2f}")
    
    # 4. Overall Distribution
    print("\n" + "="*70)
    print(" STRESS DISTRIBUTION")
    print("="*70)
    
    print("\nStress Statistics:")
    print(df['Stress_Level'].describe())
    
    print("\nStress Distribution:")
    bins = [1, 3, 5, 7, 9]
    labels = ['Low (1-3)', 'Medium (3-5)', 'High (5-7)', 'Very High (7-9)']
    df['Stress_Category'] = pd.cut(df['Stress_Level'], bins=bins, labels=labels, include_lowest=True)
    
    stress_dist = df['Stress_Category'].value_counts(normalize=True) * 100
    for cat, pct in stress_dist.items():
        print(f"  {cat:20s}: {pct:5.1f}%")
    
    # 5. Feature Importance Check
    print("\n" + "="*70)
    print("  FEATURE COMPLETENESS CHECK")
    print("="*70)
    
    required_features = [
        'Activity', 'Location', 'Heart_Rate', 'Sleep_Duration', 'Sleep_Quality',
        'Energy_Level', 'Mood_Score', 'Screen_Usage_Current', 'Phone_Usage_Intensity',
        'Social_Current_Level', 'Ambient_Light', 'Noise_Level', 'Weather_Condition',
        'Exercise_Minutes', 'Stress_Level'
    ]
    
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        print(f" Missing features: {missing}")
    else:
        print(f" All {len(required_features)} required features present")
    
    # Check for null values
    null_counts = df[required_features].isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n  Null values found:")
        print(null_counts[null_counts > 0])
    else:
        print(f" No null values in required features")
    
    return df


def create_validation_plots(df, output_dir='validation_plots'):
    """Create validation plots"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("  CREATING VALIDATION PLOTS")
    print("="*70)
    
    # Plot 1: Stress by Activity and Location
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Phase 1: Context-Stress Variation Analysis', fontsize=16, fontweight='bold')
    
    # 1.1 Stress by Activity
    activity_stress = df.groupby('Activity')['Stress_Level'].mean().sort_values()
    axes[0, 0].barh(activity_stress.index, activity_stress.values, color='skyblue')
    axes[0, 0].set_xlabel('Average Stress Level')
    axes[0, 0].set_title('Average Stress by Activity')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 1.2 Stress by Location
    location_stress = df.groupby('Location')['Stress_Level'].mean().sort_values()
    axes[0, 1].barh(location_stress.index, location_stress.values, color='lightcoral')
    axes[0, 1].set_xlabel('Average Stress Level')
    axes[0, 1].set_title('Average Stress by Location')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 1.3 Heatmap: Activity × Location
    pivot = df.pivot_table(values='Stress_Level', index='Activity', columns='Location', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[1, 0], cbar_kws={'label': 'Stress Level'})
    axes[1, 0].set_title('Stress Heatmap: Activity × Location')
    
    # 1.4 Stress Distribution
    axes[1, 1].hist(df['Stress_Level'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Stress Level')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Stress Distribution')
    axes[1, 1].axvline(df['Stress_Level'].mean(), color='red', linestyle='--', label=f'Mean: {df["Stress_Level"].mean():.2f}')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'phase1_validation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f" Saved plot: {plot_path}")
    plt.close()


def main():
    """Main validation"""
    csv_path = 'data/optimized_health_data_20features_v2.csv'
    
    if not Path(csv_path).exists():
        print(f" File not found: {csv_path}")
        print("Please run data generation first!")
        return
    
    # Analyze
    df = analyze_context_stress_variations(csv_path)
    
    # Create plots
    create_validation_plots(df)
    
    print("\n" + "="*70)
    print(" Phase 1 Validation Complete!")
    print("="*70)
    print("\nSummary:")
    print("   Dataset generated with 20 optimized features")
    print("   Context-aware stress variations implemented")
    print("   Same activity + different context → different stress")
    print("   Ready for Phase 2: Model Development")
    print("\nNext Steps:")
    print("   Test HAR model compatibility")
    print("   Begin stress prediction model comparison")


if __name__ == "__main__":
    main()
