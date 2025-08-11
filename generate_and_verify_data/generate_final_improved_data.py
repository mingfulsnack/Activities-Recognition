"""
FINAL TEST: Generate improved health data và test HAR accuracy
- Sử dụng real WISDM accelerometer data  
- Improved activity consistency
- Test với classificator_model.keras
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from health_data_generator import HealthMonitoringDataGenerator

def main():
    print("=== FINAL IMPROVED HEALTH DATA GENERATION & HAR TESTING ===")
    print("🔧 Using REAL WISDM accelerometer data for better HAR accuracy")
    print("🎯 Target: 95% HAR accuracy với realistic health monitoring data")
    
    # Generate improved dataset
    print("\n🚀 Step 1: Generating improved dataset...")
    generator = HealthMonitoringDataGenerator()
    
    df = generator.generate_enhanced_dataset("2024-01-01", 30)
    
    # Save dataset
    output_path = "data/improved_health_data_30days.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Size: {len(df):,} records")
    
    # Quick data quality check
    print(f"\n📊 === DATA QUALITY CHECK ===")
    print(f"Activities distribution:")
    for activity in df['Activity'].value_counts().head().items():
        print(f"   {activity[0]}: {activity[1]:,} samples")
    
    # Accelerometer data check
    accel_cols = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    for col in accel_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"   {col}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    # Activity change rate
    activities = df['Activity'].values
    changes = sum(1 for i in range(1, len(activities)) if activities[i] != activities[i-1])
    change_rate = changes / len(activities)
    print(f"   Activity change rate: {change_rate:.1%}")
    
    print(f"\n🎯 Dataset generated successfully!")
    print(f"   Next step: Run HAR model testing to verify accuracy")
    print(f"   Use: python test_har_accuracy.py")
    
    return output_path

if __name__ == "__main__":
    main()
