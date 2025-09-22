"""
ANALYZE ACTIVITY DISTRIBUTION
Tìm vấn đề imbalance và tạo strategy để fix
"""

import pandas as pd
import numpy as np

def analyze_current_distribution():
    """Analyze current activity distribution"""
    print("📊 === CURRENT ACTIVITY DISTRIBUTION ANALYSIS ===")
    
    # Load current dataset
    df = pd.read_csv('data/quota_balanced_health_data_30days.csv')
    print(f"✅ Loaded {len(df)} samples")
    
    # Activity distribution
    activity_counts = df['Activity'].value_counts()
    activity_percentages = df['Activity'].value_counts(normalize=True) * 100
    
    print(f"\n📈 Current Activity Distribution:")
    print(f"{'Activity':<12} {'Count':<8} {'Percentage':<10} {'Hours/Day':<10}")
    print("-" * 50)
    
    total_samples = len(df)
    samples_per_day = total_samples / 30
    hours_per_day = 16  # 8h sleep excluded
    
    for activity in activity_counts.index:
        count = activity_counts[activity]
        percentage = activity_percentages[activity]
        daily_samples = count / 30
        daily_hours = (daily_samples / samples_per_day) * hours_per_day
        
        status = "🚨" if percentage > 40 else "⚠️" if percentage > 20 else "✅"
        print(f"{activity:<12} {count:<8} {percentage:<9.1f}% {daily_hours:<9.1f}h {status}")
    
    # HAR Segment Analysis
    print(f"\n🔍 === HAR SEGMENT ANALYSIS ===")
    
    # Calculate segments needed for each activity (minimum for HAR)
    segment_size = 180  # samples per segment
    min_segments_per_activity = 50  # minimum for good HAR training
    
    print(f"HAR requirements:")
    print(f"  Segment size: {segment_size} samples (9 seconds)")
    print(f"  Minimum segments per activity: {min_segments_per_activity}")
    print(f"  Minimum samples per activity: {min_segments_per_activity * segment_size}")
    
    print(f"\n📊 Current HAR Readiness:")
    print(f"{'Activity':<12} {'Samples':<8} {'Segments':<10} {'Status':<15}")
    print("-" * 50)
    
    for activity in activity_counts.index:
        count = activity_counts[activity]
        segments = count // segment_size
        
        if segments >= min_segments_per_activity:
            status = "✅ GOOD"
        elif segments >= 20:
            status = "⚠️ ACCEPTABLE"
        else:
            status = "❌ INSUFFICIENT"
            
        print(f"{activity:<12} {count:<8} {segments:<10} {status:<15}")
    
    # Identify problems
    print(f"\n🚨 === IDENTIFIED PROBLEMS ===")
    
    # Problem 1: Sitting dominance
    sitting_pct = activity_percentages.get('Sitting', 0)
    if sitting_pct > 50:
        print(f"1. ❌ SITTING DOMINANCE: {sitting_pct:.1f}% (should be ~25-35%)")
    
    # Problem 2: Insufficient minor activities
    minor_activities = ['Jogging', 'Upstairs', 'Downstairs']
    insufficient_activities = []
    
    for activity in minor_activities:
        if activity in activity_counts:
            count = activity_counts[activity]
            segments = count // segment_size
            if segments < min_segments_per_activity:
                insufficient_activities.append(f"{activity} ({segments} segments)")
    
    if insufficient_activities:
        print(f"2. ❌ INSUFFICIENT SEQUENCES: {', '.join(insufficient_activities)}")
    
    # Problem 3: Poor distribution
    target_distribution = {
        'Sitting': 30,    # 30% instead of 75%
        'Walking': 25,    # 25%
        'Standing': 20,   # 20%
        'Jogging': 10,    # 10%
        'Upstairs': 8,    # 8%
        'Downstairs': 7   # 7%
    }
    
    print(f"\n🎯 === TARGET vs CURRENT COMPARISON ===")
    print(f"{'Activity':<12} {'Current':<8} {'Target':<8} {'Diff':<8} {'Action':<15}")
    print("-" * 60)
    
    for activity, target_pct in target_distribution.items():
        current_pct = activity_percentages.get(activity, 0)
        diff = current_pct - target_pct
        
        if diff > 5:
            action = "🔻 REDUCE"
        elif diff < -5:
            action = "🔺 INCREASE"
        else:
            action = "✅ OK"
            
        print(f"{activity:<12} {current_pct:<7.1f}% {target_pct:<7.1f}% {diff:<+7.1f}% {action:<15}")
    
    return target_distribution, activity_percentages

def calculate_required_changes(target_distribution, current_percentages):
    """Calculate specific changes needed"""
    print(f"\n📋 === REQUIRED CHANGES FOR 85-95% HAR ACCURACY ===")
    
    total_samples = 54163  # current dataset size
    samples_per_day = total_samples / 30
    
    print(f"Current dataset: {total_samples} samples over 30 days")
    print(f"Samples per day: {samples_per_day:.0f}")
    
    print(f"\n🎯 Required Daily Time Allocation:")
    print(f"{'Activity':<12} {'Current':<8} {'Target':<8} {'Change':<10}")
    print("-" * 45)
    
    hours_per_day = 16  # excluding sleep
    
    for activity, target_pct in target_distribution.items():
        current_pct = current_percentages.get(activity, 0)
        
        current_hours = (current_pct / 100) * hours_per_day
        target_hours = (target_pct / 100) * hours_per_day
        change_hours = target_hours - current_hours
        
        change_str = f"{change_hours:+.1f}h"
        print(f"{activity:<12} {current_hours:<7.1f}h {target_hours:<7.1f}h {change_str:<10}")
    
    print(f"\n🔧 === IMPLEMENTATION STRATEGY ===")
    print("1. 🕰️ Daily Activity Quotas:")
    print("   - Sitting: Max 4.8h/day (currently ~12h)")
    print("   - Walking: Min 4.0h/day (currently ~1.5h)")
    print("   - Standing: Min 3.2h/day (currently ~1.0h)")
    print("   - Jogging: Min 1.6h/day (currently ~0.2h)")
    print("   - Upstairs: Min 1.3h/day (currently ~0.1h)")
    print("   - Downstairs: Min 1.1h/day (currently ~0.1h)")
    
    print("\n2. 📐 HAR Segment Targets:")
    segment_size = 180
    for activity, target_pct in target_distribution.items():
        target_samples = int((target_pct / 100) * total_samples)
        target_segments = target_samples // segment_size
        print(f"   - {activity}: {target_segments} segments ({target_samples} samples)")
    
    print("\n3. 🔄 Generation Improvements:")
    print("   - Anti-sitting logic: Force movement every 2 hours")
    print("   - Activity quotas: Ensure minimum daily time per activity")
    print("   - Segment continuity: Longer durations for better HAR")
    print("   - Balanced transitions: Natural activity flow")

def main():
    """Main analysis function"""
    print("🔍 === ACTIVITY DISTRIBUTION OPTIMIZATION ANALYSIS ===")
    print("Goal: Fix 75% Sitting imbalance to achieve 85-95% HAR accuracy\n")
    
    # Analyze current distribution
    target_dist, current_dist = analyze_current_distribution()
    
    # Calculate required changes
    calculate_required_changes(target_dist, current_dist)
    
    print(f"\n🎯 === EXPECTED OUTCOME ===")
    print("✅ With balanced distribution:")
    print("   - HAR accuracy: 85-95% (currently 81.3%)")
    print("   - All activities: >50 segments each")
    print("   - Upstairs accuracy: 0% → 70%+")
    print("   - Walking accuracy: 57% → 80%+")
    print("   - Overall quality: Research → Production ready")

if __name__ == "__main__":
    main()
