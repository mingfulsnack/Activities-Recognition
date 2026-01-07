"""
Refactored Health Data Generator with 20 Features + Context-Stress Variations
Version 2.0 - Optimized for stress prediction research
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Import core modules
from core.user_profile import UserProfile
from core.wisdm_loader import WisdmDataLoader
from core.activity_manager import ActivityManager
from core.metrics_calculator import HealthMetricsCalculator
from core.behavioral_tracker import BehavioralTracker
from core.schedule_generator import DailyScheduleGenerator

class OptimizedHealthDataGenerator:
    """
    Optimized Health Data Generator with 20 Features
    - Reduced từ 44 → 20 fields
    - Context-aware stress variations
    - Better for model training
    """
    
    # Selected 20 features (excluding target Stress_Level)
    SELECTED_FEATURES = [
        'Timestamp', 'Activity', 'Location',
        'Heart_Rate', 'Sleep_Duration', 'Sleep_Quality', 'Energy_Level', 'Mood_Score',
        'Screen_Usage_Current', 'Screen_Usage_15min_Avg', 'Screen_Usage_Trend',
        'Phone_Usage_Intensity', 'Phone_Event_Frequency',
        'Social_Current_Level', 'Social_1hour_Avg',
        'Ambient_Light', 'Noise_Level', 'Weather_Condition', 'Exercise_Minutes',
        'Stress_Level'  # Target
    ]
    
    def __init__(self, age=28, gender='Female'):
        # Initialize core components
        self.user_profile = UserProfile(age, gender)
        self.wisdm_loader = WisdmDataLoader()
        self.activity_manager = ActivityManager()
        self.metrics_calculator = HealthMetricsCalculator(self.user_profile)
        self.behavioral_tracker = BehavioralTracker()
        self.schedule_generator = DailyScheduleGenerator(self.activity_manager)
        
        # Configuration
        self.samples_per_minute = 2  # 2 samples/phút = 2880 samples/ngày
        self.life_events = {}
        
        # Load WISDM data
        self.wisdm_loader.load_wisdm_data()
        print(f"🔍 Loaded WISDM data for {len(self.wisdm_loader.get_available_activities())} activities")
        print(f"✨ Generating {len(self.SELECTED_FEATURES)} features (reduced from 44)")

    def calculate_enhanced_daily_metrics(self, date, schedule, day_context):
        """Tính toán các metrics với context-aware approach"""
        daily_noise = self.schedule_generator.get_daily_noise_factor(date)
        
        base_sleep = self.user_profile.profile['base_sleep_duration']
        sleep_variation = daily_noise['sleep_pattern'] * 1.2
        
        if day_context['life_event']:
            event_type = day_context['life_event']['type']
            if event_type in ['sick', 'stress', 'deadline', 'exam']:
                sleep_variation -= 0.8
            elif event_type in ['vacation', 'weekend_trip']:
                sleep_variation += 0.5
        
        actual_sleep = max(4, min(12, base_sleep + sleep_variation))
        
        # Heart rate calculation using Age/Gender
        base_hr = self.user_profile.calculate_resting_heart_rate()
        hr_variation = (
            (day_context['stress_base'] - 4) * 6 +
            (1 - day_context['sleep_quality']) * 10 +
            daily_noise['health_variation'] * 12 +
            day_context['weather_effect'] * 6
        )
        max_hr = self.user_profile.calculate_max_heart_rate()
        heart_rate_baseline = max(45, min(max_hr * 0.6, base_hr + hr_variation))
        
        # Calculate total exercise minutes for the day
        total_exercise_minutes = 0
        for activity_block in schedule:
            if activity_block['activity'] in ['Jogging', 'Upstairs', 'Downstairs']:
                duration_minutes = (activity_block['time_end'] - activity_block['time_start']) * 60
                total_exercise_minutes += duration_minutes
        
        return {
            'sleep_duration': round(actual_sleep, 1),
            'sleep_quality': day_context['sleep_quality'],
            'energy_level': day_context['energy_level'],
            'heart_rate_baseline': round(heart_rate_baseline),
            'exercise_minutes': round(total_exercise_minutes, 1)
        }

    def generate_30day_data(self, num_days=30, output_csv='data/optimized_health_data_20features.csv'):
        """
        Generate 30-day dataset with 20 optimized features and context-stress variations
        """
        print(f"🚀 Generating {num_days}-day health dataset with context-aware stress...")
        
        all_data = []
        start_date = datetime(2024, 1, 1, 7, 30)
        
        for day_num in range(num_days):
            current_date = start_date + timedelta(days=day_num)
            is_weekend = current_date.weekday() >= 5
            
            print(f"\n📅 Day {day_num + 1}/{num_days}: {current_date.strftime('%Y-%m-%d %A')}")
            
            # Generate schedule and day context (method returns both)
            schedule, day_context = self.schedule_generator.generate_improved_daily_schedule(
                date=current_date,
                life_events=self.life_events
            )
            
            # Calculate daily metrics
            daily_metrics = self.calculate_enhanced_daily_metrics(current_date, schedule, day_context)
            
            # Initialize behavioral tracker for this day
            self.behavioral_tracker.reset_behavioral_state()
            
            # Storage for previous stress levels (for momentum)
            previous_stress_levels = []
            
            # Generate data for each schedule block
            for activity_block in schedule:
                activity = activity_block['activity']
                location = activity_block['location']
                time_start = activity_block['time_start']
                time_end = activity_block['time_end']
                
                # Calculate duration in hours
                duration_hours = time_end - time_start
                num_samples = max(1, int(duration_hours * 60 * self.samples_per_minute))
                
                # Generate accelerometer samples
                accel_samples = self.wisdm_loader.generate_accelerometer_with_variations(
                    activity=activity,
                    num_samples=num_samples,
                    stress_level=day_context['stress_base'],
                    fatigue_level=1 - day_context['energy_level']
                )
                
                # Generate timestamp sequence
                for i, (acc_x, acc_y, acc_z) in enumerate(accel_samples):
                    # Calculate timestamp
                    time_offset_hours = time_start + (duration_hours * i / num_samples)
                    sample_timestamp = current_date + timedelta(hours=time_offset_hours)
                    hour_of_day = sample_timestamp.hour + sample_timestamp.minute / 60
                    
                    # Calculate heart rate for this sample
                    activity_hr_boost = {
                        'Sitting': 0, 'Standing': 5, 'Walking': 15,
                        'Jogging': 45, 'Upstairs': 25, 'Downstairs': 15
                    }.get(activity, 0)
                    sample_hr = daily_metrics['heart_rate_baseline'] + activity_hr_boost + random.randint(-3, 3)
                    
                    # Calculate mood
                    mood_score = self.metrics_calculator.calculate_mood_score(
                        base_mood_factor=day_context['base_mood_factor'],
                        hour=sample_timestamp.hour,
                        activity=activity,
                        location=location,
                        stress_level=day_context['stress_base']
                    )
                    
                    # Calculate stress with context-aware variations
                    stress_level = self.metrics_calculator.calculate_realistic_stress_level(
                        base_stress=day_context['stress_base'],
                        hour=sample_timestamp.hour,
                        activity=activity,
                        location=location,
                        heart_rate=sample_hr,
                        sleep_quality=daily_metrics['sleep_quality'],
                        work_intensity=day_context['work_intensity'],
                        previous_stress_levels=previous_stress_levels,
                        sleep_duration=daily_metrics['sleep_duration'],
                        is_weekend=is_weekend
                    )
                    previous_stress_levels.append(stress_level)
                    if len(previous_stress_levels) > 10:
                        previous_stress_levels.pop(0)
                    
                    # Environmental factors
                    ambient_light = self.activity_manager.generate_ambient_light(hour_of_day, location)
                    noise_level = self.activity_manager.generate_noise_level(location, activity)
                    weather = day_context['weather_condition']
                    
                    # Behavioral sequences (screen, phone, social)
                    behavioral_data = self.behavioral_tracker.update_and_get_features(
                        timestamp=sample_timestamp,
                        activity=activity,
                        location=location,
                        stress_level=stress_level
                    )
                    
                    # Build row with 20 selected features
                    row = {
                        'Timestamp': sample_timestamp.strftime('%m/%d/%Y %H:%M'),
                        'Activity': activity,
                        'Location': location,
                        'Heart_Rate': sample_hr,
                        'Sleep_Duration': daily_metrics['sleep_duration'],
                        'Sleep_Quality': round(daily_metrics['sleep_quality'], 2),
                        'Energy_Level': round(daily_metrics['energy_level'], 2),
                        'Mood_Score': mood_score,
                        'Screen_Usage_Current': round(behavioral_data['screen_usage']['current'], 9),
                        'Screen_Usage_15min_Avg': round(behavioral_data['screen_usage']['avg_15min'], 9),
                        'Screen_Usage_Trend': round(behavioral_data['screen_usage']['trend'], 9),
                        'Phone_Usage_Intensity': round(behavioral_data['phone_usage']['intensity'], 9),
                        'Phone_Event_Frequency': round(behavioral_data['phone_usage']['frequency'], 9),
                        'Social_Current_Level': round(behavioral_data['social']['current'], 9),
                        'Social_1hour_Avg': round(behavioral_data['social']['avg_1hour'], 9),
                        'Ambient_Light': round(ambient_light, 1),
                        'Noise_Level': round(noise_level, 1),
                        'Weather_Condition': round(weather, 1),
                        'Exercise_Minutes': round(daily_metrics['exercise_minutes'], 1),
                        'Stress_Level': stress_level
                    }
                    
                    all_data.append(row)
            
            print(f"  ✓ Generated {len(all_data) - (day_num * 1800 if day_num > 0 else 0)} samples")
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"✅ Dataset generation completed!")
        print(f"{'='*70}")
        print(f"📊 Total samples: {len(df):,}")
        print(f"📁 Saved to: {output_csv}")
        print(f"📏 Features: {len(df.columns)} (reduced from 44)")
        print(f"\n📈 Activity distribution:")
        print(df['Activity'].value_counts())
        print(f"\n🏠 Location distribution:")
        print(df['Location'].value_counts())
        print(f"\n😰 Stress statistics:")
        print(df['Stress_Level'].describe())
        
        return df


def main():
    """Main execution"""
    print("="*70)
    print("Optimized Health Data Generator v2.0")
    print("20 Features + Context-Stress Variations")
    print("="*70)
    
    # Create generator
    generator = OptimizedHealthDataGenerator(age=28, gender='Female')
    
    # Generate 30-day dataset
    df = generator.generate_30day_data(
        num_days=30,
        output_csv='data/optimized_health_data_20features_v2.csv'
    )
    
    print("\n" + "="*70)
    print("🎉 Generation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
