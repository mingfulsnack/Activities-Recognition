"""
Refactored Health Data Generator
Main orchestrator sử dụng các module nhỏ hơn
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

class RefactoredHealthDataGenerator:
    """
    Refactored Health Data Generator
    Tạo dữ liệu theo dõi sức khỏe với SEQUENTIAL BEHAVIORAL DATA
    """
    
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

    def calculate_enhanced_daily_metrics(self, date, schedule, day_context):
        """Tính toán các metrics với calories và steps theo giờ thay vì tổng ngày"""
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
        stress_level = day_context['stress_base']
        
        # Heart rate calculation using Age/Gender
        base_hr = self.user_profile.calculate_resting_heart_rate()
        hr_variation = (
            (day_context['stress_base'] - 4) * 6 +
            (1 - day_context['sleep_quality']) * 10 +
            daily_noise['health_variation'] * 12 +
            day_context['weather_effect'] * 6
        )
        # Ensure HR stays within realistic range for this person
        max_hr = self.user_profile.calculate_max_heart_rate()
        heart_rate_baseline = max(45, min(max_hr * 0.6, base_hr + hr_variation))
        
        # Screen time calculation
        base_screen = self.user_profile.profile['base_screen_time']
        work_modifier = {'very_low': -2.5, 'low': -1.2, 'normal': 0, 'high': 1.3, 'very_high': 2.8, 'none': -4.5}
        screen_time_variation = (
            work_modifier.get(day_context['work_intensity'], 0) +
            daily_noise['mood'] * 2.5 +
            random.uniform(-1.5, 1.5)
        )
        screen_time = max(2, min(16, base_screen + screen_time_variation))
        
        # Calculate hourly-based step count and calories
        total_steps = 0
        total_calories = 0
        
        for activity_block in schedule:
            activity = activity_block['activity']
            duration = activity_block['time_end'] - activity_block['time_start']
            
            # Calculate steps for this activity block
            block_steps = self.metrics_calculator.calculate_hourly_steps(
                activity, duration, day_context['energy_level']
            )
            total_steps += block_steps
            
            # Calculate calories for this activity block  
            block_calories = self.metrics_calculator.calculate_hourly_calories(
                activity, duration, stress_level
            )
            total_calories += block_calories
        
        # Add baseline metabolism for sleep time (8 hours * 60 cal/h = 480 cal)
        sleep_calories = actual_sleep * 60  # 60 cal/hour during sleep
        total_calories += sleep_calories
        
        # Add some daily variation
        step_variation = random.uniform(0.85, 1.15)
        calorie_variation = random.uniform(0.9, 1.1)
        
        final_steps = max(500, int(total_steps * step_variation))  # Minimum 500 steps/day
        final_calories = max(1200, int(total_calories * calorie_variation))  # Minimum 1200 cal/day
        
        # Reaction time
        reaction_time = self.metrics_calculator.calculate_reaction_time(
            stress_level, day_context['sleep_quality'], day_context['energy_level']
        )
        
        return {
            'Sleep_Duration': round(actual_sleep, 1),
            'Stress_Level': round(stress_level, 1),
            'Heart_Rate_Baseline': round(heart_rate_baseline),
            'Screen_Time': round(screen_time, 1),
            'Step_Count': final_steps,
            'Calories': final_calories,
            'Reaction_Time': reaction_time,
            'Sleep_Quality': round(day_context['sleep_quality'], 2),
            'Energy_Level': round(day_context['energy_level'], 2),
            'Mood_Score': round(5 + day_context['mood_factor'] * 3, 1),
            'Exercise_Minutes': round(day_context['exercise_intensity'] * 75, 0),
            'Social_Interaction': round(max(0, random.uniform(-0.2, 0.7)), 2)
        }

    def generate_accelerometer_with_variations(self, activity, location, stress_modifier, base_metrics, duration_hours):
        """Tạo accelerometer data sử dụng real WISDM samples"""
        samples_per_hour = 60 * self.samples_per_minute
        total_samples = int(duration_hours * samples_per_hour)
        
        if total_samples <= 0:
            return []
        
        data = []
        for i in range(total_samples):
            # GET REAL WISDM DATA: Use real accelerometer sample
            real_accel = self.wisdm_loader.get_real_accelerometer_sample(activity)
            x_real, y_real, z_real = real_accel
            
            # Add small variations để không bị duplicate hoàn toàn
            stress_noise = stress_modifier * 0.05  # Very small noise để maintain HAR accuracy
            fatigue_noise = (1 - base_metrics.get('Energy_Level', 0.7)) * 0.03
            total_noise = stress_noise + fatigue_noise
            
            # Apply tiny variations
            x_variation = total_noise * random.gauss(0, 0.02)
            y_variation = total_noise * random.gauss(0, 0.02)
            z_variation = total_noise * random.gauss(0, 0.015)
            
            final_x = x_real + x_variation
            final_y = y_real + y_variation
            final_z = z_real + z_variation
            
            data.append({
                'x': round(final_x, 3),
                'y': round(final_y, 3),
                'z': round(final_z, 3),
                'time_offset': random.uniform(-2, 2)
            })
        
        return data

    def generate_enhanced_dataset(self, start_date_str="2024-01-01", days=30):
        """
        Tạo dataset với SEQUENTIAL BEHAVIORAL DATA để hỗ trợ LSTM
        """
        print(f"🚀 Tạo REFACTORED HEALTH DATASET từ {start_date_str} trong {days} ngày...")
        print("🧠 Với support cho LSTM sequences: Screen Time, Phone Usage, Social Interaction")
        print("🔧 Sử dụng refactored architecture với modules tách biệt")
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = start_date + timedelta(days=days-1)
        
        # Reset behavioral state
        self.behavioral_tracker.reset_behavioral_state()
        
        # Generate life events
        self.life_events = self.schedule_generator.generate_life_events(start_date, end_date)
        print(f"📅 Tạo {len(self.life_events)} life events trong {days} ngày")
        
        all_data = []
        current_date = start_date
        
        for day_num in range(days):
            print(f"📊 Generating day {day_num + 1}/{days}: {current_date.strftime('%Y-%m-%d')}")
            
            # Generate improved schedule for better activity segments
            schedule, day_context = self.schedule_generator.generate_improved_daily_schedule(
                current_date, self.life_events
            )
            base_metrics = self.calculate_enhanced_daily_metrics(current_date, schedule, day_context)
            
            print(f"    📋 Generated {len(schedule)} activity segments for improved HAR compatibility")
            
            # CUMULATIVE TRACKING - Continue from previous day
            if day_num == 0:
                daily_cumulative_calories = 0
                daily_cumulative_steps = 0
            else:
                # Get last values from previous day
                if len(all_data) > 0:
                    daily_cumulative_calories = all_data[-1]['Calories'] 
                    daily_cumulative_steps = all_data[-1]['Step_Count']
                else:
                    daily_cumulative_calories = 0
                    daily_cumulative_steps = 0
            
            for slot in schedule:
                duration = slot['time_end'] - slot['time_start']
                
                # TẠO ACCELEROMETER DATA ĐỒNG BỘ VỚI ACTIVITY
                accelerometer_data = self.generate_accelerometer_with_variations(
                    slot['activity'], 
                    slot['location'], 
                    slot['stress_modifier'],
                    base_metrics,
                    duration
                )
                
                for i, accel in enumerate(accelerometer_data):
                    # CONSISTENT TIMESTAMP: Ensure proper 30-second intervals
                    if len(accelerometer_data) > 1:
                        # Calculate exact time within slot with consistent intervals
                        time_fraction = i / (len(accelerometer_data) - 1) if len(accelerometer_data) > 1 else 0
                        hours = slot['time_start'] + time_fraction * duration
                    else:
                        hours = slot['time_start']
                    
                    # Handle day boundary properly
                    if hours >= 24:
                        sample_datetime = current_date + timedelta(days=1, hours=hours-24)
                    else:
                        sample_datetime = current_date + timedelta(hours=hours)
                    
                    # Add small random variation (max ±2 seconds) for realism
                    seconds_offset = random.uniform(-2, 2)
                    sample_datetime += timedelta(seconds=seconds_offset)
                    
                    # REALISTIC STRESS CALCULATION for prediction modeling
                    previous_stress_levels = []
                    if len(all_data) > 0:
                        # Get last few stress levels for momentum calculation
                        previous_stress_levels = [d['Stress_Level'] for d in all_data[-10:]]
                    
                    current_stress = self.metrics_calculator.calculate_realistic_stress_level(
                        base_metrics['Stress_Level'], hours, slot['activity'], slot['location'],
                        base_metrics['Heart_Rate_Baseline'], base_metrics['Sleep_Quality'],
                        day_context['work_intensity'], previous_stress_levels
                    )
                    
                    # UPDATE BEHAVIORAL STATE
                    current_data = {'stress_level': current_stress}
                    self.behavioral_tracker.update_behavioral_state(
                        sample_datetime, current_data, slot['activity'], slot['location']
                    )
                    
                    # GET BEHAVIORAL FEATURES FROM SEQUENCES
                    behavioral_features = self.behavioral_tracker.get_behavioral_features(sample_datetime)
                    
                    # Calculate heart rate using metrics calculator
                    current_hr = self.metrics_calculator.calculate_heart_rate(
                        slot['activity'], current_stress, 
                        base_metrics['Heart_Rate_Baseline'], base_metrics['Energy_Level']
                    )
                    
                    # REALISTIC MOOD SCORE với intra-day variation
                    mood_score = self.metrics_calculator.calculate_mood_score(
                        day_context['mood_factor'], hours, slot['activity'], 
                        slot['location'], current_stress
                    )
                    
                    # REALISTIC CALORIES AND STEPS TRACKING
                    sample_duration = duration / len(accelerometer_data)
                    
                    # Steps TĂNG khi Walking, Jogging, Upstairs, Downstairs
                    if slot['activity'] in ['Walking', 'Jogging', 'Upstairs', 'Downstairs']:
                        steps_increment = self.metrics_calculator.calculate_hourly_steps(
                            slot['activity'], sample_duration, base_metrics['Energy_Level']
                        )
                        daily_cumulative_steps += steps_increment
                    # Sitting và Standing KHÔNG tăng steps đáng kể
                    
                    # Calories = TỔNG NĂNG LƯỢNG TIÊU THỤ (energy expenditure)
                    # BASE METABOLIC CALORIES cho sample này
                    base_calories = self.metrics_calculator.calculate_hourly_calories(
                        slot['activity'], sample_duration, current_stress
                    )
                    daily_cumulative_calories += base_calories
                    
                    # Environmental factors
                    ambient_light = max(0, min(1200, 
                        350 + (hours - 12) * 25 + random.uniform(-80, 80)
                    ))
                    
                    noise_level = {
                        'home': random.uniform(32, 58),
                        'work': random.uniform(42, 68),
                        'outdoor': random.uniform(48, 78),
                        'commute': random.uniform(58, 85),
                        'social': random.uniform(52, 88),
                        'gym': random.uniform(45, 75)
                    }.get(slot['location'], 45)
                    
                    # STABLE WEATHER PER DAY với gradual changes
                    daily_weather = self.schedule_generator.get_daily_noise_factor(current_date)['weather_mood']
                    # Base weather for the day (stable)
                    daily_base_weather = 5.5 + daily_weather * 3  # Reduced variation
                    
                    # Gradual intra-day changes (morning cool, noon hot, evening cool)
                    time_of_day_effect = 0.5 * np.sin((hours - 6) * np.pi / 12)  # -0.5 to +0.5
                    
                    # Small random variation (much smaller than before)
                    weather_condition = max(0, min(10, 
                        daily_base_weather + time_of_day_effect + random.uniform(-0.3, 0.3)
                    ))
                    
                    # FORCE ACTIVITY CONSISTENCY for better HAR accuracy
                    consistent_activity = slot['activity']  # Use planned activity consistently
                    
                    # CREATE RECORD WITH CUMULATIVE CALORIES AND STEPS
                    record = {
                        'Timestamp': sample_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        'Age': self.user_profile.profile['Age'],
                        'Gender': self.user_profile.profile['Gender'],
                        'Sleep_Duration': base_metrics['Sleep_Duration'],
                        'Stress_Level': round(current_stress, 1),
                        'Heart_Rate': current_hr,
                        'Screen_Time': base_metrics['Screen_Time'],
                        'Step_Count': daily_cumulative_steps,  # CUMULATIVE STEPS
                        'Calories': daily_cumulative_calories,  # CUMULATIVE CALORIES
                        'Accelerometer_X': accel['x'],
                        'Accelerometer_Y': accel['y'],
                        'Accelerometer_Z': accel['z'],
                        'Activity': consistent_activity,  # CONSISTENT ACTIVITY
                        'Location': slot['location'],
                        'Ambient_Light': round(ambient_light, 1),
                        'Noise_Level': round(noise_level, 1),
                        'Weather_Condition': round(weather_condition, 1),
                        'Reaction_Time': base_metrics['Reaction_Time'],
                        'Sleep_Quality': base_metrics['Sleep_Quality'],
                        'Energy_Level': base_metrics['Energy_Level'],
                        'Mood_Score': mood_score,
                        'Exercise_Minutes': base_metrics['Exercise_Minutes'],
                        'Social_Interaction': base_metrics['Social_Interaction'],
                        
                        # SEQUENTIAL BEHAVIORAL FEATURES FOR LSTM
                        **behavioral_features
                    }
                    
                    all_data.append(record)
            
            current_date += timedelta(days=1)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        filename = f'data/quota_balanced_health_data_{days}days.csv'
        df.to_csv(filename, index=False)
        
        print(f"\n🎉 === REFACTORED DATASET SUMMARY ===")
        print(f"📈 Tổng số records: {len(df):,}")
        print(f"📅 Số ngày: {days}")
        print(f"📊 Records per day trung bình: {len(df)//days:,}")
        print(f"💾 File saved: {filename}")
        
        # VERIFICATION: Cumulative data quality
        print(f"\n✅ === CUMULATIVE DATA VERIFICATION ===")
        sample_day_data = df[df['Timestamp'].str.startswith('2024-01-01')]
        if len(sample_day_data) > 0:
            calories_progression = sample_day_data['Calories'].tolist()
            steps_progression = sample_day_data['Step_Count'].tolist()
            print(f"📊 Day 1 Calories: {calories_progression[0]} → {calories_progression[-1]} (cumulative)")
            print(f"👟 Day 1 Steps: {steps_progression[0]} → {steps_progression[-1]} (cumulative)")
            
            # Check if truly cumulative (non-decreasing)
            calories_increasing = all(calories_progression[i] <= calories_progression[i+1] for i in range(len(calories_progression)-1))
            steps_increasing = all(steps_progression[i] <= steps_progression[i+1] for i in range(len(steps_progression)-1))
            print(f"✅ Calories tăng dần: {'YES' if calories_increasing else 'NO'}")
            print(f"✅ Steps tăng dần: {'YES' if steps_increasing else 'NO'}")
        
        # VERIFICATION: Activity-Accelerometer consistency
        print(f"\n✅ === ACTIVITY-ACCELEROMETER CONSISTENCY ===")
        for activity in df['Activity'].unique():
            activity_data = df[df['Activity'] == activity]
            magnitudes = np.sqrt(activity_data['Accelerometer_X']**2 + 
                               activity_data['Accelerometer_Y']**2 + 
                               activity_data['Accelerometer_Z']**2)
            print(f"{activity:10s}: Magnitude {magnitudes.min():.1f} - {magnitudes.max():.1f} (avg: {magnitudes.mean():.1f})")
        
        # Print behavioral features summary
        print(f"\n🧠 === BEHAVIORAL SEQUENCE FEATURES ===")
        behavioral_columns = [col for col in df.columns if any(prefix in col for prefix in 
                             ['Screen_Usage', 'Phone_', 'Social_', 'Stress_Current', 'Stress_Velocity'])]
        
        print(f"📱 Screen Usage Features: {len([c for c in behavioral_columns if 'Screen' in c])}")
        print(f"📞 Phone Interaction Features: {len([c for c in behavioral_columns if 'Phone' in c])}")  
        print(f"👥 Social Sequence Features: {len([c for c in behavioral_columns if 'Social' in c])}")
        print(f"😰 Stress Sequence Features: {len([c for c in behavioral_columns if 'Stress_' in c])}")
        
        # Verify sequential data quality
        print(f"\n✅ === SEQUENCE DATA QUALITY ===")
        print(f"📊 Screen Usage Variance: {df['Screen_Usage_Variance'].mean():.4f} (>0 = có variation)")
        print(f"📱 Phone Events per 30min: {df['Phone_Events_Count_30min'].mean():.2f}")
        print(f"👥 Social Interaction Trend: {df['Social_Interaction_Trend'].std():.4f} (>0 = có variation)")
        print(f"😰 Stress Velocity: {df['Stress_Velocity'].std():.4f} (>0 = có changes)")
        
        # Life events summary
        event_days = len(self.life_events)
        print(f"\n🎭 Life Events: {event_days} ngày có sự kiện đặc biệt ({event_days/days*100:.1f}%)")
        
        print(f"\n🚀 === REFACTORED ARCHITECTURE BENEFITS ===")
        print(f"✅ Modular design: Easier maintenance và extension")
        print(f"✅ Clear separation of concerns: Mỗi module có responsibility riêng")
        print(f"✅ Reusable components: Có thể tái sử dụng cho projects khác")
        print(f"✅ Better testability: Có thể test từng module độc lập")
        print(f"✅ Improved readability: Code dễ hiểu và maintain hơn")
        
        return df


if __name__ == "__main__":
    print("=== REFACTORED HEALTH DATASET GENERATOR ===")
    print("🔧 Modular architecture với 6 core modules")
    print("🎯 Improved maintainability và extensibility")
    print("🧠 Enhanced LSTM sequence modeling")
    
    generator = RefactoredHealthDataGenerator()
    
    print("\nBắt đầu tạo refactored dataset...")
    df = generator.generate_enhanced_dataset("2024-01-01", 30)
    
    print("\n📋 === SAMPLE REFACTORED DATA ===")
    # Show key improvements in sample
    sample_cols = ['Timestamp', 'Activity', 'Calories', 'Step_Count', 'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    sample_data = df[sample_cols].head(10)
    print(sample_data)
    
    print(f"\n🎯 Refactored Dataset ready với improved architecture!")
    print(f"✅ Modular, maintainable và extensible!")
    print(f"✅ All features preserved với better organization!")
