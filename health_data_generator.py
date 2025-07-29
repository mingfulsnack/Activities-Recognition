import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import math

class HealthMonitoringDataGenerator:
    """
    Tạo dữ liệu theo dõi sức khỏe real-time cho 1 người trong 1 tháng
    Với SEQUENTIAL BEHAVIORAL DATA để hỗ trợ LSTM sequences
    """
    
    def __init__(self):
        # Thông tin cơ bản của người được theo dõi
        self.user_profile = {
            'Age': 28,
            'Gender': 'Female',
            'base_sleep_duration': 7.5,
            'base_screen_time': 8.0,
            'base_stress_level': 4,
            'base_reaction_time': 380.0
        }
        
        # Patterns hoạt động trong ngày
        self.daily_patterns = {
            'wake_up_time': (6, 8),
            'work_start': (8, 9),
            'lunch_time': (12, 13),
            'work_end': (17, 18),
            'dinner_time': (18, 20),
            'sleep_time': (22, 24),
            'exercise_days': [1, 3, 5, 6]
        }
        
        # ENHANCED: Sampling rate cao hơn để tạo sequences
        self.samples_per_minute = 2  # 2 samples/phút = 2880 samples/ngày
        
        # BEHAVIORAL PATTERN TRACKING - Lưu trữ state để tạo sequences
        self.behavioral_state = {
            'recent_screen_usage': [],      # Track 30 phút gần nhất
            'phone_interaction_history': [], # Track phone touches/unlocks
            'social_activity_timeline': [],  # Track social interactions
            'stress_accumulation': [],       # Track stress changes over time
            'activity_transitions': [],     # Track activity changes
            'environmental_changes': []     # Track environment transitions
        }
        
        # Life events
        self.life_events = {}

    def update_behavioral_state(self, timestamp, current_data, activity, location):
        """
        Cập nhật behavioral state để tạo sequential patterns
        """
        # 1. SCREEN TIME SEQUENCES
        # Tính screen usage trong 5 phút gần đây
        current_screen_intensity = self.calculate_screen_intensity(
            activity, location, current_data.get('stress_level', 4)
        )
        
        # Lưu screen usage với timestamp
        self.behavioral_state['recent_screen_usage'].append({
            'timestamp': timestamp,
            'intensity': current_screen_intensity,
            'activity': activity,
            'location': location
        })
        
        # Chỉ giữ 30 phút gần nhất (60 samples)
        cutoff_time = timestamp - timedelta(minutes=30)
        self.behavioral_state['recent_screen_usage'] = [
            x for x in self.behavioral_state['recent_screen_usage'] 
            if x['timestamp'] > cutoff_time
        ]
        
        # 2. PHONE INTERACTION SEQUENCES  
        # Mô phỏng phone unlocks/notifications
        phone_events = self.generate_phone_interactions(
            timestamp, activity, current_data.get('stress_level', 4)
        )
        
        for event in phone_events:
            self.behavioral_state['phone_interaction_history'].append({
                'timestamp': timestamp,
                'event_type': event['type'],  # 'unlock', 'notification', 'call', 'text'
                'duration': event['duration'],
                'intensity': event['intensity']
            })
        
        # Giữ 2 giờ gần nhất
        cutoff_time = timestamp - timedelta(hours=2)
        self.behavioral_state['phone_interaction_history'] = [
            x for x in self.behavioral_state['phone_interaction_history']
            if x['timestamp'] > cutoff_time
        ]
        
        # 3. SOCIAL INTERACTION SEQUENCES
        social_level = self.calculate_social_interaction(
            timestamp, activity, location
        )
        
        self.behavioral_state['social_activity_timeline'].append({
            'timestamp': timestamp,
            'social_level': social_level,
            'location': location,
            'interaction_type': self.determine_social_type(location, activity)
        })
        
        # Giữ 4 giờ gần nhất
        cutoff_time = timestamp - timedelta(hours=4)
        self.behavioral_state['social_activity_timeline'] = [
            x for x in self.behavioral_state['social_activity_timeline']
            if x['timestamp'] > cutoff_time
        ]
        
        # 4. STRESS ACCUMULATION SEQUENCES
        self.behavioral_state['stress_accumulation'].append({
            'timestamp': timestamp,
            'stress_level': current_data.get('stress_level', 4),
            'stress_trend': self.calculate_stress_trend(),
            'stress_velocity': self.calculate_stress_velocity()
        })
        
        # Giữ 6 giờ gần nhất
        cutoff_time = timestamp - timedelta(hours=6)
        self.behavioral_state['stress_accumulation'] = [
            x for x in self.behavioral_state['stress_accumulation']
            if x['timestamp'] > cutoff_time
        ]

    def calculate_screen_intensity(self, activity, location, stress_level):
        """
        Tính screen usage intensity dựa trên context
        """
        base_intensity = {
            'Sitting': 0.7,    # Ngồi thường xem nhiều
            'Standing': 0.4,   # Đứng ít xem hơn
            'Walking': 0.2,    # Đi bộ ít xem
            'Jogging': 0.05    # Chạy hầu như không xem
        }.get(activity, 0.3)
        
        location_modifier = {
            'home': 1.3,       # Ở nhà xem nhiều
            'work': 1.1,       # Làm việc vừa phải
            'commute': 0.9,    # Di chuyển ít hơn
            'outdoor': 0.6,    # Ngoài trời ít
            'social': 0.4      # Xã hội ít xem
        }.get(location, 1.0)
        
        stress_modifier = 1 + (stress_level - 4) * 0.15  # Stress cao -> xem nhiều
        
        # Random variation
        variation = random.uniform(0.7, 1.4)
        
        intensity = base_intensity * location_modifier * stress_modifier * variation
        return max(0, min(1, intensity))

    def generate_phone_interactions(self, timestamp, activity, stress_level):
        """
        Tạo phone interaction events trong 5 phút gần đây
        """
        events = []
        
        # Tần suất dựa trên activity và stress
        base_frequency = {
            'Sitting': 0.6,    # Ngồi hay dùng phone
            'Standing': 0.4,
            'Walking': 0.2,
            'Jogging': 0.05
        }.get(activity, 0.3)
        
        stress_frequency = base_frequency * (1 + (stress_level - 4) * 0.2)
        
        # Tạo events ngẫu nhiên
        if random.random() < stress_frequency:
            event_types = ['unlock', 'notification', 'call', 'text', 'app_usage']
            weights = [0.4, 0.3, 0.1, 0.15, 0.05]
            
            event_type = np.random.choice(event_types, p=weights)
            
            # Duration và intensity dựa trên event type
            if event_type == 'unlock':
                duration = random.uniform(5, 45)  # 5-45 giây
                intensity = 0.3
            elif event_type == 'notification':
                duration = random.uniform(2, 8)   # 2-8 giây
                intensity = 0.5
            elif event_type == 'call':
                duration = random.uniform(30, 600) # 30s-10min
                intensity = 0.8
            elif event_type == 'text':
                duration = random.uniform(10, 120) # 10s-2min
                intensity = 0.6
            else:  # app_usage
                duration = random.uniform(60, 300) # 1-5min
                intensity = 0.9
            
            events.append({
                'type': event_type,
                'duration': duration,
                'intensity': intensity
            })
        
        return events

    def calculate_social_interaction(self, timestamp, activity, location):
        """
        Tính social interaction level
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Base social level theo thời gian và địa điểm
        time_factor = {
            'morning': 0.3 if 6 <= hour < 9 else 0,
            'work': 0.6 if 9 <= hour < 17 and weekday < 5 else 0,
            'evening': 0.8 if 17 <= hour < 22 else 0,
            'night': 0.2 if 22 <= hour or hour < 6 else 0
        }
        
        base_social = max(time_factor.values())
        
        location_modifier = {
            'work': 1.2,
            'social': 2.0,
            'outdoor': 1.4,
            'commute': 0.8,
            'home': 0.5
        }.get(location, 0.7)
        
        # Weekend boost
        if weekday >= 5:
            location_modifier *= 1.5
        
        # Activity modifier
        activity_modifier = {
            'Sitting': 1.0,
            'Standing': 1.2,
            'Walking': 1.1,
            'Jogging': 0.7
        }.get(activity, 1.0)
        
        social_level = base_social * location_modifier * activity_modifier
        social_level += random.uniform(-0.3, 0.3)  # Random variation
        
        return max(0, min(1, social_level))

    def determine_social_type(self, location, activity):
        """
        Xác định loại tương tác xã hội
        """
        if location == 'work':
            return np.random.choice(['meeting', 'colleague_chat', 'email', 'alone'], 
                                  p=[0.3, 0.4, 0.2, 0.1])
        elif location == 'social':
            return np.random.choice(['friend_gathering', 'family_time', 'party', 'date'], 
                                  p=[0.4, 0.3, 0.2, 0.1])
        elif location == 'outdoor':
            return np.random.choice(['stranger_interaction', 'friend_meetup', 'exercise_buddy', 'alone'], 
                                  p=[0.3, 0.2, 0.2, 0.3])
        else:
            return np.random.choice(['family', 'video_call', 'text_friends', 'alone'], 
                                  p=[0.4, 0.2, 0.2, 0.2])

    def calculate_stress_trend(self):
        """
        Tính xu hướng stress trong 1 giờ gần đây
        """
        if len(self.behavioral_state['stress_accumulation']) < 2:
            return 0
        
        recent_stress = [x['stress_level'] for x in self.behavioral_state['stress_accumulation'][-12:]]
        if len(recent_stress) >= 2:
            return recent_stress[-1] - recent_stress[0]
        return 0

    def calculate_stress_velocity(self):
        """
        Tính tốc độ thay đổi stress
        """
        if len(self.behavioral_state['stress_accumulation']) < 3:
            return 0
        
        recent_stress = [x['stress_level'] for x in self.behavioral_state['stress_accumulation'][-6:]]
        if len(recent_stress) >= 3:
            # Simple derivative approximation
            return (recent_stress[-1] - recent_stress[-3]) / 2
        return 0

    def get_behavioral_features(self, timestamp):
        """
        Trích xuất behavioral features từ sequences
        """
        features = {}
        
        # 1. SCREEN TIME FEATURES
        if self.behavioral_state['recent_screen_usage']:
            screen_data = [x['intensity'] for x in self.behavioral_state['recent_screen_usage']]
            features.update({
                'Screen_Usage_Current': screen_data[-1] if screen_data else 0,
                'Screen_Usage_5min_Avg': np.mean(screen_data[-10:]) if len(screen_data) >= 10 else np.mean(screen_data),
                'Screen_Usage_15min_Avg': np.mean(screen_data[-30:]) if len(screen_data) >= 30 else np.mean(screen_data),
                'Screen_Usage_Trend': (screen_data[-1] - screen_data[0]) if len(screen_data) >= 2 else 0,
                'Screen_Usage_Variance': np.var(screen_data) if len(screen_data) >= 2 else 0
            })
        else:
            features.update({
                'Screen_Usage_Current': 0,
                'Screen_Usage_5min_Avg': 0,
                'Screen_Usage_15min_Avg': 0,
                'Screen_Usage_Trend': 0,
                'Screen_Usage_Variance': 0
            })
        
        # 2. PHONE INTERACTION FEATURES
        cutoff_30min = timestamp - timedelta(minutes=30)
        recent_phone = [x for x in self.behavioral_state['phone_interaction_history'] 
                       if x['timestamp'] > cutoff_30min]
        
        if recent_phone:
            features.update({
                'Phone_Events_Count_30min': len(recent_phone),
                'Phone_Usage_Intensity': np.mean([x['intensity'] for x in recent_phone]),
                'Phone_Avg_Duration': np.mean([x['duration'] for x in recent_phone]),
                'Phone_Last_Event_Minutes': (timestamp - recent_phone[-1]['timestamp']).total_seconds() / 60,
                'Phone_Event_Frequency': len(recent_phone) / 30  # events per minute
            })
        else:
            features.update({
                'Phone_Events_Count_30min': 0,
                'Phone_Usage_Intensity': 0,
                'Phone_Avg_Duration': 0,
                'Phone_Last_Event_Minutes': 30,
                'Phone_Event_Frequency': 0
            })
        
        # 3. SOCIAL INTERACTION FEATURES
        cutoff_2hour = timestamp - timedelta(hours=2)
        recent_social = [x for x in self.behavioral_state['social_activity_timeline']
                        if x['timestamp'] > cutoff_2hour]
        
        if recent_social:
            social_levels = [x['social_level'] for x in recent_social]
            features.update({
                'Social_Current_Level': social_levels[-1],
                'Social_30min_Avg': np.mean(social_levels[-15:]) if len(social_levels) >= 15 else np.mean(social_levels),
                'Social_1hour_Avg': np.mean(social_levels[-30:]) if len(social_levels) >= 30 else np.mean(social_levels),
                'Social_2hour_Avg': np.mean(social_levels),
                'Social_Interaction_Trend': (social_levels[-1] - social_levels[0]) if len(social_levels) >= 2 else 0,
                'Social_Stability': 1 - np.var(social_levels) if len(social_levels) >= 2 else 1
            })
        else:
            features.update({
                'Social_Current_Level': 0,
                'Social_30min_Avg': 0,
                'Social_1hour_Avg': 0,
                'Social_2hour_Avg': 0,
                'Social_Interaction_Trend': 0,
                'Social_Stability': 1
            })
        
        # 4. STRESS ACCUMULATION FEATURES
        if self.behavioral_state['stress_accumulation']:
            stress_data = self.behavioral_state['stress_accumulation']
            features.update({
                'Stress_Current_Trend': self.calculate_stress_trend(),
                'Stress_Velocity': self.calculate_stress_velocity(),
                'Stress_1hour_Avg': np.mean([x['stress_level'] for x in stress_data[-30:]]) if len(stress_data) >= 30 else np.mean([x['stress_level'] for x in stress_data]),
                'Stress_Accumulation_Score': sum([max(0, x['stress_level'] - 5) for x in stress_data]) / len(stress_data),
                'Stress_Recovery_Indicator': len([x for x in stress_data[-10:] if x['stress_level'] < 4]) / min(10, len(stress_data))
            })
        else:
            features.update({
                'Stress_Current_Trend': 0,
                'Stress_Velocity': 0,
                'Stress_1hour_Avg': 4,
                'Stress_Accumulation_Score': 0,
                'Stress_Recovery_Indicator': 0.5
            })
        
        return features

    def get_daily_noise_factor(self, date):
        """Tạo các yếu tố nhiễu thực tế cho từng ngày"""
        # [Giữ nguyên code cũ]
        day_seed = date.toordinal()
        np.random.seed(day_seed)
        
        weekday = date.weekday()
        weekly_stress = 0.15 * (weekday / 4) if weekday < 5 else -0.3
        
        day_of_month = date.day
        monthly_cycle = 0.2 * np.sin(2 * np.pi * day_of_month / 28)
        
        week_number = date.isocalendar()[1]
        weekly_fatigue = 0.05 * (week_number % 4)
        
        noise = {
            'sleep_pattern': np.random.normal(0, 0.8),
            'sleep_quality': np.random.normal(0, 0.2),
            'stress_fluctuation': weekly_stress + np.random.normal(0, 0.3),
            'mood': np.random.normal(0, 0.25),
            'energy': monthly_cycle + weekly_fatigue + np.random.normal(0, 0.15),
            'appetite': np.random.normal(0, 0.15),
            'social': np.random.normal(0, 0.2),
            'weather_mood': np.random.uniform(-0.2, 0.2),
            'work_pressure': np.random.uniform(0, 0.4) if weekday < 5 else 0,
            'health_variation': np.random.normal(0, 0.12),
        }
        
        np.random.seed(None)
        return noise

    def generate_life_events(self, start_date, end_date):
        """Tạo các sự kiện đặc biệt trong cuộc sống ảnh hưởng nhiều ngày"""
        # [Giữ nguyên code cũ]
        events = {}
        current = start_date
        
        while current <= end_date:
            if random.random() < 0.06:
                event_type = random.choice([
                    'sick', 'deadline', 'family_visit', 'vacation', 
                    'bad_news', 'good_news', 'exam', 'party',
                    'travel', 'medical_checkup', 'job_interview',
                    'presentation', 'fight_with_friend', 'promotion',
                    'menstrual_cycle', 'pms', 'exercise_rest_day'
                ])
                
                duration = random.randint(1, 4)
                intensity = random.uniform(0.3, 0.9)
                
                for i in range(duration):
                    event_date = current + timedelta(days=i)
                    if event_date <= end_date:
                        events[event_date] = {
                            'type': event_type,
                            'intensity': intensity,
                            'day_in_event': i + 1
                        }
                
                current += timedelta(days=duration)
            else:
                current += timedelta(days=1)
        
        return events

    def generate_daily_schedule(self, date):
        """Tạo lịch trình hoạt động cho 1 ngày với biến động realistic"""
        # [Giữ nguyên phần lớn code cũ, chỉ thêm behavioral state updates]
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5
        
        daily_noise = self.get_daily_noise_factor(date)
        
        life_event = self.life_events.get(date, None)
        event_modifier = 0
        if life_event:
            if life_event['type'] in ['sick', 'deadline', 'bad_news', 'exam', 'pms']:
                event_modifier = life_event['intensity']
            elif life_event['type'] in ['vacation', 'good_news', 'party', 'promotion']:
                event_modifier = -life_event['intensity']
        
        if is_weekend:
            wake_up = random.uniform(7.5, 10.5) + daily_noise['sleep_pattern']
            sleep_time = (random.uniform(23.5, 1.5) % 24) + random.uniform(-1, 1)
        else:
            wake_up = random.uniform(*self.daily_patterns['wake_up_time']) + daily_noise['sleep_pattern']
            sleep_time = random.uniform(*self.daily_patterns['sleep_time']) + random.uniform(-0.8, 0.8)
        
        wake_up = max(4.5, min(12, wake_up))
        sleep_time = max(20, min(26, sleep_time)) % 24
        
        day_context = {
            'sleep_quality': max(0.25, min(1.0, 0.8 + daily_noise['sleep_quality'])),
            'energy_level': max(0.2, min(1.0, 0.7 + daily_noise['energy'] + event_modifier * 0.3)),
            'stress_base': max(1, min(9, self.user_profile['base_stress_level'] + 
                              daily_noise['stress_fluctuation'] * 3 + event_modifier * 4)),
            'mood_factor': daily_noise['mood'] + event_modifier * 0.5,
            'has_social_event': random.random() < (0.5 if is_weekend else 0.15),
            'work_intensity': random.choice(['very_low', 'low', 'normal', 'high', 'very_high']) if not is_weekend else 'none',
            'weather_effect': daily_noise['weather_mood'],
            'appetite_change': daily_noise['appetite'],
            'exercise_intensity': random.uniform(0.4, 1.3) if day_of_week in self.daily_patterns['exercise_days'] else 0,
            'life_event': life_event
        }
        
        schedule = []
        
        # Morning activities
        morning_activity = 'Standing' if day_context['sleep_quality'] > 0.6 else 'Sitting'
        schedule.append({
            'time_start': wake_up,
            'time_end': wake_up + random.uniform(0.5, 1.8),
            'activity': morning_activity,
            'location': 'home',
            'stress_modifier': daily_noise['mood'] * 0.5
        })
        
        if day_context['energy_level'] > 0.7:
            prep_activity = 'Walking'
            prep_duration = random.uniform(1, 1.8)
        else:
            prep_activity = 'Sitting'
            prep_duration = random.uniform(0.5, 1.2)
            
        schedule.append({
            'time_start': wake_up + 1,
            'time_end': wake_up + 1 + prep_duration,
            'activity': prep_activity,
            'location': 'home',
            'stress_modifier': daily_noise['work_pressure'] * 0.3
        })
        
        if not is_weekend:
            # Work day activities
            commute_stress = 0.5 + daily_noise['work_pressure']
            work_intensity_map = {'very_high': 0.4, 'high': 0.3, 'normal': 0, 'low': -0.1, 'very_low': -0.2}
            commute_stress += work_intensity_map.get(day_context['work_intensity'], 0)
                
            schedule.append({
                'time_start': wake_up + 2.5,
                'time_end': wake_up + 3.5,
                'activity': 'Walking',
                'location': 'commute',
                'stress_modifier': commute_stress
            })
            
            work_hours = 8 + random.uniform(-1.5, 2.5)
            work_stress_base = {'very_low': 0, 'low': 0.3, 'normal': 0.6, 'high': 1.0, 'very_high': 1.5}
            
            for i in range(int(work_hours)):
                hour_stress = work_stress_base[day_context['work_intensity']]
                if i < 2:
                    hour_stress *= 0.7
                elif i > 6:
                    hour_stress *= 1.3
                    
                if random.random() < 0.35:
                    hour_stress += 0.4
                    work_activity = 'Sitting'
                else:
                    work_activity = np.random.choice(['Sitting', 'Standing', 'Walking'], 
                                                    p=[0.6, 0.25, 0.15])
                    
                schedule.append({
                    'time_start': wake_up + 3.5 + i,
                    'time_end': wake_up + 4.5 + i,
                    'activity': work_activity,
                    'location': 'work',
                    'stress_modifier': hour_stress + random.uniform(-0.3, 0.3)
                })
            
            schedule.append({
                'time_start': wake_up + 3.5 + work_hours,
                'time_end': wake_up + 4.5 + work_hours,
                'activity': 'Walking',
                'location': 'commute',
                'stress_modifier': 0.2 + daily_noise['mood'] * 0.3
            })
        else:
            # Weekend activities
            weekend_activities = ['Standing', 'Walking', 'Sitting']
            for i in range(6):
                activity = random.choice(weekend_activities)
                location = random.choice(['home', 'outdoor', 'social'])
                
                if day_context['has_social_event']:
                    stress_mod = -0.3 + daily_noise['social']
                else:
                    stress_mod = -0.15 + random.uniform(-0.15, 0.15)
                    
                schedule.append({
                    'time_start': wake_up + 2.5 + i,
                    'time_end': wake_up + 3.5 + i,
                    'activity': activity,
                    'location': location,
                    'stress_modifier': stress_mod
                })
        
        # Exercise
        if day_context['exercise_intensity'] > 0:
            exercise_time = random.uniform(17, 20)
            exercise_duration = 0.5 + day_context['exercise_intensity'] * 0.7
            
            schedule.append({
                'time_start': exercise_time,
                'time_end': exercise_time + exercise_duration,
                'activity': 'Jogging',
                'location': 'outdoor',
                'stress_modifier': -0.4 * day_context['exercise_intensity']
            })
        
        # Evening
        evening_stress = daily_noise['mood'] * 0.5
        if day_context['mood_factor'] > 0:
            evening_activity = random.choice(['Standing', 'Walking'])
        else:
            evening_activity = 'Sitting'
            
        schedule.append({
            'time_start': sleep_time - 2.5,
            'time_end': sleep_time,
            'activity': evening_activity,
            'location': 'home',
            'stress_modifier': evening_stress
        })
        
        return schedule, day_context

    def calculate_enhanced_daily_metrics(self, date, schedule, day_context):
        """Tính toán các metrics với nhiều biến động realistic"""
        # [Giữ nguyên code cũ]
        daily_noise = self.get_daily_noise_factor(date)
        
        base_sleep = self.user_profile['base_sleep_duration']
        sleep_variation = daily_noise['sleep_pattern'] * 1.2
        
        if day_context['life_event']:
            event_type = day_context['life_event']['type']
            if event_type in ['sick', 'stress', 'deadline', 'exam']:
                sleep_variation -= 0.8
            elif event_type in ['vacation', 'weekend_trip']:
                sleep_variation += 0.5
        
        actual_sleep = max(4, min(12, base_sleep + sleep_variation))
        
        stress_level = day_context['stress_base']
        
        base_hr = 70
        hr_variation = (
            (day_context['stress_base'] - 4) * 6 +
            (1 - day_context['sleep_quality']) * 10 +
            daily_noise['health_variation'] * 12 +
            day_context['weather_effect'] * 6
        )
        heart_rate_baseline = max(55, min(95, base_hr + hr_variation))
        
        base_screen = self.user_profile['base_screen_time']
        work_modifier = {'very_low': -2.5, 'low': -1.2, 'normal': 0, 'high': 1.3, 'very_high': 2.8, 'none': -4.5}
        screen_time_variation = (
            work_modifier.get(day_context['work_intensity'], 0) +
            daily_noise['mood'] * 2.5 +
            random.uniform(-1.5, 1.5)
        )
        screen_time = max(2, min(16, base_screen + screen_time_variation))
        
        base_steps = 8500
        step_variation = (
            day_context['energy_level'] * 3500 +
            day_context['exercise_intensity'] * 6000 +
            daily_noise['social'] * 2500 +
            random.randint(-2000, 2000)
        )
        step_count = max(1500, min(25000, int(base_steps + step_variation)))
        
        base_calories = 1850
        calorie_variation = (
            step_count * 0.035 +
            day_context['exercise_intensity'] * 400 +
            daily_noise['appetite'] * 250 +
            stress_level * 25
        )
        calories = max(1200, min(3500, int(base_calories + calorie_variation)))
        
        base_reaction = self.user_profile['base_reaction_time']
        reaction_variation = (
            (1 - day_context['sleep_quality']) * 60 +
            stress_level * 10 +
            (1 - day_context['energy_level']) * 40 +
            random.uniform(-25, 25)
        )
        reaction_time = max(250, min(650, base_reaction + reaction_variation))
        
        return {
            'Sleep_Duration': round(actual_sleep, 1),
            'Stress_Level': round(stress_level, 1),
            'Heart_Rate_Baseline': round(heart_rate_baseline),
            'Screen_Time': round(screen_time, 1),
            'Step_Count': step_count,
            'Calories': calories,
            'Reaction_Time': round(reaction_time, 1),
            'Sleep_Quality': round(day_context['sleep_quality'], 2),
            'Energy_Level': round(day_context['energy_level'], 2),
            'Mood_Score': round(5 + day_context['mood_factor'] * 3, 1),
            'Exercise_Minutes': round(day_context['exercise_intensity'] * 75, 0),
            'Social_Interaction': round(max(0, daily_noise['social'] + 0.5), 2)
        }

    def generate_accelerometer_with_variations(self, activity, location, stress_modifier, base_metrics, duration_hours):
        """Tạo dữ liệu accelerometer với nhiều biến động realistic"""
        # [Giữ nguyên code cũ]
        samples_per_hour = 60 * self.samples_per_minute
        total_samples = int(duration_hours * samples_per_hour)
        
        if total_samples <= 0:
            return []
        
        activity_patterns = {
            'Walking': {'x_range': (0, 4.2), 'y_range': (-2.5, 2.5), 'z_range': (7.5, 12.5), 'frequency': 2.1},
            'Jogging': {'x_range': (-3.8, 3.8), 'y_range': (-4.5, 4.5), 'z_range': (5.5, 14.5), 'frequency': 3.2},
            'Standing': {'x_range': (-0.6, 0.6), 'y_range': (-0.6, 0.6), 'z_range': (9.2, 10.8), 'frequency': 0.15},
            'Sitting': {'x_range': (-0.4, 0.4), 'y_range': (-0.4, 0.4), 'z_range': (9.6, 10.4), 'frequency': 0.08}
        }
        
        pattern = activity_patterns.get(activity, activity_patterns['Sitting'])
        
        stress_noise = stress_modifier * 0.4
        fatigue_noise = (1 - base_metrics.get('Energy_Level', 0.7)) * 0.25
        health_noise = random.uniform(-0.12, 0.12)
        
        total_noise = stress_noise + fatigue_noise + health_noise
        
        data = []
        for i in range(total_samples):
            time_noise = random.uniform(-3, 3)
            
            t = i / samples_per_hour
            freq_variation = random.uniform(0.75, 1.25)
            
            x_base = random.uniform(*pattern['x_range'])
            y_base = random.uniform(*pattern['y_range'])
            z_base = random.uniform(*pattern['z_range'])
            
            if activity in ['Walking', 'Jogging']:
                rhythm = np.sin(2 * np.pi * pattern['frequency'] * freq_variation * t)
                x_base += rhythm * 0.6
                y_base += rhythm * 0.4
                z_base += rhythm * 1.0
            
            x_noise = (
                random.gauss(0, 0.12) +
                total_noise * random.gauss(0, 0.25) +
                random.uniform(-0.08, 0.08)
            )
            y_noise = (
                random.gauss(0, 0.12) +
                total_noise * random.gauss(0, 0.25) +
                random.uniform(-0.08, 0.08)
            )
            z_noise = (
                random.gauss(0, 0.12) +
                total_noise * random.gauss(0, 0.18) +
                random.uniform(-0.08, 0.08)
            )
            
            data.append({
                'x': round(x_base + x_noise, 3),
                'y': round(y_base + y_noise, 3),
                'z': round(z_base + z_noise, 3),
                'time_offset': time_noise
            })
        
        return data

    def generate_enhanced_dataset(self, start_date_str="2024-01-01", days=30):
        """
        Tạo dataset với SEQUENTIAL BEHAVIORAL DATA để hỗ trợ LSTM
        """
        print(f"🚀 Tạo SEQUENTIAL BEHAVIORAL DATASET từ {start_date_str} trong {days} ngày...")
        print("🧠 Với support cho LSTM sequences: Screen Time, Phone Usage, Social Interaction")
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = start_date + timedelta(days=days-1)
        
        # Reset behavioral state
        self.behavioral_state = {
            'recent_screen_usage': [],
            'phone_interaction_history': [],
            'social_activity_timeline': [],
            'stress_accumulation': [],
            'activity_transitions': [],
            'environmental_changes': []
        }
        
        self.life_events = self.generate_life_events(start_date, end_date)
        print(f"📅 Tạo {len(self.life_events)} life events trong {days} ngày")
        
        all_data = []
        current_date = start_date
        
        for day_num in range(days):
            print(f"📊 Generating day {day_num + 1}/{days}: {current_date.strftime('%Y-%m-%d')}")
            
            schedule, day_context = self.generate_daily_schedule(current_date)
            base_metrics = self.calculate_enhanced_daily_metrics(current_date, schedule, day_context)
            
            for slot in schedule:
                duration = slot['time_end'] - slot['time_start']
                accelerometer_data = self.generate_accelerometer_with_variations(
                    slot['activity'], 
                    slot['location'], 
                    slot['stress_modifier'],
                    base_metrics,
                    duration
                )
                
                for i, accel in enumerate(accelerometer_data):
                    # Calculate exact timestamp
                    hours = slot['time_start'] + (i / len(accelerometer_data)) * duration
                    minutes = int((hours % 1) * 60)
                    seconds = int(((hours % 1) * 60 % 1) * 60) + accel['time_offset']
                    
                    if hours >= 24:
                        sample_datetime = current_date + timedelta(days=1, hours=hours-24, minutes=minutes, seconds=seconds)
                    else:
                        sample_datetime = current_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)
                    
                    # Current stress level with slot modifier
                    current_stress = max(1, min(10, base_metrics['Stress_Level'] + slot['stress_modifier']))
                    
                    # UPDATE BEHAVIORAL STATE - This is the key enhancement!
                    current_data = {'stress_level': current_stress}
                    self.update_behavioral_state(sample_datetime, current_data, slot['activity'], slot['location'])
                    
                    # GET BEHAVIORAL FEATURES FROM SEQUENCES
                    behavioral_features = self.get_behavioral_features(sample_datetime)
                    
                    # Heart rate calculation
                    activity_hr_modifier = {
                        'Walking': 18, 'Jogging': 40, 'Standing': 6, 'Sitting': 0
                    }
                    current_hr = (
                        base_metrics['Heart_Rate_Baseline'] + 
                        activity_hr_modifier.get(slot['activity'], 0) +
                        slot['stress_modifier'] * 10 +
                        random.uniform(-4, 4)
                    )
                    current_hr = max(50, min(190, int(current_hr)))
                    
                    # Environmental factors
                    ambient_light = max(0, min(1200, 
                        350 + (hours - 12) * 25 + random.uniform(-80, 80)
                    ))
                    
                    noise_level = {
                        'home': random.uniform(32, 58),
                        'work': random.uniform(42, 68),
                        'outdoor': random.uniform(48, 78),
                        'commute': random.uniform(58, 85),
                        'social': random.uniform(52, 88)
                    }.get(slot['location'], 45)
                    
                    daily_weather = self.get_daily_noise_factor(current_date)['weather_mood']
                    weather_condition = max(0, min(10, 5.5 + daily_weather * 12 + random.uniform(-1.2, 1.2)))
                    
                    # CREATE RECORD WITH SEQUENTIAL BEHAVIORAL FEATURES
                    record = {
                        'Timestamp': sample_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        'Age': self.user_profile['Age'],
                        'Gender': self.user_profile['Gender'],
                        'Sleep_Duration': base_metrics['Sleep_Duration'],
                        'Stress_Level': round(current_stress, 1),
                        'Heart_Rate': current_hr,
                        'Screen_Time': base_metrics['Screen_Time'],
                        'Step_Count': base_metrics['Step_Count'],
                        'Calories': base_metrics['Calories'],
                        'Accelerometer_X': accel['x'],
                        'Accelerometer_Y': accel['y'],
                        'Accelerometer_Z': accel['z'],
                        'Activity': slot['activity'],
                        'Location': slot['location'],
                        'Ambient_Light': round(ambient_light, 1),
                        'Noise_Level': round(noise_level, 1),
                        'Weather_Condition': round(weather_condition, 1),
                        'Reaction_Time': base_metrics['Reaction_Time'],
                        'Sleep_Quality': base_metrics['Sleep_Quality'],
                        'Energy_Level': base_metrics['Energy_Level'],
                        'Mood_Score': base_metrics['Mood_Score'],
                        'Exercise_Minutes': base_metrics['Exercise_Minutes'],
                        'Social_Interaction': base_metrics['Social_Interaction'],
                        
                        # 🧠 NEW: SEQUENTIAL BEHAVIORAL FEATURES FOR LSTM
                        **behavioral_features  # This adds all behavioral sequence features
                    }
                    
                    all_data.append(record)
            
            current_date += timedelta(days=1)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        filename = f'data/sequential_behavioral_health_data_{days}days.csv'
        df.to_csv(filename, index=False)
        
        print(f"\n🎉 === SEQUENTIAL BEHAVIORAL DATASET SUMMARY ===")
        print(f"📈 Tổng số records: {len(df):,}")
        print(f"📅 Số ngày: {days}")
        print(f"📊 Records per day trung bình: {len(df)//days:,}")
        print(f"💾 File saved: {filename}")
        
        # Print behavioral features summary
        print(f"\n🧠 === BEHAVIORAL SEQUENCE FEATURES ===")
        behavioral_columns = [col for col in df.columns if any(prefix in col for prefix in 
                             ['Screen_Usage', 'Phone_', 'Social_', 'Stress_Current', 'Stress_Velocity'])]
        
        print(f"📱 Screen Usage Features: {len([c for c in behavioral_columns if 'Screen' in c])}")
        print(f"📞 Phone Interaction Features: {len([c for c in behavioral_columns if 'Phone' in c])}")  
        print(f"👥 Social Sequence Features: {len([c for c in behavioral_columns if 'Social' in c])}")
        print(f"😰 Stress Sequence Features: {len([c for c in behavioral_columns if 'Stress_' in c])}")
        
        print(f"\n📋 Behavioral Feature Examples:")
        for feature in behavioral_columns[:8]:  # Show first 8 features
            print(f"  • {feature}: {df[feature].min():.3f} - {df[feature].max():.3f}")
        
        # Verify sequential data quality
        print(f"\n✅ === SEQUENCE DATA QUALITY ===")
        print(f"📊 Screen Usage Variance: {df['Screen_Usage_Variance'].mean():.4f} (>0 = có variation)")
        print(f"📱 Phone Events per 30min: {df['Phone_Events_Count_30min'].mean():.2f}")
        print(f"👥 Social Interaction Trend: {df['Social_Interaction_Trend'].std():.4f} (>0 = có variation)")
        print(f"😰 Stress Velocity: {df['Stress_Velocity'].std():.4f} (>0 = có changes)")
        
        # Life events summary
        event_days = len(self.life_events)
        print(f"\n🎭 Life Events: {event_days} ngày có sự kiện đặc biệt ({event_days/days*100:.1f}%)")
        
        print(f"\n🚀 === READY FOR LSTM TRAINING ===")
        print(f"✅ Sequential behavioral patterns: GENERATED")
        print(f"✅ Temporal dependencies: CAPTURED")
        print(f"✅ Multi-modal fusion ready: YES")
        print(f"✅ Architecture support: Bidirectional LSTM + Dense fusion")
        
        return df

if __name__ == "__main__":
    print("=== SEQUENTIAL BEHAVIORAL HEALTH DATASET GENERATOR ===")
    print("🧠 Enhanced for LSTM sequence modeling")
    
    generator = HealthMonitoringDataGenerator()
    
    print("Bắt đầu tạo sequential behavioral dataset...")
    df = generator.generate_enhanced_dataset("2024-01-01", 30)
    
    print("\n📋 === SAMPLE SEQUENTIAL DATA ===")
    # Show behavioral features in sample
    behavioral_cols = ['Timestamp', 'Activity', 'Screen_Usage_Current', 'Screen_Usage_Trend', 
                      'Phone_Events_Count_30min', 'Social_Current_Level', 'Stress_Velocity']
    print(df[behavioral_cols].head(10))
    
    print(f"\n🎯 Dataset ready for Phase 1 LSTM enhancement!")
