"""
Daily Schedule Generator
Tạo lịch trình hàng ngày thực tế cho người Việt Nam
"""

import random
import numpy as np
from datetime import timedelta

class DailyScheduleGenerator:
    """Tạo lịch trình hoạt động hàng ngày thực tế"""
    
    def __init__(self, activity_manager):
        self.activity_manager = activity_manager
        self.daily_patterns = {
            'wake_up_time': (6, 8),
            'work_start': (8, 9),
            'lunch_time': (12, 13),
            'work_end': (17, 18),
            'dinner_time': (18, 20),
            'sleep_time': (22, 24),
            'exercise_days': [1, 3, 5, 6]  # Mon, Wed, Fri, Sat
        }

    def get_daily_noise_factor(self, date):
        """Tạo các yếu tố nhiễu thực tế cho từng ngày"""
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

    def generate_improved_daily_schedule(self, date, life_events=None):
        """
        HAR IMPROVED: Tạo schedule với activity segments dài hơn để cải thiện HAR accuracy
        """
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5
        
        daily_noise = self.get_daily_noise_factor(date)
        life_event = life_events.get(date, None) if life_events else None
        event_modifier = 0
        if life_event:
            if life_event['type'] in ['sick', 'deadline', 'bad_news', 'exam', 'pms']:
                event_modifier = life_event['intensity']
            elif life_event['type'] in ['vacation', 'good_news', 'party', 'promotion']:
                event_modifier = -life_event['intensity']
        
        # REALISTIC TIMES - Thời gian thực tế người Việt
        if is_weekend:
            wake_up = random.uniform(7.5, 9.5)  # Weekend dậy muộn hơn
            sleep_time = random.uniform(23.0, 24.5)  # Ngủ muộn hơn
        else:
            wake_up = random.uniform(6.5, 8.0)   # Weekday dậy sớm
            sleep_time = random.uniform(22.5, 23.5)  # Ngủ sớm hơn
        
        day_context = {
            'sleep_quality': max(0.25, min(1.0, 0.8 + daily_noise['sleep_quality'])),
            'energy_level': max(0.2, min(1.0, 0.7 + daily_noise['energy'] + event_modifier * 0.3)),
            'stress_base': max(1, min(9, 4 + daily_noise['stress_fluctuation'] * 3 + event_modifier * 4)),
            'mood_factor': daily_noise['mood'] + event_modifier * 0.5,
            'has_social_event': random.random() < (0.4 if is_weekend else 0.1),
            'work_intensity': random.choice(['low', 'normal', 'high']) if not is_weekend else 'none',
            'weather_effect': daily_noise['weather_mood'],
            'exercise_intensity': random.uniform(0.4, 1.2) if day_of_week in self.daily_patterns['exercise_days'] else 0,
            'life_event': life_event
        }
        
        schedule = []
        current_time = wake_up
        previous_activity = 'Sitting'  # Start with sitting
        
        # HAR IMPROVED: Generate longer activity segments
        while current_time < sleep_time - 0.5:
            # Choose activity based on time of day and context
            activity = self.activity_manager.choose_contextual_activity(
                current_time, is_weekend, day_context, previous_activity
            )
            
            # Get realistic duration for this activity
            duration_minutes = self.activity_manager.get_improved_activity_duration(
                activity, current_time, is_weekend
            )
            duration_hours = duration_minutes / 60.0
            
            # Determine location and stress
            location = self._determine_activity_location(activity, current_time, is_weekend)
            stress_modifier = self._calculate_activity_stress(activity, location, day_context)
            
            # Add to schedule
            schedule.append({
                'time_start': current_time,
                'time_end': current_time + duration_hours,
                'activity': activity,
                'location': location,
                'stress_modifier': stress_modifier,
                'duration_minutes': duration_minutes
            })
            
            current_time += duration_hours
            previous_activity = activity
        
        return schedule, day_context

    def _determine_activity_location(self, activity, current_time, is_weekend):
        """Determine realistic location for activity"""
        hour = current_time
        
        if activity == 'Jogging':
            return 'outdoor'
        elif activity in ['Upstairs', 'Downstairs']:
            if 9 <= hour <= 17 and not is_weekend:
                return 'work'
            else:
                return 'home'
        elif 9 <= hour <= 17 and not is_weekend:
            return 'work'
        elif 12 <= hour <= 13:
            return 'outdoor' if random.random() < 0.4 else 'work'
        else:
            return 'home'

    def _calculate_activity_stress(self, activity, location, day_context):
        """Calculate stress modifier based on activity and context"""
        base_stress = 0
        
        if activity == 'Jogging':
            base_stress = -0.4  # Exercise reduces stress
        elif activity == 'Walking':
            base_stress = -0.1
        elif activity == 'Sitting':
            if location == 'work':
                base_stress = 0.2
            else:
                base_stress = -0.1
        elif activity in ['Upstairs', 'Downstairs']:
            base_stress = 0.1  # Slight physical exertion
        
        # Apply day context modifiers
        if day_context['work_intensity'] == 'high' and location == 'work':
            base_stress += 0.3
        elif day_context['work_intensity'] == 'low' and location == 'work':
            base_stress -= 0.1
            
        return base_stress + random.uniform(-0.1, 0.1)

    def determine_realistic_location(self, activity, time_of_day, is_weekend, context_location=None):
        """
        Xác định location THỰC TẾ dựa trên activity, thời gian và thói quen người Việt
        """
        hour = time_of_day
        
        # LOGIC THỰC TẾ: Sáng ở nhà → đi làm → về nhà tối
        
        # Jogging - chỉ outdoor hoặc gym, phù hợp với thời gian
        if activity == 'Jogging':
            if 6 <= hour <= 8 or 17 <= hour <= 19:  # Sáng sớm hoặc chiều
                return 'outdoor'  # Chạy ngoài trời
            else:
                return 'gym'  # Chạy trong gym
        
        # Upstairs/Downstairs activities
        elif activity == 'Upstairs':
            if 9 <= hour <= 17 and not is_weekend:
                return 'work'  # Lên cầu thang ở văn phòng
            elif 18 <= hour <= 22:
                return 'home'  # Lên tầng ở nhà
            else:
                return random.choice(['home', 'work', 'social'])
                
        elif activity == 'Downstairs':
            if 9 <= hour <= 17 and not is_weekend:
                return 'work'  # Xuống cầu thang ở văn phòng
            elif 18 <= hour <= 22:
                return 'home'  # Xuống tầng ở nhà
            else:
                return random.choice(['home', 'work', 'social'])
        
        # Walking - phụ thuộc nhiều vào thời gian và context
        elif activity == 'Walking':
            if context_location == 'commute':
                return 'commute'
            elif 7 <= hour <= 9:  # Sáng đi làm
                return 'commute' if not is_weekend else 'outdoor'
            elif 12 <= hour <= 13:  # Giờ ăn trưa
                return 'outdoor'  # Ra ngoài ăn trưa
            elif 17 <= hour <= 18:  # Tan tầm về nhà
                return 'commute' if not is_weekend else 'outdoor'
            elif is_weekend:
                if 9 <= hour <= 17:
                    return random.choice(['outdoor', 'social'])  # Weekend đi chơi
                else:
                    return 'home'
            else:
                return 'outdoor'  # Đi bộ ngoài trời
        
        # Standing - tùy theo thời gian trong ngày
        elif activity == 'Standing':
            if 9 <= hour <= 17 and not is_weekend:
                return 'work'  # Đứng ở công ty
            elif 18 <= hour <= 22:
                if random.random() < 0.3:
                    return 'social'  # Gặp bạn bè
                else:
                    return 'home'  # Ở nhà
            else:
                return 'home'
        
        # Sitting - location chính xác theo thời gian
        elif activity == 'Sitting':
            if 6 <= hour <= 8:  # Sáng sớm
                return 'home'
            elif 9 <= hour <= 17 and not is_weekend:  # Giờ làm việc
                return 'work'
            elif 22 <= hour or hour <= 6:  # Tối muộn/đêm
                return 'home'
            elif is_weekend:
                if random.random() < 0.2:
                    return 'social'  # Ngồi cafe, nhà bạn
                else:
                    return 'home'
            else:
                return 'home'
        
        return 'home'  # Default
