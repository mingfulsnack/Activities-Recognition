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
        ENHANCED: Tạo schedule phức tạp và cân bằng với nhiều hoạt động đa dạng
        Giảm thiểu sitting quá nhiều, thêm micro-activities và transitions thực tế
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
            'life_event': life_event,
            'activity_diversity_factor': random.uniform(0.6, 1.4),  # Yếu tố đa dạng hoạt động
            'restlessness': random.uniform(0.3, 1.2)  # Mức độ không thể ngồi yên
        }
        
        schedule = []
        current_time = wake_up
        previous_activity = 'Sitting'  # Start with sitting
        sitting_accumulation = 0  # Track sitting time để force breaks
        continuous_sitting_limit = random.uniform(45, 90)  # 45-90 minutes max sitting
        
        # ENHANCED: Generate more diverse and realistic schedule
        while current_time < sleep_time - 0.5:
            # ANTI-SITTING LOGIC: Force activity breaks if sitting too long
            if previous_activity == 'Sitting' and sitting_accumulation > continuous_sitting_limit:
                # Force a movement activity
                movement_activities = ['Standing', 'Walking', 'Upstairs', 'Downstairs']
                # Only allow jogging during appropriate times
                if (day_context['exercise_intensity'] > 0 and random.random() < 0.2 and
                    ((6 <= current_time <= 8) or (17 <= current_time <= 19)) and
                    (is_weekend or not (9 <= current_time <= 17))):  # Not during work hours
                    movement_activities.append('Jogging')
                
                activity = random.choice(movement_activities)
                sitting_accumulation = 0  # Reset sitting counter
                continuous_sitting_limit = random.uniform(45, 90)  # New limit for next sitting session
            else:
                # CONTEXTUAL ACTIVITY SELECTION with enhanced diversity
                activity = self._choose_enhanced_contextual_activity(
                    current_time, is_weekend, day_context, previous_activity
                )
            
            # FINAL JOGGING RESTRICTION CHECK
            if activity == 'Jogging':
                # Absolutely forbidden times
                if (not is_weekend and 9 <= current_time <= 17) or current_time > 20:
                    # Replace with appropriate alternative
                    if 17 <= current_time <= 19:
                        activity = 'Walking'  # Evening walk instead
                    else:
                        activity = 'Standing'  # Default safe activity
            
            # ENHANCED DURATION: More realistic and varied durations
            duration_minutes = self._get_enhanced_activity_duration(
                activity, current_time, is_weekend, day_context, previous_activity
            )
            duration_hours = duration_minutes / 60.0
            
            # Track sitting accumulation
            if activity == 'Sitting':
                sitting_accumulation += duration_minutes
            else:
                sitting_accumulation = 0
            
            # Determine location and stress
            location = self._determine_enhanced_location(activity, current_time, is_weekend, day_context)
            stress_modifier = self._calculate_activity_stress(activity, location, day_context)
            
            # Add micro-activities và natural breaks
            segments = self._create_activity_segments_with_breaks(
                activity, duration_minutes, current_time, day_context
            )
            
            # Add all segments to schedule
            for segment in segments:
                schedule.append({
                    'time_start': segment['time_start'],
                    'time_end': segment['time_end'],
                    'activity': segment['activity'],
                    'location': location,
                    'stress_modifier': stress_modifier,
                    'duration_minutes': segment['duration_minutes']
                })
            
            current_time += duration_hours
            previous_activity = activity
        
        return schedule, day_context

    def _determine_activity_location(self, activity, current_time, is_weekend):
        """
        LEGACY METHOD: Determine realistic location for activity
        Updated to match realistic Vietnamese schedule
        """
        hour = current_time
        
        # Use the new enhanced location logic
        day_context = {'has_social_event': False, 'weather_effect': 0}
        return self._determine_enhanced_location(activity, current_time, is_weekend, day_context)

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

    def _choose_enhanced_contextual_activity(self, current_time, is_weekend, day_context, previous_activity):
        """
        Enhanced activity selection với nhiều yếu tố ảnh hưởng và đa dạng hơn
        """
        hour = current_time
        
        # TIME-BASED ACTIVITY PREFERENCES with REALISTIC JOGGING TIMES
        if is_weekend:
            if 6 <= hour <= 9:  # Weekend morning - BEST TIME FOR JOGGING
                candidates = ['Sitting', 'Standing', 'Walking', 'Jogging']
                weights = [0.35, 0.15, 0.25, 0.25]  # Higher jogging chance
            elif 9 <= hour <= 12:  # Weekend late morning - NO JOGGING (too hot/busy)
                candidates = ['Walking', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
                weights = [0.3, 0.3, 0.2, 0.1, 0.1]
            elif 12 <= hour <= 18:  # Weekend afternoon - LIGHT JOGGING only if cool
                candidates = ['Walking', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
                if day_context['weather_effect'] > 0.1:  # Good weather only
                    candidates.append('Jogging')
                    weights = [0.25, 0.25, 0.2, 0.1, 0.1, 0.1]
                else:
                    weights = [0.3, 0.3, 0.25, 0.075, 0.075]
            else:  # Weekend evening
                candidates = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']
                weights = [0.4, 0.25, 0.2, 0.075, 0.075]
        else:
            if 6 <= hour <= 8:  # EARLY MORNING - PRIME JOGGING TIME (before work)
                candidates = ['Sitting', 'Standing', 'Walking', 'Jogging']
                if day_context['exercise_intensity'] > 0:
                    weights = [0.3, 0.15, 0.25, 0.3]  # High jogging chance
                else:
                    weights = [0.4, 0.25, 0.3, 0.05]  # Low jogging chance
            elif 8 <= hour <= 9:  # Workday morning commute - NO JOGGING
                candidates = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']
                weights = [0.35, 0.25, 0.25, 0.075, 0.075]
            elif 9 <= hour <= 12:  # Work morning - ABSOLUTELY NO JOGGING
                candidates = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']
                if day_context['work_intensity'] == 'high':
                    weights = [0.6, 0.2, 0.1, 0.05, 0.05]
                else:
                    weights = [0.45, 0.25, 0.15, 0.075, 0.075]
            elif 12 <= hour <= 13:  # Lunch time - NO JOGGING (too short time)
                candidates = ['Walking', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
                weights = [0.4, 0.3, 0.15, 0.075, 0.075]
            elif 13 <= hour <= 17:  # Work afternoon - ABSOLUTELY NO JOGGING
                candidates = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']
                if day_context['work_intensity'] == 'high':
                    weights = [0.65, 0.15, 0.1, 0.05, 0.05]
                else:
                    weights = [0.5, 0.2, 0.15, 0.075, 0.075]
            elif 17 <= hour <= 19:  # EVENING - SECOND BEST TIME FOR JOGGING
                candidates = ['Walking', 'Sitting', 'Standing', 'Jogging', 'Upstairs', 'Downstairs']
                if day_context['exercise_intensity'] > 0:
                    weights = [0.25, 0.2, 0.15, 0.3, 0.05, 0.05]  # High jogging chance
                else:
                    weights = [0.4, 0.3, 0.15, 0.05, 0.05, 0.05]  # Low jogging chance
            else:  # Late evening - ABSOLUTELY NO JOGGING (too late, unsafe)
                candidates = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']
                weights = [0.5, 0.2, 0.2, 0.05, 0.05]
        
        # CONTEXTUAL MODIFIERS
        # Activity diversity factor - encourage variety
        diversity_boost = day_context['activity_diversity_factor']
        if previous_activity in candidates:
            prev_idx = candidates.index(previous_activity)
            weights[prev_idx] *= (2 - diversity_boost)  # Reduce chance of repeating
        
        # Restlessness factor - encourage movement
        restlessness = day_context['restlessness']
        movement_activities = ['Walking', 'Standing', 'Jogging', 'Upstairs', 'Downstairs']
        for i, activity in enumerate(candidates):
            if activity in movement_activities:
                weights[i] *= (1 + restlessness * 0.3)
        
        # Energy level affects activity intensity
        energy = day_context['energy_level']
        if 'Jogging' in candidates:
            jog_idx = candidates.index('Jogging')
            weights[jog_idx] *= energy  # Low energy = less jogging
        
        # Stress affects sitting preference
        stress = day_context['stress_base']
        if 'Sitting' in candidates:
            sit_idx = candidates.index('Sitting')
            if stress > 6:
                weights[sit_idx] *= 1.2  # High stress = more sitting
            elif stress < 3:
                weights[sit_idx] *= 0.8  # Low stress = less sitting
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Choose activity
        chosen_activity = np.random.choice(candidates, p=weights)
        
        return chosen_activity

    def _get_enhanced_activity_duration(self, activity, current_time, is_weekend, day_context, previous_activity):
        """
        Enhanced duration calculation với realistic variations và context awareness
        """
        hour = current_time
        
        # BASE DURATIONS với realistic ranges
        base_durations = {
            'Sitting': {
                'work_hours': (20, 60),      # 20-60 min during work
                'evening': (30, 90),         # 30-90 min in evening
                'weekend': (25, 75),         # 25-75 min on weekend
                'default': (15, 45)
            },
            'Standing': {
                'work_hours': (5, 25),       # 5-25 min during work
                'meeting': (10, 45),         # 10-45 min for meetings
                'break': (3, 15),            # 3-15 min for breaks
                'exercise': (5, 20),         # 5-20 min exercise standing
                'default': (5, 20)
            },
            'Walking': {
                'commute': (15, 45),         # 15-45 min commuting
                'lunch': (20, 60),           # 20-60 min lunch walk
                'exercise': (30, 90),        # 30-90 min exercise walk
                'casual': (10, 30),          # 10-30 min casual walk
                'default': (15, 35)
            },
            'Jogging': {
                'morning': (20, 60),         # 20-60 min morning jog
                'evening': (25, 75),         # 25-75 min evening jog
                'weekend': (30, 120),        # 30-120 min weekend jog
                'default': (20, 45)
            },
            'Upstairs': {
                'single_trip': (1, 3),       # 1-3 min single trip
                'multiple_trips': (3, 10),   # 3-10 min multiple trips
                'exercise': (5, 15),         # 5-15 min stair exercise
                'default': (2, 8)
            },
            'Downstairs': {
                'single_trip': (1, 3),       # 1-3 min single trip
                'multiple_trips': (3, 10),   # 3-10 min multiple trips
                'default': (2, 6)
            }
        }
        
        # CONTEXTUAL DURATION SELECTION
        duration_key = 'default'
        
        if activity == 'Sitting':
            if not is_weekend and 9 <= hour <= 17:
                duration_key = 'work_hours'
            elif 18 <= hour <= 22:
                duration_key = 'evening'
            elif is_weekend:
                duration_key = 'weekend'
        
        elif activity == 'Standing':
            if not is_weekend and 9 <= hour <= 17:
                if random.random() < 0.3:
                    duration_key = 'meeting'
                else:
                    duration_key = 'work_hours'
            elif previous_activity == 'Sitting':
                duration_key = 'break'
            elif day_context['exercise_intensity'] > 0 and random.random() < 0.2:
                duration_key = 'exercise'
        
        elif activity == 'Walking':
            if 7 <= hour <= 9 or 17 <= hour <= 18:
                duration_key = 'commute'
            elif 12 <= hour <= 13:
                duration_key = 'lunch'
            elif day_context['exercise_intensity'] > 0 and random.random() < 0.4:
                duration_key = 'exercise'
            else:
                duration_key = 'casual'
        
        elif activity == 'Jogging':
            if 6 <= hour <= 9:
                duration_key = 'morning'
            elif 17 <= hour <= 20:
                duration_key = 'evening'
            elif is_weekend:
                duration_key = 'weekend'
        
        elif activity in ['Upstairs', 'Downstairs']:
            if day_context['exercise_intensity'] > 0 and random.random() < 0.2:
                duration_key = 'exercise'
            elif random.random() < 0.6:
                duration_key = 'single_trip'
            else:
                duration_key = 'multiple_trips'
        
        # Get duration range with safe fallback
        try:
            duration_range = base_durations[activity][duration_key]
        except KeyError:
            # Fallback to default if key not found
            duration_range = base_durations[activity]['default']
        
        base_duration = random.uniform(duration_range[0], duration_range[1])
        
        # CONTEXTUAL MODIFIERS
        # Energy level affects duration
        energy_modifier = 0.7 + day_context['energy_level'] * 0.6  # 0.7 - 1.3
        
        # Stress affects duration differently by activity
        stress = day_context['stress_base']
        if activity == 'Sitting':
            stress_modifier = 1 + (stress - 4) * 0.05  # More stress = longer sitting
        elif activity in ['Jogging', 'Walking']:
            stress_modifier = 1 + (4 - stress) * 0.05  # Less stress = longer exercise
        else:
            stress_modifier = 1.0
        
        # Time pressure modifier
        if not is_weekend and day_context['work_intensity'] == 'high':
            if activity in ['Walking', 'Standing', 'Jogging']:
                time_pressure_modifier = 0.8  # Less time for activities
            else:
                time_pressure_modifier = 1.1  # More time sitting/working
        else:
            time_pressure_modifier = 1.0
        
        # Apply all modifiers
        final_duration = base_duration * energy_modifier * stress_modifier * time_pressure_modifier
        
        # Ensure reasonable bounds
        min_duration = 2 if activity in ['Upstairs', 'Downstairs'] else 5
        max_duration = 180 if activity == 'Jogging' else 120
        
        return max(min_duration, min(max_duration, final_duration))

    def _determine_enhanced_location(self, activity, current_time, is_weekend, day_context):
        """
        REALISTIC LOCATION: Dựa trên lịch trình di chuyển thực tế của người Việt Nam
        Sáng ở nhà → đi làm → về nhà tối
        """
        hour = current_time
        
        # JOGGING-SPECIFIC LOGIC - ALWAYS OUTDOOR/GYM
        if activity == 'Jogging':
            if day_context.get('weather_effect', 0) < -0.1:  # Bad weather
                return 'gym'
            elif 6 <= hour <= 8 or 17 <= hour <= 19:  # Prime jogging times
                return 'outdoor'
            else:
                return 'gym'  # Fallback to gym if somehow jogging at odd hours
        
        # REALISTIC DAILY MOVEMENT SCHEDULE for Vietnamese lifestyle
        if not is_weekend:  # WEEKDAY SCHEDULE
            if 6 <= hour <= 8.5:  # MORNING: Ở nhà (ăn sáng, chuẩn bị)
                return 'home'
            
            elif 8.5 <= hour <= 9.0:  # COMMUTE TO WORK: Đang di chuyển
                if activity == 'Walking':
                    return 'commute'
                elif activity in ['Sitting', 'Standing']:
                    return 'commute'  # Ngồi/đứng trên xe bus/tàu
                else:
                    return 'commute'
            
            elif 9.0 <= hour <= 12.0:  # WORK MORNING: STRICTLY AT WORK
                return 'work'  # NO EXCEPTIONS during core work hours
            
            elif 12.0 <= hour <= 13.0:  # LUNCH BREAK: Ra ngoài ăn trưa
                if activity == 'Walking':
                    return 'outdoor'  # Đi ăn trưa
                elif activity == 'Sitting':
                    return random.choice(['outdoor', 'work'])  # Ăn ngoài hoặc ăn ở công ty
                else:
                    return 'outdoor'
            
            elif 13.0 <= hour <= 17.5:  # WORK AFTERNOON: STRICTLY AT WORK
                return 'work'  # NO EXCEPTIONS during core work hours
            
            elif 17.5 <= hour <= 18.5:  # COMMUTE HOME: Đang về nhà
                if activity == 'Walking':
                    return 'commute'
                elif activity in ['Sitting', 'Standing']:
                    return 'commute'  # Trên phương tiện
                else:
                    return 'commute'
            
            elif 18.5 <= hour <= 22.5:  # EVENING: Về nhà
                if activity == 'Jogging':
                    if day_context['weather_effect'] < -0.1:
                        return 'gym'
                    else:
                        return 'outdoor'
                elif day_context['has_social_event'] and random.random() < 0.4:
                    return 'social'  # Gặp bạn bè buổi tối
                else:
                    return 'home'
            
            else:  # LATE EVENING/NIGHT: Chắc chắn ở nhà
                return 'home'
                
        else:  # WEEKEND SCHEDULE - More flexible
            if 6 <= hour <= 9:  # WEEKEND MORNING: Ở nhà
                return 'home'
            
            elif 9 <= hour <= 18:  # WEEKEND DAY: Linh hoạt
                if day_context['has_social_event'] and random.random() < 0.6:
                    return 'social'  # Weekend hay gặp bạn bè
                elif activity == 'Walking':
                    return random.choice(['outdoor', 'social'])
                elif activity in ['Upstairs', 'Downstairs']:
                    return random.choice(['home', 'social'])  # Có thể ở nhà bạn
                else:
                    return random.choice(['home', 'social', 'outdoor'])
            
            else:  # WEEKEND EVENING: Về nhà hoặc social
                if day_context['has_social_event'] and random.random() < 0.5:
                    return 'social'
                else:
                    return 'home'
        
        return 'home'  # Safe default

    def _create_activity_segments_with_breaks(self, main_activity, total_duration, start_time, day_context):
        """
        Tạo segments với micro-breaks và natural transitions để realistic hơn
        """
        segments = []
        
        # If activity is very short, don't break it up
        if total_duration <= 10:
            segments.append({
                'time_start': start_time,
                'time_end': start_time + total_duration / 60.0,
                'activity': main_activity,
                'duration_minutes': total_duration
            })
            return segments
        
        # For longer activities, add micro-breaks based on restlessness
        restlessness = day_context['restlessness']
        
        # Determine if we should add breaks
        should_break = False
        if main_activity == 'Sitting' and total_duration > 30:
            should_break = random.random() < (restlessness * 0.6)
        elif main_activity in ['Standing', 'Walking'] and total_duration > 45:
            should_break = random.random() < (restlessness * 0.4)
        elif main_activity == 'Jogging' and total_duration > 20:
            should_break = random.random() < 0.3  # Natural water/rest breaks
        
        if not should_break:
            # No breaks, single segment
            segments.append({
                'time_start': start_time,
                'time_end': start_time + total_duration / 60.0,
                'activity': main_activity,
                'duration_minutes': total_duration
            })
        else:
            # Add micro-breaks
            current_time = start_time
            remaining_duration = total_duration
            
            while remaining_duration > 5:
                # Main activity segment duration
                if main_activity == 'Sitting':
                    segment_duration = random.uniform(15, 35)
                elif main_activity == 'Jogging':
                    segment_duration = random.uniform(8, 15)
                else:
                    segment_duration = random.uniform(10, 25)
                
                segment_duration = min(segment_duration, remaining_duration - 2)
                
                # Add main activity segment
                segments.append({
                    'time_start': current_time,
                    'time_end': current_time + segment_duration / 60.0,
                    'activity': main_activity,
                    'duration_minutes': segment_duration
                })
                
                current_time += segment_duration / 60.0
                remaining_duration -= segment_duration
                
                # Add micro-break if there's time left
                if remaining_duration > 3:
                    break_duration = random.uniform(1, 3)
                    break_duration = min(break_duration, remaining_duration)
                    
                    # Choose break activity
                    if main_activity == 'Sitting':
                        break_activity = random.choice(['Standing', 'Walking'])
                    elif main_activity == 'Standing':
                        break_activity = random.choice(['Sitting', 'Walking'])
                    elif main_activity == 'Walking':
                        break_activity = random.choice(['Standing', 'Sitting'])
                    elif main_activity == 'Jogging':
                        break_activity = random.choice(['Standing', 'Walking'])
                    else:
                        break_activity = 'Standing'
                    
                    segments.append({
                        'time_start': current_time,
                        'time_end': current_time + break_duration / 60.0,
                        'activity': break_activity,
                        'duration_minutes': break_duration
                    })
                    
                    current_time += break_duration / 60.0
                    remaining_duration -= break_duration
        
        return segments
