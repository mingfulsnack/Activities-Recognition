"""
Behavioral State Tracker
Theo dõi và tính toán behavioral patterns như screen time, phone usage, social interactions
"""

import random
import numpy as np
from datetime import timedelta

class BehavioralTracker:
    """Theo dõi behavioral patterns để tạo sequential data cho LSTM"""
    
    def __init__(self):
        # BEHAVIORAL PATTERN TRACKING - Lưu trữ state để tạo sequences
        self.behavioral_state = {
            'recent_screen_usage': [],      # Track 30 phút gần nhất
            'phone_interaction_history': [], # Track phone touches/unlocks
            'social_activity_timeline': [],  # Track social interactions
            'stress_accumulation': [],       # Track stress changes over time
            'activity_transitions': [],     # Track activity changes
            'environmental_changes': []     # Track environment transitions
        }

    def reset_behavioral_state(self):
        """Reset behavioral state cho dataset mới"""
        self.behavioral_state = {
            'recent_screen_usage': [],
            'phone_interaction_history': [],
            'social_activity_timeline': [],
            'stress_accumulation': [],
            'activity_transitions': [],
            'environmental_changes': []
        }

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
        # Base intensity bao gồm tất cả 6 HAR activities
        base_intensity = {
            'Sitting': 0.7,      # Ngồi thường xem nhiều
            'Standing': 0.4,     # Đứng ít xem hơn
            'Walking': 0.2,      # Đi bộ ít xem
            'Jogging': 0.05,     # Chạy hầu như không xem
            'Upstairs': 0.1,     # Lên cầu thang ít xem (tập trung)
            'Downstairs': 0.15   # Xuống cầu thang ít xem nhưng dễ hơn upstairs
        }.get(activity, 0.3)
        
        location_modifier = {
            'home': 1.3,       # Ở nhà xem nhiều
            'work': 1.1,       # Làm việc vừa phải
            'commute': 0.9,    # Di chuyển ít hơn
            'outdoor': 0.6,    # Ngoài trời ít
            'social': 0.4,     # Xã hội ít xem
            'gym': 0.3         # Gym ít xem (tập trung exercise)
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
        
        # Tần suất dựa trên activity và stress - bao gồm tất cả activities
        base_frequency = {
            'Sitting': 0.6,      # Ngồi hay dùng phone
            'Standing': 0.4,
            'Walking': 0.2,
            'Jogging': 0.05,
            'Upstairs': 0.1,     # Lên cầu thang ít dùng phone (cần tập trung)
            'Downstairs': 0.15   # Xuống cầu thang ít dùng phone nhưng dễ hơn upstairs
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
        
        # Activity modifier bao gồm tất cả 6 HAR activities
        activity_modifier = {
            'Sitting': 1.0,
            'Standing': 1.2,
            'Walking': 1.1,
            'Jogging': 0.7,
            'Upstairs': 0.9,     # Lên cầu thang ít social hơn (tập trung)
            'Downstairs': 1.0    # Xuống cầu thang bình thường
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
