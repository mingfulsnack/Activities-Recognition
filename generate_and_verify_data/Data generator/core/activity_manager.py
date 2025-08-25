"""
Activity Management
Quản lý activities, transitions và validation
"""

import math
import random
import numpy as np

class ActivityManager:
    """Quản lý activities và transitions"""
    
    def __init__(self):
        self.har_activities = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.segment_time_size = 180  # HAR model requirement
        
        # Activity duration ranges for better HAR compatibility
        self.activity_durations = {
            'Sitting': (15, 60),      # 15-60 minutes continuous sitting
            'Standing': (3, 20),      # 3-20 minutes standing
            'Walking': (10, 45),      # 10-45 minutes walking
            'Jogging': (15, 90),      # 15-90 minutes jogging session  
            'Upstairs': (1, 5),       # 1-5min climbing stairs
            'Downstairs': (1, 5)      # 1-5min going downstairs
        }
        
        # Activity transition probabilities for better sequence consistency
        self.activity_transitions = {
            'Sitting': {
                'Standing': 0.4,
                'Walking': 0.3, 
                'Upstairs': 0.05,
                'Downstairs': 0.05,
                'Jogging': 0.05,
                'Sitting': 0.15  # Stay sitting longer
            },
            'Standing': {
                'Sitting': 0.5,
                'Walking': 0.3,
                'Upstairs': 0.05,
                'Downstairs': 0.05,
                'Jogging': 0.05,
                'Standing': 0.05
            },
            'Walking': {
                'Sitting': 0.25,
                'Standing': 0.15,
                'Upstairs': 0.15,
                'Downstairs': 0.15,
                'Jogging': 0.2,
                'Walking': 0.1
            },
            'Jogging': {
                'Walking': 0.4,
                'Standing': 0.25,
                'Sitting': 0.3,
                'Downstairs': 0.03,
                'Upstairs': 0.02,
                'Jogging': 0.0
            },
            'Upstairs': {
                'Standing': 0.4,
                'Sitting': 0.3,
                'Walking': 0.25,
                'Downstairs': 0.03,
                'Jogging': 0.02,
                'Upstairs': 0.0
            },
            'Downstairs': {
                'Walking': 0.4,
                'Standing': 0.3,
                'Sitting': 0.25,
                'Upstairs': 0.03,
                'Jogging': 0.02,
                'Downstairs': 0.0
            }
        }

    def verify_activity_from_accelerometer(self, x, y, z, intended_activity):
        """
        Verify và adjust activity dựa trên accelerometer data để đảm bảo consistency
        Support all 6 HAR activities
        """
        # Tính magnitude của accelerometer
        magnitude = math.sqrt(x*x + y*y + z*z)
        
        # Tính variance để detect movement patterns
        recent_variance = abs(x) + abs(y) + abs(z - 9.8)  # Z should be ~9.8 when stationary
        
        # Activity classification thresholds cho tất cả 6 HAR activities
        if magnitude < 9.5 or recent_variance < 1.0:
            # Very low movement - likely sitting
            return 'Sitting'
        elif magnitude < 10.5 and recent_variance < 2.0:
            # Low movement - likely standing
            return 'Standing'
        elif magnitude >= 10.5 and magnitude < 12.0:
            # Moderate movement - walking or light stairs
            if abs(z) > 10.5:  # Higher Z suggests upward motion
                return 'Upstairs' if random.random() < 0.3 else 'Walking'
            elif abs(z) < 8.5:  # Lower Z suggests downward motion
                return 'Downstairs' if random.random() < 0.3 else 'Walking'
            else:
                return 'Walking'
        elif magnitude >= 12.0 and magnitude < 14.0:
            # Higher movement - could be stairs or fast walking
            if abs(z) > 11.0:
                return 'Upstairs'
            elif abs(z) < 8.0:
                return 'Downstairs'
            else:
                return 'Walking'
        elif magnitude >= 14.0:
            # Very high movement - likely jogging
            return 'Jogging'
        else:
            # Fallback to intended activity if it's valid
            return intended_activity if intended_activity in self.har_activities else 'Sitting'

    def validate_har_sequence_consistency(self, data_sequence):
        """
        HAR COMPATIBILITY: Validate that generated sequence makes sense for HAR model
        Ensure that consecutive samples form coherent activity patterns
        """
        if len(data_sequence) < self.segment_time_size:
            return True  # Too short to validate properly
            
        # Take last 180 samples for HAR validation
        recent_sequence = data_sequence[-self.segment_time_size:]
        
        # Extract accelerometer data and activities
        accel_data = [(row['Accelerometer_X'], row['Accelerometer_Y'], row['Accelerometer_Z']) 
                     for row in recent_sequence]
        activities = [row['Activity'] for row in recent_sequence]
        
        # Check for activity consistency within sequence
        unique_activities = set(activities)
        if len(unique_activities) > 2:  # Too many different activities in one sequence
            return False
            
        # Check accelerometer magnitude consistency with activities
        magnitudes = [math.sqrt(x*x + y*y + z*z) for x, y, z in accel_data]
        avg_magnitude = np.mean(magnitudes)
        
        dominant_activity = max(set(activities), key=activities.count)
        
        # Validate magnitude ranges for each activity
        expected_ranges = {
            'Sitting': (9.0, 10.0),
            'Standing': (9.5, 10.5), 
            'Walking': (10.5, 13.0),
            'Jogging': (13.0, 16.0),
            'Upstairs': (11.5, 14.0),
            'Downstairs': (10.8, 13.5)
        }
        
        if dominant_activity in expected_ranges:
            min_mag, max_mag = expected_ranges[dominant_activity]
            if not (min_mag <= avg_magnitude <= max_mag):
                return False
                
        return True

    def choose_contextual_activity(self, current_time, is_weekend, day_context, previous_activity='Sitting'):
        """Choose activity based on time context and transitions"""
        hour = current_time
        
        # Time-based activity preferences
        if is_weekend:
            if 6 <= hour <= 9:  # Weekend morning
                candidates = ['Sitting', 'Standing', 'Walking']
                weights = [0.5, 0.2, 0.3]
            elif 9 <= hour <= 12:  # Weekend late morning
                candidates = ['Sitting', 'Walking', 'Jogging', 'Standing']
                weights = [0.3, 0.3, 0.25, 0.15]
            elif 12 <= hour <= 18:  # Weekend afternoon
                candidates = ['Walking', 'Sitting', 'Jogging', 'Standing', 'Upstairs', 'Downstairs']
                weights = [0.25, 0.25, 0.2, 0.15, 0.075, 0.075]
            else:  # Weekend evening
                candidates = ['Sitting', 'Standing', 'Walking']
                weights = [0.6, 0.2, 0.2]
        else:
            if 6 <= hour <= 9:  # Workday morning
                candidates = ['Sitting', 'Standing', 'Walking', 'Upstairs', 'Downstairs']
                weights = [0.4, 0.2, 0.25, 0.075, 0.075]
            elif 9 <= hour <= 12:  # Work morning
                candidates = ['Sitting', 'Standing', 'Upstairs', 'Downstairs']
                weights = [0.7, 0.2, 0.05, 0.05]
            elif 12 <= hour <= 13:  # Lunch time
                candidates = ['Walking', 'Sitting', 'Standing']
                weights = [0.5, 0.3, 0.2]
            elif 13 <= hour <= 17:  # Work afternoon
                candidates = ['Sitting', 'Standing', 'Upstairs', 'Downstairs']
                weights = [0.75, 0.15, 0.05, 0.05]
            elif 17 <= hour <= 19:  # Commute/exercise time
                if day_context.get('exercise_intensity', 0) > 0 and random.random() < 0.3:
                    return 'Jogging'
                candidates = ['Walking', 'Sitting', 'Standing']
                weights = [0.5, 0.3, 0.2]
            else:  # Evening
                candidates = ['Sitting', 'Standing', 'Walking']
                weights = [0.6, 0.2, 0.2]
        
        # Apply transition probabilities if we have previous activity
        if previous_activity in self.activity_transitions:
            transition_probs = self.activity_transitions[previous_activity]
            
            # Modify weights based on transition probabilities
            for i, candidate in enumerate(candidates):
                if candidate in transition_probs:
                    weights[i] *= (1 + transition_probs[candidate])
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Choose activity
        chosen_activity = np.random.choice(candidates, p=weights)
        
        return chosen_activity

    def get_improved_activity_duration(self, activity, current_time, is_weekend):
        """
        HAR OPTIMIZED: Get much longer, realistic duration để đảm bảo HAR sequences consistency
        HAR needs 180 samples = 90 minutes of consistent activity (at 2 samples/minute)
        """
        hour = current_time
        
        # FORCE LONGER SEGMENTS: Dramatically increase durations for HAR compatibility
        if activity == 'Sitting':
            if 9 <= hour <= 17 and not is_weekend:  # Work hours - MUCH longer sitting
                duration_min, duration_max = 120, 240  # 2-4 hours continuous sitting
            elif 19 <= hour <= 22:  # Evening meals/TV/relaxation
                duration_min, duration_max = 90, 180   # 1.5-3 hours sitting
            else:
                duration_min, duration_max = 60, 120   # 1-2 hours other times
                
        elif activity == 'Walking':
            if 7 <= hour <= 9 or 17 <= hour <= 18:  # Commute - longer walks
                duration_min, duration_max = 45, 90    # 45-90 minutes
            elif 12 <= hour <= 13:  # Lunch walk
                duration_min, duration_max = 30, 60    # 30-60 minutes
            elif is_weekend:
                duration_min, duration_max = 60, 120   # Weekend long walks
            else:
                duration_min, duration_max = 45, 90    # Default longer walks
                
        elif activity == 'Jogging':
            if 6 <= hour <= 8 or 17 <= hour <= 19:  # Prime exercise times
                duration_min, duration_max = 60, 120   # 1-2 hours exercise
            elif is_weekend:
                duration_min, duration_max = 90, 150   # Weekend longer exercise
            else:
                duration_min, duration_max = 45, 90    # 45-90 minutes
                
        elif activity == 'Standing':
            if 9 <= hour <= 17 and not is_weekend:  # Work standing
                duration_min, duration_max = 30, 90    # 30-90 minutes
            else:
                duration_min, duration_max = 15, 45    # 15-45 minutes
        
        elif activity in ['Upstairs', 'Downstairs']:
            # Stairs - extend for multiple floors or multiple trips
            duration_min, duration_max = 10, 30        # 10-30 minutes total
            
        else:
            duration_min, duration_max = 45, 90        # Default: 45-90 minutes
        
        return random.uniform(duration_min, duration_max)
