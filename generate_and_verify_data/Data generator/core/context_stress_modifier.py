"""
Context-Stress Variation Generator
Tạo variations để cùng activity nhưng khác context có thể có stress levels khác nhau
"""

import random


class ContextStressModifier:
    """
    Modifier để tạo context-aware stress variations
    Đảm bảo cùng activity nhưng khác context → stress khác
    """
    
    # Context-specific stress patterns
    CONTEXT_PATTERNS = {
        'Walking': {
            # Same activity, different stress based on context
            ('work', 'morning', 'high_workload'): 1.4,      # Walking to meeting under deadline
            ('work', 'afternoon', 'high_workload'): 1.6,    # Walking during peak stress time
            ('outdoor', 'evening', 'low_workload'): -1.2,   # Relaxing evening walk
            ('outdoor', 'morning', 'weekend'): -1.0,        # Weekend morning walk
            ('commute', 'morning', 'weekday'): 0.8,         # Rush hour commute
            ('commute', 'evening', 'weekday'): 1.0,         # Evening rush stress
            ('home', 'evening', 'low_workload'): -0.5,      # Walking at home after work
            ('gym', 'morning', 'weekend'): -0.3,            # Light gym walking
        },
        
        'Sitting': {
            # Same sitting, different stress
            ('work', 'afternoon', 'high_workload'): 2.0,    # Deadline work sitting
            ('work', 'morning', 'high_workload'): 1.5,      # Morning work pressure
            ('work', 'afternoon', 'normal_workload'): 0.5,  # Normal work sitting
            ('home', 'evening', 'low_workload'): -1.0,      # Relaxing at home
            ('home', 'night', 'low_workload'): -1.5,        # Evening relaxation
            ('social', 'evening', 'weekend'): -0.8,         # Social sitting (cafe/restaurant)
            ('commute', 'morning', 'weekday'): 0.3,         # Sitting in transport
        },
        
        'Standing': {
            ('work', 'morning', 'high_workload'): 1.2,      # Standing during presentation
            ('work', 'afternoon', 'high_workload'): 1.4,    # Standing meeting under pressure
            ('home', 'morning', 'weekend'): -0.5,           # Standing cooking/relaxed
            ('home', 'evening', 'low_workload'): -0.3,      # Standing at home
            ('commute', 'morning', 'weekday'): 0.6,         # Standing in crowded transport
            ('outdoor', 'evening', 'weekend'): -0.4,        # Standing outdoor relaxed
        },
        
        'Jogging': {
            ('outdoor', 'morning', 'weekend'): -1.5,        # Weekend morning jog (best)
            ('outdoor', 'morning', 'weekday'): -1.2,        # Weekday morning jog
            ('outdoor', 'evening', 'weekday'): -1.0,        # Evening jog after work
            ('gym', 'morning', 'weekend'): -1.0,            # Gym jog weekend
            ('gym', 'evening', 'weekday'): -0.8,            # Gym jog after work (tired)
        },
        
        'Upstairs': {
            ('work', 'morning', 'high_workload'): 0.8,      # Rushing upstairs for meeting
            ('work', 'afternoon', 'high_workload'): 1.0,    # Afternoon rush upstairs
            ('work', 'morning', 'normal_workload'): 0.3,    # Normal work upstairs
            ('home', 'morning', 'weekend'): -0.2,           # Relaxed home upstairs
            ('home', 'evening', 'low_workload'): -0.1,      # Home upstairs evening
            ('gym', 'morning', 'weekend'): 0.0,             # Exercise upstairs
        },
        
        'Downstairs': {
            ('work', 'evening', 'high_workload'): 0.5,      # Leaving work stressed
            ('work', 'morning', 'normal_workload'): 0.2,    # Normal work downstairs
            ('home', 'morning', 'weekend'): -0.3,           # Relaxed home downstairs
            ('home', 'evening', 'low_workload'): -0.2,      # Home downstairs evening
            ('gym', 'morning', 'weekend'): -0.1,            # Exercise downstairs
        }
    }
    
    # Environmental noise factors
    NOISE_FACTORS = {
        'heavy_traffic': 0.8,
        'crowded_space': 0.6,
        'quiet_environment': -0.4,
        'nature_sounds': -0.6,
    }
    
    # Social context modifiers
    SOCIAL_MODIFIERS = {
        'conflict': 1.5,
        'supportive': -0.8,
        'neutral': 0.0,
        'collaborative': -0.4,
    }
    
    # Sleep deprivation amplifier
    SLEEP_AMPLIFIERS = {
        'poor': 1.3,      # < 6 hours
        'fair': 1.1,      # 6-7 hours
        'good': 1.0,      # 7-8 hours
        'excellent': 0.9  # > 8 hours
    }
    
    
    @staticmethod
    def get_context_key(location, time_period, workload_or_day_type):
        """
        Build context key for lookup
        """
        return (location, time_period, workload_or_day_type)
    
    
    @staticmethod
    def get_time_period(hour):
        """
        Convert hour to time period
        """
        if hour < 7:
            return 'early_morning'
        elif hour < 12:
            return 'morning'
        elif hour < 17:
            return 'afternoon'
        elif hour < 20:
            return 'evening'
        else:
            return 'night'
    
    
    @staticmethod
    def get_workload_type(work_intensity, is_weekend):
        """
        Convert work intensity to workload type
        """
        if is_weekend:
            return 'weekend'
        
        if work_intensity == 'high':
            return 'high_workload'
        elif work_intensity == 'normal':
            return 'normal_workload'
        else:
            return 'low_workload'
    
    
    @staticmethod
    def get_sleep_quality_category(sleep_duration):
        """
        Categorize sleep quality
        """
        if sleep_duration < 6:
            return 'poor'
        elif sleep_duration < 7:
            return 'fair'
        elif sleep_duration < 8.5:
            return 'good'
        else:
            return 'excellent'
    
    
    @classmethod
    def calculate_context_stress_modifier(cls, activity, location, hour, work_intensity, 
                                          is_weekend, sleep_duration, 
                                          noise_environment=None, social_context=None):
        """
        Calculate stress modifier based on full context
        Returns modifier to add to base stress
        
        Args:
            activity: Current activity (Walking, Sitting, etc.)
            location: Current location (work, home, outdoor, etc.)
            hour: Hour of day (0-23)
            work_intensity: Work intensity (high, normal, low, none)
            is_weekend: Boolean indicating weekend
            sleep_duration: Hours of sleep
            noise_environment: Optional noise factor
            social_context: Optional social context
        
        Returns:
            float: Stress modifier to add to base stress
        """
        total_modifier = 0.0
        
        # 1. Core context pattern lookup
        time_period = cls.get_time_period(hour)
        workload_type = cls.get_workload_type(work_intensity, is_weekend)
        context_key = cls.get_context_key(location, time_period, workload_type)
        
        activity_patterns = cls.CONTEXT_PATTERNS.get(activity, {})
        pattern_modifier = activity_patterns.get(context_key, 0.0)
        total_modifier += pattern_modifier
        
        # 2. Sleep quality amplifier
        sleep_category = cls.get_sleep_quality_category(sleep_duration)
        sleep_amplifier = cls.SLEEP_AMPLIFIERS.get(sleep_category, 1.0)
        
        # Apply sleep amplifier to pattern (poor sleep amplifies stress)
        if pattern_modifier > 0:  # Only amplify positive stress
            total_modifier = pattern_modifier * sleep_amplifier
        
        # 3. Noise environment modifier
        if noise_environment:
            noise_modifier = cls.NOISE_FACTORS.get(noise_environment, 0.0)
            total_modifier += noise_modifier
        
        # 4. Social context modifier
        if social_context:
            social_modifier = cls.SOCIAL_MODIFIERS.get(social_context, 0.0)
            total_modifier += social_modifier
        
        # 5. Add small random variation to create more diversity
        total_modifier += random.uniform(-0.2, 0.2)
        
        return round(total_modifier, 2)
    
    
    @classmethod
    def determine_noise_environment(cls, location, hour, activity):
        """
        Determine likely noise environment based on context
        """
        if location == 'commute':
            return random.choice(['heavy_traffic', 'crowded_space'])
        elif location == 'work':
            if hour >= 14 and hour < 17:  # Afternoon peak
                return random.choice(['crowded_space', 'quiet_environment'])
            return 'quiet_environment'
        elif location == 'outdoor':
            if activity == 'Jogging':
                return 'nature_sounds'
            return random.choice(['nature_sounds', 'quiet_environment'])
        elif location == 'home':
            return 'quiet_environment'
        else:
            return None
    
    
    @classmethod
    def determine_social_context(cls, location, hour, activity):
        """
        Determine likely social context
        """
        if location == 'social':
            return random.choice(['supportive', 'neutral', 'collaborative'])
        elif location == 'work':
            if hour >= 14 and hour < 17:  # Afternoon meetings
                return random.choice(['collaborative', 'neutral', 'conflict'])
            return 'neutral'
        elif location == 'home':
            return random.choice(['supportive', 'neutral'])
        else:
            return None
    
    
    @classmethod
    def apply_context_variations(cls, base_stress, activity, location, hour, 
                                 work_intensity, is_weekend, sleep_duration):
        """
        Main method to apply all context-aware variations
        
        Returns:
            float: Modified stress level with context variations
        """
        # Determine environmental contexts
        noise_env = cls.determine_noise_environment(location, hour, activity)
        social_ctx = cls.determine_social_context(location, hour, activity)
        
        # Calculate total modifier
        modifier = cls.calculate_context_stress_modifier(
            activity=activity,
            location=location,
            hour=hour,
            work_intensity=work_intensity,
            is_weekend=is_weekend,
            sleep_duration=sleep_duration,
            noise_environment=noise_env,
            social_context=social_ctx
        )
        
        # Apply modifier to base stress
        modified_stress = base_stress + modifier
        
        # Ensure valid range [1-9]
        modified_stress = max(1.0, min(9.0, modified_stress))
        
        return round(modified_stress, 1)
