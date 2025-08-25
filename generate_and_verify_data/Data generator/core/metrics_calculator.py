"""
Health Metrics Calculator
Tính toán calories, steps, heart rate và các metrics sức khỏe khác
"""

import random

class HealthMetricsCalculator:
    """Tính toán các health metrics dựa trên activity và context"""
    
    def __init__(self, user_profile):
        self.user_profile = user_profile

    def calculate_hourly_calories(self, activity, duration_hours, stress_level, base_metabolic_rate=None):
        """
        Tính calories tiêu thụ theo giờ dựa trên activity (cho sample nhỏ)
        """
        # Calculate BMR based on Age/Gender if not provided
        if base_metabolic_rate is None:
            base_metabolic_rate = self.user_profile.calculate_bmr()
        
        # Activity multipliers (calories per hour) - including stairs
        activity_calories = {
            'Sitting': base_metabolic_rate * 1.0,      # 75 cal/h
            'Standing': base_metabolic_rate * 1.2,     # 90 cal/h  
            'Walking': base_metabolic_rate * 3.0,      # 225 cal/h
            'Jogging': base_metabolic_rate * 8.0,      # 600 cal/h
            'Upstairs': base_metabolic_rate * 5.0,     # 375 cal/h - climbing burns more
            'Downstairs': base_metabolic_rate * 3.5    # 262 cal/h - descending still effort
        }
        
        base_calories_per_hour = activity_calories.get(activity, base_metabolic_rate)
        
        # Stress modifier - stress cao tiêu thụ calories nhiều hơn
        stress_modifier = 1 + (stress_level - 4) * 0.05  # +/-10% based on stress
        
        # For very short durations (samples), calculate proportionally but add minimum base
        if duration_hours < 0.1:  # Less than 6 minutes
            # Add a base calorie burn even for short samples
            base_sample_calories = 1.0  # Minimum per sample
            proportional_calories = base_calories_per_hour * duration_hours * stress_modifier
            total_calories = base_sample_calories + proportional_calories
        else:
            # Duration modifier với diminishing returns cho high-intensity activities
            if activity == 'Jogging' and duration_hours > 1:
                # Jogging lâu -> hiệu quả giảm
                effective_duration = 1 + (duration_hours - 1) * 0.7
            else:
                effective_duration = duration_hours
            
            total_calories = base_calories_per_hour * effective_duration * stress_modifier
        
        # Add some realistic variation
        variation = random.uniform(0.9, 1.1)
        
        return max(1, int(total_calories * variation))  # Minimum 1 calorie per sample

    def calculate_hourly_steps(self, activity, duration_hours, energy_level=0.7):
        """
        Tính step count theo giờ dựa trên activity (cho sample nhỏ)
        """
        # Steps per hour by activity - including Upstairs/Downstairs
        activity_steps_per_hour = {
            'Sitting': 30,       # Very minimal movement per hour
            'Standing': 150,     # Small movements per hour  
            'Walking': 4000,     # Normal walking pace per hour
            'Jogging': 7500,     # Running pace per hour
            'Upstairs': 5500,    # Climbing stairs - more effort than walking
            'Downstairs': 4800   # Going downstairs - less effort than upstairs but more than walking
        }
        
        base_steps_per_hour = activity_steps_per_hour.get(activity, 100)
        
        # Energy level modifier
        energy_modifier = 0.7 + energy_level * 0.6  # 0.7 - 1.3 range
        
        # For short durations, scale proportionally with minimum
        if duration_hours < 0.1:  # Less than 6 minutes
            if activity in ['Sitting', 'Standing']:
                # Minimal steps for stationary activities
                steps = random.randint(0, int(5 * duration_hours / 0.01667))  # 0-5 steps per minute
            else:
                # Proportional steps for active activities
                proportional_steps = base_steps_per_hour * duration_hours * energy_modifier
                steps = max(1, int(proportional_steps + random.uniform(-0.3, 0.3) * proportional_steps))
        else:
            # Duration scaling for longer periods
            total_steps = int(base_steps_per_hour * duration_hours * energy_modifier)
            # Add realistic variation
            variation = random.uniform(0.8, 1.2)
            steps = int(total_steps * variation)
        
        return max(0, steps)

    def calculate_heart_rate(self, activity, stress_level, base_hr, energy_level=0.7):
        """
        Tính heart rate dựa trên activity và stress level
        """
        # Activity HR modifiers - including stairs activities
        activity_hr_modifier = {
            'Walking': 18, 'Jogging': 40, 'Standing': 6, 'Sitting': 0,
            'Upstairs': 28,   # Higher than walking - climbing is harder
            'Downstairs': 22  # Moderate - less than upstairs but more than walking
        }
        
        # Use age-appropriate heart rate limits
        max_hr = self.user_profile.calculate_max_heart_rate()
        min_hr = self.user_profile.calculate_resting_heart_rate() - 10  # Allow 10 bpm below resting
        
        # Calculate current HR
        current_hr = (
            base_hr + 
            activity_hr_modifier.get(activity, 0) +
            (stress_level - 4) * 3 +  # Stress effect
            (1 - energy_level) * 5 +  # Fatigue effect
            random.uniform(-4, 4)     # Random variation
        )
        
        # Ensure HR stays within realistic range
        current_hr = max(min_hr, min(max_hr, int(current_hr)))
        
        return current_hr

    def calculate_reaction_time(self, stress_level, sleep_quality, energy_level):
        """
        Tính reaction time dựa trên stress, sleep quality và energy level
        """
        base_reaction = self.user_profile.profile['base_reaction_time']
        
        reaction_variation = (
            (1 - sleep_quality) * 60 +      # Poor sleep slows reaction
            (stress_level - 4) * 10 +       # High stress affects reaction
            (1 - energy_level) * 40 +       # Low energy slows reaction
            random.uniform(-25, 25)         # Random variation
        )
        
        reaction_time = max(250, min(650, base_reaction + reaction_variation))
        
        return round(reaction_time, 1)
