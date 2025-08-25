"""
Health Metrics Calculator
Tính toán calories, steps, heart rate và các metrics sức khỏe khác
"""

import random
import numpy as np

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
            # Scale calories appropriately for samples (30-second intervals)
            # Normal BMR ~75 cal/hour = 1.25 cal/minute = 0.625 cal per 30-second sample
            proportional_calories = base_calories_per_hour * duration_hours * stress_modifier
            # Minimum calories per sample should be reasonable
            total_calories = max(0.5, proportional_calories)  # At least 0.5 cal per sample
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
        
        return max(0.5, round(total_calories * variation, 1))  # Minimum 0.5 calorie per sample

    def calculate_hourly_steps(self, activity, duration_hours, energy_level=0.7):
        """
        Tính step count theo giờ dựa trên activity (cho sample nhỏ)
        """
        # REALISTIC Steps per hour by activity  
        # TARGET: 8,000-15,000 steps total per day
        activity_steps_per_hour = {
            'Sitting': 2,        # Almost no movement per hour
            'Standing': 10,      # Very small movements per hour  
            'Walking': 1500,     # Conservative walking pace (was 2000)
            'Jogging': 4800,     # Conservative running pace (was 6000)
            'Upstairs': 400,     # Short climbing bursts (was 800)
            'Downstairs': 300    # Short descending bursts (was 600)
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

    def calculate_mood_score(self, base_mood_factor, hour, activity, location, stress_level):
        """
        Tính mood score với gradual intra-day variation
        """
        # Base mood from daily context
        daily_base_mood = 5 + base_mood_factor * 3
        
        # NATURAL DAILY MOOD RHYTHM 
        # Morning: neutral → lunch: peak → afternoon: dip → evening: recovery
        if hour < 7:
            time_mood_effect = -0.5  # Early morning grogginess
        elif hour < 12:
            time_mood_effect = (hour - 7) * 0.2  # Gradual improvement
        elif hour < 14:
            time_mood_effect = 1.0  # Peak mood around lunch
        elif hour < 16:
            time_mood_effect = 1.0 - (hour - 14) * 0.5  # Afternoon dip
        elif hour < 19:
            time_mood_effect = 0.0 + (hour - 16) * 0.3  # Evening recovery
        elif hour < 22:
            time_mood_effect = 0.9  # Good evening mood
        else:
            time_mood_effect = 0.9 - (hour - 22) * 0.3  # Late night decline
        
        # ACTIVITY MOOD EFFECTS
        activity_mood_effects = {
            'Jogging': 1.2,      # Exercise improves mood
            'Walking': 0.5,      # Light activity is good
            'Standing': 0.1,     # Neutral
            'Sitting': -0.2,     # Sedentary can lower mood
            'Upstairs': 0.3,     # Small accomplishment feeling
            'Downstairs': 0.1    # Neutral
        }
        
        activity_effect = activity_mood_effects.get(activity, 0)
        
        # LOCATION MOOD EFFECTS
        location_mood_effects = {
            'outdoor': 0.8,      # Nature/fresh air boost
            'gym': 0.6,          # Accomplishment from exercise
            'social': 1.0,       # Social interaction boost  
            'home': 0.3,         # Comfort but neutral
            'work': -0.3,        # Work stress
            'commute': -0.5      # Transportation stress
        }
        
        location_effect = location_mood_effects.get(location, 0)
        
        # STRESS IMPACT ON MOOD
        stress_effect = -(stress_level - 4) * 0.3  # Higher stress → lower mood
        
        # COMBINE ALL EFFECTS with realistic constraints
        final_mood = daily_base_mood + time_mood_effect + activity_effect + location_effect + stress_effect
        
        # Add small random variation (much smaller than before)
        final_mood += random.uniform(-0.1, 0.1)
        
        # Ensure mood stays in realistic range [1-10]
        final_mood = max(1, min(10, final_mood))
        
        return round(final_mood, 1)

    def calculate_realistic_stress_level(self, base_stress, hour, activity, location, 
                                       heart_rate, sleep_quality, work_intensity, 
                                       previous_stress_levels=None):
        """
        Tính stress level realistic cho stress prediction modeling
        """
        # DAILY STRESS RHYTHM - office worker pattern
        if hour < 7:
            time_stress_modifier = -1.0  # Early morning calm
        elif hour < 9:
            time_stress_modifier = 0.5   # Getting ready stress
        elif hour < 12:
            time_stress_modifier = 1.0   # Work morning pressure
        elif hour < 13:
            time_stress_modifier = 0.0   # Lunch break relief
        elif hour < 17:
            time_stress_modifier = 1.5   # Afternoon work peak stress
        elif hour < 18:
            time_stress_modifier = 0.8   # End of work transition
        elif hour < 20:
            time_stress_modifier = -0.5  # Evening relaxation
        else:
            time_stress_modifier = -0.8  # Night calm
        
        # ACTIVITY-BASED STRESS
        activity_stress = {
            'Sitting': 0.2,      # Sedentary can increase stress
            'Standing': 0.1,     # Neutral
            'Walking': -0.3,     # Light exercise reduces stress
            'Jogging': -0.8,     # Exercise significantly reduces stress
            'Upstairs': 0.4,     # Physical exertion increases momentary stress
            'Downstairs': 0.2    # Less stressful than upstairs
        }.get(activity, 0)
        
        # LOCATION-BASED STRESS
        location_stress = {
            'work': 1.5,         # Work environment stress
            'commute': 1.0,      # Transportation stress
            'home': -0.5,        # Home comfort reduces stress
            'outdoor': -0.7,     # Nature reduces stress
            'gym': -0.3,         # Exercise environment moderately good
            'social': -0.4       # Social support reduces stress
        }.get(location, 0)
        
        # WORK INTENSITY MODIFIER
        work_stress_modifier = {
            'low': -0.5,
            'normal': 0,
            'high': 1.5,
            'none': -1.0  # Weekend/no work
        }.get(work_intensity, 0)
        
        # PHYSIOLOGICAL INDICATORS
        # Heart rate correlation with stress
        if heart_rate > 85:
            hr_stress = 1.0
        elif heart_rate > 75:
            hr_stress = 0.5
        elif heart_rate < 60:
            hr_stress = -0.3
        else:
            hr_stress = 0
        
        # Sleep quality impact
        sleep_stress = (1 - sleep_quality) * 2  # Poor sleep -> high stress
        
        # STRESS MOMENTUM - stress tends to persist
        momentum_effect = 0
        if previous_stress_levels and len(previous_stress_levels) > 0:
            recent_avg = np.mean(previous_stress_levels[-3:])  # Last 3 samples
            momentum_effect = (recent_avg - 4) * 0.3  # Trend continuation
        
        # COMBINE ALL FACTORS
        calculated_stress = (
            base_stress +
            time_stress_modifier +
            activity_stress +
            location_stress +
            work_stress_modifier +
            hr_stress +
            sleep_stress +
            momentum_effect
        )
        
        # Add small realistic variation
        calculated_stress += random.uniform(-0.2, 0.2)
        
        # Ensure stress level stays in realistic range [1-9]
        calculated_stress = max(1, min(9, calculated_stress))
        
        return round(calculated_stress, 1)
