"""
User Profile Management
Quản lý thông tin cơ bản của người dùng và tính toán các metrics sinh lý
"""

class UserProfile:
    """Quản lý thông tin cá nhân và tính toán metrics sinh lý cơ bản"""
    
    def __init__(self, age=28, gender='Female'):
        self.profile = {
            'Age': age,
            'Gender': gender,
            'base_sleep_duration': 7.5,
            'base_screen_time': 8.0,
            'base_stress_level': 4,
            'base_reaction_time': 380.0
        }
    
    def calculate_bmr(self):
        """
        Tính Base Metabolic Rate (BMR) dựa trên Age và Gender
        BMR per hour = Daily BMR / 24 hours
        """
        age = self.profile['Age']
        gender = self.profile['Gender'].lower()
        
        # Approximate BMR calculation (Mifflin-St Jeor Equation simplified)
        # Assuming average height/weight for Vietnamese: Female ~52kg/155cm, Male ~62kg/168cm
        if gender == 'female':
            # BMR = (10 × 52) + (6.25 × 155) - (5 × age) - 161
            daily_bmr = (10 * 52) + (6.25 * 155) - (5 * age) - 161
        else:  # male
            # BMR = (10 × 62) + (6.25 × 168) - (5 × age) + 5  
            daily_bmr = (10 * 62) + (6.25 * 168) - (5 * age) + 5
            
        # Convert to hourly BMR
        hourly_bmr = daily_bmr / 24
        
        # Ensure reasonable range 60-90 cal/hour
        return max(60, min(90, hourly_bmr))

    def calculate_max_heart_rate(self):
        """Tính Maximum Heart Rate dựa trên tuổi"""
        age = self.profile['Age']
        return 220 - age

    def calculate_resting_heart_rate(self):
        """Tính Resting Heart Rate dựa trên tuổi và giới tính"""
        age = self.profile['Age']
        gender = self.profile['Gender'].lower()
        
        # Base resting HR by age group and gender
        if gender == 'female':
            if age < 30:
                base_rhr = 68
            elif age < 50:
                base_rhr = 70
            else:
                base_rhr = 72
        else:  # male
            if age < 30:
                base_rhr = 65
            elif age < 50:
                base_rhr = 67
            else:
                base_rhr = 69
                
        return base_rhr
    
    def get_profile_dict(self):
        """Trả về dictionary chứa thông tin profile"""
        return self.profile.copy()
