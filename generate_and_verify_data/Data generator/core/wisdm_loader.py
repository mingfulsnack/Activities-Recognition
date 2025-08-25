"""
WISDM Data Loader
Tải và quản lý dữ liệu accelerometer từ WISDM dataset
"""

import os
import random

class WisdmDataLoader:
    """Tải và quản lý dữ liệu accelerometer thực từ WISDM dataset"""
    
    def __init__(self):
        self.wisdm_data = {}
        self._wisdm_indices = {}
        self.har_activities = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
    def load_wisdm_data(self):
        """Load real accelerometer data from WISDM dataset"""
        wisdm_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'WISDM_ar_v1.1_raw.txt')
        
        if not os.path.exists(wisdm_path):
            print(f"⚠️ WISDM data not found: {wisdm_path}")
            return {}
        
        print("📂 Loading real WISDM accelerometer data...")
        
        data_by_activity = {}
        
        try:
            with open(wisdm_path, 'r') as f:
                for line in f:
                    try:
                        # Parse: user,activity,timestamp,x,y,z;
                        parts = line.strip().rstrip(';').split(',')
                        if len(parts) >= 6:
                            user = int(parts[0])
                            activity = parts[1].strip()
                            timestamp = int(parts[2])
                            x = float(parts[3])
                            y = float(parts[4])
                            z = float(parts[5])
                            
                            if activity not in data_by_activity:
                                data_by_activity[activity] = []
                            
                            data_by_activity[activity].append([x, y, z])
                    except:
                        continue
            
            # Show stats
            for activity, samples in data_by_activity.items():
                print(f"   {activity}: {len(samples):,} real samples")
            
            self.wisdm_data = data_by_activity
            return data_by_activity
            
        except Exception as e:
            print(f"❌ Error loading WISDM: {e}")
            return {}

    def get_real_accelerometer_sample(self, activity, add_noise=True):
        """
        Get CONSISTENT real accelerometer sample with temporal coherence
        """
        if activity not in self.wisdm_data or len(self.wisdm_data[activity]) == 0:
            return self._generate_synthetic_accelerometer(activity)
        
        # Use sequential sampling instead of random sampling
        # This maintains temporal consistency within activity segments
        
        if activity not in self._wisdm_indices:
            self._wisdm_indices[activity] = 0
        
        # Get current sample sequentially
        current_index = self._wisdm_indices[activity]
        activity_data = self.wisdm_data[activity]
        
        sample = activity_data[current_index % len(activity_data)]
        
        # Advance index for next call
        self._wisdm_indices[activity] = (current_index + 1) % len(activity_data)
        
        if add_noise:
            # Add minimal noise to avoid exact repetition
            x, y, z = sample
            noise_level = 0.05  # Very small noise
            x += random.uniform(-noise_level, noise_level)
            y += random.uniform(-noise_level, noise_level) 
            z += random.uniform(-noise_level, noise_level)
            return [x, y, z]
        
        return sample

    def _generate_synthetic_accelerometer(self, activity):
        """
        Enhanced physics-based synthetic accelerometer with better activity matching
        """
        import numpy as np
        
        # More accurate physics-based patterns based on WISDM analysis
        patterns = {
            'Sitting': {
                'x_base': 0, 'x_var': 0.8, 
                'y_base': 0, 'y_var': 0.8,
                'z_base': 9.8, 'z_var': 0.5
            },
            'Standing': {
                'x_base': 0, 'x_var': 1.5,
                'y_base': 0, 'y_var': 1.0,
                'z_base': 9.5, 'z_var': 1.0
            },
            'Walking': {
                'x_base': 0, 'x_var': 4.0,
                'y_base': 10, 'y_var': 5.0,
                'z_base': 1, 'z_var': 3.0
            },
            'Jogging': {
                'x_base': 0, 'x_var': 6.0,
                'y_base': 12, 'y_var': 8.0,
                'z_base': 2, 'z_var': 4.0
            },
            'Upstairs': {
                'x_base': 0, 'x_var': 4.5,
                'y_base': 8, 'y_var': 4.0,
                'z_base': 1, 'z_var': 3.5
            },
            'Downstairs': {
                'x_base': 0, 'x_var': 4.0,
                'y_base': 8, 'y_var': 4.0,
                'z_base': 1, 'z_var': 3.0
            }
        }
        
        pattern = patterns.get(activity, patterns['Sitting'])
        
        # Generate with Gaussian distribution for more realistic values
        x = np.random.normal(pattern['x_base'], pattern['x_var'])
        y = np.random.normal(pattern['y_base'], pattern['y_var'])
        z = np.random.normal(pattern['z_base'], pattern['z_var'])
        
        # Clip to realistic ranges
        x = np.clip(x, -20, 20)
        y = np.clip(y, -20, 20)  
        z = np.clip(z, -20, 20)
        
        return [x, y, z]
    
    def get_available_activities(self):
        """Trả về danh sách activities có sẵn trong WISDM data"""
        return list(self.wisdm_data.keys())
    
    def get_activity_sample_count(self, activity):
        """Trả về số lượng samples của một activity"""
        return len(self.wisdm_data.get(activity, []))
