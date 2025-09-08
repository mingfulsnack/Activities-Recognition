"""
IMPROVE ACCELEROMETER PATTERNS 
Fix Upstairs (0% accuracy) và Walking (55.3% accuracy) issues
"""

import pandas as pd
import numpy as np
import sys
import os

# Add HAR directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'HAR'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

class AccelerometerPatternAnalyzer:
    """Analyze real WISDM patterns để improve generation"""
    
    def __init__(self):
        self.wisdm_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'WISDM_ar_v1.1_raw.txt')
        self.activities = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
    def load_real_wisdm_patterns(self):
        """Load và analyze real WISDM patterns"""
        print("🔍 === ANALYZING REAL WISDM PATTERNS ===")
        
        try:
            # Load WISDM data
            wisdm_data = pd.read_csv(
                self.wisdm_path, 
                header=None, 
                names=['user', 'activity', 'timestamp', 'x', 'y', 'z'],
                sep=',', 
                on_bad_lines='skip'
            )
            
            # Clean data
            wisdm_data['z'] = wisdm_data['z'].astype(str).str.replace(';', '', regex=True)
            for col in ['x', 'y', 'z']:
                wisdm_data[col] = pd.to_numeric(wisdm_data[col], errors='coerce')
            wisdm_data = wisdm_data.dropna()
            
            print(f"✅ Loaded {len(wisdm_data)} WISDM samples")
            
            # Analyze patterns by activity
            patterns = {}
            
            for activity in self.activities:
                activity_data = wisdm_data[wisdm_data['activity'] == activity]
                
                if len(activity_data) == 0:
                    print(f"⚠️  No data for {activity}")
                    continue
                    
                # Calculate statistics
                patterns[activity] = {
                    'count': len(activity_data),
                    'x_stats': {
                        'mean': activity_data['x'].mean(),
                        'std': activity_data['x'].std(),
                        'min': activity_data['x'].min(),
                        'max': activity_data['x'].max(),
                        'q25': activity_data['x'].quantile(0.25),
                        'q75': activity_data['x'].quantile(0.75)
                    },
                    'y_stats': {
                        'mean': activity_data['y'].mean(),
                        'std': activity_data['y'].std(),
                        'min': activity_data['y'].min(),
                        'max': activity_data['y'].max(),
                        'q25': activity_data['y'].quantile(0.25),
                        'q75': activity_data['y'].quantile(0.75)
                    },
                    'z_stats': {
                        'mean': activity_data['z'].mean(),
                        'std': activity_data['z'].std(),
                        'min': activity_data['z'].min(),
                        'max': activity_data['z'].max(),
                        'q25': activity_data['z'].quantile(0.25),
                        'q75': activity_data['z'].quantile(0.75)
                    },
                    'magnitude_stats': {
                        'mean': np.sqrt(activity_data['x']**2 + activity_data['y']**2 + activity_data['z']**2).mean(),
                        'std': np.sqrt(activity_data['x']**2 + activity_data['y']**2 + activity_data['z']**2).std()
                    }
                }
                
                print(f"\n📊 {activity} ({patterns[activity]['count']} samples):")
                print(f"  X: {patterns[activity]['x_stats']['mean']:.2f} ± {patterns[activity]['x_stats']['std']:.2f}")
                print(f"  Y: {patterns[activity]['y_stats']['mean']:.2f} ± {patterns[activity]['y_stats']['std']:.2f}")
                print(f"  Z: {patterns[activity]['z_stats']['mean']:.2f} ± {patterns[activity]['z_stats']['std']:.2f}")
                print(f"  Magnitude: {patterns[activity]['magnitude_stats']['mean']:.2f} ± {patterns[activity]['magnitude_stats']['std']:.2f}")
            
            return patterns
            
        except Exception as e:
            print(f"❌ Error loading WISDM: {e}")
            return None

    def generate_improved_patterns(self, patterns):
        """Generate improved synthetic patterns based on analysis"""
        print(f"\n🔧 === GENERATING IMPROVED PATTERNS ===")
        
        if not patterns:
            print("❌ No patterns available")
            return None
            
        improved_patterns = {}
        
        for activity in self.activities:
            if activity not in patterns:
                print(f"⚠️  Skipping {activity} - no WISDM data")
                continue
                
            stats = patterns[activity]
            
            # Create more accurate patterns based on real data
            improved_patterns[activity] = {
                'x_base': stats['x_stats']['mean'],
                'x_var': stats['x_stats']['std'],
                'x_range': (stats['x_stats']['min'], stats['x_stats']['max']),
                'y_base': stats['y_stats']['mean'], 
                'y_var': stats['y_stats']['std'],
                'y_range': (stats['y_stats']['min'], stats['y_stats']['max']),
                'z_base': stats['z_stats']['mean'],
                'z_var': stats['z_stats']['std'], 
                'z_range': (stats['z_stats']['min'], stats['z_stats']['max']),
                'magnitude_target': stats['magnitude_stats']['mean'],
                'magnitude_var': stats['magnitude_stats']['std']
            }
            
            print(f"✅ {activity}: X({improved_patterns[activity]['x_base']:.2f}±{improved_patterns[activity]['x_var']:.2f}), "
                  f"Y({improved_patterns[activity]['y_base']:.2f}±{improved_patterns[activity]['y_var']:.2f}), "
                  f"Z({improved_patterns[activity]['z_base']:.2f}±{improved_patterns[activity]['z_var']:.2f})")
        
        return improved_patterns

    def update_wisdm_loader(self, improved_patterns):
        """Update wisdm_loader.py với improved patterns"""
        print(f"\n📝 === UPDATING WISDM_LOADER.PY ===")
        
        if not improved_patterns:
            print("❌ No improved patterns to update")
            return
            
        # Read current file
        wisdm_loader_path = os.path.join(os.path.dirname(__file__), 'core', 'wisdm_loader.py')
        
        try:
            with open(wisdm_loader_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find and replace patterns section
            start_marker = "        # More accurate physics-based patterns based on WISDM analysis"
            end_marker = "        }"
            
            start_idx = content.find(start_marker)
            if start_idx == -1:
                print("❌ Cannot find patterns section in wisdm_loader.py")
                return
                
            # Find the end of the patterns dict
            end_idx = content.find(end_marker, start_idx)
            if end_idx == -1:
                print("❌ Cannot find end of patterns section")
                return
            end_idx += len(end_marker)
            
            # Generate new patterns section
            new_patterns = "        # IMPROVED patterns based on real WISDM analysis\n"
            new_patterns += "        patterns = {\n"
            
            for activity, pattern in improved_patterns.items():
                new_patterns += f"            '{activity}': {{\n"
                new_patterns += f"                'x_base': {pattern['x_base']:.3f}, 'x_var': {pattern['x_var']:.3f}, \n"
                new_patterns += f"                'y_base': {pattern['y_base']:.3f}, 'y_var': {pattern['y_var']:.3f},\n"
                new_patterns += f"                'z_base': {pattern['z_base']:.3f}, 'z_var': {pattern['z_var']:.3f}\n"
                new_patterns += f"            }},\n"
            
            new_patterns += "        }"
            
            # Replace the old patterns with new ones
            new_content = content[:start_idx] + new_patterns + content[end_idx:]
            
            # Write back to file
            with open(wisdm_loader_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ Updated wisdm_loader.py with improved patterns")
            
        except Exception as e:
            print(f"❌ Error updating wisdm_loader.py: {e}")

    def compare_before_after(self, improved_patterns):
        """Compare patterns before and after improvement"""
        print(f"\n📊 === BEFORE vs AFTER COMPARISON ===")
        
        # Current patterns from wisdm_loader.py
        current_patterns = {
            'Sitting': {'x_base': 0, 'x_var': 0.8, 'y_base': 0, 'y_var': 0.8, 'z_base': 9.8, 'z_var': 0.5},
            'Standing': {'x_base': 0, 'x_var': 1.5, 'y_base': 0, 'y_var': 1.0, 'z_base': 9.5, 'z_var': 1.0},
            'Walking': {'x_base': 0, 'x_var': 4.0, 'y_base': 10, 'y_var': 5.0, 'z_base': 1, 'z_var': 3.0},
            'Jogging': {'x_base': 0, 'x_var': 6.0, 'y_base': 12, 'y_var': 8.0, 'z_base': 2, 'z_var': 4.0},
            'Upstairs': {'x_base': 0, 'x_var': 4.5, 'y_base': 8, 'y_var': 4.0, 'z_base': 1, 'z_var': 3.5},
            'Downstairs': {'x_base': 0, 'x_var': 4.0, 'y_base': 8, 'y_var': 4.0, 'z_base': 1, 'z_var': 3.0}
        }
        
        for activity in self.activities:
            if activity in improved_patterns and activity in current_patterns:
                old = current_patterns[activity]
                new = improved_patterns[activity]
                
                print(f"\n🔄 {activity}:")
                print(f"  X: {old['x_base']:.2f}±{old['x_var']:.2f} → {new['x_base']:.2f}±{new['x_var']:.2f}")
                print(f"  Y: {old['y_base']:.2f}±{old['y_var']:.2f} → {new['y_base']:.2f}±{new['y_var']:.2f}")
                print(f"  Z: {old['z_base']:.2f}±{old['z_var']:.2f} → {new['z_base']:.2f}±{new['z_var']:.2f}")
                
                # Highlight major changes
                x_change = abs(new['x_base'] - old['x_base']) + abs(new['x_var'] - old['x_var'])
                y_change = abs(new['y_base'] - old['y_base']) + abs(new['y_var'] - old['y_var'])
                z_change = abs(new['z_base'] - old['z_base']) + abs(new['z_var'] - old['z_var'])
                
                if x_change > 2 or y_change > 2 or z_change > 2:
                    print(f"  🚨 MAJOR CHANGE - expect significant accuracy improvement")

def main():
    """Main improvement function"""
    print("🔧 === ACCELEROMETER PATTERN IMPROVEMENT ===")
    print("Goal: Fix Upstairs (0% accuracy) và Walking (55.3% accuracy)")
    
    analyzer = AccelerometerPatternAnalyzer()
    
    # Step 1: Analyze real WISDM patterns
    real_patterns = analyzer.load_real_wisdm_patterns()
    
    if not real_patterns:
        print("❌ Cannot proceed without real WISDM patterns")
        return
    
    # Step 2: Generate improved patterns
    improved_patterns = analyzer.generate_improved_patterns(real_patterns)
    
    # Step 3: Compare before/after
    analyzer.compare_before_after(improved_patterns)
    
    # Step 4: Update wisdm_loader.py
    analyzer.update_wisdm_loader(improved_patterns)
    
    print(f"\n🎯 === IMPROVEMENT COMPLETE ===")
    print("✅ Patterns updated in wisdm_loader.py")
    print("✅ Ready to regenerate dataset with improved accuracy")
    print("\n📋 Next steps:")
    print("1. Run refactored_health_data_generator.py to create new dataset")
    print("2. Run validate_accelerometer_with_har.py to verify improvements")
    print("3. Expected improvements: Upstairs >50%, Walking >70%")

if __name__ == "__main__":
    main()
