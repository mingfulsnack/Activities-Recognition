"""
Quick Generator - Uses existing generator + feature selection
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run existing generator then apply feature selection"""
    print("="*70)
    print("Phase 1: Refactored Data Generation")
    print("="*70)
    
    # Step 1: Generate full 44-field dataset
    print("\n[1/2] Generating 44-field dataset...")
    result = subprocess.run([
        sys.executable,
        "refactored_health_data_generator.py"
    ], capture_output=False)
    
    if result.returncode != 0:
        print("\n❌ Generation failed!")
        return
    
    # Step 2: Apply feature selection
    print("\n" + "="*70)
    print("[2/2] Applying feature selection (44 → 20 fields)...")
    print("="*70)
    
    result = subprocess.run([
        sys.executable,
        "feature_selector.py"
    ], capture_output=False)
    
    if result.returncode != 0:
        print("\n❌ Feature selection failed!")
        return
    
    print("\n" + "="*70)
    print("✅ Phase 1 completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. data/quota_balanced_health_data_30days.csv (44 fields)")
    print("  2. data/quota_balanced_health_data_20features.csv (20 fields) ← USE THIS")
    print("\nNext steps:")
    print("  - Validate data quality")
    print("  - Test with HAR model")
    print("  - Train stress prediction models")

if __name__ == "__main__":
    main()
