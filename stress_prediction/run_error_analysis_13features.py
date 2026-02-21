"""
Run comprehensive error analysis for 13-feature LSTM model
"""

from pathlib import Path
from error_analysis import ErrorAnalyzer

def main():
    """Run error analysis for 13-feature model"""
    
    # Paths for 13-feature model
    base_dir = Path(__file__).parent
    model_path = base_dir.parent / 'models' / 'lstm_13features_best.keras'
    data_path = base_dir.parent / 'data' / 'optimized_health_data_13features.csv'
    results_dir = base_dir.parent / 'results' / 'error_analysis_13features'
    
    print("="*70)
    print("  ERROR ANALYSIS - 13-FEATURE LSTM MODEL")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Data: {data_path}")
    print(f"Results: {results_dir}")
    
    # Verify paths exist
    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found at {model_path}")
        return
    
    if not data_path.exists():
        print(f"\n❌ ERROR: Data not found at {data_path}")
        return
    
    print("\n✅ All files found. Starting analysis...")
    
    # Run analysis
    analyzer = ErrorAnalyzer(
        model_path=str(model_path),
        data_path=str(data_path),
        results_dir=str(results_dir)
    )
    
    analyzer.run_full_analysis()
    
    print("\n" + "="*70)
    print("  ✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    print("\nNext steps:")
    print("  1. Review ERROR_ANALYSIS_REPORT.md")
    print("  2. Check visualizations (PNG files)")
    print("  3. Compare with 10-feature model results")
    print("  4. Document insights for thesis defense")


if __name__ == '__main__':
    main()
