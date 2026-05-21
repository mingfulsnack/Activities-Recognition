"""
Model Comparison Analysis Script
=================================
Purpose: Compare performance between:
- Baseline Model: 23 features (R² = 0.9343, MAE = 0.5095)
- Reduced Model: 10 features (R² = ?, MAE = ?)

Validates feature selection hypothesis: Top 10 features (98% importance) 
should achieve comparable performance with 43% fewer features.
"""

import pandas as pd
import numpy as np
import os
import json
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelComparator:
    """Class to compare baseline and reduced feature models."""
    
    def __init__(self):
        self.baseline_metrics = {
            'name': '23-Feature Model (Baseline)',
            'n_features': 21,  # After encoding
            'r2': 0.9343,
            'mae': 0.5095,
            'rmse': 0.8123
        }
        self.reduced_metrics = None
        self.results_dir = 'results/feature_comparison/'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_reduced_metrics(self, metrics_file='results/feature_comparison/metrics_10features.txt'):
        """Load metrics from 10-feature model."""
        print(f" Loading reduced model metrics from: {metrics_file}")
        
        if not os.path.exists(metrics_file):
            print(f" Error: Metrics file not found: {metrics_file}")
            print(f"  Please run train_lstm_10features.py first.")
            return False
        
        # Parse metrics file
        metrics = {}
        with open(metrics_file, 'r') as f:
            for line in f:
                if ':' in line and not '=' in line:
                    key, value = line.strip().split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except:
                        metrics[key.strip()] = value.strip()
        
        self.reduced_metrics = {
            'name': '10-Feature Model (Reduced)',
            'n_features': 10,
            'r2': metrics.get('r2', 0),
            'mae': metrics.get('mae', 0),
            'rmse': metrics.get('rmse', 0)
        }
        
        print(f" Loaded metrics:")
        print(f"  R²:   {self.reduced_metrics['r2']:.4f}")
        print(f"  MAE:  {self.reduced_metrics['mae']:.4f}")
        print(f"  RMSE: {self.reduced_metrics['rmse']:.4f}")
        
        return True
        
    def create_comparison_table(self):
        """Create comparison table."""
        print("\n Creating comparison table...")
        
        # Calculate differences
        r2_diff = self.reduced_metrics['r2'] - self.baseline_metrics['r2']
        mae_diff = self.reduced_metrics['mae'] - self.baseline_metrics['mae']
        rmse_diff = self.reduced_metrics['rmse'] - self.baseline_metrics['rmse']
        
        r2_diff_pct = (r2_diff / self.baseline_metrics['r2']) * 100
        mae_diff_pct = (mae_diff / self.baseline_metrics['mae']) * 100
        rmse_diff_pct = (rmse_diff / self.baseline_metrics['rmse']) * 100
        
        feature_reduction = ((self.baseline_metrics['n_features'] - self.reduced_metrics['n_features']) 
                            / self.baseline_metrics['n_features']) * 100
        
        # Create table
        comparison = {
            'Metric': ['Features', 'R²', 'MAE', 'RMSE'],
            'Baseline (23)': [
                self.baseline_metrics['n_features'],
                f"{self.baseline_metrics['r2']:.4f}",
                f"{self.baseline_metrics['mae']:.4f}",
                f"{self.baseline_metrics['rmse']:.4f}"
            ],
            'Reduced (10)': [
                self.reduced_metrics['n_features'],
                f"{self.reduced_metrics['r2']:.4f}",
                f"{self.reduced_metrics['mae']:.4f}",
                f"{self.reduced_metrics['rmse']:.4f}"
            ],
            'Difference': [
                f"{-feature_reduction:.1f}%",
                f"{r2_diff:+.4f} ({r2_diff_pct:+.2f}%)",
                f"{mae_diff:+.4f} ({mae_diff_pct:+.2f}%)",
                f"{rmse_diff:+.4f} ({rmse_diff_pct:+.2f}%)"
            ]
        }
        
        df = pd.DataFrame(comparison)
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        # Save table
        table_path = os.path.join(self.results_dir, 'comparison_table.csv')
        df.to_csv(table_path, index=False)
        print(f"\n✓ Table saved to: {table_path}")
        
        return df
        
    def create_comparison_visualizations(self):
        """Create comparison visualizations."""
        print("\n Creating comparison visualizations...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Feature count comparison
        ax1 = fig.add_subplot(gs[0, 0])
        models = ['Baseline\n(23 features)', 'Reduced\n(10 features)']
        features = [self.baseline_metrics['n_features'], self.reduced_metrics['n_features']]
        bars1 = ax1.bar(models, features, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
        ax1.set_title('Feature Count Comparison', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, max(features) * 1.2)
        
        # Add value labels
        for bar, val in zip(bars1, features):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add reduction percentage
        reduction = ((features[0] - features[1]) / features[0]) * 100
        ax1.text(0.5, max(features) * 1.1, f'{reduction:.1f}% reduction',
                ha='center', transform=ax1.transData, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 2. R² Score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        r2_scores = [self.baseline_metrics['r2'], self.reduced_metrics['r2']]
        bars2 = ax2.bar(models, r2_scores, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
        ax2.set_title('R² Score Comparison', fontsize=13, fontweight='bold')
        ax2.set_ylim(0.85, 1.0)
        ax2.axhline(y=0.90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Good threshold (0.90)')
        ax2.legend(fontsize=9)
        
        # Add value labels
        for bar, val in zip(bars2, r2_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. MAE comparison
        ax3 = fig.add_subplot(gs[0, 2])
        mae_scores = [self.baseline_metrics['mae'], self.reduced_metrics['mae']]
        bars3 = ax3.bar(models, mae_scores, color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax3.set_title('Mean Absolute Error Comparison', fontsize=13, fontweight='bold')
        ax3.axhline(y=0.60, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Acceptable threshold (0.60)')
        ax3.legend(fontsize=9)
        
        # Add value labels
        for bar, val in zip(bars3, mae_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Performance radar chart
        ax4 = fig.add_subplot(gs[1, 0], projection='polar')
        
        # Normalize metrics to 0-1 scale for radar chart
        categories = ['R²', 'MAE\n(inverted)', 'RMSE\n(inverted)']
        baseline_values = [
            self.baseline_metrics['r2'],
            1 - self.baseline_metrics['mae'] / 10,  # Invert (lower is better)
            1 - self.baseline_metrics['rmse'] / 10
        ]
        reduced_values = [
            self.reduced_metrics['r2'],
            1 - self.reduced_metrics['mae'] / 10,
            1 - self.reduced_metrics['rmse'] / 10
        ]
        
        # Close the plot
        baseline_values += baseline_values[:1]
        reduced_values += reduced_values[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax4.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='#3498db')
        ax4.fill(angles, baseline_values, alpha=0.25, color='#3498db')
        ax4.plot(angles, reduced_values, 'o-', linewidth=2, label='Reduced', color='#2ecc71')
        ax4.fill(angles, reduced_values, alpha=0.25, color='#2ecc71')
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories, fontsize=10)
        ax4.set_ylim(0.85, 1.0)
        ax4.set_title('Performance Radar Chart', fontsize=13, fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax4.grid(True)
        
        # 5. Metric differences (bar chart)
        ax5 = fig.add_subplot(gs[1, 1])
        
        r2_diff = (self.reduced_metrics['r2'] - self.baseline_metrics['r2']) / self.baseline_metrics['r2'] * 100
        mae_diff = (self.reduced_metrics['mae'] - self.baseline_metrics['mae']) / self.baseline_metrics['mae'] * 100
        rmse_diff = (self.reduced_metrics['rmse'] - self.baseline_metrics['rmse']) / self.baseline_metrics['rmse'] * 100
        
        metrics_names = ['R²', 'MAE', 'RMSE']
        differences = [r2_diff, mae_diff, rmse_diff]
        colors_diff = ['green' if d >= 0 and m == 'R²' or d <= 0 and m != 'R²' else 'red' 
                       for d, m in zip(differences, metrics_names)]
        
        bars5 = ax5.barh(metrics_names, differences, color=colors_diff, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('% Change from Baseline', fontsize=11, fontweight='bold')
        ax5.set_title('Performance Change (%)', fontsize=13, fontweight='bold')
        ax5.axvline(x=0, color='black', linewidth=2)
        ax5.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars5, differences):
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:+.2f}%',
                    ha='left' if width >= 0 else 'right',
                    va='center', fontsize=11, fontweight='bold')
        
        # 6. Summary text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Determine validation result
        r2_threshold = 0.92  # Within 1.5% of baseline
        if self.reduced_metrics['r2'] >= r2_threshold:
            validation_status = " VALIDATION SUCCESS"
            status_color = 'green'
            conclusion = "10 features achieve comparable\nperformance with 43% reduction!"
        elif self.reduced_metrics['r2'] >= 0.90:
            validation_status = " ACCEPTABLE"
            status_color = 'orange'
            conclusion = "Performance slightly lower\nbut still acceptable (R² ≥ 0.90)"
        else:
            validation_status = " PERFORMANCE DROP"
            status_color = 'red'
            conclusion = "10 features may not be\nsufficient (R² < 0.90)"
        
        summary_text = f"""
FEATURE SELECTION VALIDATION
{'=' * 35}

{validation_status}

Feature Reduction:
  23 → 10 features (-52.4%)

Performance Metrics:
  R²:   {self.baseline_metrics['r2']:.4f} → {self.reduced_metrics['r2']:.4f}
  MAE:  {self.baseline_metrics['mae']:.4f} → {self.reduced_metrics['mae']:.4f}
  RMSE: {self.baseline_metrics['rmse']:.4f} → {self.reduced_metrics['rmse']:.4f}

Conclusion:
  {conclusion}

Benefits of 10-Feature Model:
  ✓ 52% fewer features
  ✓ Faster training
  ✓ Better interpretability
  ✓ Lower overfitting risk
        """
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add main title
        fig.suptitle('LSTM Model Comparison: 23 Features vs 10 Features', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save figure
        plot_path = os.path.join(self.results_dir, 'model_comparison_comprehensive.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison visualization saved to: {plot_path}")
        
        plt.close()
        
    def generate_final_report(self):
        """Generate detailed comparison report."""
        print("\n Generating final report...")
        
        # Calculate metrics
        r2_diff = self.reduced_metrics['r2'] - self.baseline_metrics['r2']
        mae_diff = self.reduced_metrics['mae'] - self.baseline_metrics['mae']
        rmse_diff = self.reduced_metrics['rmse'] - self.baseline_metrics['rmse']
        
        r2_diff_pct = (r2_diff / self.baseline_metrics['r2']) * 100
        mae_diff_pct = (mae_diff / self.baseline_metrics['mae']) * 100
        
        feature_reduction = ((self.baseline_metrics['n_features'] - self.reduced_metrics['n_features']) 
                            / self.baseline_metrics['n_features']) * 100
        
        # Determine conclusion
        if self.reduced_metrics['r2'] >= 0.92:
            validation = " VALIDATION SUCCESS"
            conclusion = "The feature selection hypothesis is confirmed. Top 10 features (98% cumulative importance) achieve comparable performance with 52% fewer features."
        elif self.reduced_metrics['r2'] >= 0.90:
            validation = " ACCEPTABLE PERFORMANCE"
            conclusion = "Performance is slightly lower but still acceptable. The 10-feature model provides a good trade-off between simplicity and accuracy."
        else:
            validation = " PERFORMANCE DROP"
            conclusion = "The 10-feature model shows significant performance degradation. May need to include more features or investigate feature engineering."
        
        # Create report
        report = []
        report.append("=" * 80)
        report.append("FEATURE SELECTION VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("Date: February 2026")
        report.append("Experiment: Compare LSTM performance with 23 vs 10 features")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Status: {validation}")
        report.append("")
        report.append(conclusion)
        report.append("")
        
        # Model Comparison
        report.append("MODEL COMPARISON")
        report.append("-" * 80)
        report.append(f"{'Metric':<20} {'Baseline (23)':<15} {'Reduced (10)':<15} {'Difference':<20}")
        report.append("-" * 80)
        report.append(f"{'Features':<20} {self.baseline_metrics['n_features']:<15} {self.reduced_metrics['n_features']:<15} {-feature_reduction:>+6.1f}%")
        report.append(f"{'R² Score':<20} {self.baseline_metrics['r2']:<15.4f} {self.reduced_metrics['r2']:<15.4f} {r2_diff:>+7.4f} ({r2_diff_pct:>+6.2f}%)")
        report.append(f"{'MAE':<20} {self.baseline_metrics['mae']:<15.4f} {self.reduced_metrics['mae']:<15.4f} {mae_diff:>+7.4f} ({mae_diff_pct:>+6.2f}%)")
        report.append(f"{'RMSE':<20} {self.baseline_metrics['rmse']:<15.4f} {self.reduced_metrics['rmse']:<15.4f} {rmse_diff:>+7.4f}")
        report.append("")
        
        # Top 10 Features
        report.append("TOP 10 SELECTED FEATURES")
        report.append("-" * 80)
        top_features = [
            ("1.  Location", "64.98%"),
            ("2.  Heart_Rate", "13.93%"),
            ("3.  Screen_Usage_Current", "7.46%"),
            ("4.  Phone_Event_Frequency", "3.35%"),
            ("5.  Mood_Score", "2.55%"),
            ("6.  Context_Stress_Modifier", "1.99%"),
            ("7.  Social_Interaction_Current", "1.50%"),
            ("8.  Activity", "0.97%"),
            ("9.  Sleep_Hours", "0.71%"),
            ("10. Hour", "0.63%")
        ]
        for feature, importance in top_features:
            report.append(f"  {feature:<35} {importance:>10}")
        report.append("")
        report.append("  Total Cumulative Importance: 98.07%")
        report.append("")
        
        # Advantages of 10-Feature Model
        report.append("ADVANTAGES OF 10-FEATURE MODEL")
        report.append("-" * 80)
        report.append("  ✓ Simpler Model:")
        report.append("    - 52% fewer features")
        report.append("    - Easier to interpret and explain")
        report.append("    - Reduced data collection requirements")
        report.append("")
        report.append("  ✓ Better Generalization:")
        report.append("    - Lower risk of overfitting")
        report.append("    - More robust to noise in less important features")
        report.append("")
        report.append("  ✓ Computational Efficiency:")
        report.append("    - Faster training time")
        report.append("    - Faster inference")
        report.append("    - Lower memory footprint")
        report.append("")
        report.append("  ✓ Practical Benefits:")
        report.append("    - Focus on most important features")
        report.append("    - Easier to deploy on resource-constrained devices")
        report.append("    - Better for real-time applications")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        if self.reduced_metrics['r2'] >= 0.92:
            report.append("  1.  Use 10-feature model for production")
            report.append("  2. Continue with GRU/TCN/Transformer comparison using 10 features")
            report.append("  3. Focus feature engineering on top 10 features")
            report.append("  4. Consider further reduction (top 5-7 features) for mobile deployment")
        elif self.reduced_metrics['r2'] >= 0.90:
            report.append("  1. 10-feature model acceptable for most use cases")
            report.append("  2. Keep 23-feature model as backup for high-accuracy requirements")
            report.append("  3. Investigate if performance can be improved with feature engineering")
            report.append("  4. Consider ensemble of 10-feature and 23-feature models")
        else:
            report.append("  1.  Need to include more features (top 15-20)")
            report.append("  2. Investigate why performance dropped significantly")
            report.append("  3. Consider feature engineering to compensate for missing features")
            report.append("  4. Re-evaluate feature importance with different methods")
        report.append("")
        
        # Next Steps
        report.append("NEXT STEPS")
        report.append("-" * 80)
        report.append("  Phase 2 Step 4: Model Comparison")
        report.append("    - Implement GRU model (10 features)")
        report.append("    - Implement TCN model (10 features)")
        report.append("    - Implement Transformer model (10 features)")
        report.append("    - Compare all models: LSTM vs GRU vs TCN vs Transformer")
        report.append("")
        report.append("=" * 80)
        
        # Print report
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        report_path = os.path.join(self.results_dir, 'FEATURE_SELECTION_VALIDATION_REPORT.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to: {report_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("MODEL COMPARISON ANALYSIS: 23 Features vs 10 Features")
    print("=" * 80)
    print("")
    
    # Create comparator
    comparator = ModelComparator()
    
    # Load reduced model metrics
    if not comparator.load_reduced_metrics():
        print("\n Cannot proceed without 10-feature model metrics.")
        print("  Please run train_lstm_10features.py first.")
        return
    
    # Perform comparison
    try:
        comparator.create_comparison_table()
        comparator.create_comparison_visualizations()
        comparator.generate_final_report()
        
        print("\n" + "=" * 80)
        print("✓ COMPARISON ANALYSIS COMPLETED!")
        print("=" * 80)
        print(f"\nResults saved to: {comparator.results_dir}")
        
    except Exception as e:
        print(f"\n Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
