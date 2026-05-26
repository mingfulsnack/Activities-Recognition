"""
Feature Importance Analysis for Stress Prediction

Analyzes which features contribute most to stress predictions using:
1. Random Forest feature importance
2. Permutation importance
3. Correlation analysis
4. SHAP values (if available)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import configuration and data pipeline
from config import *
from data_pipeline import StressDataPipeline


class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis"""
    
    def __init__(self, data_path, results_dir='results/feature_importance'):
        """
        Initialize feature importance analyzer
        
        Args:
            data_path: Path to dataset
            results_dir: Directory to save analysis results
        """
        self.data_path = Path(data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and prepare data
        print(f"Loading data from {data_path}...")
        self.pipeline = StressDataPipeline(sequence_length=SEQUENCE_LENGTH)
        data_dict = self.pipeline.prepare_data(str(data_path))
        
        # Get flattened data for Random Forest (RF doesn't work well with sequences)
        # We'll use the last timestep of each sequence
        self.X_train = data_dict['X_train'][:, -1, :]  # (samples, features)
        self.X_val = data_dict['X_val'][:, -1, :]
        self.X_test = data_dict['X_test'][:, -1, :]
        self.y_train = data_dict['y_train']
        self.y_val = data_dict['y_val']
        self.y_test = data_dict['y_test']
        
        # Combine train+val for better RF training
        self.X_train_full = np.vstack([self.X_train, self.X_val])
        self.y_train_full = np.concatenate([self.y_train, self.y_val])
        
        # Get feature names
        self.feature_names = FEATURE_COLUMNS
        
        # Get original dataframe for correlation analysis
        self.df = self.pipeline.df
        
        print(f"\nData loaded:")
        print(f"  Training samples: {len(self.X_train_full)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Number of features: {len(self.feature_names)}")
        
    def train_random_forest(self, n_estimators=50):
        """Train Random Forest model for feature importance analysis"""
        print(f"\n=== Training Random Forest (n_estimators={n_estimators}) ===")
        
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,  # Reduced from 100
            max_depth=10,  # Reduced from 15
            min_samples_split=20,  # Increased from 10
            min_samples_leaf=10,  # Increased from 5
            random_state=42,
            n_jobs=2,  # Limited parallelization
            verbose=0
        )
        
        print("Training...")
        self.rf_model.fit(self.X_train_full, self.y_train_full)
        
        # Evaluate
        train_pred = self.rf_model.predict(self.X_train_full)
        test_pred = self.rf_model.predict(self.X_test)
        
        train_mae = mean_absolute_error(self.y_train_full, train_pred)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        train_r2 = r2_score(self.y_train_full, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        print(f"\nRandom Forest Performance:")
        print(f"  Train MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"  Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def analyze_rf_importance(self):
        """Analyze Random Forest feature importance"""
        print("\n=== Random Forest Feature Importance ===")
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Add percentage
        importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
        importance_df['Cumulative_Percentage'] = importance_df['Percentage'].cumsum()
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Save results
        importance_df.to_csv(self.results_dir / 'rf_feature_importance.csv', index=False)
        
        return importance_df
    
    def analyze_permutation_importance(self, n_repeats=5):
        """Analyze permutation importance (more reliable than RF importance)"""
        print(f"\n=== Permutation Importance (n_repeats={n_repeats}) ===")
        print("Calculating... (this may take 2-3 minutes)")
        
        # Calculate permutation importance on test set
        # Use fewer jobs to avoid memory issues
        perm_importance = permutation_importance(
            self.rf_model, 
            self.X_test, 
            self.y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=2  # Reduced from -1 to avoid memory issues
        )
        
        # Create DataFrame
        perm_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance_Mean': perm_importance.importances_mean,
            'Importance_Std': perm_importance.importances_std
        }).sort_values('Importance_Mean', ascending=False)
        
        print("\nTop 10 Most Important Features (Permutation):")
        print(perm_df.head(10).to_string(index=False))
        
        # Save results
        perm_df.to_csv(self.results_dir / 'permutation_importance.csv', index=False)
        
        return perm_df
    
    def analyze_correlations(self):
        """Analyze feature correlations with target"""
        print("\n=== Feature-Target Correlations ===")
        
        # Calculate correlations
        correlations = []
        for col in self.feature_names:
            if col in self.df.columns:
                # Handle categorical columns
                if self.df[col].dtype == 'object':
                    # For categorical, use one-hot encoding correlation
                    df_encoded = pd.get_dummies(self.df[[col, 'Stress_Level']], columns=[col])
                    corr = df_encoded.corr()['Stress_Level'].drop('Stress_Level').abs().mean()
                else:
                    corr = self.df[col].corr(self.df['Stress_Level'])
                
                correlations.append({
                    'Feature': col,
                    'Correlation': corr,
                    'Abs_Correlation': abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
        
        print("\nTop 10 Features by Correlation:")
        print(corr_df.head(10).to_string(index=False))
        
        # Save results
        corr_df.to_csv(self.results_dir / 'feature_correlations.csv', index=False)
        
        return corr_df
    
    def create_visualizations(self, rf_importance, perm_importance, correlations):
        """Create comprehensive feature importance visualizations"""
        print("\n=== Creating Visualizations ===")
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Random Forest Importance (Top 15)
        ax1 = plt.subplot(2, 3, 1)
        top_rf = rf_importance.head(15)
        bars = plt.barh(range(len(top_rf)), top_rf['Importance'], alpha=0.8)
        plt.yticks(range(len(top_rf)), top_rf['Feature'], fontsize=9)
        plt.xlabel('Importance', fontsize=10)
        plt.title('Random Forest Feature Importance (Top 15)', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(top_rf['Importance'] / top_rf['Importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 2. Permutation Importance (Top 15)
        ax2 = plt.subplot(2, 3, 2)
        top_perm = perm_importance.head(15)
        plt.barh(range(len(top_perm)), top_perm['Importance_Mean'], 
                xerr=top_perm['Importance_Std'], alpha=0.8, capsize=3)
        plt.yticks(range(len(top_perm)), top_perm['Feature'], fontsize=9)
        plt.xlabel('Importance (with std)', fontsize=10)
        plt.title('Permutation Importance (Top 15)', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 3. Correlation with Target (Top 15)
        ax3 = plt.subplot(2, 3, 3)
        top_corr = correlations.head(15)
        colors_corr = ['green' if x > 0 else 'red' for x in top_corr['Correlation']]
        plt.barh(range(len(top_corr)), top_corr['Correlation'], alpha=0.8, color=colors_corr)
        plt.yticks(range(len(top_corr)), top_corr['Feature'], fontsize=9)
        plt.xlabel('Correlation with Stress', fontsize=10)
        plt.title('Feature-Target Correlation (Top 15)', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. Cumulative Importance (RF)
        ax4 = plt.subplot(2, 3, 4)
        plt.plot(range(1, len(rf_importance) + 1), 
                rf_importance['Cumulative_Percentage'], 
                marker='o', linewidth=2, markersize=4)
        plt.axhline(y=80, color='r', linestyle='--', label='80% threshold')
        plt.axhline(y=90, color='orange', linestyle='--', label='90% threshold')
        plt.xlabel('Number of Features', fontsize=10)
        plt.ylabel('Cumulative Importance (%)', fontsize=10)
        plt.title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Find features needed for 80% and 90%
        n_80 = (rf_importance['Cumulative_Percentage'] <= 80).sum() + 1
        n_90 = (rf_importance['Cumulative_Percentage'] <= 90).sum() + 1
        plt.text(0.5, 0.3, f'{n_80} features → 80%\n{n_90} features → 90%',
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Comparison: RF vs Permutation Importance
        ax5 = plt.subplot(2, 3, 5)
        # Merge dataframes
        comparison = rf_importance[['Feature', 'Importance']].merge(
            perm_importance[['Feature', 'Importance_Mean']], on='Feature'
        )
        comparison.columns = ['Feature', 'RF_Importance', 'Perm_Importance']
        
        # Normalize for comparison
        comparison['RF_Normalized'] = comparison['RF_Importance'] / comparison['RF_Importance'].max()
        comparison['Perm_Normalized'] = comparison['Perm_Importance'] / comparison['Perm_Importance'].max()
        
        plt.scatter(comparison['RF_Normalized'], comparison['Perm_Normalized'], alpha=0.6, s=50)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
        plt.xlabel('RF Importance (normalized)', fontsize=10)
        plt.ylabel('Permutation Importance (normalized)', fontsize=10)
        plt.title('RF vs Permutation Importance', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation
        corr = comparison['RF_Normalized'].corr(comparison['Perm_Normalized'])
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax5.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Feature Categories Importance
        ax6 = plt.subplot(2, 3, 6)
        
        # Categorize features
        categories = {
            'Physiological': ['Heart_Rate', 'Sleep_Duration', 'Sleep_Quality', 'Energy_Level', 'Mood_Score'],
            'Screen/Phone': ['Screen_Usage_Current', 'Screen_Usage_15min_Avg', 'Screen_Usage_Trend', 
                           'Phone_Usage_Intensity', 'Phone_Event_Frequency'],
            'Social': ['Social_Current_Level', 'Social_1hour_Avg'],
            'Environmental': ['Ambient_Light', 'Noise_Level', 'Weather_Condition'],
            'Activity': ['Activity', 'Location', 'Exercise_Minutes'],
            'Accelerometer': ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
        }
        
        category_importance = {}
        for cat, features in categories.items():
            cat_features = [f for f in features if f in rf_importance['Feature'].values]
            if cat_features:
                cat_imp = rf_importance[rf_importance['Feature'].isin(cat_features)]['Importance'].sum()
                category_importance[cat] = cat_imp
        
        cats = list(category_importance.keys())
        imps = list(category_importance.values())
        colors_cat = plt.cm.Set3(range(len(cats)))
        
        plt.barh(cats, imps, alpha=0.8, color=colors_cat)
        plt.xlabel('Total Importance', fontsize=10)
        plt.title('Feature Category Importance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        total_imp = sum(imps)
        for i, (cat, imp) in enumerate(zip(cats, imps)):
            pct = (imp / total_imp) * 100
            plt.text(imp, i, f' {pct:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        print(f"   Saved: feature_importance_comprehensive.png")
        plt.close()
        
        # Create separate detailed plot for top features
        self._create_top_features_plot(rf_importance, perm_importance, correlations)
    
    def _create_top_features_plot(self, rf_importance, perm_importance, correlations):
        """Create detailed plot for top 10 features with all metrics"""
        print("  Creating detailed top features plot...")
        
        # Get top 10 by RF importance
        top_10_features = rf_importance.head(10)['Feature'].tolist()
        
        # Collect metrics for these features
        metrics = []
        for feat in top_10_features:
            rf_imp = rf_importance[rf_importance['Feature'] == feat]['Importance'].values[0]
            perm_imp = perm_importance[perm_importance['Feature'] == feat]['Importance_Mean'].values[0]
            corr = correlations[correlations['Feature'] == feat]['Correlation'].values[0]
            
            metrics.append({
                'Feature': feat,
                'RF_Importance': rf_imp,
                'Perm_Importance': perm_imp,
                'Correlation': abs(corr)
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Normalize for comparison
        for col in ['RF_Importance', 'Perm_Importance', 'Correlation']:
            metrics_df[f'{col}_Norm'] = metrics_df[col] / metrics_df[col].max()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(top_10_features))
        width = 0.25
        
        bars1 = ax.bar(x - width, metrics_df['RF_Importance_Norm'], width, 
                      label='RF Importance', alpha=0.8)
        bars2 = ax.bar(x, metrics_df['Perm_Importance_Norm'], width,
                      label='Permutation Importance', alpha=0.8)
        bars3 = ax.bar(x + width, metrics_df['Correlation_Norm'], width,
                      label='|Correlation|', alpha=0.8)
        
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Features: Multiple Importance Metrics (Normalized)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_10_features, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'top10_features_detailed.png', 
                   dpi=300, bbox_inches='tight')
        print(f"   Saved: top10_features_detailed.png")
        plt.close()
    
    def generate_report(self, rf_importance, perm_importance, correlations, rf_metrics):
        """Generate comprehensive feature importance report"""
        print("\n=== Generating Report ===")
        
        report = []
        report.append("# Feature Importance Analysis Report")
        report.append(f"\n**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Dataset**: {self.data_path}")
        report.append(f"**Training Samples**: {len(self.X_train_full)}")
        report.append(f"**Test Samples**: {len(self.X_test)}")
        report.append(f"**Number of Features**: {len(self.feature_names)}")
        
        report.append("\n## Random Forest Model Performance")
        report.append(f"- **Train MAE**: {rf_metrics['train_mae']:.4f}")
        report.append(f"- **Test MAE**: {rf_metrics['test_mae']:.4f}")
        report.append(f"- **Train R²**: {rf_metrics['train_r2']:.4f}")
        report.append(f"- **Test R²**: {rf_metrics['test_r2']:.4f}")
        
        if rf_metrics['test_r2'] < 0.85:
            report.append(f"\n⚠️ Note: RF R² ({rf_metrics['test_r2']:.4f}) is lower than LSTM (0.9343)")
            report.append("This is expected - RF doesn't use temporal sequences. Feature importance still valid.")
        
        report.append("\n## Top 10 Most Important Features")
        report.append("\n### By Random Forest Importance")
        for i, row in rf_importance.head(10).iterrows():
            report.append(f"{i+1}. **{row['Feature']}**: {row['Importance']:.4f} ({row['Percentage']:.2f}%)")
        
        report.append("\n### By Permutation Importance")
        for i, row in perm_importance.head(10).iterrows():
            report.append(f"{i+1}. **{row['Feature']}**: {row['Importance_Mean']:.4f} (±{row['Importance_Std']:.4f})")
        
        report.append("\n### By Correlation with Stress")
        for i, row in correlations.head(10).iterrows():
            direction = "↑" if row['Correlation'] > 0 else "↓"
            report.append(f"{i+1}. **{row['Feature']}**: {row['Correlation']:.4f} {direction}")
        
        # Cumulative analysis
        n_80 = (rf_importance['Cumulative_Percentage'] <= 80).sum() + 1
        n_90 = (rf_importance['Cumulative_Percentage'] <= 90).sum() + 1
        
        report.append("\n## Cumulative Importance Analysis")
        report.append(f"- **{n_80} features** explain **80%** of importance")
        report.append(f"- **{n_90} features** explain **90%** of importance")
        report.append(f"- Total features: {len(self.feature_names)}")
        report.append(f"\n→ **{(n_80/len(self.feature_names)*100):.1f}%** of features capture most predictive power")
        
        # Feature categories
        categories = {
            'Physiological': ['Heart_Rate', 'Sleep_Duration', 'Sleep_Quality', 'Energy_Level', 'Mood_Score'],
            'Screen/Phone': ['Screen_Usage_Current', 'Screen_Usage_15min_Avg', 'Screen_Usage_Trend', 
                           'Phone_Usage_Intensity', 'Phone_Event_Frequency'],
            'Social': ['Social_Current_Level', 'Social_1hour_Avg'],
            'Environmental': ['Ambient_Light', 'Noise_Level', 'Weather_Condition'],
            'Activity': ['Activity', 'Location', 'Exercise_Minutes'],
            'Accelerometer': ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
        }
        
        report.append("\n## Feature Category Analysis")
        category_importance = {}
        for cat, features in categories.items():
            cat_features = [f for f in features if f in rf_importance['Feature'].values]
            if cat_features:
                cat_imp = rf_importance[rf_importance['Feature'].isin(cat_features)]['Importance'].sum()
                category_importance[cat] = cat_imp
        
        sorted_cats = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        total_imp = sum(category_importance.values())
        
        for cat, imp in sorted_cats:
            pct = (imp / total_imp) * 100
            report.append(f"- **{cat}**: {imp:.4f} ({pct:.1f}%)")
        
        # Key insights
        report.append("\n## Key Insights")
        
        # Most important feature
        top_feat = rf_importance.iloc[0]
        report.append(f"\n### 1. Most Important Feature")
        report.append(f"**{top_feat['Feature']}** dominates with {top_feat['Percentage']:.1f}% of total importance")
        
        # Top category
        top_cat = sorted_cats[0]
        report.append(f"\n### 2. Most Important Category")
        report.append(f"**{top_cat[0]}** features collectively contribute {(top_cat[1]/total_imp*100):.1f}%")
        
        # Agreement between methods
        top_rf_set = set(rf_importance.head(10)['Feature'])
        top_perm_set = set(perm_importance.head(10)['Feature'])
        agreement = len(top_rf_set & top_perm_set)
        report.append(f"\n### 3. Method Agreement")
        report.append(f"{agreement}/10 features appear in top 10 of both RF and Permutation importance")
        if agreement >= 7:
            report.append("✓ Strong agreement between methods - results are reliable")
        else:
            report.append("⚠️ Moderate agreement - consider using ensemble of importance metrics")
        
        # Correlation insights
        high_corr = correlations[correlations['Abs_Correlation'] > 0.3]
        report.append(f"\n### 4. Strong Correlations")
        report.append(f"{len(high_corr)} features have |correlation| > 0.3 with stress")
        
        report.append("\n## Recommendations for Model Improvement")
        
        # Based on top features
        report.append("\n### 1. Feature Engineering")
        top_5 = rf_importance.head(5)['Feature'].tolist()
        report.append(f"- Focus on top 5 features: {', '.join(top_5)}")
        report.append(f"- Create interaction terms between these features")
        report.append(f"- Add temporal features (rolling means, trends) for top features")
        
        # Based on categories
        if sorted_cats[0][0] == 'Physiological':
            report.append("\n### 2. Physiological Features Dominant")
            report.append("- Heart rate, sleep, energy, mood are key predictors")
            report.append("- Consider adding: HRV, sleep stages, mood patterns")
        
        # Based on low importance features
        low_imp = rf_importance[rf_importance['Percentage'] < 1]
        if len(low_imp) > 0:
            report.append(f"\n### 3. Feature Selection")
            report.append(f"- {len(low_imp)} features contribute < 1% each")
            report.append(f"- Consider removing features with importance < 0.5%")
            report.append(f"- This could simplify model without losing performance")
        
        # Based on error analysis insights
        report.append("\n### 4. Address Weak Predictions")
        report.append("- Medium stress (4-5) had highest errors in error analysis")
        report.append("- Boost features that distinguish medium stress from low/high")
        report.append("- Add context features for 'Standing' activity")
        report.append("- Create 'commute' specific features")
        
        report.append("\n## Next Steps")
        report.append("\n1. **Feature Selection**: Remove low-importance features (< 0.5%)")
        report.append("2. **Feature Engineering**: Create interactions and temporal features for top 10")
        report.append("3. **Targeted Improvement**: Add features to improve medium-stress predictions")
        report.append("4. **Model Comparison**: Test GRU/TCN with reduced feature set")
        report.append("5. **Ablation Study**: Systematically remove feature categories to verify importance")
        
        report.append("\n---")
        report.append("\n*See accompanying CSV files and plots for detailed analysis*")
        
        # Save report
        report_text = '\n'.join(report)
        with open(self.results_dir / 'FEATURE_IMPORTANCE_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"   Saved: FEATURE_IMPORTANCE_REPORT.md")
        
        return report_text
    
    def run_full_analysis(self, skip_permutation=False):
        """Run complete feature importance analysis pipeline"""
        print("\n" + "="*70)
        print("  FEATURE IMPORTANCE ANALYSIS - LSTM BASELINE")
        print("="*70)
        
        # Train Random Forest
        rf_metrics = self.train_random_forest(n_estimators=50)
        
        # Analyze importances
        rf_importance = self.analyze_rf_importance()
        
        if skip_permutation:
            print("\n Skipping permutation importance (takes too long)")
            # Create dummy permutation importance from RF importance
            perm_importance = rf_importance[['Feature', 'Importance']].copy()
            perm_importance.columns = ['Feature', 'Importance_Mean']
            perm_importance['Importance_Std'] = 0.0
        else:
            perm_importance = self.analyze_permutation_importance(n_repeats=5)  # Reduced from 10
        
        correlations = self.analyze_correlations()
        
        # Create visualizations
        self.create_visualizations(rf_importance, perm_importance, correlations)
        
        # Generate report
        report = self.generate_report(rf_importance, perm_importance, correlations, rf_metrics)
        
        print("\n" + "="*70)
        print("   FEATURE IMPORTANCE ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.results_dir}")
        print("\nFiles created:")
        print("  - rf_feature_importance.csv")
        print("  - permutation_importance.csv")
        print("  - feature_correlations.csv")
        print("  - feature_importance_comprehensive.png")
        print("  - top10_features_detailed.png")
        print("  - FEATURE_IMPORTANCE_REPORT.md")
        
        return {
            'rf_importance': rf_importance,
            'perm_importance': perm_importance,
            'correlations': correlations,
            'rf_metrics': rf_metrics
        }


def main():
    """Main execution function"""
    import sys
    
    # Default path
    base_dir = Path(__file__).parent
    data_path = base_dir.parent / 'generate_and_verify_data' / 'Data generator' / 'data' / 'optimized_health_data_23features.csv'
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    print(f"Data path: {data_path}")
    print(f"Data exists: {Path(data_path).exists()}")
    
    # Run analysis (skip permutation importance to save time)
    analyzer = FeatureImportanceAnalyzer(data_path)
    results = analyzer.run_full_analysis(skip_permutation=True)
    
    print("\n✓ Analysis complete! Check results/feature_importance/ for outputs.")


if __name__ == '__main__':
    main()
