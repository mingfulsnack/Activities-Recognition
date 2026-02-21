"""
Error Analysis for 13-Feature LSTM Model
==========================================
Comprehensive error analysis adapted for 13-feature model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from tensorflow import keras


class ErrorAnalyzer13Features:
    """Error analysis specifically for 13-feature model"""
    
    def __init__(self, model_path, data_path, results_dir='results/error_analysis_13features'):
        """
        Initialize error analyzer for 13-feature model
        
        Args:
            model_path: Path to trained 13-feature model
            data_path: Path to 13-feature dataset
            results_dir: Directory to save analysis results
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"📂 Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        
        # Load preprocessors
        print("📂 Loading preprocessors...")
        model_dir = self.model_path.parent
        with open(model_dir / 'scaler_13features.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(model_dir / 'label_encoder_13features_Activity.pkl', 'rb') as f:
            self.activity_encoder = pickle.load(f)
        with open(model_dir / 'label_encoder_13features_Location.pkl', 'rb') as f:
            self.location_encoder = pickle.load(f)
        
        # Load and prepare data
        print(f"📂 Loading data from {data_path}...")
        self.prepare_data()
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        self.metrics = {
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'MSE': mean_squared_error(self.y_true, self.y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'R2': r2_score(self.y_true, self.y_pred)
        }
        
        print(f"\n✅ Test Metrics (13-Feature Model):")
        print(f"  MAE:  {self.metrics['MAE']:.4f}")
        print(f"  RMSE: {self.metrics['RMSE']:.4f}")
        print(f"  R²:   {self.metrics['R2']:.4f}")
    
    def prepare_data(self):
        """Load and prepare data using same pipeline as training"""
        # Load raw data
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df):,} samples")
        
        # Store full dataframe for contextual analysis
        self.df_full = df.copy()
        
        # Separate features and target
        X = df.drop('Stress_Level', axis=1)
        y = df['Stress_Level'].values
        
        # Split (same as training: 70/15/15)
        train_size = int(0.70 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X.iloc[:train_size]
        X_val = X.iloc[train_size:train_size+val_size]
        X_test = X.iloc[train_size+val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size+val_size]
        y_test = y[train_size+val_size:]
        
        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        
        # Encode categorical features (Activity, Location)
        X_test_encoded = X_test.copy()
        X_test_encoded['Activity'] = self.activity_encoder.transform(X_test['Activity'])
        X_test_encoded['Location'] = self.location_encoder.transform(X_test['Location'])
        
        # Normalize (using pre-fitted scaler)
        X_test_scaled = self.scaler.transform(X_test_encoded)
        
        # Create sequences (60 timesteps)
        sequence_length = 60
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test, sequence_length)
        
        print(f"  Sequences created: {X_test_seq.shape}")
        
        # Make predictions
        print("🔮 Making predictions...")
        self.y_true = y_test_seq
        self.y_pred = self.model.predict(X_test_seq, verbose=0).flatten()
        
        # Calculate errors
        self.errors = self.y_true - self.y_pred
        self.abs_errors = np.abs(self.errors)
        
        # Store test dataframe for analysis (after sequence creation)
        test_start_idx = train_size + val_size
        self.df_test = self.df_full.iloc[test_start_idx + sequence_length - 1:].reset_index(drop=True)
        if len(self.df_test) > len(self.y_pred):
            self.df_test = self.df_test.iloc[:len(self.y_pred)]
    
    def create_sequences(self, X, y, sequence_length=60):
        """Create sequences from data"""
        X_seq, y_seq = [], []
        for i in range(sequence_length - 1, len(X)):
            X_seq.append(X[i - sequence_length + 1:i + 1])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def analyze_error_distribution(self):
        """Analyze overall error distribution"""
        print("\n=== Error Distribution Analysis ===")
        
        stats_dict = {
            'Mean Error': np.mean(self.errors),
            'Std Error': np.std(self.errors),
            'Mean Absolute Error': np.mean(self.abs_errors),
            'Median Absolute Error': np.median(self.abs_errors),
            'Max Error': np.max(self.abs_errors),
            'Min Error': np.min(self.abs_errors),
            '90th Percentile': np.percentile(self.abs_errors, 90),
            '95th Percentile': np.percentile(self.abs_errors, 95),
            '99th Percentile': np.percentile(self.abs_errors, 99)
        }
        
        for key, value in stats_dict.items():
            print(f"  {key}: {value:.4f}")
        
        # Normality test
        _, p_value = stats.normaltest(self.errors)
        print(f"\n  Normality test p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("  → Errors appear normally distributed ✓")
        else:
            print("  → Errors may not be normally distributed ⚠")
        
        # Save
        pd.DataFrame([stats_dict]).to_csv(self.results_dir / 'error_statistics.csv', index=False)
        return stats_dict
    
    def analyze_by_stress_level(self):
        """Analyze errors by stress level"""
        print("\n=== Error Analysis by Stress Level ===")
        
        bins = [0, 3, 5, 7, 10]
        labels = ['Low (1-3)', 'Medium (4-5)', 'High (6-7)', 'Very High (8-9)']
        stress_bins = pd.cut(self.y_true, bins=bins, labels=labels, include_lowest=True)
        
        bin_analysis = []
        for label in labels:
            mask = stress_bins == label
            if mask.sum() > 0:
                bin_data = {
                    'Stress Level': label,
                    'Count': mask.sum(),
                    'Percentage': mask.sum() / len(self.y_true) * 100,
                    'MAE': np.mean(self.abs_errors[mask]),
                    'RMSE': np.sqrt(np.mean(self.errors[mask]**2)),
                    'Mean Actual': np.mean(self.y_true[mask]),
                    'Mean Predicted': np.mean(self.y_pred[mask]),
                    'Bias': np.mean(self.errors[mask])
                }
                bin_analysis.append(bin_data)
                
                print(f"\n  {label}:")
                print(f"    Count: {bin_data['Count']} ({bin_data['Percentage']:.1f}%)")
                print(f"    MAE: {bin_data['MAE']:.4f}")
                print(f"    Bias: {bin_data['Bias']:.4f}")
        
        bin_df = pd.DataFrame(bin_analysis)
        bin_df.to_csv(self.results_dir / 'error_by_stress_level.csv', index=False)
        return bin_df
    
    def analyze_by_activity(self):
        """Analyze errors by activity"""
        print("\n=== Error Analysis by Activity ===")
        
        df_analysis = self.df_test.copy()
        df_analysis['predicted_stress'] = self.y_pred[:len(df_analysis)]
        df_analysis['abs_error'] = self.abs_errors[:len(df_analysis)]
        
        activity_analysis = df_analysis.groupby('Activity').agg({
            'abs_error': ['count', 'mean', 'std'],
            'Stress_Level': 'mean',
            'predicted_stress': 'mean'
        }).round(4)
        
        activity_analysis.columns = ['Count', 'MAE', 'Std', 'Actual_Mean', 'Pred_Mean']
        activity_analysis = activity_analysis.sort_values('MAE', ascending=False)
        
        print("\n  Top activities by error:")
        print(activity_analysis.head(10).to_string())
        
        activity_analysis.to_csv(self.results_dir / 'error_by_activity.csv')
        return activity_analysis
    
    def analyze_by_time(self):
        """Analyze errors by time of day"""
        print("\n=== Error Analysis by Time ===")
        
        df_analysis = self.df_test.copy()
        df_analysis['abs_error'] = self.abs_errors[:len(df_analysis)]
        
        # Group by hour
        time_analysis = df_analysis.groupby('Hour').agg({
            'abs_error': ['count', 'mean'],
            'Stress_Level': 'mean'
        }).round(4)
        
        time_analysis.columns = ['Count', 'MAE', 'Actual_Mean']
        
        print("\n  Error by hour:")
        print(time_analysis.to_string())
        
        time_analysis.to_csv(self.results_dir / 'error_by_time.csv')
        return time_analysis
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n=== Creating Visualizations ===")
        
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Predictions vs Actual
        ax1 = plt.subplot(2, 3, 1)
        plt.scatter(self.y_true, self.y_pred, alpha=0.3, s=10)
        plt.plot([self.y_true.min(), self.y_true.max()], 
                [self.y_true.min(), self.y_true.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Stress', fontsize=10)
        plt.ylabel('Predicted Stress', fontsize=10)
        plt.title('13-Feature Model: Predictions vs Actual', fontsize=12, fontweight='bold')
        plt.text(0.05, 0.95, f'R² = {self.metrics["R2"]:.4f}', 
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.grid(True, alpha=0.3)
        
        # 2. Error Distribution
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(self.errors, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.axvline(x=np.mean(self.errors), color='g', linestyle='--', linewidth=2)
        plt.xlabel('Error', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Error Distribution', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 3. Absolute Error Distribution
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(self.abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.axvline(x=np.mean(self.abs_errors), color='r', linestyle='--', linewidth=2,
                   label=f'MAE = {self.metrics["MAE"]:.3f}')
        plt.xlabel('Absolute Error', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Error by Stress Level
        ax4 = plt.subplot(2, 3, 4)
        bins = [0, 3, 5, 7, 10]
        labels = ['Low\n(1-3)', 'Medium\n(4-5)', 'High\n(6-7)', 'Very High\n(8-9)']
        stress_bins = pd.cut(self.y_true, bins=bins, labels=labels, include_lowest=True)
        
        bin_errors = [self.abs_errors[stress_bins == label].mean() for label in labels]
        bin_counts = [np.sum(stress_bins == label) for label in labels]
        
        bars = plt.bar(labels, bin_errors, edgecolor='black', alpha=0.7)
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        plt.ylabel('MAE', fontsize=10)
        plt.title('Error by Stress Level', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. Residual Plot
        ax5 = plt.subplot(2, 3, 5)
        plt.scatter(self.y_pred, self.errors, alpha=0.3, s=10)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Stress', fontsize=10)
        plt.ylabel('Residual', fontsize=10)
        plt.title('Residual Plot', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 6. Error by Activity
        ax6 = plt.subplot(2, 3, 6)
        df_analysis = self.df_test.copy()
        df_analysis['abs_error'] = self.abs_errors[:len(df_analysis)]
        
        activity_errors = df_analysis.groupby('Activity')['abs_error'].agg(['mean', 'count'])
        activity_errors = activity_errors[activity_errors['count'] >= 30]
        activity_errors = activity_errors.sort_values('mean', ascending=False).head(8)
        
        plt.barh(range(len(activity_errors)), activity_errors['mean'], edgecolor='black', alpha=0.7)
        plt.yticks(range(len(activity_errors)), activity_errors.index, fontsize=8)
        plt.xlabel('MAE', fontsize=10)
        plt.title('Error by Activity (Top 8)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'error_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: error_analysis_comprehensive.png")
        plt.close()
        
        # Q-Q plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        stats.probplot(self.errors, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'qq_plot.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: qq_plot.png")
        plt.close()
    
    def generate_report(self):
        """Generate markdown report"""
        print("\n=== Generating Report ===")
        
        report = []
        report.append("# Error Analysis Report - 13-Feature LSTM Model")
        report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nModel: 13-feature LSTM (Core + High-Importance)")
        report.append(f"Dataset: {self.data_path}")
        report.append(f"Test Samples: {len(self.y_true):,}")
        
        report.append("\n## Overall Performance")
        report.append(f"- **MAE**: {self.metrics['MAE']:.4f}")
        report.append(f"- **RMSE**: {self.metrics['RMSE']:.4f}")
        report.append(f"- **R²**: {self.metrics['R2']:.4f}")
        
        report.append("\n## Error Statistics")
        report.append(f"- Mean Error: {np.mean(self.errors):.4f}")
        report.append(f"- Std Error: {np.std(self.errors):.4f}")
        report.append(f"- Median Absolute Error: {np.median(self.abs_errors):.4f}")
        report.append(f"- 95th Percentile: {np.percentile(self.abs_errors, 95):.4f}")
        
        # Save
        with open(self.results_dir / 'ERROR_ANALYSIS_REPORT.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"  ✅ Saved: ERROR_ANALYSIS_REPORT.md")
    
    def run_full_analysis(self):
        """Run complete error analysis"""
        print("\n" + "="*70)
        print("  13-FEATURE LSTM MODEL - ERROR ANALYSIS")
        print("="*70)
        
        self.analyze_error_distribution()
        self.analyze_by_stress_level()
        self.analyze_by_activity()
        self.analyze_by_time()
        self.create_visualizations()
        self.generate_report()
        
        print("\n" + "="*70)
        print("  ✅ ERROR ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults: {self.results_dir}/")


def main():
    """Main execution"""
    base_dir = Path(__file__).parent
    model_path = base_dir.parent / 'models' / 'lstm_13features_best.keras'
    data_path = base_dir.parent / 'data' / 'optimized_health_data_13features.csv'
    
    analyzer = ErrorAnalyzer13Features(model_path, data_path)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
