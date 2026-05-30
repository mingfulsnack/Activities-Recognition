"""
Error Analysis for 13-Feature TUNED LSTM Model
- Reuses ErrorAnalyzer13Features from error_analysis_13features.py
- Points to tuned model + tuned preprocessors
- Adds baseline vs tuned comparison
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from tensorflow import keras
from datetime import datetime


class ErrorAnalyzerTuned:
    """Error analysis for tuned 13-feature LSTM model with baseline comparison."""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / 'results' / 'error_analysis_13features_tuned'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        data_path = self.base_dir / 'data' / 'optimized_health_data_13features.csv'
        
        # Load data & prepare (shared pipeline)
        print("[1] Loading data...")
        self.df = pd.read_csv(data_path)
        print(f"    Loaded {len(self.df):,} samples")
        
        X = self.df.drop('Stress_Level', axis=1)
        y = self.df['Stress_Level'].values
        
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        self.X_test_raw = X.iloc[val_end:]
        self.y_test_raw = y[val_end:]
        self.df_test = self.df.iloc[val_end:].copy()
        
        print(f"    Test set: {len(self.X_test_raw):,} samples")
        
        # --- Load BASELINE model ---
        print("\n[2] Loading baseline model...")
        self.baseline = self._load_model_and_predict(
            model_path=self.base_dir / 'models' / 'lstm_13features_best.keras',
            scaler_path=self.base_dir / 'models' / 'scaler_13features.pkl',
            activity_enc_path=self.base_dir / 'models' / 'label_encoder_13features_Activity.pkl',
            location_enc_path=self.base_dir / 'models' / 'label_encoder_13features_Location.pkl',
            label='Baseline'
        )
        
        # --- Load TUNED model ---
        print("\n[3] Loading tuned model...")
        self.tuned = self._load_model_and_predict(
            model_path=self.base_dir / 'models' / 'lstm_13features_tuned.keras',
            scaler_path=self.base_dir / 'models' / 'scaler_13features_tuned.pkl',
            activity_enc_path=self.base_dir / 'models' / 'label_encoder_13features_tuned_Activity.pkl',
            location_enc_path=self.base_dir / 'models' / 'label_encoder_13features_tuned_Location.pkl',
            label='Tuned'
        )
    
    # DA: ERROR_ANALYSIS_LOAD_PREDICT
    # Loads a saved model/preprocessors and predicts on the shared test split.
    def _load_model_and_predict(self, model_path, scaler_path, activity_enc_path, location_enc_path, label):
        """Load model, preprocess test data, make predictions."""
        model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(activity_enc_path, 'rb') as f:
            act_enc = pickle.load(f)
        with open(location_enc_path, 'rb') as f:
            loc_enc = pickle.load(f)
        
        # Encode
        X_test = self.X_test_raw.copy()
        X_test['Activity'] = act_enc.transform(X_test['Activity'].astype(str))
        X_test['Location'] = loc_enc.transform(X_test['Location'].astype(str))
        
        # Scale
        X_test_scaled = scaler.transform(X_test.values)
        
        # Sequences
        seq_len = 60
        X_seq, y_seq = [], []
        for i in range(len(X_test_scaled) - seq_len):
            X_seq.append(X_test_scaled[i:i+seq_len])
            y_seq.append(self.y_test_raw[i+seq_len])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Predict
        y_pred = model.predict(X_seq, verbose=0).flatten()
        
        errors = y_seq - y_pred
        abs_errors = np.abs(errors)
        
        metrics = {
            'MAE': mean_absolute_error(y_seq, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_seq, y_pred)),
            'R2': r2_score(y_seq, y_pred),
            'MSE': mean_squared_error(y_seq, y_pred)
        }
        
        print(f"    {label} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")
        
        # df for analysis (aligned with sequences)
        df_aligned = self.df_test.iloc[seq_len:].reset_index(drop=True)
        if len(df_aligned) > len(y_pred):
            df_aligned = df_aligned.iloc[:len(y_pred)]
        
        return {
            'model': model, 'y_true': y_seq, 'y_pred': y_pred,
            'errors': errors, 'abs_errors': abs_errors,
            'metrics': metrics, 'df': df_aligned, 'label': label
        }
    
    # DA: ERROR_ANALYSIS_OVERALL
    # Compares baseline vs tuned model on MAE/RMSE/R2.
    def compare_overall(self):
        """Compare overall metrics."""
        print("\n" + "=" * 60)
        print("  OVERALL COMPARISON: BASELINE vs TUNED")
        print("=" * 60)
        
        fmt = "  {:<8} {:>10} {:>10} {:>12}"
        print(fmt.format('Metric', 'Baseline', 'Tuned', 'Change'))
        print(f"  {'-'*42}")
        
        comparison = {}
        for m in ['MAE', 'RMSE', 'R2']:
            b = self.baseline['metrics'][m]
            t = self.tuned['metrics'][m]
            change = t - b
            pct = (change / b) * 100
            
            if m == 'R2':
                better = '[+]' if change > 0 else '[-]'
            else:
                better = '[+]' if change < 0 else '[-]'
            
            print(fmt.format(m, f"{b:.4f}", f"{t:.4f}", f"{change:+.4f} ({pct:+.1f}%) {better}"))
            comparison[m] = {'baseline': b, 'tuned': t, 'change': change, 'pct_change': pct}
        
        return comparison
    
    def compare_by_stress_level(self):
        """Compare errors by stress level bins."""
        print("\n" + "=" * 60)
        print("  ERROR BY STRESS LEVEL: BASELINE vs TUNED")
        print("=" * 60)
        
        bins = [0, 3, 5, 7, 10]
        labels = ['Low (1-3)', 'Medium (4-5)', 'High (6-7)', 'Very High (8-9)']
        
        results = []
        for label_name in labels:
            row = {'Stress_Level': label_name}
            
            for data_key, prefix in [('baseline', 'Base'), ('tuned', 'Tuned')]:
                d = self.__dict__[data_key]
                stress_bins = pd.cut(d['y_true'], bins=bins, labels=labels, include_lowest=True)
                mask = stress_bins == label_name
                
                if mask.sum() > 0:
                    row[f'{prefix}_Count'] = int(mask.sum())
                    row[f'{prefix}_MAE'] = float(np.mean(d['abs_errors'][mask]))
                    row[f'{prefix}_RMSE'] = float(np.sqrt(np.mean(d['errors'][mask]**2)))
                    row[f'{prefix}_Bias'] = float(np.mean(d['errors'][mask]))
            
            if 'Base_MAE' in row and 'Tuned_MAE' in row:
                row['MAE_Change'] = row['Tuned_MAE'] - row['Base_MAE']
                row['MAE_Change_Pct'] = (row['MAE_Change'] / row['Base_MAE']) * 100
            
            results.append(row)
        
        df_result = pd.DataFrame(results)
        
        print(f"\n  {'Level':<16} {'Base MAE':>10} {'Tuned MAE':>10} {'Change':>12}")
        print(f"  {'-'*50}")
        for _, row in df_result.iterrows():
            if 'MAE_Change' in row and not pd.isna(row.get('MAE_Change')):
                print(f"  {row['Stress_Level']:<16} {row['Base_MAE']:>10.4f} {row['Tuned_MAE']:>10.4f} {row['MAE_Change']:>+10.4f} ({row['MAE_Change_Pct']:>+.1f}%)")
        
        df_result.to_csv(self.results_dir / 'comparison_by_stress_level.csv', index=False)
        return df_result
    
    # DA: ERROR_ANALYSIS_BY_ACTIVITY
    # Breaks prediction error down by activity context.
    def compare_by_activity(self):
        """Compare errors by activity."""
        print("\n" + "=" * 60)
        print("  ERROR BY ACTIVITY: BASELINE vs TUNED")
        print("=" * 60)
        
        results = []
        for data_key, prefix in [('baseline', 'Base'), ('tuned', 'Tuned')]:
            d = self.__dict__[data_key]
            df_a = d['df'].copy()
            df_a['abs_error'] = d['abs_errors'][:len(df_a)]
            
            act_mae = df_a.groupby('Activity')['abs_error'].agg(['mean', 'count']).rename(
                columns={'mean': f'{prefix}_MAE', 'count': f'{prefix}_Count'}
            )
            results.append(act_mae)
        
        merged = results[0].join(results[1], how='outer')
        merged['MAE_Change'] = merged['Tuned_MAE'] - merged['Base_MAE']
        merged['MAE_Change_Pct'] = (merged['MAE_Change'] / merged['Base_MAE']) * 100
        merged = merged.sort_values('Base_MAE', ascending=False)
        
        print(f"\n  {'Activity':<16} {'Base MAE':>10} {'Tuned MAE':>10} {'Change':>12}")
        print(f"  {'-'*50}")
        for act, row in merged.iterrows():
            print(f"  {act:<16} {row['Base_MAE']:>10.4f} {row['Tuned_MAE']:>10.4f} {row['MAE_Change']:>+10.4f} ({row['MAE_Change_Pct']:>+.1f}%)")
        
        merged.to_csv(self.results_dir / 'comparison_by_activity.csv')
        return merged
    
    # DA: ERROR_ANALYSIS_BY_TIME
    # Breaks prediction error down by time-of-day context.
    def compare_by_time(self):
        """Compare errors by time of day."""
        print("\n" + "=" * 60)
        print("  ERROR BY TIME PERIOD: BASELINE vs TUNED")
        print("=" * 60)
        
        time_bins = [(0, 6, 'Night (0-5)'), (6, 12, 'Morning (6-11)'),
                     (12, 18, 'Afternoon (12-17)'), (18, 24, 'Evening (18-23)')]
        
        results = []
        for start, end, period_name in time_bins:
            row = {'Period': period_name}
            
            for data_key, prefix in [('baseline', 'Base'), ('tuned', 'Tuned')]:
                d = self.__dict__[data_key]
                df_t = d['df'].copy()
                df_t['abs_error'] = d['abs_errors'][:len(df_t)]
                
                mask = (df_t['Hour'] >= start) & (df_t['Hour'] < end)
                if mask.sum() > 0:
                    row[f'{prefix}_MAE'] = float(df_t.loc[mask, 'abs_error'].mean())
                    row[f'{prefix}_Count'] = int(mask.sum())
            
            if 'Base_MAE' in row and 'Tuned_MAE' in row:
                row['MAE_Change'] = row['Tuned_MAE'] - row['Base_MAE']
                row['MAE_Change_Pct'] = (row['MAE_Change'] / row['Base_MAE']) * 100
            
            results.append(row)
        
        df_result = pd.DataFrame(results)
        
        print(f"\n  {'Period':<20} {'Base MAE':>10} {'Tuned MAE':>10} {'Change':>12}")
        print(f"  {'-'*55}")
        for _, row in df_result.iterrows():
            if 'MAE_Change' in row:
                print(f"  {row['Period']:<20} {row['Base_MAE']:>10.4f} {row['Tuned_MAE']:>10.4f} {row['MAE_Change']:>+10.4f} ({row['MAE_Change_Pct']:>+.1f}%)")
        
        df_result.to_csv(self.results_dir / 'comparison_by_time.csv', index=False)
        return df_result
    
    def compare_error_distribution(self):
        """Compare error distribution statistics."""
        print("\n" + "=" * 60)
        print("  ERROR DISTRIBUTION COMPARISON")
        print("=" * 60)
        
        stats_list = []
        for data_key in ['baseline', 'tuned']:
            d = self.__dict__[data_key]
            stats_list.append({
                'Model': d['label'],
                'Mean_Error': np.mean(d['errors']),
                'Std_Error': np.std(d['errors']),
                'MAE': np.mean(d['abs_errors']),
                'Median_AE': np.median(d['abs_errors']),
                'P90': np.percentile(d['abs_errors'], 90),
                'P95': np.percentile(d['abs_errors'], 95),
                'P99': np.percentile(d['abs_errors'], 99),
                'Max_Error': np.max(d['abs_errors']),
                'Pct_Under_0.5': (d['abs_errors'] < 0.5).mean() * 100,
                'Pct_Under_1.0': (d['abs_errors'] < 1.0).mean() * 100,
                'Pct_Under_2.0': (d['abs_errors'] < 2.0).mean() * 100,
            })
        
        df_stats = pd.DataFrame(stats_list)
        
        print(f"\n  {'Statistic':<20} {'Baseline':>10} {'Tuned':>10}")
        print(f"  {'-'*42}")
        for col in df_stats.columns[1:]:
            b = df_stats.loc[0, col]
            t = df_stats.loc[1, col]
            print(f"  {col:<20} {b:>10.4f} {t:>10.4f}")
        
        df_stats.to_csv(self.results_dir / 'error_distribution_comparison.csv', index=False)
        return df_stats
    
    def create_comparison_plots(self):
        """Create side-by-side comparison visualizations."""
        print("\n  Creating comparison plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Error Analysis: Baseline vs Tuned (13-Feature LSTM)', fontsize=16, fontweight='bold')
        
        colors = {'baseline': '#3498db', 'tuned': '#e74c3c'}
        
        # --- 1. Predictions vs Actual (both models) ---
        ax = axes[0, 0]
        ax.scatter(self.baseline['y_true'], self.baseline['y_pred'], alpha=0.15, s=8, c=colors['baseline'], label='Baseline')
        ax.scatter(self.tuned['y_true'], self.tuned['y_pred'], alpha=0.15, s=8, c=colors['tuned'], label='Tuned')
        lims = [min(self.baseline['y_true'].min(), self.tuned['y_true'].min()),
                max(self.baseline['y_true'].max(), self.tuned['y_true'].max())]
        ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.6)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predictions vs Actual')
        ax.legend(fontsize=8)
        ax.text(0.05, 0.95, f"Base R2={self.baseline['metrics']['R2']:.4f}\nTuned R2={self.tuned['metrics']['R2']:.4f}",
                transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # --- 2. Error Distribution (overlaid) ---
        ax = axes[0, 1]
        ax.hist(self.baseline['errors'], bins=50, alpha=0.5, color=colors['baseline'], label=f"Base (MAE={self.baseline['metrics']['MAE']:.3f})")
        ax.hist(self.tuned['errors'], bins=50, alpha=0.5, color=colors['tuned'], label=f"Tuned (MAE={self.tuned['metrics']['MAE']:.3f})")
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Error')
        ax.set_title('Error Distribution')
        ax.legend(fontsize=8)
        
        # --- 3. MAE by Stress Level ---
        ax = axes[0, 2]
        bins = [0, 3, 5, 7, 10]
        labels_s = ['Low\n(1-3)', 'Med\n(4-5)', 'High\n(6-7)', 'VHigh\n(8-9)']
        
        base_maes, tuned_maes = [], []
        for label_name, full_label in zip(labels_s, ['Low (1-3)', 'Medium (4-5)', 'High (6-7)', 'Very High (8-9)']):
            for d, out_list in [(self.baseline, base_maes), (self.tuned, tuned_maes)]:
                stress_bins_d = pd.cut(d['y_true'], bins=bins, labels=['Low (1-3)', 'Medium (4-5)', 'High (6-7)', 'Very High (8-9)'], include_lowest=True)
                mask = stress_bins_d == full_label
                out_list.append(np.mean(d['abs_errors'][mask]) if mask.sum() > 0 else 0)
        
        x = np.arange(len(labels_s))
        w = 0.35
        ax.bar(x - w/2, base_maes, w, color=colors['baseline'], label='Baseline', alpha=0.8)
        ax.bar(x + w/2, tuned_maes, w, color=colors['tuned'], label='Tuned', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_s)
        ax.set_ylabel('MAE')
        ax.set_title('MAE by Stress Level')
        ax.legend(fontsize=8)
        
        # --- 4. MAE by Activity ---
        ax = axes[1, 0]
        activities = self.baseline['df']['Activity'].unique()
        act_base, act_tuned = {}, {}
        for d, out_dict in [(self.baseline, act_base), (self.tuned, act_tuned)]:
            df_a = d['df'].copy()
            df_a['abs_error'] = d['abs_errors'][:len(df_a)]
            for act in activities:
                mask = df_a['Activity'] == act
                if mask.sum() > 0:
                    out_dict[act] = df_a.loc[mask, 'abs_error'].mean()
        
        acts_sorted = sorted(act_base.keys(), key=lambda a: act_base.get(a, 0), reverse=True)
        x = np.arange(len(acts_sorted))
        w = 0.35
        ax.barh(x - w/2, [act_base.get(a, 0) for a in acts_sorted], w, color=colors['baseline'], label='Baseline', alpha=0.8)
        ax.barh(x + w/2, [act_tuned.get(a, 0) for a in acts_sorted], w, color=colors['tuned'], label='Tuned', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(acts_sorted, fontsize=8)
        ax.set_xlabel('MAE')
        ax.set_title('MAE by Activity')
        ax.legend(fontsize=8)
        
        # --- 5. MAE by Time Period ---
        ax = axes[1, 1]
        time_periods = ['Night\n(0-5)', 'Morning\n(6-11)', 'Afternoon\n(12-17)', 'Evening\n(18-23)']
        time_ranges = [(0, 6), (6, 12), (12, 18), (18, 24)]
        
        time_base, time_tuned = [], []
        for start, end in time_ranges:
            for d, out_list in [(self.baseline, time_base), (self.tuned, time_tuned)]:
                df_t = d['df'].copy()
                df_t['abs_error'] = d['abs_errors'][:len(df_t)]
                mask = (df_t['Hour'] >= start) & (df_t['Hour'] < end)
                out_list.append(df_t.loc[mask, 'abs_error'].mean() if mask.sum() > 0 else 0)
        
        x = np.arange(len(time_periods))
        ax.bar(x - w/2, time_base, w, color=colors['baseline'], label='Baseline', alpha=0.8)
        ax.bar(x + w/2, time_tuned, w, color=colors['tuned'], label='Tuned', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(time_periods, fontsize=8)
        ax.set_ylabel('MAE')
        ax.set_title('MAE by Time Period')
        ax.legend(fontsize=8)
        
        # --- 6. Cumulative Error Distribution ---
        ax = axes[1, 2]
        for data_key, color in colors.items():
            d = self.__dict__[data_key]
            sorted_errors = np.sort(d['abs_errors'])
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax.plot(sorted_errors, cumulative, color=color, label=d['label'], linewidth=2)
        
        ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90%')
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='95%')
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Cumulative Error Distribution')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 4)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'error_comparison_baseline_vs_tuned.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("    [OK] Saved: error_comparison_baseline_vs_tuned.png")
    
    def save_summary(self, comparison, stress_df, activity_df, time_df, dist_df):
        """Save JSON summary."""
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall': comparison,
            'baseline_metrics': self.baseline['metrics'],
            'tuned_metrics': self.tuned['metrics'],
            'improvement': {
                'mae_reduction_pct': comparison['MAE']['pct_change'],
                'rmse_reduction_pct': comparison['RMSE']['pct_change'],
                'r2_improvement_pct': comparison['R2']['pct_change'],
            }
        }
        
        with open(self.results_dir / 'error_analysis_tuned_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print("    [OK] Saved: error_analysis_tuned_summary.json")
    
    def run(self):
        """Run full comparison analysis."""
        print("\n" + "=" * 70)
        print("  ERROR ANALYSIS - BASELINE vs TUNED (13-Feature LSTM)")
        print("=" * 70)
        
        comparison = self.compare_overall()
        stress_df = self.compare_by_stress_level()
        activity_df = self.compare_by_activity()
        time_df = self.compare_by_time()
        dist_df = self.compare_error_distribution()
        self.create_comparison_plots()
        self.save_summary(comparison, stress_df, activity_df, time_df, dist_df)
        
        print("\n" + "=" * 70)
        print("  [DONE] ERROR ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\n  Baseline: MAE={self.baseline['metrics']['MAE']:.4f}, R2={self.baseline['metrics']['R2']:.4f}")
        print(f"  Tuned:    MAE={self.tuned['metrics']['MAE']:.4f}, R2={self.tuned['metrics']['R2']:.4f}")
        print(f"  Results:  {self.results_dir}")


def main():
    base_dir = Path(__file__).parent.parent
    analyzer = ErrorAnalyzerTuned(base_dir)
    analyzer.run()


if __name__ == '__main__':
    main()
