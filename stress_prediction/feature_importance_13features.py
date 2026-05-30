"""
Feature Importance Analysis - 13-Feature Tuned LSTM Model
Methods:
1. Permutation Importance (truc tiep tren LSTM model)
2. SHAP Values (DeepExplainer cho deep learning)
3. Correlation Analysis
4. Random Forest surrogate importance

"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import json
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance as sklearn_perm_importance

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ============================================================
# Configuration
# ============================================================
FEATURE_NAMES = [
    'Hour', 'Day_of_Week', 'Activity', 'Accelerometer_X',
    'Accelerometer_Y', 'Accelerometer_Z', 'Heart_Rate',
    'Location', 'Screen_Usage_Current', 'Phone_Event_Frequency',
    'Mood_Score', 'Energy_Level', 'Sleep_Duration'
]

CATEGORICAL_FEATURES = ['Activity', 'Location']
SEQ_LENGTH = 60
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Data Pipeline (standalone, same as other 13-feature scripts)
# ============================================================
class DataPreprocessor:
    """Load & preprocess 13-feature data with correct pipeline."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_prepare(self):
        """Full pipeline: Load → Split → Encode → Normalize → Sequences."""
        df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(df):,} samples, {len(df.columns)} columns")
        
        X = df.drop('Stress_Level', axis=1)
        y = df['Stress_Level']
        
        # Split 70/15/15
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
        y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
        
        # Store raw for correlation analysis
        self.df = df
        self.X_train_raw = X_train.copy()
        self.X_test_raw = X_test.copy()
        
        # Encode categorical (fit train only)
        X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
        for col in CATEGORICAL_FEATURES:
            if col in X_train.columns:
                enc = LabelEncoder()
                X_train[col] = enc.fit_transform(X_train[col].astype(str))
                X_val[col] = enc.transform(X_val[col].astype(str))
                X_test[col] = enc.transform(X_test[col].astype(str))
                self.label_encoders[col] = enc
        
        # Store encoded (flat) for RF
        self.X_train_encoded = X_train.copy()
        self.X_test_encoded = X_test.copy()
        self.y_train_flat = y_train.copy()
        self.y_test_flat = y_test.copy()
        
        # Normalize (fit train only)
        X_train_s = self.scaler.fit_transform(X_train.values)
        X_val_s = self.scaler.transform(X_val.values)
        X_test_s = self.scaler.transform(X_test.values)
        
        # Store scaled flat data
        self.X_train_scaled = X_train_s
        self.X_test_scaled = X_test_s
        
        # Create sequences
        def make_seq(X, y, seq_len):
            Xs, ys = [], []
            yv = y.values if hasattr(y, 'values') else y
            for i in range(len(X) - seq_len):
                Xs.append(X[i:i+seq_len])
                ys.append(yv[i+seq_len])
            return np.array(Xs), np.array(ys)
        
        X_train_seq, y_train_seq = make_seq(X_train_s, y_train, SEQ_LENGTH)
        X_val_seq, y_val_seq = make_seq(X_val_s, y_val, SEQ_LENGTH)
        X_test_seq, y_test_seq = make_seq(X_test_s, y_test, SEQ_LENGTH)
        
        print(f"  Train: {X_train_seq.shape} | Val: {X_val_seq.shape} | Test: {X_test_seq.shape}")
        
        return {
            'X_train': X_train_seq, 'y_train': y_train_seq,
            'X_val': X_val_seq, 'y_val': y_val_seq,
            'X_test': X_test_seq, 'y_test': y_test_seq,
        }


# ============================================================
# 1. Permutation Importance (direct on LSTM)
# ============================================================
# DA: FEATURE_IMPORTANCE_PERMUTATION
# Measures LSTM feature importance by permuting each feature and tracking MAE degradation.
def permutation_importance_lstm(model, X_test, y_test, feature_names, n_repeats=10):
    """
    Permutation importance trực tiếp trên LSTM model.
    Shuffle từng feature across all timesteps, đo sự thay đổi MAE.
    """
    print("\n" + "=" * 60)
    print("  1. PERMUTATION IMPORTANCE (LSTM)")
    print("=" * 60)
    
    # Baseline MAE
    y_pred_base = model.predict(X_test, verbose=0).flatten()
    base_mae = mean_absolute_error(y_test, y_pred_base)
    print(f"  Baseline MAE: {base_mae:.4f}")
    
    n_features = X_test.shape[2]
    importances = np.zeros((n_repeats, n_features))
    
    for feat_idx in range(n_features):
        print(f"  Shuffling feature {feat_idx+1}/{n_features}: {feature_names[feat_idx]}...", end=' ')
        
        for rep in range(n_repeats):
            X_permuted = X_test.copy()
            # Shuffle this feature across all timesteps simultaneously
            perm_idx = np.random.permutation(X_test.shape[0])
            X_permuted[:, :, feat_idx] = X_test[perm_idx, :, feat_idx]
            
            y_pred_perm = model.predict(X_permuted, verbose=0).flatten()
            perm_mae = mean_absolute_error(y_test, y_pred_perm)
            importances[rep, feat_idx] = perm_mae - base_mae
        
        mean_imp = importances[:, feat_idx].mean()
        print(f"dMAE = {mean_imp:+.4f}")
    
    # Build results DataFrame
    results = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': importances.mean(axis=0),
        'Importance_Std': importances.std(axis=0)
    }).sort_values('Importance_Mean', ascending=False).reset_index(drop=True)
    
    print(f"\n  Permutation Importance (sorted by impact):")
    for _, row in results.iterrows():
        bar = '#' * int(row['Importance_Mean'] / results['Importance_Mean'].max() * 30) if results['Importance_Mean'].max() > 0 else ''
        print(f"    {row['Feature']:<25} {row['Importance_Mean']:>+.4f} ± {row['Importance_Std']:.4f}  {bar}")
    
    return results, base_mae


# ============================================================
# 2. SHAP Values (DeepExplainer)
# ============================================================
# DA: FEATURE_IMPORTANCE_SHAP
# Uses SHAP to explain which features contribute most to model predictions.
def shap_analysis(model, X_train, X_test, y_test, feature_names, n_background=200, n_explain=500):
    """
    SHAP DeepExplainer for LSTM model.
    """
    print("\n" + "=" * 60)
    print("  2. SHAP VALUES (DeepExplainer)")
    print("=" * 60)
    
    try:
        import shap
        
        # Background samples for DeepExplainer
        bg_idx = np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)
        background = X_train[bg_idx]
        
        # Samples to explain
        exp_idx = np.random.choice(len(X_test), min(n_explain, len(X_test)), replace=False)
        X_explain = X_test[exp_idx]
        
        print(f"  Background: {len(background)} samples")
        print(f"  Explaining: {len(X_explain)} samples")
        
        # Try DeepExplainer first, fall back to GradientExplainer
        try:
            print("  Using DeepExplainer...")
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_explain)
        except Exception as e1:
            print(f"  DeepExplainer failed: {e1}")
            print("  Falling back to GradientExplainer...")
            try:
                explainer = shap.GradientExplainer(model, background)
                shap_values = explainer.shap_values(X_explain)
            except Exception as e2:
                print(f"  GradientExplainer failed: {e2}")
                print("  Falling back to KernelExplainer (slower)...")
                
                # For KernelExplainer, use flattened last timestep
                def predict_fn(X_flat):
                    # Reshape to sequence: repeat last timestep
                    X_seq = np.repeat(X_flat[:, np.newaxis, :], SEQ_LENGTH, axis=1)
                    return model.predict(X_seq, verbose=0).flatten()
                
                bg_flat = background[:, -1, :]
                X_explain_flat = X_explain[:, -1, :]
                explainer = shap.KernelExplainer(predict_fn, bg_flat[:50])
                shap_values_flat = explainer.shap_values(X_explain_flat[:100])
                
                # Aggregate
                if isinstance(shap_values_flat, list):
                    shap_values_flat = shap_values_flat[0]
                
                mean_abs_shap = np.abs(shap_values_flat).mean(axis=0)
                results = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Mean_Abs': mean_abs_shap
                }).sort_values('SHAP_Mean_Abs', ascending=False).reset_index(drop=True)
                
                return results, shap_values_flat, X_explain_flat
        
        # Process SHAP values (shape: [n_samples, seq_len, n_features] or list)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        print(f"  SHAP values shape: {shap_values.shape}")
        
        # Aggregate across timesteps (mean absolute SHAP per feature)
        # shap_values shape: (n_explain, seq_length, n_features)
        shap_per_feature = np.abs(shap_values).mean(axis=(0, 1))  # mean over samples and timesteps
        
        results = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Mean_Abs': shap_per_feature
        }).sort_values('SHAP_Mean_Abs', ascending=False).reset_index(drop=True)
        
        print(f"\n  SHAP Feature Importance (mean |SHAP|):")
        for _, row in results.iterrows():
            bar = '#' * int(row['SHAP_Mean_Abs'] / results['SHAP_Mean_Abs'].max() * 30) if results['SHAP_Mean_Abs'].max() > 0 else ''
            print(f"    {row['Feature']:<25} {row['SHAP_Mean_Abs']:.4f}  {bar}")
        
        # Get last-timestep SHAP for summary plot
        shap_last = shap_values[:, -1, :]  # (n_explain, n_features)
        X_last = X_explain[:, -1, :]
        
        return results, shap_last, X_last
        
    except Exception as e:
        print(f"  [ERROR] SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ============================================================
# 3. Correlation Analysis
# ============================================================
# DA: FEATURE_IMPORTANCE_CORRELATION
# Computes feature-target correlations for interpretable evidence.
def correlation_analysis(df, feature_names):
    """Analyze feature correlations with Stress_Level."""
    print("\n" + "=" * 60)
    print("  3. CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Encode categorical for correlation
    df_encoded = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    
    # Correlation with target
    correlations = df_encoded[feature_names + ['Stress_Level']].corr()['Stress_Level'].drop('Stress_Level')
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values,
        'Abs_Correlation': np.abs(correlations.values)
    }).sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
    
    print(f"\n  Correlations with Stress_Level:")
    for _, row in corr_df.iterrows():
        direction = '+' if row['Correlation'] > 0 else '-'
        bar = '#' * int(row['Abs_Correlation'] / corr_df['Abs_Correlation'].max() * 30)
        print(f"    {row['Feature']:<25} {row['Correlation']:>+.4f}  {direction} {bar}")
    
    return corr_df


# ============================================================
# 4. Random Forest Surrogate
# ============================================================
# DA: FEATURE_IMPORTANCE_RF_SURROGATE
# Trains a Random Forest surrogate to estimate feature importance.
def rf_surrogate_importance(X_train_flat, y_train, X_test_flat, y_test, feature_names):
    """Train Random Forest on flat data for comparison importance."""
    print("\n" + "=" * 60)
    print("  4. RANDOM FOREST SURROGATE IMPORTANCE")
    print("=" * 60)
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_flat, y_train)
    
    # Tree-based importance
    tree_imp = rf.feature_importances_
    
    # Permutation importance on RF
    perm_result = sklearn_perm_importance(
        rf, X_test_flat, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_pred = rf.predict(X_test_flat)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"  RF Test MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
    
    results = pd.DataFrame({
        'Feature': feature_names,
        'RF_Tree_Importance': tree_imp,
        'RF_Perm_Importance': perm_result.importances_mean,
        'RF_Perm_Std': perm_result.importances_std
    }).sort_values('RF_Tree_Importance', ascending=False).reset_index(drop=True)
    
    print(f"\n  RF Feature Importance (Gini / Permutation):")
    for _, row in results.iterrows():
        bar = '#' * int(row['RF_Tree_Importance'] / results['RF_Tree_Importance'].max() * 30)
        print(f"    {row['Feature']:<25} Gini: {row['RF_Tree_Importance']:.4f}  Perm: {row['RF_Perm_Importance']:+.4f}  {bar}")
    
    return results


# ============================================================
# Visualization
# ============================================================
def create_visualizations(perm_results, shap_results, corr_results, rf_results,
                          shap_values_last, X_last, feature_names, results_dir):
    """Create comprehensive visualization plots."""
    print("\n  Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Feature Importance Analysis - 13-Feature Tuned LSTM', fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 13))
    
    # --- Plot 1: Permutation Importance (LSTM) ---
    ax = axes[0, 0]
    perm_sorted = perm_results.sort_values('Importance_Mean', ascending=True)
    bars = ax.barh(range(len(perm_sorted)), perm_sorted['Importance_Mean'],
                   xerr=perm_sorted['Importance_Std'], color=colors[::1],
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(perm_sorted)))
    ax.set_yticklabels(perm_sorted['Feature'], fontsize=9)
    ax.set_xlabel('Delta MAE (higher = more important)')
    ax.set_title('Permutation Importance (LSTM)', fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # --- Plot 2: SHAP Values ---
    ax = axes[0, 1]
    if shap_results is not None:
        shap_sorted = shap_results.sort_values('SHAP_Mean_Abs', ascending=True)
        ax.barh(range(len(shap_sorted)), shap_sorted['SHAP_Mean_Abs'],
                color=colors[::1], edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(shap_sorted)))
        ax.set_yticklabels(shap_sorted['Feature'], fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('SHAP Feature Importance', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'SHAP Analysis\nNot Available', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_title('SHAP Feature Importance', fontweight='bold')
    
    # --- Plot 3: Correlation with Stress Level ---
    ax = axes[1, 0]
    corr_sorted = corr_results.sort_values('Correlation', ascending=True)
    bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in corr_sorted['Correlation']]
    ax.barh(range(len(corr_sorted)), corr_sorted['Correlation'],
            color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels(corr_sorted['Feature'], fontsize=9)
    ax.set_xlabel('Pearson Correlation')
    ax.set_title('Correlation with Stress Level', fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # --- Plot 4: Random Forest Importance ---
    ax = axes[1, 1]
    rf_sorted = rf_results.sort_values('RF_Tree_Importance', ascending=True)
    ax.barh(range(len(rf_sorted)), rf_sorted['RF_Tree_Importance'],
            color=colors[::1], edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(rf_sorted)))
    ax.set_yticklabels(rf_sorted['Feature'], fontsize=9)
    ax.set_xlabel('Gini Importance')
    ax.set_title('Random Forest Surrogate Importance', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance_13features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: feature_importance_13features.png")
    
    # --- SHAP Summary Plot ---
    if shap_values_last is not None and X_last is not None:
        try:
            import shap
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values_last, X_last,
                              feature_names=feature_names,
                              show=False, max_display=13)
            plt.title('SHAP Summary Plot - 13-Feature Tuned LSTM', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'shap_summary_13features.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: shap_summary_13features.png")
        except Exception as e:
            print(f"  [WARN] SHAP summary plot failed: {e}")
    
    # --- Combined ranking plot ---
    create_ranking_comparison(perm_results, shap_results, corr_results, rf_results, results_dir)


def create_ranking_comparison(perm_results, shap_results, corr_results, rf_results, results_dir):
    """Create a combined ranking comparison across all methods."""
    
    ranking = pd.DataFrame({'Feature': FEATURE_NAMES})
    
    # Permutation rank
    perm_rank = perm_results.sort_values('Importance_Mean', ascending=False).reset_index(drop=True)
    perm_rank['Perm_Rank'] = range(1, len(perm_rank) + 1)
    ranking = ranking.merge(perm_rank[['Feature', 'Perm_Rank']], on='Feature')
    
    # SHAP rank
    if shap_results is not None:
        shap_rank = shap_results.sort_values('SHAP_Mean_Abs', ascending=False).reset_index(drop=True)
        shap_rank['SHAP_Rank'] = range(1, len(shap_rank) + 1)
        ranking = ranking.merge(shap_rank[['Feature', 'SHAP_Rank']], on='Feature')
    
    # Correlation rank
    corr_rank = corr_results.sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
    corr_rank['Corr_Rank'] = range(1, len(corr_rank) + 1)
    ranking = ranking.merge(corr_rank[['Feature', 'Corr_Rank']], on='Feature')
    
    # RF rank
    rf_rank = rf_results.sort_values('RF_Tree_Importance', ascending=False).reset_index(drop=True)
    rf_rank['RF_Rank'] = range(1, len(rf_rank) + 1)
    ranking = ranking.merge(rf_rank[['Feature', 'RF_Rank']], on='Feature')
    
    # Average rank
    rank_cols = [c for c in ranking.columns if c.endswith('_Rank')]
    ranking['Avg_Rank'] = ranking[rank_cols].mean(axis=1)
    ranking = ranking.sort_values('Avg_Rank').reset_index(drop=True)
    ranking['Overall_Rank'] = range(1, len(ranking) + 1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(ranking))
    width = 0.2
    
    colors_map = {'Perm_Rank': '#3498db', 'SHAP_Rank': '#e74c3c', 'Corr_Rank': '#2ecc71', 'RF_Rank': '#f39c12'}
    labels_map = {'Perm_Rank': 'Permutation', 'SHAP_Rank': 'SHAP', 'Corr_Rank': 'Correlation', 'RF_Rank': 'Random Forest'}
    
    for i, col in enumerate(rank_cols):
        offset = (i - len(rank_cols)/2 + 0.5) * width
        ax.bar(x + offset, ranking[col], width, label=labels_map.get(col, col),
               color=colors_map.get(col, 'gray'), alpha=0.8, edgecolor='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(ranking['Feature'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Rank (lower = more important)')
    ax.set_title('Feature Importance Ranking Comparison (All Methods)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ranking_comparison_13features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: ranking_comparison_13features.png")
    
    return ranking


# ============================================================
# Save Results
# ============================================================
def save_all_results(perm_results, shap_results, corr_results, rf_results,
                     ranking, base_mae, results_dir):
    """Save all results to files."""
    
    # CSV files
    perm_results.to_csv(os.path.join(results_dir, 'permutation_importance_13features.csv'), index=False)
    corr_results.to_csv(os.path.join(results_dir, 'correlation_analysis_13features.csv'), index=False)
    rf_results.to_csv(os.path.join(results_dir, 'rf_surrogate_importance_13features.csv'), index=False)
    
    if shap_results is not None:
        shap_results.to_csv(os.path.join(results_dir, 'shap_importance_13features.csv'), index=False)
    
    if ranking is not None:
        ranking.to_csv(os.path.join(results_dir, 'feature_ranking_combined_13features.csv'), index=False)
    
    # JSON summary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'lstm_13features_tuned',
        'baseline_mae': float(base_mae),
        'permutation_importance': {
            row['Feature']: {
                'importance': float(row['Importance_Mean']),
                'std': float(row['Importance_Std'])
            } for _, row in perm_results.iterrows()
        },
        'top5_permutation': perm_results.head(5)['Feature'].tolist(),
    }
    
    if shap_results is not None:
        summary['shap_importance'] = {
            row['Feature']: float(row['SHAP_Mean_Abs'])
            for _, row in shap_results.iterrows()
        }
        summary['top5_shap'] = shap_results.head(5)['Feature'].tolist()
    
    summary['correlation'] = {
        row['Feature']: float(row['Correlation'])
        for _, row in corr_results.iterrows()
    }
    
    summary['rf_gini_importance'] = {
        row['Feature']: float(row['RF_Tree_Importance'])
        for _, row in rf_results.iterrows()
    }
    
    if ranking is not None:
        summary['overall_ranking'] = ranking[['Feature', 'Overall_Rank', 'Avg_Rank']].to_dict('records')
    
    with open(os.path.join(results_dir, 'feature_importance_summary_13features.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  [SAVED] All results saved to: {results_dir}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  FEATURE IMPORTANCE ANALYSIS - 13-FEATURE TUNED LSTM")
    print("=" * 70)
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Setup paths
    data_path = os.path.join(BASE_DIR, 'data', 'optimized_health_data_13features.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'lstm_13features_tuned.keras')
    results_dir = os.path.join(BASE_DIR, 'results', 'feature_importance_13features')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load data
    print("\n[1] Loading data...")
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.load_and_prepare()
    
    # 2. Load tuned model
    print("\n[2] Loading tuned model...")
    model = keras.models.load_model(model_path)
    print(f"  [OK] Model loaded: {model_path}")
    
    # Quick test
    y_pred = model.predict(data['X_test'], verbose=0).flatten()
    test_mae = mean_absolute_error(data['y_test'], y_pred)
    test_r2 = r2_score(data['y_test'], y_pred)
    print(f"  Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # 3. Permutation Importance
    perm_results, base_mae = permutation_importance_lstm(
        model, data['X_test'], data['y_test'], FEATURE_NAMES, n_repeats=10
    )
    
    # 4. SHAP Analysis
    shap_results, shap_values_last, X_last = shap_analysis(
        model, data['X_train'], data['X_test'], data['y_test'],
        FEATURE_NAMES, n_background=200, n_explain=500
    )
    
    # 5. Correlation Analysis
    corr_results = correlation_analysis(preprocessor.df, FEATURE_NAMES)
    
    # 6. Random Forest Surrogate
    rf_results = rf_surrogate_importance(
        preprocessor.X_train_encoded.values, preprocessor.y_train_flat.values,
        preprocessor.X_test_encoded.values, preprocessor.y_test_flat.values,
        FEATURE_NAMES
    )
    
    # 7. Create visualizations
    create_visualizations(
        perm_results, shap_results, corr_results, rf_results,
        shap_values_last, X_last, FEATURE_NAMES, results_dir
    )
    
    # 8. Build ranking
    ranking = None
    try:
        ranking_df = pd.DataFrame({'Feature': FEATURE_NAMES})
        
        perm_r = perm_results.sort_values('Importance_Mean', ascending=False).reset_index(drop=True)
        perm_r['Perm_Rank'] = range(1, len(perm_r)+1)
        ranking_df = ranking_df.merge(perm_r[['Feature', 'Perm_Rank']], on='Feature')
        
        corr_r = corr_results.sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
        corr_r['Corr_Rank'] = range(1, len(corr_r)+1)
        ranking_df = ranking_df.merge(corr_r[['Feature', 'Corr_Rank']], on='Feature')
        
        rf_r = rf_results.sort_values('RF_Tree_Importance', ascending=False).reset_index(drop=True)
        rf_r['RF_Rank'] = range(1, len(rf_r)+1)
        ranking_df = ranking_df.merge(rf_r[['Feature', 'RF_Rank']], on='Feature')
        
        if shap_results is not None:
            shap_r = shap_results.sort_values('SHAP_Mean_Abs', ascending=False).reset_index(drop=True)
            shap_r['SHAP_Rank'] = range(1, len(shap_r)+1)
            ranking_df = ranking_df.merge(shap_r[['Feature', 'SHAP_Rank']], on='Feature')
        
        rank_cols = [c for c in ranking_df.columns if c.endswith('_Rank')]
        ranking_df['Avg_Rank'] = ranking_df[rank_cols].mean(axis=1)
        ranking_df = ranking_df.sort_values('Avg_Rank').reset_index(drop=True)
        ranking_df['Overall_Rank'] = range(1, len(ranking_df)+1)
        ranking = ranking_df
        
        print("\n" + "=" * 60)
        print("  OVERALL FEATURE RANKING (Combined)")
        print("=" * 60)
        print(f"\n  {'Rank':<5} {'Feature':<25} {'Avg Rank':<10} {' | '.join([c.replace('_Rank','') for c in rank_cols])}")
        print(f"  {'-'*70}")
        for _, row in ranking_df.iterrows():
            ranks_str = ' | '.join([f"{int(row[c]):>5}" for c in rank_cols])
            print(f"  {int(row['Overall_Rank']):<5} {row['Feature']:<25} {row['Avg_Rank']:<10.2f} {ranks_str}")
    except Exception as e:
        print(f"  [WARN] Ranking computation error: {e}")
    
    # 9. Save all results
    save_all_results(perm_results, shap_results, corr_results, rf_results,
                     ranking, base_mae, results_dir)
    
    print("\n" + "=" * 70)
    print("  [DONE] FEATURE IMPORTANCE ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n  Top 5 features (Permutation):")
    for i, row in perm_results.head(5).iterrows():
        print(f"    {i+1}. {row['Feature']} (dMAE: {row['Importance_Mean']:+.4f})")
    
    if shap_results is not None:
        print(f"\n  Top 5 features (SHAP):")
        for i, row in shap_results.head(5).iterrows():
            print(f"    {i+1}. {row['Feature']} (|SHAP|: {row['SHAP_Mean_Abs']:.4f})")
    
    print(f"\n  Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
