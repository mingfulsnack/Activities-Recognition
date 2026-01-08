# Error Analysis Report - LSTM Baseline

**Generated**: January 9, 2026  
**Model**: lstm_baseline_best.keras  
**Dataset**: optimized_health_data_23features.csv  
**Test Samples**: 8,159 sequences

---

## 📊 Executive Summary

The LSTM baseline model demonstrates **excellent performance** with an R² score of **0.9343** (93.43%), explaining most variance in stress levels. Average prediction error is only **0.51 stress units** (on 1-9 scale).

### Key Metrics
- **R² Score**: 0.9343 (93.43%) ✅
- **MAE**: 0.5095 ✅  
- **RMSE**: 0.8123 ✅
- **Median Absolute Error**: 0.2145

---

## 🔍 Error Distribution Analysis

### Overall Statistics
- **Mean Error**: -0.1174 (slight underestimation bias)
- **Std Error**: 0.8038
- **90th Percentile Error**: 1.5508
- **95th Percentile Error**: 1.7892  
- **99th Percentile Error**: 2.6557
- **Max Error**: 4.9610

### Distribution Shape
- **Normality Test p-value**: 0.0000 ⚠️
- **Status**: Errors are **NOT normally distributed**
- **Implication**: Suggests systematic bias or outliers exist

---

## 📈 Performance by Stress Level

| Stress Level | Count | % of Test | MAE | Mean Actual | Mean Predicted | Bias |
|-------------|-------|-----------|-----|-------------|----------------|------|
| **Very High (8-9)** | 2,083 | 25.5% | **0.1223** ✅ | 8.92 | 8.91 | +0.01 |
| **Low (1-3)** | 4,221 | 51.7% | 0.5368 | 1.44 | 1.70 | -0.26 |
| **High (6-7)** | 603 | 7.4% | 0.7925 | 5.63 | 5.92 | -0.29 |
| **Medium (4-5)** | 1,252 | 15.3% | **0.9257** ⚠️ | 3.93 | 3.69 | +0.24 |

### Key Insights

**✅ Best Performance:**
- **Very High Stress (8-9)**: MAE = 0.1223
  - Excellent accuracy for critical high-stress situations
  - Almost perfect predictions (bias = 0.01)
  - Most important for real-world stress monitoring

**⚠️ Worst Performance:**
- **Medium Stress (4-5)**: MAE = 0.9257 (7.6x worse than best)
  - Moderate stress levels are hardest to predict
  - Likely due to high variability in this range
  - May represent transition states between low/high stress

---

## 🏃 Performance by Activity Type

| Activity | Count | MAE | Mean Actual | Mean Predicted | Bias |
|----------|-------|-----|-------------|----------------|------|
| **Standing** | 1,807 | **0.6578** ⚠️ | 3.44 | 3.56 | -0.15 |
| **Jogging** | 814 | 0.5591 | 2.29 | 2.81 | -0.37 |
| **Upstairs** | 467 | 0.5216 | 6.23 | 6.27 | -0.10 |
| **Sitting** | 2,367 | 0.5048 | 4.51 | 4.53 | -0.06 |
| **Downstairs** | 446 | 0.4475 | 6.41 | 6.66 | -0.24 |
| **Walking** | 2,258 | **0.3878** ✅ | 3.75 | 3.81 | -0.04 |

### Key Insights

**✅ Best Accuracy:**
- **Walking**: MAE = 0.3878
  - Consistent, predictable stress patterns during walking
  - Most common activity in dataset (2,258 samples)

**⚠️ Worst Accuracy:**
- **Standing**: MAE = 0.6578 (1.7x worse than walking)
  - Static posture with variable context
  - Standing can occur in many different stress contexts (waiting, commuting, working)

---

## ⏰ Performance by Time of Day

| Time Period | Count | MAE | Mean Actual | Mean Predicted |
|------------|-------|-----|-------------|----------------|
| **Night (22-6)** | 602 | **0.4305** ✅ | 1.80 | 2.10 |
| **Morning (6-12)** | 2,109 | 0.4428 | 5.07 | 5.11 |
| **Afternoon (12-18)** | 3,084 | 0.5234 | 5.22 | 5.12 |
| **Evening (18-22)** | 2,364 | **0.5711** ⚠️ | 2.17 | 2.59 |

### Key Insights

**✅ Best Time:**
- **Night (22-6)** and **Morning (6-12)**: Consistent routines lead to better predictions

**⚠️ Worst Time:**
- **Evening (18-22)**: MAE = 0.5711
  - Variable evening activities (dinner, entertainment, work, relaxation)
  - Transition from work to personal time creates unpredictability

---

## 🔴 Worst Predictions Analysis

**Top 100 Worst Predictions:**

### Common Patterns
- **Most Common Activities**: 
  - Standing: 79/100 (79%) ⚠️
  - Sitting: 10/100 (10%)
  - Downstairs: 8/100 (8%)
  
- **Most Common Locations**:
  - Commute: 68/100 (68%) ⚠️
  - Work: 18/100 (18%)
  - Home: 5/100 (5%)

- **Error Characteristics**:
  - Average Actual Stress: 2.92 (Low-Medium)
  - Average Predicted Stress: 4.84 (Medium)
  - Average Error: **2.95** (systematic overestimation)

### Critical Finding ⚠️

**"Standing during commute"** combination produces largest errors:
- Model consistently **overestimates** stress by ~2 levels
- Standing in commute context may have variable stress (sometimes relaxing, sometimes stressful)
- Model may not capture context-specific variations well

---

## 💡 Key Findings

### ✅ Strengths

1. **Excellent Overall Performance**
   - R² = 93.43% demonstrates strong predictive power
   - MAE = 0.51 means predictions are typically within half a stress level
   - Model works well across most scenarios

2. **Critical Stress Detection**
   - **Very High Stress (8-9)**: MAE = 0.12 - almost perfect
   - Essential for real-world stress monitoring applications
   - Can reliably identify when users need intervention

3. **Activity-Specific Accuracy**
   - Walking predictions are very accurate (MAE = 0.39)
   - Consistent activities produce consistent predictions

4. **Time-Based Patterns**
   - Morning and Night predictions are reliable
   - Model captures circadian rhythm effects

### ⚠️ Weaknesses

1. **Medium Stress Levels (4-5)**
   - MAE = 0.93 - significantly worse than other ranges
   - Represents transition states that are inherently variable
   - May need additional features or modeling approaches

2. **Standing Activity**
   - MAE = 0.66 - highest error among activities
   - Static posture with highly variable context
   - "Standing" alone doesn't provide enough information

3. **Evening Predictions**
   - MAE = 0.57 - worse than other time periods
   - Variable evening routines are hard to predict
   - Transition from work to personal time creates uncertainty

4. **Commute Context**
   - 68% of worst predictions occur during commute
   - Systematic overestimation by ~2 stress levels
   - Model doesn't capture context-specific stress variations

5. **Non-Normal Error Distribution**
   - Errors not normally distributed (p < 0.001)
   - Suggests systematic biases exist
   - May benefit from outlier analysis

---

## 🎯 Recommendations

### For Model Improvement

1. **Address Medium Stress (4-5) Predictions**
   - **Approach**: Use class weights or focal loss to balance training
   - **Data**: Augment medium-stress scenarios
   - **Features**: Add more contextual features to distinguish transition states

2. **Improve "Standing" and "Commute" Predictions**
   - **Context Features**: Add more location/time context for standing
   - **Commute-Specific Model**: Consider separate modeling for commute scenarios
   - **Feature Engineering**: Create interaction features (Activity × Location × Time)

3. **Handle Non-Normal Error Distribution**
   - **Outlier Analysis**: Investigate samples with error > 2.5
   - **Robust Loss**: Try Huber loss or quantile regression
   - **Ensemble Methods**: Combine predictions to reduce systematic biases

4. **Feature Importance Analysis** (Next Step)
   - Identify which features contribute most to errors
   - Use Random Forest or SHAP values
   - Guide feature engineering efforts

5. **Hyperparameter Tuning**
   - Current model may not be optimal
   - Try different LSTM units, dropout rates, learning rates
   - Use grid search or Bayesian optimization

6. **Ensemble Approach**
   - Combine LSTM with other models (GRU, TCN, Transformer)
   - Use voting or stacking to reduce variance
   - Leverage strengths of different architectures

### For Thesis Documentation

1. **Excellent Baseline Established**
   - R² = 93.43% provides strong comparison benchmark
   - Future models must beat or provide other advantages (speed, interpretability)

2. **Comprehensive Error Analysis**
   - Demonstrates understanding of model behavior
   - Identifies specific weaknesses for future improvement
   - Provides evidence for systematic research approach

3. **Real-World Implications**
   - High accuracy for critical high-stress detection
   - Identifies challenging scenarios (commute, standing, medium stress)
   - Informs practical deployment considerations

4. **Visualization Support**
   - Error analysis plots provide visual evidence
   - Support methodology and results sections
   - Demonstrate scientific rigor

---

## 📁 Generated Files

All results saved to: `stress_prediction/results/error_analysis/`

- **error_statistics.csv** - Overall error metrics
- **error_by_stress_level.csv** - Performance by stress ranges
- **error_by_activity.csv** - Performance by activity type
- **error_by_time.csv** - Performance by time of day  
- **worst_predictions.csv** - Top 100 worst predictions for analysis
- **error_analysis_comprehensive.png** - 6-panel visualization
- **qq_plot.png** - Normality check
- **ERROR_ANALYSIS_REPORT.md** - This comprehensive report

---

## 🚀 Next Steps

**Immediate Actions:**
1. ✅ **Error Analysis** - COMPLETED
2. **Feature Importance** - Use Random Forest to identify key features
3. **Hyperparameter Optimization** - Fine-tune LSTM architecture
4. **Model Comparison** - Implement GRU, TCN, Transformer variants

**Phase 2 Continuation:**
- Build comparison framework for multiple models
- Create model evaluation dashboard
- Statistical significance testing between models
- Prepare results for thesis writeup

---

**Report Generated by**: error_analysis.py  
**Date**: January 9, 2026  
**Status**: Phase 2 Step 2 - COMPLETED ✅
