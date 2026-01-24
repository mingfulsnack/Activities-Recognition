# Phase 2 - Step 3: Feature Importance Analysis - Summary

## ✅ HOÀN THÀNH - 23 Tháng 1, 2026

---

## 🎯 Phát Hiện Quan Trọng Nhất

### 🥇 Top 3 Features Chi Phối Model (86.37% importance)

1. **Location (64.98%)** 🏆
   - Vị trí là yếu tố quan trọng nhất
   - Chiếm hơn 64% tổng importance
   - Work vs Home vs Commute vs Outdoor có stress patterns rất khác nhau

2. **Heart_Rate (13.93%)**
   - Nhịp tim phản ánh trực tiếp stress sinh lý
   - Feature quan trọng thứ 2
   - Correlation = 0.62 với stress

3. **Screen_Usage_Current (7.46%)**
   - Thời gian dùng màn hình hiện tại
   - Chỉ số stress từ công nghệ
   - Correlation = 0.55 với stress

### 📊 Random Forest Performance
- **Train R²**: 0.9832
- **Test R²**: 0.9311 (rất gần LSTM: 0.9343)
- **Test MAE**: 0.5916 (tương đương LSTM: 0.5095)

→ RF validation cho thấy feature importance đáng tin cậy

---

## 🔍 Top 10 Features Chi Tiết

| Rank | Feature | RF Importance | % | Cumulative % | Correlation |
|------|---------|---------------|---|--------------|-------------|
| 1 | **Location** | 0.6498 | 64.98% | 64.98% | 0.34 |
| 2 | **Heart_Rate** | 0.1393 | 13.93% | 78.92% | **0.62** |
| 3 | **Screen_Usage_Current** | 0.0746 | 7.46% | 86.37% | 0.55 |
| 4 | **Phone_Event_Frequency** | 0.0335 | 3.35% | 89.72% | **0.74** |
| 5 | **Mood_Score** | 0.0255 | 2.55% | 92.27% | **-0.74** |
| 6 | **Energy_Level** | 0.0199 | 1.99% | 94.26% | -0.22 |
| 7 | **Exercise_Minutes** | 0.0109 | 1.09% | 95.34% | -0.07 |
| 8 | **Sleep_Duration** | 0.0106 | 1.06% | 96.40% | -0.13 |
| 9 | **Screen_Usage_15min_Avg** | 0.0105 | 1.05% | 97.45% | 0.58 |
| 10 | **Sleep_Quality** | 0.0063 | 0.63% | 98.07% | -0.20 |

---

## 💡 Insight Quan Trọng: RF Importance vs Correlation

### Sự Khác Biệt Thú Vị:

**RF Importance (dựa vào tree splits):**
1. Location (65%)
2. Heart_Rate (14%)
3. Screen_Usage_Current (7.5%)

**Correlation (quan hệ tuyến tính):**
1. **Mood_Score** (-0.74) - Cao nhất nhưng chỉ rank #5 trong RF
2. **Phone_Event_Frequency** (0.74) - Cao nhưng rank #4
3. **Heart_Rate** (0.62) - Rank #2 cả hai metrics

### Giải Thích:

- **Location** có importance cao vì là **categorical feature** với nhiều interactions
  - Location chia data thành các nhóm rõ ràng (work, home, commute)
  - Mỗi location có stress pattern khác nhau
  - RF trees dùng Location để split data hiệu quả

- **Mood_Score & Phone_Event_Frequency** có correlation cao nhưng importance thấp hơn vì:
  - Là **numerical features** với quan hệ tuyến tính mạnh
  - Nhưng khi Location đã split data, chúng ít được dùng thêm
  - Có thể bị "che" bởi Location trong tree structure

---

## 📈 Cumulative Importance Analysis

### Feature Selection Recommendations:

- **10 features** → 98.07% importance ✅
- **6 features** → 94.26% importance ✅
- **3 features** → 86.37% importance ⚠️

**Đề xuất:**
- Giữ **top 10 features** để đảm bảo performance
- Có thể remove **11 features** còn lại (chỉ contribute 1.93%)
- Tiết kiệm: 52% features, mất chỉ 2% importance

---

## 🎨 Feature Categories Analysis

### Physiological Features (Heart Rate, Sleep, Energy, Mood)
- **Total Importance**: ~16%
- **Key Features**: Heart_Rate (14%), Mood_Score (2.5%)
- **Insight**: Mood_Score có correlation -0.74 (cao nhất) nhưng importance chỉ 2.5%

### Screen/Phone Features
- **Total Importance**: ~12%
- **Key Features**: Screen_Usage_Current (7.5%), Phone_Event_Frequency (3.3%)
- **Insight**: Phone usage indicators are strong stress predictors

### Activity/Location Features
- **Total Importance**: ~66%
- **Dominated by**: Location (65%)
- **Insight**: Contextual features (where you are) matter most

### Accelerometer Features (X, Y, Z)
- **Total Importance**: <0.5%
- **Insight**: Kept for HAR validation, but don't contribute to stress prediction

---

## 🎯 Key Insights for Model Improvement

### 1. Location là Game Changer
- 64.98% importance - dominates all other features
- **Error Analysis insight**: "Standing during commute" had worst errors
- **Action**: Create location-specific sub-models hoặc interaction features

### 2. Mood-Stress Paradox
- Mood_Score có correlation -0.74 (highest) nhưng importance chỉ 2.5%
- **Possible reasons**:
  - Mood là consequence của stress (not cause)
  - Location already captures context that determines both mood & stress
  - RF prioritizes features that create clear splits

### 3. Phone Usage Patterns Matter
- Phone_Event_Frequency + Screen_Usage = 10.81% combined
- Correlation 0.74 and 0.55 respectively
- **Insight**: Digital behavior reflects stress levels

### 4. Physiological Features Important but Secondary
- Heart_Rate (14%) + others (~2%)
- **Insight**: Body responses are reliable but location context matters more

### 5. Low-Importance Features
- **11 features** < 0.5% each (total 1.93%)
- Candidates for removal: Accelerometer X/Y/Z, Weather, Ambient_Light, etc.

---

## 🚀 Recommendations - Bước Tiếp Theo

### 1. Feature Engineering Priorities

**High Priority:**
- **Location Interactions**: Location × Time, Location × Activity
- **Phone/Screen Patterns**: Rolling averages, usage spikes, nighttime usage
- **Heart Rate Variability**: HRV from Heart_Rate, stress recovery patterns

**Medium Priority:**
- **Mood Patterns**: Mood trends, mood volatility
- **Social Context**: Social × Location interactions

### 2. Feature Selection Strategy

**Option A: Conservative (Keep 10 features - 98% importance)**
```
Location, Heart_Rate, Screen_Usage_Current, Phone_Event_Frequency,
Mood_Score, Energy_Level, Exercise_Minutes, Sleep_Duration,
Screen_Usage_15min_Avg, Sleep_Quality
```

**Option B: Aggressive (Keep 6 features - 94% importance)**
```
Location, Heart_Rate, Screen_Usage_Current, Phone_Event_Frequency,
Mood_Score, Energy_Level
```

**Option C: Minimal (Keep 3 features - 86% importance)**
```
Location, Heart_Rate, Screen_Usage_Current
```

### 3. Model Architecture Adjustments

**For LSTM/RNN models:**
- Add **attention mechanism** focusing on Location feature
- Create **location-specific LSTM cells**
- Use **embedding layers** for Location (not just encoding)

**For next models (GRU, TCN):**
- Test with reduced feature set (10 features)
- Compare performance: 10 vs 23 features
- Faster training, less overfitting risk

### 4. Address Medium Stress (4-5) Weakness
- Error analysis showed MAE=0.93 for stress 4-5
- Top features (Location, Heart_Rate) might not capture subtle differences
- **Solution**: Boost importance of Mood_Score, Energy_Level for mid-range stress

---

## 📊 Correlation vs Importance - Deep Dive

| Feature | RF Importance | Correlation | Agreement |
|---------|---------------|-------------|-----------|
| Location | **#1 (65%)** | #8 (0.34) | ❌ Mismatch |
| Heart_Rate | **#2 (14%)** | **#3 (0.62)** | ✅ Strong |
| Screen_Usage_Current | **#3 (7.5%)** | #5 (0.55) | ✅ Good |
| Phone_Event_Frequency | #4 (3.3%) | **#2 (0.74)** | ⚠️ Gap |
| Mood_Score | #5 (2.5%) | **#1 (-0.74)** | ❌ Major gap |

**Interpretation:**
- ✅ **Heart_Rate**: Consistent across both metrics - truly important
- ⚠️ **Phone_Event_Frequency**: High correlation but lower importance - useful but redundant with other features
- ❌ **Mood_Score**: Highest correlation but low importance - captured by Location?
- ❌ **Location**: Highest importance but lower correlation - enables complex interactions

---

## 📁 Files Created

### Data
- `results/feature_importance/rf_feature_importance.csv` - RF importance scores
- `results/feature_importance/permutation_importance.csv` - Permutation scores (skipped)
- `results/feature_importance/feature_correlations.csv` - Feature-target correlations

### Visualizations
- `results/feature_importance/feature_importance_comprehensive.png` - 6-panel analysis
- `results/feature_importance/top10_features_detailed.png` - Top 10 comparison

### Reports
- `results/feature_importance/FEATURE_IMPORTANCE_REPORT.md` - Full analysis

---

## 🎓 Ý Nghĩa Cho Luận Văn

### 1. Feature Understanding
- Chứng minh được **Location** là yếu tố quan trọng nhất
- Giải thích tại sao error analysis cho thấy "Standing during commute" khó dự đoán
- Hỗ trợ discussion về context-aware stress monitoring

### 2. Model Design Justification
- Justify việc giữ 23 features hay reduce xuống 10
- Explain trade-off giữa complexity và performance
- Support ablation study design

### 3. Future Work Direction
- Location-specific modeling
- Interaction feature engineering
- Attention mechanisms focusing on key features

### 4. Methodology Rigor
- Multiple importance metrics (RF + Correlation)
- Validation through RF performance (R²=0.9311)
- Comprehensive visualization and analysis

---

## 🔄 So Sánh Với Error Analysis

### Error Analysis Results (Step 2):
- **Worst**: Medium stress (4-5), MAE = 0.93
- **Worst Activity**: Standing, MAE = 0.66
- **Worst Context**: "Standing during commute"

### Feature Importance Insights (Step 3):
- **Location dominates** (65%) → Explains why context matters
- **Heart_Rate #2** (14%) → Good for high stress detection
- **Mood_Score high correlation** (-0.74) → Could help with medium stress

### Integration:
**To fix medium stress predictions:**
1. Boost Mood_Score importance (currently underutilized)
2. Add Location × Mood interactions
3. Create features that distinguish medium from low/high stress

**To fix "Standing during commute":**
1. Location already important, but needs interaction features
2. Add Activity × Location × Time features
3. Context-specific embeddings

---

## ✅ Completion Status

- ✅ Random Forest trained successfully (R²=0.9311)
- ✅ Feature importance calculated and ranked
- ✅ Correlation analysis completed
- ✅ Visualizations generated (2 plots)
- ✅ Comprehensive report created
- ⚠️ Permutation importance skipped (too slow)

**Overall: Phase 2 Step 3 - COMPLETED** 🎉

---

## 🚀 Next Steps - Options

**Option 1: Implement Feature Selection** (Recommended)
- Create reduced dataset with top 10 features
- Retrain LSTM with reduced features
- Compare performance: 10 vs 23 features
- Duration: ~1 day

**Option 2: Feature Engineering**
- Create interaction features (Location × Activity, etc.)
- Add temporal features (rolling means, trends)
- Test with LSTM
- Duration: ~2 days

**Option 3: Implement Next Model (GRU)**
- Use insights from feature importance
- Test with both 10 and 23 features
- Compare with LSTM baseline
- Duration: ~2 days

**Option 4: Build Model Comparison Framework**
- Setup infrastructure for comparing multiple models
- Standardized evaluation metrics
- Prepare for GRU, TCN, Transformer
- Duration: ~1-2 days

---

**Tôi recommend Option 1 (Feature Selection)** để kiểm tra xem model có perform tốt với ít features hơn không. Điều này sẽ:
1. Làm model đơn giản hơn
2. Faster training
3. Giảm overfitting risk
4. Validate feature importance findings

**Bạn muốn làm option nào?**

---

**Status**: Phase 2 Step 3 - COMPLETED ✅  
**Date**: January 23, 2026  
**Duration**: <1 day  
**Next**: Feature Selection or Next Model
