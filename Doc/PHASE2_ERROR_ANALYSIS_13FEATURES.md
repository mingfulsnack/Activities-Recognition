# Phase 2 - Error Analysis: 13-Feature LSTM Model

**Date**: February 19, 2026  
**Model**: Stacked Bidirectional LSTM (128→64 units, Dropout 0.3)  
**Dataset**: 54,448 samples, 13 features, 60-timestep sequences  
**Pipeline**: Split → Encode (fit train) → Normalize → Sequences (no data leakage)

---

## 1. Tổng quan Performance

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **R²** | **0.9271** | Model giải thích 92.71% variance của stress |
| **MAE** | **0.6757** | Sai số trung bình ±0.68 mức stress (thang 0-10) |
| **RMSE** | **0.8571** | Root mean squared error |
| **Test Samples** | 8,109 | Sequences từ 8,168 test samples |

---

## 2. Error Distribution

| Thống kê | Giá trị |
|----------|---------|
| Mean Error (Bias) | +0.2953 (model hơi under-predict) |
| Std Error | 0.8046 |
| Median Absolute Error | 0.5470 |
| 90th Percentile | 1.3795 |
| 95th Percentile | 1.6573 |
| 99th Percentile | 2.1742 |
| Max Error | 3.8444 |

**Normality Test**: p-value = 0.0000 → Errors **không phân phối chuẩn** (có systematic bias)

**Nhận xét**:
- 50% predictions sai < 0.55 stress units (rất tốt)
- 90% predictions sai < 1.38 (chấp nhận được)
- Bias dương (+0.30) → model có xu hướng predict thấp hơn actual

---

## 3. Error theo Stress Level

| Mức Stress | Count | % | MAE | Bias | Đánh giá |
|------------|-------|---|-----|------|----------|
| **Low (1-3)** | 4,221 | 52.1% | **0.3140** | -0.09 | ✅ Rất tốt |
| **Medium (4-5)** | 1,237 | 15.3% | 0.8458 | +0.16 | ⚠️ Khá |
| **High (6-7)** | 573 | 7.1% | 0.8842 | -0.02 | ⚠️ Khá |
| **Very High (8-9)** | 2,078 | 25.6% | **1.2518** | +1.25 | ❌ Cần cải thiện |

**Phân tích**:
- **Low Stress (1-3)**: Tốt nhất - MAE chỉ 0.31, chiếm 52% dữ liệu
- **Medium/High (4-7)**: MAE ~0.85, chấp nhận được
- **Very High (8-9)**: MAE cao nhất (1.25) với bias +1.25 → model **under-predict stress cao**
  - Nguyên nhân: Model hồi quy về mean, stress rất cao bị kéo xuống
  - Giải pháp tiềm năng: Class weights, focal loss, hoặc data augmentation

---

## 4. Error theo Activity

| Activity | Count | MAE | Actual Mean | Pred Mean | Đánh giá |
|----------|-------|-----|-------------|-----------|----------|
| **Upstairs** | 467 | **0.9316** | 6.23 | 5.79 | ❌ Cao nhất |
| **Downstairs** | 417 | **0.9028** | 6.48 | 5.79 | ❌ Cao |
| **Sitting** | 2,337 | 0.7195 | 4.49 | 3.99 | ⚠️ Khá |
| **Jogging** | 814 | 0.6405 | 2.29 | 2.64 | ✅ Tốt |
| **Walking** | 2,258 | 0.6302 | 3.75 | 3.48 | ✅ Tốt |
| **Standing** | 1,816 | **0.5737** | 3.45 | 3.22 | ✅ Tốt nhất |

**Phân tích**:
- **Standing, Walking, Jogging**: Error thấp (0.57-0.64) → Model dự đoán tốt cho hoạt động phổ biến
- **Upstairs/Downstairs**: Error cao nhất (~0.93) → phải leo cầu thang thường kèm stress cao
- Activities có stress cao (Upstairs: 6.23, Downstairs: 6.48) bị under-predict → liên quan bias ở Very High stress

---

## 5. Error theo Thời gian (Hour)

| Khoảng giờ | MAE Trung bình | Stress Trung bình | Đánh giá |
|-------------|---------------|-------------------|----------|
| **Sáng sớm (6-8)** | ~0.91 | 3.0-4.5 | ⚠️ Cao |
| **Sáng (9-12)** | ~0.84 | 4.3-5.5 | ⚠️ Khá |
| **Chiều (13-16)** | ~0.91 | 5.4-5.8 | ⚠️ Cao |
| **Chiều tối (17-18)** | ~0.63 | 2.6-4.4 | ✅ Tốt |
| **Tối (19-23)** | **~0.32** | 1.0-2.2 | ✅ Rất tốt |

**Phân tích**:
- **Tối (19-23)**: Error thấp nhất (0.25-0.38) → stress thấp, dễ predict
- **Giờ làm việc (9-16)**: Error cao (0.81-1.03) → stress biến động mạnh, khó predict
- **Đỉnh error**: 6h sáng (MAE=1.15) và 13h (MAE=1.03) → transition periods
- Hour feature giúp model nhận diện pattern circadian nhưng vẫn khó ở peak hours

---

## 6. Visualizations

Các biểu đồ đã được tạo tại `results/error_analysis_13features/`:

1. **error_analysis_comprehensive.png**: 6 biểu đồ tổng hợp
   - Predictions vs Actual (scatter)
   - Error Distribution (histogram)
   - Absolute Error Distribution
   - Error by Stress Level (bar chart)
   - Residual Plot
   - Error by Activity (horizontal bar)

2. **qq_plot.png**: Q-Q plot kiểm tra phân phối chuẩn

---

## 7. Tóm tắt & Insights chính

### Điểm mạnh ✅
1. **R² = 0.9271** - Model giải thích >92% variance
2. **Low stress prediction rất tốt** - MAE = 0.31 cho stress 1-3
3. **Standing/Walking/Jogging** - Error thấp, ổn định
4. **Thời gian tối** - Predictions chính xác nhất
5. **No data leakage** - Pipeline đúng chuẩn, kết quả đáng tin

### Điểm yếu ❌ & Hướng cải thiện
1. **Very High Stress (8-9) bị under-predict** (bias +1.25)
   - → Thử weighted loss hoặc class rebalancing
2. **Upstairs/Downstairs error cao** (~0.93)
   - → Ít sample hơn + stress cao → double penalty
3. **Giờ làm việc (9-16) khó predict** (MAE ~0.85-1.03)
   - → Stress biến động mạnh trong giờ work
4. **Error không phân phối chuẩn**
   - → Có systematic bias, cần investigate thêm

### Đánh giá tổng quan

Model 13-feature LSTM đạt kết quả **tốt** cho bài toán stress prediction:
- **92.71% variance explained** - xuất sắc cho dữ liệu health
- **MAE = 0.68** - sai số chưa tới 1 mức stress trên thang 0-10
- Bao gồm đầy đủ **HAR core features** + evidence-based selection
- Pipeline **không có data leakage** - kết quả đáng tin cậy

---

## 8. Bước tiếp theo

- [x] ~~Error Analysis cho 13-feature model~~
- [ ] Feature Importance Analysis (SHAP/Permutation)
- [ ] Hyperparameter Tuning (Bayesian Optimization)
- [ ] Cập nhật báo cáo luận văn
- [ ] Chuẩn bị slide thuyết trình

---

**Files liên quan**:
- Model: `models/lstm_13features_best.keras`
- Data: `data/optimized_health_data_13features.csv`
- Script: `stress_prediction/error_analysis_13features.py`
- Results: `results/error_analysis_13features/`
