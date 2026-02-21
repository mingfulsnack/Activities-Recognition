# Phase 2 - Hyperparameter Tuning (13-Feature LSTM)

## 1. Phương pháp: Bayesian Optimization (Keras Tuner)

### Tại sao chọn Bayesian Optimization?
- Hiệu quả hơn Grid Search và Random Search
- Sử dụng mô hình xác suất (surrogate model) để dự đoán vùng hyperparameter tốt nhất
- Hội tụ nhanh hơn với ít thử nghiệm hơn

### Cấu hình
- **Số trials**: 20
- **Epochs/trial**: 30 (Early Stopping patience=7)
- **Initial random exploration**: 5 trials
- **Objective**: Minimize val_mae

## 2. Không gian tìm kiếm (Search Space)

| Hyperparameter | Giá trị | Loại |
|---------------|---------|------|
| LSTM Layer 1 units | [64, 128, 256] | Choice |
| LSTM Layer 2 units | [32, 64, 128] | Choice |
| Dropout rate | 0.1 - 0.5 (step 0.1) | Float |
| Dense units | [32, 64, 128] | Choice |
| Learning rate | 1e-4 → 1e-2 | Float (log scale) |

**Kiến trúc cố định**: Stacked Bidirectional LSTM (2 layers) + 2 Dense layers

## 3. Kết quả tối ưu

### Best Hyperparameters

| Parameter | Baseline | **Tuned (Best)** |
|-----------|----------|-------------------|
| LSTM Layer 1 | 128 | **64** |
| LSTM Layer 2 | 64 | **64** |
| Dropout | 0.3 | **0.1** |
| Dense units | 64 | **128** |
| Learning rate | Adam default (~0.001) | **0.01** |

### Nhận xét:
- **LSTM units nhỏ hơn** (64→64 thay vì 128→64): Model baseline bị overparameterized
- **Dropout thấp (0.1)**: Dữ liệu đủ lớn, không cần regularization mạnh
- **Dense units lớn hơn (128)**: Cần capacity ở dense layer để học non-linear patterns
- **Learning rate cao hơn (0.01)**: Cho phép converge nhanh hơn, kết hợp ReduceLROnPlateau

## 4. So sánh Performance

### Test Set Metrics

| Metric | Baseline | Tuned | Thay đổi | % |
|--------|----------|-------|----------|---|
| **MAE** | 0.6757 | **0.5292** | -0.1465 | **-21.7%** ✅ |
| **RMSE** | 0.8571 | **0.7483** | -0.1088 | **-12.7%** ✅ |
| **R²** | 0.9271 | **0.9444** | +0.0173 | **+1.9%** ✅ |

### Phân tích cải thiện:
- **MAE giảm 21.7%**: Dự đoán trung bình sai lệch ~0.53 điểm stress (thang 1-10)
- **RMSE giảm 12.7%**: Cải thiện rõ rệt ở các trường hợp sai lệch lớn
- **R² tăng lên 0.9444**: Mô hình giải thích được 94.4% phương sai của stress level

## 5. Kiến trúc mô hình tối ưu

```
Input (60 timesteps × 13 features)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.1)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.1)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.1)
    ↓
Dense (64, ReLU)
    ↓
Dense (1, Linear) → Stress Level
```

## 6. Training Configuration (Final Model)

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.01) |
| Loss | MSE |
| Epochs | 80 (Early Stopping patience=15) |
| Batch size | 32 |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Best weights | ModelCheckpoint (monitor val_loss) |

## 7. Data Pipeline (No Data Leakage)

```
Raw Data (54,448 samples)
    ↓
Split 70/15/15 (TRƯỚC encoding)
    ↓
Encode categorical (fit TRAIN only)  
    ↓
Normalize features (fit TRAIN only)
    ↓
Create sequences (seq_length=60)
    ↓
Train: 37,993 | Val: 8,097 | Test: 8,107
```

## 8. Files & Artifacts

| File | Mô tả |
|------|--------|
| `models/lstm_13features_tuned.keras` | Mô hình đã tối ưu |
| `models/scaler_13features_tuned.pkl` | StandardScaler |
| `models/label_encoder_13features_tuned_*.pkl` | Label encoders |
| `results/hp_tuning/tuning_results.json` | Kết quả chi tiết |
| `results/hp_tuning/tuned_training_history.json` | Training history |
| `results/metrics_13features_tuned.txt` | Metrics text |
| `stress_prediction/hyperparameter_tuning.py` | Script tuning |

## 9. Kết luận

Bayesian Optimization đã cải thiện mô hình **đáng kể**:
- MAE giảm từ 0.6757 → **0.5292** (-21.7%)
- R² tăng từ 0.9271 → **0.9444** (+1.9%)
- Mô hình nhỏ gọn hơn (ít LSTM units) nhưng hiệu quả hơn
- Dropout thấp (0.1) cho thấy baseline đã regularize quá mức

**Bước tiếp theo**: Feature Importance Analysis (SHAP/Permutation) để hiểu đóng góp của từng feature.
