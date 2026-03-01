# Context-Aware Stress Prediction using Stacked Bidirectional LSTM

Hệ thống dự đoán mức độ **stress liên tục (thang 1–9)** từ dữ liệu cảm biến và hành vi người dùng, tích hợp module **Nhận dạng hoạt động (HAR)** và kiến trúc **Stacked Bidirectional LSTM** được tối ưu bằng Bayesian Optimization.

> **Kết quả tốt nhất:** R² = 0.9555 | MAE = 0.4414 (thang 10 điểm)

---

## Yêu cầu hệ thống

- Python **3.10 – 3.12**
- pip >= 23.0
- RAM >= 8 GB (khuyến nghị 16 GB)
- OS: Windows 10/11, Ubuntu 20.04+, macOS 12+

---

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd har-wisdm-bidirectional-lstm-rnns-stacked_lstm_wihout_BO
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install tensorflow==2.16.1
pip install "numpy<2"
pip install pandas scikit-learn matplotlib seaborn
pip install keras-tuner
pip install "shap<0.47" --no-deps
```

> **Lưu ý quan trọng:**
> - Phải cài `numpy<2` trước SHAP để tránh xung đột với TensorFlow 2.16.x
> - SHAP phiên bản `<0.47` tương thích với numpy 1.x và TensorFlow 2.16.x
> - Không dùng `pip install shap` trực tiếp (sẽ kéo numpy 2.x gây lỗi)

Kiểm tra cài đặt:

```bash
python -c "import tensorflow as tf; import numpy as np; import shap; print('TF:', tf.__version__, '| NP:', np.__version__, '| SHAP:', shap.__version__)"
```

Kết quả mong đợi:
```
TF: 2.16.1 | NP: 1.26.x | SHAP: 0.46.x
```

---

## Cấu trúc project

```
.
├── data/
│   ├── WISDM_ar_v1.1_raw.txt                    # Dữ liệu gia tốc kế WISDM thô
│   ├── optimized_health_data_13features.csv      # Tập dữ liệu 13-feature (dùng để train)
│   ├── optimized_health_data_17features.csv      # Phiên bản 17-feature (thử nghiệm)
│   └── feature_selection_13features_report.txt   # Báo cáo lựa chọn đặc trưng
│
├── HAR/                                          # Module nhận dạng hoạt động
│   ├── HAR_Recognition.py                        # Script huấn luyện HAR
│   ├── classificator.py                          # Inference / phân loại hoạt động
│   ├── preprocessing.py                          # Tiền xử lý WISDM (sliding window)
│   ├── config.py                                 # Cấu hình HAR
│   └── visualization.py                          # Vẽ kết quả HAR
│
├── generate_and_verify_data/
│   └── Data generator/
│       ├── refactored_health_data_generator.py   # Sinh 54K mẫu 30 ngày
│       ├── validate_accelerometer_with_har.py    # Xác thực với HAR model
│       ├── create_23feature_dataset.py           # Giảm 44 → 23 features
│       └── core/                                 # 7 sub-modules sinh dữ liệu
│
├── stress_prediction/                            # Module dự đoán stress
│   ├── train_lstm_13features.py                  # Huấn luyện Bi-LSTM Baseline
│   ├── hyperparameter_tuning.py                  # Bayesian Optimization (HP Tuning)
│   ├── feature_importance_13features.py          # Phân tích tầm quan trọng đặc trưng
│   ├── error_analysis_tuned.py                   # Phân tích lỗi (baseline vs tuned)
│   ├── model_comparison.py                       # So sánh 5 kiến trúc
│   ├── create_13features.py                      # Tạo dataset 13 features
│   ├── data_pipeline.py                          # Pipeline xử lý dữ liệu
│   └── config.py                                 # Cấu hình chung
│
├── models/                                       # Các model đã huấn luyện (.keras, .pkl)
├── results/                                      # Kết quả thực nghiệm (CSV, PNG, JSON)
├── Doc/                                          # Tài liệu nghiên cứu và outline báo cáo
└── classificator_model.keras                     # HAR model đã huấn luyện
```

---

## Hướng dẫn chạy từng bước

### Bước 1 — Huấn luyện mô hình HAR

```bash
cd HAR
python HAR_Recognition.py
```

Đầu ra: `classificator_model.keras` (~96% accuracy trên WISDM test set)

---

### Bước 2 — Sinh tập dữ liệu đa phương thức

```bash
cd "generate_and_verify_data/Data generator"
python refactored_health_data_generator.py
```

Đầu ra: `data/quota_balanced_health_data_30days.csv` (~54,448 mẫu, 44 features)

**Xác thực dữ liệu với HAR model:**

```bash
python validate_accelerometer_with_har.py
```

---

### Bước 3 — Tạo tập dữ liệu 13 features

```bash
cd ../../
python -m stress_prediction.create_13features
```

Đầu ra: `data/optimized_health_data_13features.csv`

---

### Bước 4 — Huấn luyện Stacked Bi-LSTM Baseline

```bash
python -m stress_prediction.train_lstm_13features
```

Đầu ra:
- `models/lstm_13features_best.keras`
- `models/scaler_13features.pkl`
- `results/metrics_13features.txt`

---

### Bước 5 — Tối ưu Hyperparameters (Bayesian Optimization)

```bash
python -m stress_prediction.hyperparameter_tuning
```

> Thời gian: ~60–90 phút (20 trials). Có thể giảm `max_trials` trong file config.

Đầu ra:
- `models/lstm_13features_tuned.keras`
- `results/hp_tuning/tuning_results.json`

---

### Bước 6 — Phân tích tầm quan trọng đặc trưng

```bash
python -m stress_prediction.feature_importance_13features
```

> Thời gian: ~60–90 phút (SHAP KernelExplainer trên 100 mẫu)

Đầu ra: `results/feature_importance_13features/` (PNG, CSV, JSON)

---

### Bước 7 — Phân tích lỗi (Baseline vs Tuned)

```bash
python -m stress_prediction.error_analysis_tuned
```

Đầu ra: `results/error_analysis_13features_tuned/` (CSV, PNG, JSON)

---

### Bước 8 — So sánh 5 kiến trúc model

```bash
python -m stress_prediction.model_comparison
```

> Thời gian: ~90–120 phút (train 5 models)

Đầu ra: `results/model_comparison/` (CSV, PNG, JSON)

---

## Kết quả tổng hợp

### HAR (Human Activity Recognition)

| Hoạt động | Accuracy |
|-----------|----------|
| Walking | 80.1% |
| Jogging | 100% |
| Upstairs | 34.4% |
| Downstairs | 100% |
| Sitting | — |
| Standing | 100% |
| **Overall** | **86.2%** |

### So sánh 5 kiến trúc Stress Prediction

| Model | MAE | RMSE | R² | Params |
|-------|-----|------|----|--------|
| MLP (Dense) | 0.9310 | 1.2968 | 0.8331 | 241K |
| Simple LSTM | 0.5213 | 0.7603 | 0.9426 | 83K |
| Stacked Bi-LSTM (Baseline) | 0.7159 | 0.9698 | 0.9067 | 320K |
| Stacked Bi-GRU | 0.7551 | 0.9103 | 0.9178 | 244K |
| **Stacked Bi-LSTM (Tuned)** | **0.4414** | **0.6697** | **0.9555** | **164K** |

### Top-5 Features quan trọng nhất

| Rank | Feature | Phương pháp nhất quán |
|------|---------|----------------------|
| 1 | Heart_Rate | Permutation, SHAP, Correlation, RF |
| 2 | Mood_Score | Permutation, SHAP, Correlation, RF |
| 3 | Screen_Usage_Current | Permutation, SHAP, RF |
| 4 | Energy_Level | Permutation, Correlation |
| 5 | Day_of_Week | SHAP, RF |

---

## Phiên bản thư viện

| Thư viện | Phiên bản | Ghi chú |
|----------|-----------|---------|
| Python | 3.12.4 | |
| TensorFlow | 2.16.1 | |
| NumPy | 1.26.4 | **Phải < 2.0** |
| pandas | 2.x | |
| scikit-learn | 1.x | |
| matplotlib | 3.x | |
| seaborn | 0.x | |
| keras-tuner | 1.x | Bayesian Optimization |
| SHAP | 0.46.0 | **Phải < 0.47** |

---

## Xử lý lỗi thường gặp

**Lỗi UnicodeEncodeError trên Windows:**
```
UnicodeEncodeError: 'cp1252' codec can't encode character
```
→ Tất cả script đã được fix bằng `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`. Nếu vẫn gặp, chạy:
```bash
set PYTHONIOENCODING=utf-8
```

**Lỗi numpy/TensorFlow incompatible:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
→ Chạy: `pip install "numpy<2" --force-reinstall`

**Lỗi SHAP với TensorFlow:**
→ Chạy: `pip install "shap<0.47" --force-reinstall --no-deps`

---

**Last Updated:** March 2026 | **Status:** Complete
