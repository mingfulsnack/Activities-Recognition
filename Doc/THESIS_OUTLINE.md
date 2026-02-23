# OUTLINE KHÓA LUẬN TỐT NGHIỆP

## Đề tài: Dự đoán mức độ Stress dựa trên dữ liệu Cảm biến và Hành vi Người dùng sử dụng Mạng LSTM hai chiều (Bidirectional LSTM)

**Sinh viên**: [Tên]  
**GVHD**: [Tên giảng viên]  
**Trường**: [Tên trường]  
**Năm**: 2026

---

## PHẦN MỞ ĐẦU (Trang i–viii)

| Mục | Nội dung | Ghi chú |
|-----|----------|---------|
| Tóm tắt | Tóm tắt toàn bộ nghiên cứu (tiếng Việt + tiếng Anh) | ~1 trang |
| Lời cảm ơn | | |
| Lời cam đoan | | |
| Mục lục | | Tự động sinh |
| Danh sách thuật ngữ viết tắt | HAR, LSTM, Bi-LSTM, GRU, SHAP, RNN, MLP, MAE, RMSE, R², HP, BO | ~1 trang |
| Danh sách hình vẽ | | |
| Danh sách bảng | | |

---

## CHƯƠNG 1: GIỚI THIỆU (~5–6 trang)

### 1.1 Đặt vấn đề
- Tầm quan trọng của sức khỏe tinh thần trong xã hội hiện đại
- Stress ảnh hưởng đến năng suất, sức khỏe thể chất và tinh thần
- Smartphone phổ biến → nguồn dữ liệu cảm biến phong phú (gia tốc kế, GPS, screen time...)
- Cơ hội: dự đoán stress mức độ liên tục (continuous) thay vì chỉ phân loại nhị phân

### 1.2 Mục tiêu nghiên cứu
- **Mục tiêu chính**: Xây dựng hệ thống dự đoán mức độ stress (thang 1–9) từ dữ liệu cảm biến và hành vi người dùng
- **Mục tiêu cụ thể**:
  1. Xây dựng module HAR (Human Activity Recognition) nhận dạng hoạt động từ dữ liệu gia tốc kế
  2. Tạo tập dữ liệu đa phương thức (multi-modal) kết hợp cảm biến + hành vi + ngữ cảnh
  3. Lựa chọn đặc trưng (feature selection) tối ưu cho bài toán dự đoán stress
  4. Huấn luyện và tối ưu mô hình Stacked Bidirectional LSTM
  5. So sánh hiệu năng với các kiến trúc khác (MLP, LSTM, Bi-GRU)

### 1.3 Phạm vi nghiên cứu
- Dữ liệu: WISDM v1.1 (gia tốc kế) + dữ liệu hành vi mô phỏng thực tế
- Mô hình: Deep Learning (LSTM family), không bao gồm phương pháp ML truyền thống
- Đầu ra: Hồi quy mức stress liên tục (1–9), không phải phân loại

### 1.4 Đóng góp của khóa luận
1. Hệ thống end-to-end từ nhận dạng hoạt động → tạo dữ liệu → dự đoán stress
2. Pipeline xử lý dữ liệu chống data leakage cho time-series
3. Phân tích tầm quan trọng đặc trưng bằng 4 phương pháp (Permutation, SHAP, Correlation, RF Surrogate)
4. So sánh hệ thống 5 kiến trúc deep learning trên cùng pipeline

### 1.5 Bố cục khóa luận
- Mô tả ngắn gọn nội dung từng chương

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ NGHIÊN CỨU LIÊN QUAN (~15–18 trang)

### 2.1 Nhận dạng hoạt động người dùng (Human Activity Recognition - HAR)
- 2.1.1 Khái niệm và ứng dụng của HAR
  - Định nghĩa HAR
  - Các phương pháp: cảm biến (accelerometer, gyroscope) vs thị giác máy tính
  - Ứng dụng: theo dõi sức khỏe, thể thao, người già, smart home
- 2.1.2 Tập dữ liệu WISDM v1.1
  - Nguồn: Kwapisz, Weiss & Moore (2010), KDD Workshop
  - 1,098,207 mẫu, 36 người dùng, tần số 20 Hz
  - 6 hoạt động: Walking (38.6%), Jogging (31.2%), Upstairs (11.2%), Downstairs (9.1%), Sitting (5.5%), Standing (4.4%)
  - Định dạng: [user, activity, timestamp, x, y, z] — phạm vi [-20, 20], 10 = 1g = 9.81 m/s²
- 2.1.3 Các phương pháp HAR dựa trên Deep Learning
  - CNN cho chuỗi thời gian (tham khảo Yang et al. 2015)
  - RNN/LSTM (tham khảo Ordóñez & Roggen 2016)
  - Bi-LSTM cho HAR (tham khảo Hämäläinen et al. 2011)

### 2.2 Mạng nơ-ron hồi quy (Recurrent Neural Networks)
- 2.2.1 Kiến trúc RNN cơ bản
  - Cấu trúc: hidden state $h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$
  - Ưu điểm: xử lý chuỗi tuần tự
  - Hạn chế: vanishing/exploding gradient (Hochreiter 1991, Bengio et al. 1994)
- 2.2.2 Long Short-Term Memory (LSTM)
  - Kiến trúc: cell state $C_t$, forget gate $f_t$, input gate $i_t$, output gate $o_t$
  - Công thức chi tiết:
    - $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ — Forget gate
    - $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ — Input gate
    - $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ — Candidate cell
    - $C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$ — Cell state update
    - $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ — Output gate
    - $h_t = o_t \cdot \tanh(C_t)$ — Hidden state
  - Giải quyết vanishing gradient qua cell state highway (Hochreiter & Schmidhuber 1997)
  - Hình minh họa kiến trúc LSTM cell
- 2.2.3 Bidirectional LSTM (Bi-LSTM)
  - Kiến trúc: forward LSTM $\overrightarrow{h_t}$ + backward LSTM $\overleftarrow{h_t}$
  - $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$ — Nối (concatenate)
  - Ưu điểm: bắt được ngữ cảnh cả quá khứ lẫn tương lai
  - Stacked Bi-LSTM: nhiều lớp chồng nhau để học đặc trưng phức tạp hơn
  - Hình minh họa kiến trúc Stacked Bi-LSTM
- 2.2.4 Gated Recurrent Unit (GRU)
  - Kiến trúc đơn giản hơn LSTM: update gate $z_t$, reset gate $r_t$
  - $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$
  - $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$
  - $\tilde{h}_t = \tanh(W \cdot [r_t \cdot h_{t-1}, x_t])$
  - $h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$
  - So sánh LSTM vs GRU (Chung et al. 2014)

### 2.3 Multilayer Perceptron (MLP)
- Kiến trúc: Input → Hidden layers (ReLU) → Output
- Ưu/nhược điểm cho dữ liệu chuỗi thời gian
- Vai trò: baseline đơn giản nhất để so sánh

### 2.4 Các kỹ thuật huấn luyện và tối ưu
- 2.4.1 Hàm mất mát (Loss Function)
  - MSE cho bài toán hồi quy: $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- 2.4.2 Optimizer — Adam (Kingma & Ba 2014)
  - Cập nhật adaptive learning rate: $m_t$, $v_t$, bias correction
- 2.4.3 Regularization
  - Dropout (Srivastava et al. 2014): ngẫu nhiên tắt neurons với xác suất $p$
  - Early Stopping: dừng khi val_loss không giảm sau $k$ epochs
  - ReduceLROnPlateau: giảm learning rate khi plateau
- 2.4.4 Bayesian Optimization cho Hyperparameter Tuning
  - Gaussian Process surrogate, Acquisition function (Expected Improvement)
  - So sánh với Grid Search, Random Search (Bergstra & Bengio 2012)

### 2.5 Các chỉ số đánh giá (Evaluation Metrics)
- MAE: $\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$ — Sai số tuyệt đối trung bình
- RMSE: $\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ — Nhạy với outlier
- R² (Coefficient of Determination): $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ — Tỷ lệ phương sai giải thích được
- Accuracy, Precision, Recall, F1-score (cho bài toán HAR classification)

### 2.6 Phương pháp phân tích tầm quan trọng đặc trưng (Feature Importance)
- 2.6.1 Permutation Importance (Breiman 2001)
- 2.6.2 SHAP — SHapley Additive exPlanations (Lundberg & Lee 2017)
- 2.6.3 Correlation Analysis
- 2.6.4 Random Forest Surrogate Model

### 2.7 Nghiên cứu liên quan về dự đoán stress
- Các công trình liên quan:
  - Hovsepian et al. (2015) — cStress: continuous stress assessment from physiological sensors
  - Garcia-Ceja et al. (2018) — Stress detection from wearable sensor data
  - Kusserow et al. (2013) — Monitoring stress with body-worn sensors
  - Schlotz et al. (2004) — Diurnal cortisol pattern and stress
  - Schmidt et al. (2018) — WESAD dataset for wearable stress detection
- Khoảng trống nghiên cứu (Research Gap):
  - Phần lớn là binary classification (stress vs no-stress)
  - Ít kết hợp HAR + behavioral data + physiological signals
  - Chưa có context-aware modifier (hoạt động ảnh hưởng đến ý nghĩa của nhịp tim)

---

## CHƯƠNG 3: PHƯƠNG PHÁP ĐỀ XUẤT (~18–22 trang)

### 3.1 Tổng quan kiến trúc hệ thống
- Sơ đồ tổng thể (System Architecture Diagram):
```
WISDM Dataset ──► HAR Module (Bi-LSTM) ──► Activity Labels
                                              │
Raw Sensor + Behavioral Data ──► Data Generator ──► 44-field Dataset
                                                        │
                                              Feature Selection (44→13)
                                                        │
                                              Data Pipeline (chống leakage)
                                                        │
                                              Stacked Bi-LSTM Training
                                                        │
                                              HP Tuning (Bayesian Opt.)
                                                        │
                                              Model Evaluation & Analysis
```
- Giải thích 5 module chính và luồng dữ liệu

### 3.2 Module 1: Nhận dạng hoạt động (HAR Module)
- 3.2.1 Tiền xử lý dữ liệu WISDM
  - Cửa sổ trượt (sliding window): 180 mẫu (~9 giây ở 20Hz), bước nhảy 100 mẫu
  - **Bảng 3.1**: Thống kê WISDM — phân bố 6 hoạt động
  - Chuẩn hóa: StandardScaler trên 3 trục (x, y, z)
  - Label: mode (hoạt động phổ biến nhất trong cửa sổ), one-hot encoding
- 3.2.2 Kiến trúc mô hình HAR
  - **Hình 3.1**: Sơ đồ kiến trúc Bi-LSTM HAR
  - Input shape: `(180, 3)` — 180 timesteps × 3 trục gia tốc
  - Layer 1: `Bidirectional(LSTM(30, return_sequences=True, dropout=0.2))`
  - Layer 2: `Bidirectional(LSTM(30, dropout=0.2))`
  - Output: `Dense(6, softmax)`
  - Optimizer: Adam (lr=0.001, clipnorm=1.0)
  - **Bảng 3.2**: Hyperparameters HAR model
- 3.2.3 Kết quả huấn luyện HAR
  - **Bảng 3.3**: Confusion matrix
  - Accuracy ~96% trên tập test (30% holdout)
  - Nhận xét kết quả theo từng hoạt động

### 3.3 Module 2: Tạo tập dữ liệu đa phương thức (Multi-modal Data Generation)
- 3.3.1 Thiết kế hệ thống sinh dữ liệu
  - Mục đích: kết hợp WISDM thực + hành vi mô phỏng thực tế
  - 7 sub-modules: UserProfile, WISDMLoader, ActivityManager, ScheduleGenerator, MetricsCalculator, BehavioralTracker, Orchestrator
  - **Hình 3.2**: Sơ đồ luồng sinh dữ liệu
- 3.3.2 Mô phỏng lịch trình sinh hoạt
  - Lịch trình Vietnam lifestyle: thức 6:30–8:00, làm việc 9:00–17:00, ngủ 22:30–23:30
  - Hạn ngạch hoạt động hàng ngày (daily activity quotas):
    - Sitting ≤4.8h, Walking ≥4h, Standing ≥3.2h, Jogging ≥1.6h, Stairs ≥2.4h
  - Tần suất: 2 mẫu/phút → ~2,880 mẫu/ngày → 54,448 mẫu/30 ngày
- 3.3.3 Tính toán chỉ số stress đa yếu tố
  - Context-Stress Modifier: hoạt động ảnh hưởng đến ý nghĩa của nhịp tim
    - VD: HR cao khi jogging → stress modifier thấp; HR cao khi sitting → stress modifier cao
  - Các yếu tố: thời gian, hoạt động, location, sleep quality, work intensity, momentum
  - **Bảng 3.4**: Context-Stress Modifier theo hoạt động × nhịp tim
- 3.3.4 Xác thực dữ liệu
  - Dùng HAR model đã train để classify activity từ accelerometer data sinh ra
  - Kết quả: 86.2% accuracy (100% Jogging/Standing/Downstairs, 80.1% Walking, 34.4% Upstairs)
  - **Bảng 3.5**: Kết quả xác thực HAR trên dữ liệu sinh
- 3.3.5 Mô tả 44 trường dữ liệu ban đầu
  - **Bảng 3.6**: Danh sách đầy đủ 44 features theo nhóm (sensor, physiological, behavioral, environmental, stress)

### 3.4 Module 3: Lựa chọn đặc trưng (Feature Selection: 44 → 13)
- 3.4.1 Bước 1: Giảm từ 44 → 23 features
  - Loại bỏ redundant (derived features, identifiers, timestamps)
  - Giữ: 3 sensor + 9 core + 7 behavioral + 4 environmental
- 3.4.2 Bước 2: Giảm từ 23 → 17 features
  - Thêm 6 engineered features có cơ sở khoa học (Schlotz, Garcia-Ceja, Hovsepian, Kusserow)
  - **Vấn đề phát hiện**: Rolling features gây data leakage, training loss ~10.23 thay vì ~0.92
- 3.4.3 Bước 3: Tinh giản xuống 13 features (giải pháp cuối)
  - Dùng Random Forest importance để xác định features quan trọng nhất
  - **Bảng 3.7**: 13 features cuối cùng với RF importance score
  - Phân nhóm:
    - Temporal (2): Hour, Day_of_Week
    - Activity & Sensor (4): Activity, Accelerometer_X/Y/Z
    - Physiological (1): Heart_Rate
    - Behavioral (3): Screen_Usage_Current, Phone_Event_Frequency, Mood_Score
    - Contextual (3): Location, Energy_Level, Sleep_Duration
  - **Bảng 3.8**: Thống kê mô tả 13 features (mean, std, min, max, distribution)

### 3.5 Module 4: Pipeline xử lý dữ liệu (chống Data Leakage)
- 3.5.1 Vấn đề Data Leakage trong time-series
  - Tại sao không shuffle data time-series
  - Tại sao phải split trước khi encode/normalize
- 3.5.2 Data Pipeline chi tiết
  - **Hình 3.3**: Sơ đồ pipeline
  - Bước 1: Split 70/15/15 (train/val/test) — sequential, không shuffle
  - Bước 2: Encode categorical — LabelEncoder, fit trên train only
    - Activity: 6 classes (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
    - Location: 6 classes (Home, Office, Gym, Commute, Outdoor, Other)
  - Bước 3: Normalize — StandardScaler, fit trên train only
  - Bước 4: Create Sequences — sliding window seq_length=60 (tương đương 1 giờ ở 2 mẫu/phút)
  - **Bảng 3.9**: Kích thước dữ liệu qua từng bước

### 3.6 Module 5: Kiến trúc mô hình Stacked Bidirectional LSTM
- 3.6.1 Kiến trúc Baseline
  - **Hình 3.4**: Sơ đồ kiến trúc chi tiết
  - Input: `(60, 13)` — 60 timesteps × 13 features
  - Layer 1: `Bidirectional(LSTM(128, return_sequences=True))` → 256 outputs
  - Dropout(0.3)
  - Layer 2: `Bidirectional(LSTM(64))` → 128 outputs
  - Dropout(0.3)
  - Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(1, Linear)
  - **Bảng 3.10**: Bảng tham số (tổng: 320,129 parameters)
  - Optimizer: Adam(lr=0.001), Loss: MSE
  - Callbacks: EarlyStopping(patience=15), ReduceLROnPlateau(factor=0.5, patience=5)
- 3.6.2 Tối ưu Hyperparameters bằng Bayesian Optimization
  - Search space:
    - lstm_units_1: [32, 64, 128, 256]
    - lstm_units_2: [32, 64, 128]
    - dropout_rate: [0.1, 0.2, 0.3, 0.4, 0.5]
    - dense_units: [32, 64, 128]
    - learning_rate: [0.0001, 0.001, 0.01]
  - 20 trials, Bayesian Optimization (keras-tuner)
  - **Bảng 3.11**: Top-5 trials và hyperparameters
- 3.6.3 Kiến trúc Tuned (tối ưu)
  - **Hình 3.5**: Sơ đồ kiến trúc tuned
  - Best HP: lstm_units=64→64, dropout=0.1, dense=128, lr=0.01
  - So sánh thay đổi: dropout 0.3→0.1, units 128→64, lr 0.001→0.01
  - **Bảng 3.12**: So sánh kiến trúc Baseline vs Tuned (params, layers, config)

### 3.7 Các mô hình so sánh
- 3.7.1 MLP (Dense only) — Flatten(60×13) → Dense(256) → Dense(128) → Dense(64) → Dense(1)
- 3.7.2 Simple LSTM (1 layer, unidirectional) — LSTM(128) → Dense(64) → Dense(32) → Dense(1)
- 3.7.3 Stacked Bi-GRU — tương tự Bi-LSTM nhưng dùng GRU cells
- **Bảng 3.13**: Tóm tắt 5 kiến trúc (layers, params, đặc điểm)

---

## CHƯƠNG 4: THỰC NGHIỆM VÀ KẾT QUẢ (~15–18 trang)

### 4.1 Môi trường thực nghiệm
- **Bảng 4.1**: Cấu hình phần cứng và phần mềm
  - Python 3.12.4, TensorFlow 2.16.1, NumPy 1.26.4
  - SHAP 0.46.0, keras-tuner, scikit-learn
  - Windows, CPU training

### 4.2 Kết quả HAR (Human Activity Recognition)
- Accuracy ~96% trên WISDM test set
- **Bảng 4.2**: Classification report theo từng hoạt động
- **Hình 4.1**: Confusion Matrix HAR
- **Hình 4.2**: Training curves HAR
- Nhận xét: Jogging dễ nhận dạng nhất (biên độ cao, có nhịp); Standing/Sitting khó phân biệt

### 4.3 Kết quả Baseline Model (Stacked Bi-LSTM)
- **Bảng 4.3**: Metrics trên tập test
  - MAE = 0.6855, RMSE = 0.8723, R² = 0.9245
- **Hình 4.3**: Training history (loss và MAE theo epoch)
- **Hình 4.4**: Predicted vs Actual scatter plot
- Nhận xét: R²=0.9245 → mô hình giải thích 92.5% phương sai

### 4.4 Kết quả Hyperparameter Tuning
- 20 trials Bayesian Optimization
- **Bảng 4.4**: Best hyperparameters vs baseline
  | Parameter | Baseline | Tuned |
  |-----------|----------|-------|
  | lstm_units_1 | 128 | **64** |
  | lstm_units_2 | 64 | **64** |
  | dropout | 0.3 | **0.1** |
  | dense_units | 64 | **128** |
  | learning_rate | 0.001 | **0.01** |
- **Bảng 4.5**: Cải thiện sau HP Tuning
  | Metric | Baseline | Tuned | Thay đổi |
  |--------|----------|-------|----------|
  | MAE | 0.6855 | **0.5292** | **-22.8%** |
  | RMSE | 0.8723 | **0.7483** | **-14.2%** |
  | R² | 0.9245 | **0.9444** | **+2.2%** |
- **Hình 4.5**: Training history tuned model
- Nhận xét: Dropout thấp hơn (0.1) + lr cao hơn (0.01) cho phép model hội tụ nhanh hơn

### 4.5 Phân tích tầm quan trọng đặc trưng (Feature Importance)
- 4 phương pháp: Permutation, SHAP (KernelExplainer), Correlation, RF Surrogate
- **Bảng 4.6**: Top-5 features theo từng phương pháp
  | Rank | Permutation | SHAP | Correlation | RF Surrogate |
  |------|-------------|------|-------------|--------------|
  | 1 | Heart_Rate | Heart_Rate | Heart_Rate | Heart_Rate |
  | 2 | Mood_Score | Mood_Score | Mood_Score | Mood_Score |
  | 3 | Screen_Usage | Screen_Usage | Energy_Level | Screen_Usage |
  | 4 | Energy_Level | Day_of_Week | Sleep_Duration | Energy_Level |
  | 5 | Day_of_Week | Hour | Screen_Usage | Day_of_Week |
- **Hình 4.6**: Feature importance — bar chart 4 phương pháp
- **Hình 4.7**: Feature ranking comparison chart
- Nhận xét:
  - **Heart_Rate và Mood_Score** nhất quán đứng top-2 ở cả 4 phương pháp
  - Accelerometer features có importance thấp (model stress dựa trên physiological + behavioral)
  - Kết quả phù hợp với nghiên cứu: Hovsepian et al. (2015) — HR là chỉ số stress quan trọng nhất

### 4.6 Phân tích lỗi (Error Analysis)
- 4.6.1 Phân tích theo mức stress
  - **Bảng 4.7**: MAE theo stress level (baseline vs tuned)
  - Tuned cải thiện mạnh nhất ở Very High stress: -46.1%
  - Medium stress (4–6) vẫn là khó dự đoán nhất
- 4.6.2 Phân tích theo hoạt động
  - **Bảng 4.8**: MAE theo activity type (baseline vs tuned)
  - Sitting cải thiện -31.2%; Walking có MAE thấp nhất
- 4.6.3 Phân tích theo thời gian
  - **Bảng 4.9**: MAE theo time period (Morning, Afternoon, Evening, Night)
- **Hình 4.8**: Error comparison baseline vs tuned (bar chart)
- **Hình 4.9**: Error distribution comparison (histogram)

### 4.7 So sánh các mô hình (Model Comparison)
- 5 kiến trúc trên cùng pipeline, cùng data, cùng callbacks
- **Bảng 4.10**: Bảng so sánh tổng hợp
  | Model | MAE | RMSE | R² | Params | Time (s) |
  |-------|-----|------|----|--------|----------|
  | MLP (Dense) | 0.9310 | 1.2968 | 0.8331 | 241K | 60 |
  | Simple LSTM | 0.5213 | 0.7603 | 0.9426 | 83K | 452 |
  | Stacked Bi-LSTM (Baseline) | 0.7159 | 0.9698 | 0.9067 | 320K | 1,148 |
  | Stacked Bi-GRU | 0.7551 | 0.9103 | 0.9178 | 244K | 2,855 |
  | **Stacked Bi-LSTM (Tuned)** | **0.4414** | **0.6697** | **0.9555** | **164K** | **825** |
- **Hình 4.10**: Bar chart 6 metrics (MAE, RMSE, R², Time, Params, Learning Curves)
- **Hình 4.11**: Radar chart so sánh đa chiều
- **Hình 4.12**: Learning curves 5 models
- Nhận xét chi tiết:
  1. **MLP thấp nhất** (R²=0.83) → chứng minh temporal dependency quan trọng
  2. **Simple LSTM** (R²=0.94) bất ngờ tốt hơn Bi-LSTM Baseline (R²=0.91)
     → Bi-LSTM cần HP tuning tốt hơn do complexity cao hơn
  3. **Bi-GRU ≈ Bi-LSTM Baseline** → hai kiến trúc tương đương với default config
  4. **HP Tuning quyết định** — cùng Bi-LSTM, tuned cải thiện 38.3% MAE
  5. **Bi-LSTM Tuned tối ưu nhất**: vừa chính xác (MAE=0.4414), vừa hiệu quả (164K params)

### 4.8 So sánh với nghiên cứu liên quan
- **Bảng 4.11**: So sánh với các công trình đã công bố
  | Nghiên cứu | Phương pháp | Task | Dataset | Kết quả |
  |------------|-------------|------|---------|---------|
  | Hovsepian et al. (2015) | SVM + ECG/RIP | Binary | cStress | F1=0.72 |
  | Schmidt et al. (2018) | Random Forest | 3-class | WESAD | Acc=0.80 |
  | Garcia-Ceja et al. (2018) | Neural Network | Binary | Custom | AUC=0.82 |
  | **Nghiên cứu này** | **Stacked Bi-LSTM** | **Continuous (1–9)** | **Custom** | **R²=0.9555** |
- Nhận xét: Hệ thống này đạt R² cao nhất và giải quyết bài toán regression thay vì classification

---

## CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (~3–4 trang)

### 5.1 Kết luận
- Tóm tắt đóng góp:
  1. Xây dựng thành công hệ thống end-to-end: HAR → Data Generation → Feature Selection → Stress Prediction
  2. Stacked Bi-LSTM (Tuned) đạt R²=0.9555, MAE=0.4414 — dự đoán stress chính xác
  3. HP Tuning bằng Bayesian Optimization hiệu quả: cải thiện 22.8% MAE
  4. Feature Importance: Heart_Rate và Mood_Score là 2 features quan trọng nhất
  5. So sánh 5 kiến trúc: chứng minh Bi-LSTM + HP Tuning tối ưu

### 5.2 Hạn chế
- Dữ liệu: mô phỏng, chưa thu thập thực tế từ người dùng
- Mô hình: chỉ thử nghiệm LSTM family, chưa có Transformer/Attention
- Hardware: CPU training → giới hạn số trials HP tuning
- Cỡ mẫu: 30 ngày cho 1 profile, chưa đa dạng nhiều người dùng

### 5.3 Hướng phát triển
- **Ngắn hạn**:
  - Thu thập dữ liệu thực tế qua ứng dụng smartphone
  - Thêm Transformer/Attention model vào so sánh
- **Trung hạn**:
  - Xây dựng ứng dụng mobile real-time stress monitoring
  - Áp dụng Transfer Learning cho người dùng mới
  - Federated Learning bảo vệ dữ liệu cá nhân
- **Dài hạn**:
  - Kết hợp thêm EEG, skin conductance, cortisol
  - Đề xuất can thiệp (intervention) dựa trên mức stress dự đoán
  - Mở rộng sang bài toán dự đoán sức khỏe tâm thần tổng quát

---

## TÀI LIỆU THAM KHẢO (~25–30 mục)

### Nhóm 1: Deep Learning — Lý thuyết nền tảng
1. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
2. Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.
3. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS Workshop*.
4. Schuster, M. & Paliwal, K. (1997). Bidirectional Recurrent Neural Networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.
5. Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15, 1929–1958.
6. Kingma, D. P. & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *ICLR*.
7. Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE TNN*, 5(2), 157–166.

### Nhóm 2: HAR — Human Activity Recognition
8. Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2010). Activity Recognition using Cell Phone Accelerometers. *KDD Workshop on SensorKDD*.
9. Ordóñez, F. J. & Roggen, D. (2016). Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. *Sensors*, 16(1), 115.
10. Yang, J., Nguyen, M. N., San, P. P., Li, X. L., & Krishnaswamy, S. (2015). Deep Convolutional Neural Networks on Multichannel Time Series for Human Activity Recognition. *IJCAI*.
11. Hammerla, N. Y., Halloran, S., & Plötz, T. (2016). Deep, Convolutional, and Recurrent Models for Human Activity Recognition using Wearables. *IJCAI*.

### Nhóm 3: Stress Detection — Nghiên cứu liên quan
12. Hovsepian, K. et al. (2015). cStress: Towards a Gold Standard for Continuous Stress Assessment in the Mobile Environment. *UbiComp*, 493–504.
13. Schmidt, P. et al. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. *ICMI*.
14. Garcia-Ceja, E. et al. (2018). Multi-sensor fusion for stress detection and classification. *Pervasive and Mobile Computing*.
15. Kusserow, M. et al. (2013). Monitoring stress with a wrist device using context. *Journal of Biomedical Informatics*, 46(2), 287–295.
16. Schlotz, W. et al. (2004). Perceived work overload and chronic worrying predict weekend-weekday differences in the cortisol awakening response. *Psychosomatic Medicine*, 66(2), 207–214.
17. Can, Y. S. et al. (2019). Stress detection in daily life scenarios using smart phones and wearable sensors. *Journal of Biomedical Informatics*, 92, 103139.

### Nhóm 4: Feature Importance & Explainability
18. Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
19. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

### Nhóm 5: Hyperparameter Tuning
20. Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR*, 13, 281–305.
21. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *NeurIPS*.

### Nhóm 6: Framework & Công cụ
22. Abadi, M. et al. (2016). TensorFlow: A System for Large-Scale Machine Learning. *OSDI*.
23. Chollet, F. (2015). Keras. https://keras.io
24. O'Malley, T. et al. (2019). KerasTuner. https://github.com/keras-team/keras-tuner

---

## PHỤ LỤC

### Phụ lục A: Cấu trúc mã nguồn
- Cây thư mục dự án
- Mô tả chức năng từng file chính

### Phụ lục B: Chi tiết 44 features gốc
- Bảng: tên, kiểu dữ liệu, mô tả, nguồn

### Phụ lục C: Kết quả chi tiết Hyperparameter Tuning
- Bảng 20 trials đầy đủ

### Phụ lục D: Source code các module chính
- (Có thể trích đoạn code quan trọng)

---

## HƯỚNG DẪN SỐ TRANG ƯỚC TÍNH

| Phần | Số trang ước tính |
|------|-------------------|
| Phần mở đầu (v–viii) | 4–5 |
| Chương 1: Giới thiệu | 5–6 |
| Chương 2: Cơ sở lý thuyết | 15–18 |
| Chương 3: Phương pháp đề xuất | 18–22 |
| Chương 4: Thực nghiệm & kết quả | 15–18 |
| Chương 5: Kết luận | 3–4 |
| Tài liệu tham khảo | 2–3 |
| Phụ lục | 5–8 |
| **Tổng** | **~67–84 trang** |

---

## DANH SÁCH HÌNH VẼ DỰ KIẾN

| STT | Hình | Nội dung | Nguồn |
|-----|------|----------|-------|
| 1 | 2.1 | Kiến trúc RNN cơ bản | Vẽ mới |
| 2 | 2.2 | Kiến trúc LSTM cell (4 gates) | Vẽ mới hoặc trích |
| 3 | 2.3 | Bidirectional LSTM | Vẽ mới |
| 4 | 2.4 | GRU cell | Vẽ mới hoặc trích |
| 5 | 3.1 | Sơ đồ tổng thể hệ thống (5 modules) | Vẽ mới |
| 6 | 3.2 | Kiến trúc Bi-LSTM HAR | Vẽ mới |
| 7 | 3.3 | Luồng sinh dữ liệu (7 sub-modules) | Vẽ mới |
| 8 | 3.4 | Data pipeline (chống leakage) | Vẽ mới |
| 9 | 3.5 | Kiến trúc Stacked Bi-LSTM Stress (baseline) | Vẽ mới |
| 10 | 3.6 | Kiến trúc Stacked Bi-LSTM Stress (tuned) | Vẽ mới |
| 11 | 4.1 | Confusion Matrix HAR | Có sẵn |
| 12 | 4.2 | Training history baseline | Có sẵn |
| 13 | 4.3 | Training history tuned | Có sẵn |
| 14 | 4.4 | Feature importance 4 phương pháp | Có sẵn |
| 15 | 4.5 | Feature ranking comparison | Có sẵn |
| 16 | 4.6 | Error comparison baseline vs tuned | Có sẵn |
| 17 | 4.7 | Model comparison bar chart (6 metrics) | Có sẵn |
| 18 | 4.8 | Model comparison radar chart | Có sẵn |
| 19 | 4.9 | Learning curves 5 models | Có sẵn trong 4.7 |

## DANH SÁCH BẢNG DỰ KIẾN

| STT | Bảng | Nội dung |
|-----|------|----------|
| 1 | 3.1 | Thống kê WISDM dataset |
| 2 | 3.2 | Hyperparameters HAR model |
| 3 | 3.3 | Confusion matrix HAR |
| 4 | 3.4 | Context-Stress Modifier |
| 5 | 3.5 | Kết quả xác thực HAR trên dữ liệu sinh |
| 6 | 3.6 | 44 features gốc |
| 7 | 3.7 | 13 features cuối + RF importance |
| 8 | 3.8 | Thống kê mô tả 13 features |
| 9 | 3.9 | Kích thước data qua pipeline |
| 10 | 3.10 | Tham số kiến trúc baseline |
| 11 | 3.11 | Top-5 HP tuning trials |
| 12 | 3.12 | So sánh baseline vs tuned config |
| 13 | 3.13 | 5 kiến trúc so sánh |
| 14 | 4.1 | Cấu hình phần cứng/phần mềm |
| 15 | 4.2 | Classification report HAR |
| 16 | 4.3 | Metrics baseline |
| 17 | 4.4 | Best HP vs baseline |
| 18 | 4.5 | Cải thiện sau tuning |
| 19 | 4.6 | Feature importance top-5 (4 methods) |
| 20 | 4.7 | Error by stress level |
| 21 | 4.8 | Error by activity |
| 22 | 4.9 | Error by time period |
| 23 | 4.10 | So sánh 5 models |
| 24 | 4.11 | So sánh với nghiên cứu khác |
