# Giải thích luồng training stress prediction và quan hệ giữa các file

Tài liệu này giải thích các file chính trong thư mục `stress_prediction`:

- `data_pipeline.py`
- `lstm_baseline.py`
- `train_lstm_13features.py`
- `hyperparameter_tuning.py`
- `model_comparison.py`

Mục tiêu là giúp nắm rõ: file nào làm gì, file nào dùng pipeline nào, các mô hình trong `model_comparison.py` có được so sánh công bằng không, và vì sao kết quả ở từng file có thể khác nhau.

## 1. Bức tranh tổng thể

Trong project hiện có hai thế hệ pipeline:

| Nhóm | File chính | Dataset | Số đặc trưng | Vai trò |
|---|---|---:|---:|---|
| Pipeline cũ | `data_pipeline.py`, `lstm_baseline.py` | `optimized_health_data_23features.csv` | 22 input features + target | Baseline LSTM giai đoạn đầu |
| Pipeline final 13 features | `train_lstm_13features.py`, `hyperparameter_tuning.py`, `model_comparison.py` | `optimized_health_data_13features.csv` | 13 input features + target | Baseline final, tuning, benchmark 5 mô hình |

Điểm quan trọng: `train_lstm_13features.py`, `hyperparameter_tuning.py` và `model_comparison.py` đều dùng cùng triết lý pipeline:

```text
Load data
-> Split raw data theo thời gian
-> Encode categorical features, fit trên train
-> Normalize numerical/features, fit trên train
-> Create sequence length = 60
-> Train / validate / test
```

Tuy nhiên, ba file này không import chung một class pipeline. Mỗi file tự định nghĩa lại `DataPreprocessor` hoặc `prepare_data()` riêng. Vì vậy khi nói “dùng chung pipeline”, nên hiểu là chung logic xử lý và nguyên tắc chống leakage, không phải chung một module code duy nhất.

## 2. `data_pipeline.py`

`data_pipeline.py` định nghĩa class `StressDataPipeline`. Đây là pipeline tổng quát ban đầu cho mô hình LSTM baseline cũ.

### Vai trò chính

File này làm 4 việc:

- Load dữ liệu từ CSV.
- Encode các biến phân loại bằng `LabelEncoder`.
- Chuẩn hóa đặc trưng bằng `StandardScaler`.
- Tạo chuỗi thời gian cho LSTM và chia train/validation/test.

### Dataset và feature set

`data_pipeline.py` lấy cấu hình từ `config.py`.

Trong `config.py`, dữ liệu mặc định là:

```python
DATA_FILE = optimized_health_data_23features.csv
```

Danh sách `FEATURE_COLUMNS` gồm 22 đặc trưng đầu vào, ví dụ:

- `Accelerometer_X`, `Accelerometer_Y`, `Accelerometer_Z`
- `Activity`, `Location`
- `Heart_Rate`, `Sleep_Duration`, `Sleep_Quality`
- `Energy_Level`, `Mood_Score`
- Các đặc trưng screen usage rolling/trend
- Social, environment, exercise features

Target là:

```python
TARGET_COLUMN = 'Stress_Level'
```

### Luồng xử lý trong `prepare_data()`

Thứ tự trong `data_pipeline.py` là:

```text
Load full DataFrame
-> preprocess_features(df, fit=True)
-> create_sequences(features, targets)
-> split_data(X, y)
```

Trong đó `preprocess_features(df, fit=True)` fit `LabelEncoder` và `StandardScaler` trên toàn bộ DataFrame trước khi split.

### Điểm cần hiểu khi bảo vệ

Pipeline này thuộc giai đoạn baseline cũ. Nó thuận tiện để chạy nhanh, nhưng không phải pipeline chống leakage tốt nhất vì scaler/encoder được fit trước khi chia train/test.

Nếu hội đồng hỏi “pipeline cuối cùng chống leakage ở đâu?”, không nên chỉ vào `data_pipeline.py`. Nên chỉ vào các file 13-feature, nhất là `train_lstm_13features.py`, `hyperparameter_tuning.py`, `model_comparison.py`, vì các file đó dùng thứ tự:

```text
Split raw -> fit encoder/scaler trên train -> transform val/test
```

## 3. `lstm_baseline.py`

`lstm_baseline.py` là script huấn luyện LSTM baseline cũ dựa trên `StressDataPipeline`.

### Quan hệ với `data_pipeline.py`

File này import trực tiếp:

```python
from data_pipeline import StressDataPipeline
```

Trong `main()`, nó gọi:

```python
pipeline = StressDataPipeline(
    sequence_length=config.SEQUENCE_LENGTH,
    prediction_horizon=config.PREDICTION_HORIZON
)
data = pipeline.prepare_data()
```

Vì vậy toàn bộ dữ liệu train/val/test của `lstm_baseline.py` do `data_pipeline.py` chuẩn bị.

### Kiến trúc mô hình

Class chính là:

```python
LSTMStressPredictor
```

Kiến trúc:

```text
Input: (sequence_length, num_features)
-> LSTM(128)
-> Dropout(0.3)
-> Dense(64, relu)
-> Dropout(0.3)
-> Dense(1, linear)
```

Đây là LSTM một chiều, một lớp LSTM chính. Nó không phải Stacked Bi-LSTM final trong báo cáo.

### Setting training

Các setting lấy từ `config.py`:

| Setting | Giá trị |
|---|---:|
| Sequence length | 60 |
| Prediction horizon | 1 |
| Train/Val/Test | 70/15/15 |
| LSTM units | 128 |
| Dropout | 0.3 |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 100 |
| Early stopping patience | 15 |

Callbacks:

- `EarlyStopping`
- `ModelCheckpoint`
- `ReduceLROnPlateau`
- `TensorBoard`

### Vai trò trong project

File này là baseline giai đoạn đầu để kiểm tra khả năng dự đoán stress bằng LSTM với bộ feature lớn hơn. Trong bản final 13 features, vai trò của nó chủ yếu là lịch sử phát triển/đối chiếu, không phải benchmark chính.

Khi thuyết trình, nếu nói về kết quả final trong Chương 4, nên ưu tiên `train_lstm_13features.py`, `hyperparameter_tuning.py`, `model_comparison.py` thay vì `lstm_baseline.py`.

## 4. `train_lstm_13features.py`

Đây là script huấn luyện baseline final với 13 đặc trưng.

### Vai trò chính

File này tạo kết quả baseline 13-feature trước khi tối ưu siêu tham số.

Kết quả đang có trong:

```text
results/metrics_13features.txt
```

Với kết quả:

| Metric | Giá trị |
|---|---:|
| MAE | 0.6855 |
| RMSE | 0.8723 |
| R2 | 0.9245 |

### Dataset

File dùng:

```python
data/optimized_health_data_13features.csv
```

13 đặc trưng gồm:

- `Hour`
- `Day_of_Week`
- `Activity`
- `Accelerometer_X`
- `Accelerometer_Y`
- `Accelerometer_Z`
- `Heart_Rate`
- `Location`
- `Screen_Usage_Current`
- `Phone_Event_Frequency`
- `Mood_Score`
- `Energy_Level`
- `Sleep_Duration`

Target:

```python
Stress_Level
```

### Pipeline chống leakage

File này tự định nghĩa class:

```python
DataPreprocessor
```

Thứ tự xử lý:

```text
1. Load CSV
2. Split raw data thành train/val/test
3. Fit LabelEncoder trên train, transform val/test
4. Fit StandardScaler trên train, transform val/test
5. Tạo sequence riêng cho từng split
```

Đây là điểm rất quan trọng.

Lý do chống leakage:

- Encoder không nhìn thấy phân phối category từ test khi fit.
- Scaler không dùng mean/std của val/test.
- Sequence được tạo sau khi split nên không có cửa sổ nào trộn dữ liệu train với val/test.

### Cách split

Trong `split_data()`:

```python
train_test_split(..., shuffle=False)
```

Vì `shuffle=False`, dữ liệu được chia theo thứ tự thời gian. Điều này phù hợp với bài toán time-series hơn là random split.

Tỷ lệ:

```text
Train: 70%
Validation: 15%
Test: 15%
```

### Tạo sequence

Hàm:

```python
create_sequences(X, y, seq_length=60)
```

Với mỗi vị trí `i`:

```text
Input X_seq = X[i : i + 60]
Target y_seq = y[i + 60]
```

Nói cách khác, mô hình dùng 60 bước thời gian trước đó để dự đoán stress ở bước kế tiếp sau cửa sổ.

### Kiến trúc baseline 13-feature

Class:

```python
LSTMModel
```

Kiến trúc:

```text
Input(shape=(60, 13))
-> Bidirectional(LSTM(128, return_sequences=True))
-> Dropout(0.3)
-> Bidirectional(LSTM(64))
-> Dropout(0.3)
-> Dense(64, relu)
-> Dropout(0.3)
-> Dense(32, relu)
-> Dense(1)
```

Compile:

```python
optimizer='adam'
loss='mse'
metrics=['mae']
```

Training:

| Setting | Giá trị |
|---|---:|
| Epochs | 50 |
| Batch size | 32 |
| Early stopping patience | 10 |
| Checkpoint | `models/lstm_13features_best.keras` |

### File output

Sau khi chạy, file này lưu:

- Model tốt nhất: `models/lstm_13features_best.keras`
- Scaler: `models/scaler_13features.pkl`
- Label encoders: `models/label_encoder_13features_Activity.pkl`, `models/label_encoder_13features_Location.pkl`
- Metrics: `results/metrics_13features.txt`

## 5. `hyperparameter_tuning.py`

Đây là file tối ưu siêu tham số cho Stacked Bi-LSTM 13-feature bằng Bayesian Optimization.

### Quan hệ với `train_lstm_13features.py`

File này không import `DataPreprocessor` từ `train_lstm_13features.py`, nhưng copy lại cùng logic pipeline:

```text
Split raw
-> Encode fit train
-> Normalize fit train
-> Create sequences
```

Vì vậy, về mặt phương pháp, nó tương thích với baseline 13-feature. Nhưng về mặt code, pipeline bị lặp.

### Search space

Hàm chính để build model:

```python
build_model(hp)
```

Các siêu tham số được tune:

| Hyperparameter | Search space |
|---|---|
| `lstm_units_1` | 64, 128, 256 |
| `lstm_units_2` | 32, 64, 128 |
| `dropout_rate` | 0.1 đến 0.5, step 0.1 |
| `dense_units` | 32, 64, 128 |
| `learning_rate` | 1e-4 đến 1e-2, log scale |

Lưu ý quan trọng: phần docstring có nhắc batch size `[16, 32, 64]`, nhưng trong code Keras Tuner hiện tại batch size không được tune trong `build_model(hp)`. Khi gọi `tuner.search()`, batch size đang cố định:

```python
batch_size=32
```

Vì vậy khi bảo vệ, nếu bị hỏi “batch size có được Bayesian Optimization tối ưu không?”, câu trả lời đúng là:

> Trong code hiện tại, Bayesian Optimization tối ưu units, dropout, dense units và learning rate. Batch size được giữ cố định ở 32 trong quá trình tuning để kiểm soát thí nghiệm.

### Quy trình tuning

Trong `main()`:

```text
1. prepare_data(seq_length=60)
2. run_tuning(max_trials=20, epochs_per_trial=30)
3. Lấy top 3 trials
4. Retrain best model với epochs=80
5. Evaluate trên test set
6. Save model, metrics, history, scaler/encoders
```

Keras Tuner:

```python
kt.BayesianOptimization(
    objective='val_mae',
    max_trials=20,
    num_initial_points=5
)
```

Nghĩa là tuner chọn cấu hình dựa trên `val_mae`, không phải test MAE. Test set chỉ dùng ở bước đánh giá cuối.

### Best hyperparameters hiện tại

Theo `results/hp_tuning/tuning_results.json`:

| Hyperparameter | Giá trị tốt nhất |
|---|---:|
| `lstm_units_1` | 64 |
| `lstm_units_2` | 64 |
| `dropout_rate` | 0.1 |
| `dense_units` | 128 |
| `learning_rate` | 0.01 |

Kết quả tuned trong giai đoạn tuning riêng:

| Metric | Giá trị |
|---|---:|
| MAE | 0.5292 |
| RMSE | 0.7483 |
| R2 | 0.9444 |

### Vì sao kết quả tuning khác model comparison?

`hyperparameter_tuning.py` và `model_comparison.py` đều dùng cấu hình tuned giống nhau về ý tưởng, nhưng chúng là hai lần chạy/đánh giá khác ngữ cảnh:

- `hyperparameter_tuning.py`: chạy Keras Tuner, chọn best HP theo validation, rồi retrain best model.
- `model_comparison.py`: tự build lại 5 mô hình, trong đó mô hình tuned dùng best HP đã biết, rồi train lại trong cùng benchmark với các mô hình khác.

Do deep learning có yếu tố ngẫu nhiên và ngữ cảnh chạy khác nhau, kết quả tuned trong tuning riêng có thể khác benchmark cuối. Trong báo cáo nên tách rõ hai ngữ cảnh này.

## 6. `model_comparison.py`

Đây là file benchmark 5 kiến trúc mô hình trên cùng bộ dữ liệu 13-feature.

### Câu hỏi quan trọng: các model có cùng pipeline không?

Có. Trong `model_comparison.py`, tất cả mô hình dùng cùng một hàm:

```python
prepare_data()
```

Hàm này chạy một lần ở đầu `main()`:

```python
X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
```

Sau đó cùng các mảng `X_train`, `X_val`, `X_test` này được truyền cho toàn bộ 5 model.

Vì vậy, các mô hình trong file này giống nhau ở:

- Cùng dataset: `data/optimized_health_data_13features.csv`
- Cùng số feature: 13
- Cùng target: `Stress_Level`
- Cùng split: 70/15/15 theo thời gian
- Cùng sequence length: 60
- Cùng cách encode: fit `LabelEncoder` trên train
- Cùng cách normalize: fit `StandardScaler` trên train
- Cùng train/val/test arrays
- Cùng loss: MSE
- Cùng metric evaluate: MAE, RMSE, R2
- Cùng batch size: 32
- Cùng max epochs: 80
- Cùng early stopping patience: 15
- Cùng callback `ReduceLROnPlateau`

### Các model khác nhau ở đâu?

Các mô hình khác nhau ở kiến trúc và một số siêu tham số thuộc kiến trúc:

| Model | Kiến trúc chính | Learning rate | Dropout |
|---|---|---:|---:|
| MLP | Flatten + Dense | 0.001 | 0.3 / 0.2 |
| Simple LSTM | 1 LSTM layer | 0.001 | 0.3 / 0.2 |
| Stacked Bi-LSTM Baseline | Bi-LSTM 128 -> 64 | 0.001 | 0.3 |
| Stacked Bi-GRU | Bi-GRU 128 -> 64 | 0.001 | 0.3 |
| Stacked Bi-LSTM Tuned | Bi-LSTM 64 -> 64 | 0.01 | 0.1 |

Mô hình tuned khác learning rate và dropout vì nó đại diện cho cấu hình đã được tối ưu bởi `hyperparameter_tuning.py`. Đây là điểm hợp lý nếu mục tiêu là so sánh “cấu hình tốt nhất sau tuning” với các kiến trúc còn lại.

Nếu hội đồng hỏi “như vậy có công bằng không?”, câu trả lời nên là:

> Công bằng ở mức pipeline dữ liệu, tập train/validation/test, sequence length, budget huấn luyện, loss và metric đánh giá. Tuy nhiên, mô hình tuned được phép dùng siêu tham số tối ưu vì mục tiêu của benchmark là đánh giá hiệu quả sau tối ưu, không phải chỉ so sánh kiến trúc thô. Do đó báo cáo cần nói rõ đây là benchmark giữa các cấu hình mô hình, trong đó một cấu hình là Bi-LSTM đã tuning.

### 5 mô hình trong benchmark

#### Model 1: MLP

```text
Input (60, 13)
-> Flatten
-> Dense(256)
-> Dropout(0.3)
-> Dense(128)
-> Dropout(0.3)
-> Dense(64)
-> Dropout(0.2)
-> Dense(1)
```

MLP không có cơ chế nhớ chuỗi. Nó dùng toàn bộ cửa sổ 60 bước sau khi flatten thành vector lớn.

Vai trò: baseline phi chuỗi để kiểm tra xem temporal modeling có thật sự cần thiết không.

#### Model 2: Simple LSTM

```text
Input (60, 13)
-> LSTM(128)
-> Dropout(0.3)
-> Dense(64)
-> Dropout(0.2)
-> Dense(32)
-> Dense(1)
```

Vai trò: baseline chuỗi đơn giản, một chiều.

#### Model 3: Stacked Bi-LSTM Baseline

```text
Input (60, 13)
-> Bidirectional LSTM(128, return_sequences=True)
-> Dropout(0.3)
-> Bidirectional LSTM(64)
-> Dropout(0.3)
-> Dense(64)
-> Dropout(0.3)
-> Dense(32)
-> Dense(1)
```

Vai trò: cấu hình Bi-LSTM baseline chưa tuning trong benchmark cuối.

#### Model 4: Stacked Bi-GRU

```text
Input (60, 13)
-> Bidirectional GRU(128, return_sequences=True)
-> Dropout(0.3)
-> Bidirectional GRU(64)
-> Dropout(0.3)
-> Dense(64)
-> Dropout(0.3)
-> Dense(32)
-> Dense(1)
```

Vai trò: kiểm tra biến thể recurrent khác LSTM. GRU thường ít cổng hơn LSTM, có thể nhanh hơn hoặc ít tham số hơn trong một số bài toán.

#### Model 5: Stacked Bi-LSTM Tuned

```text
Input (60, 13)
-> Bidirectional LSTM(64, return_sequences=True)
-> Dropout(0.1)
-> Bidirectional LSTM(64)
-> Dropout(0.1)
-> Dense(128)
-> Dropout(0.1)
-> Dense(64)
-> Dense(1)
```

Vai trò: cấu hình tốt nhất lấy từ quá trình Bayesian Optimization.

### Kết quả benchmark cuối

Theo `results/model_comparison/comparison_summary.csv`:

| Model | MAE | RMSE | R2 | Params | Epochs |
|---|---:|---:|---:|---:|---:|
| MLP | 0.9310 | 1.2968 | 0.8331 | 241,153 | 19 |
| Simple LSTM | 0.5213 | 0.7603 | 0.9426 | 83,073 | 17 |
| Stacked Bi-LSTM Baseline | 0.7159 | 0.9698 | 0.9067 | 320,129 | 16 |
| Stacked Bi-GRU | 0.7551 | 0.9103 | 0.9178 | 243,841 | 46 |
| Stacked Bi-LSTM Tuned | 0.4414 | 0.6697 | 0.9555 | 163,585 | 20 |

Kết luận benchmark:

- MLP thấp nhất, chứng minh dữ liệu có yếu tố chuỗi quan trọng.
- Simple LSTM khá tốt dù đơn giản, cho thấy temporal dependency có giá trị.
- Bi-LSTM baseline chưa tuning bị kém hơn kỳ vọng, nhiều khả năng do cấu hình quá lớn/dropout cao.
- Bi-LSTM tuned tốt nhất, vừa MAE thấp nhất vừa R2 cao nhất.
- Tuning giúp giảm số tham số so với Bi-LSTM baseline nhưng tăng hiệu quả dự đoán.

## 7. Quan hệ giữa các file theo luồng chạy

Có thể hình dung như sau:

```text
config.py
  -> data_pipeline.py
      -> lstm_baseline.py

data/optimized_health_data_13features.csv
  -> train_lstm_13features.py
      -> results/metrics_13features.txt
      -> models/lstm_13features_best.keras

data/optimized_health_data_13features.csv
  -> hyperparameter_tuning.py
      -> results/hp_tuning/tuning_results.json
      -> results/metrics_13features_tuned.txt
      -> models/lstm_13features_tuned.keras

data/optimized_health_data_13features.csv
  -> model_comparison.py
      -> results/model_comparison/comparison_summary.csv
      -> results/model_comparison/comparison_results.json
      -> results/model_comparison/model_comparison.png
      -> results/model_comparison/model_comparison_radar.png
```

### Nên hiểu vai trò từng file như thế nào?

| File | Vai trò ngắn gọn |
|---|---|
| `data_pipeline.py` | Pipeline cũ/tổng quát cho 23-feature LSTM baseline |
| `lstm_baseline.py` | Train Simple LSTM baseline cũ dựa trên `data_pipeline.py` |
| `train_lstm_13features.py` | Train Stacked Bi-LSTM baseline final với 13 features |
| `hyperparameter_tuning.py` | Tìm cấu hình Bi-LSTM tốt hơn bằng Bayesian Optimization |
| `model_comparison.py` | Benchmark 5 kiến trúc/cấu hình trên cùng pipeline 13-feature |

## 8. Các điểm dễ bị hỏi khi bảo vệ code

### Câu hỏi 1: “Tại sao nói chống data leakage?”

Vì trong pipeline 13-feature, thứ tự là:

```text
Split raw data trước
-> fit encoder/scaler chỉ trên train
-> transform validation/test
```

Điều này tránh việc thông tin thống kê của test set đi vào scaler hoặc encoder trong quá trình train.

### Câu hỏi 2: “Vậy `data_pipeline.py` có leakage không?”

Nên trả lời thẳng:

> `data_pipeline.py` là pipeline baseline cũ, tiện cho giai đoạn đầu. Pipeline final 13-feature đã sửa thứ tự xử lý để hạn chế leakage. Vì vậy, kết quả final trong báo cáo dựa vào pipeline 13-feature chứ không dựa vào `data_pipeline.py`.

### Câu hỏi 3: “Các model trong `model_comparison.py` có dùng cùng dữ liệu không?”

Có.

Tất cả model dùng cùng output từ một lần gọi:

```python
prepare_data()
```

Nên cùng train/val/test split, cùng scaler/encoder, cùng sequence length và cùng target.

### Câu hỏi 4: “Các model có cùng setting huấn luyện không?”

Cùng ở các điểm:

- Dataset
- Split
- Sequence length
- Batch size
- Max epochs
- Early stopping patience
- Loss
- Metrics
- Test set

Khác ở:

- Kiến trúc
- Số tham số
- Dropout theo cấu hình
- Learning rate của tuned model

Mục tiêu không phải bắt tất cả model có cùng số tham số hoặc cùng dropout, mà là so sánh các cấu hình đại diện trong cùng điều kiện dữ liệu và cùng ngân sách huấn luyện.

### Câu hỏi 5: “Tại sao `hyperparameter_tuning.py` và `model_comparison.py` có kết quả tuned khác nhau?”

Vì đó là hai ngữ cảnh chạy khác nhau:

- `hyperparameter_tuning.py`: chọn best HP bằng Bayesian Optimization rồi retrain/evaluate.
- `model_comparison.py`: train lại mô hình tuned cùng lúc với 4 model khác trong benchmark cuối.

Ngoài ra deep learning có dao động do khởi tạo trọng số, early stopping và quá trình tối ưu. Vì vậy báo cáo đã tách hai kết quả:

- Tuning riêng: MAE 0.5292, RMSE 0.7483, R2 0.9444.
- Benchmark cuối: MAE 0.4414, RMSE 0.6697, R2 0.9555.

### Câu hỏi 6: “Tại sao Simple LSTM tốt hơn Bi-LSTM baseline trong benchmark?”

Theo code và kết quả:

- Simple LSTM có ít tham số hơn.
- Bi-LSTM baseline có 320,129 tham số và dropout 0.3.
- Với dữ liệu semi-synthetic 13 features, Bi-LSTM baseline có thể bị cấu hình chưa phù hợp, học chưa tối ưu.
- Sau tuning, Bi-LSTM giảm units, giảm dropout, tăng dense units và đổi learning rate, nên hiệu quả tốt hơn rõ rệt.

Điểm nên nói:

> Kết quả này cho thấy kiến trúc phức tạp hơn không tự động tốt hơn. Tối ưu siêu tham số và pipeline đánh giá công bằng mới là yếu tố quyết định.

### Câu hỏi 7: “Batch size có được tuning không?”

Trong code hiện tại: không.

Mặc dù docstring có nhắc batch size, phần Keras Tuner đang cố định:

```python
batch_size=32
```

Nên nếu bị hỏi, trả lời:

> Batch size được giữ cố định ở 32 để kiểm soát thí nghiệm. Các siêu tham số được Bayesian Optimization tối ưu gồm số units, dropout, dense units và learning rate.

## 9. Cách nói ngắn gọn khi thuyết trình

Nếu cần nói gọn trong bảo vệ:

> Em có hai lớp pipeline. Pipeline cũ trong `data_pipeline.py` phục vụ baseline ban đầu với 23 features. Với kết quả final, em chuyển sang bộ 13 features và dùng pipeline chống leakage: chia dữ liệu thô theo thời gian trước, sau đó encoder/scaler chỉ fit trên train rồi transform validation/test, cuối cùng mới tạo sequence 60 bước. `train_lstm_13features.py` tạo baseline Stacked Bi-LSTM, `hyperparameter_tuning.py` dùng Bayesian Optimization để tìm cấu hình tốt hơn, còn `model_comparison.py` train 5 kiến trúc trên cùng một tập train/val/test và cùng setting chung để so sánh công bằng. Mô hình tuned khác learning rate/dropout/units vì đó là cấu hình đã tối ưu, nhưng dữ liệu, split, batch size, epochs, callbacks và metric đánh giá là thống nhất.

## 10. Ghi chú kỹ thuật nên nhớ

- `data_pipeline.py` là pipeline dùng `config.py` và dataset 23-feature.
- `train_lstm_13features.py`, `hyperparameter_tuning.py`, `model_comparison.py` dùng dataset 13-feature.
- Pipeline 13-feature chống leakage tốt hơn vì split trước rồi mới fit preprocessing.
- `model_comparison.py` không load model tuned đã train sẵn; nó build lại tuned architecture và train lại trong benchmark.
- Kết quả tuning riêng và benchmark cuối không nên trộn lẫn.
- Trong báo cáo, nếu dùng kết quả benchmark cuối, nên dùng `MAE = 0.4414`, `RMSE = 0.6697`, `R2 = 0.9555`.
- Nếu nói về kết quả tuning riêng, dùng `MAE = 0.5292`, `RMSE = 0.7483`, `R2 = 0.9444`.
