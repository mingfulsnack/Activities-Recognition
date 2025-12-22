# TÀI LIỆU HỆ THỐNG GENERATE DATA

## 📋 MỤC LỤC

1. [Tổng quan hệ thống](#tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
3. [Chi tiết từng file](#chi-tiết-từng-file)
4. [Luồng xử lý dữ liệu](#luồng-xử-lý-dữ-liệu)
5. [Cấu hình và tùy chỉnh](#cấu-hình-và-tùy-chỉnh)

---

## 🎯 TỔNG QUAN HỆ THỐNG

### Mục đích
Hệ thống này được thiết kế để **tạo dữ liệu sức khỏe và hoạt động (health & activity data)** thực tế cho người Việt Nam, phục vụ cho:
- **HAR (Human Activity Recognition)**: Nhận diện hoạt động từ dữ liệu accelerometer
- **Stress Prediction**: Dự đoán mức độ stress
- **Health Monitoring**: Theo dõi sức khỏe tổng quan

### Đặc điểm chính
- ✅ **Dữ liệu thực tế**: Mô phỏng lịch trình người Việt Nam (giờ làm, ăn, ngủ)
- ✅ **Accelerometer data thật**: Sử dụng WISDM dataset
- ✅ **Sequential patterns**: Tạo chuỗi dữ liệu liên tục cho LSTM/RNN
- ✅ **Cân bằng cho HAR**: Đảm bảo đủ dữ liệu cho cả 6 hoạt động
- ✅ **Behavioral tracking**: Theo dõi screen time, phone usage, social interactions

### Output
- **File CSV**: `quota_balanced_health_data_30days.csv` 
- **Số lượng**: ~54,000 samples (30 ngày × 1,800 samples/ngày)
- **Tần suất**: 2 samples/phút (mỗi sample = 30 giây)

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

```
generate_and_verify_data/
├── Data generator/
│   ├── core/                          # 📦 Thư viện core modules
│   │   ├── __init__.py               # Package initialization
│   │   ├── activity_manager.py       # Quản lý activities & transitions
│   │   ├── behavioral_tracker.py     # Theo dõi behavioral patterns
│   │   ├── metrics_calculator.py     # Tính toán health metrics
│   │   ├── schedule_generator.py     # Tạo lịch trình hàng ngày
│   │   ├── user_profile.py          # Quản lý thông tin user
│   │   └── wisdm_loader.py          # Load dữ liệu WISDM
│   │
│   ├── refactored_health_data_generator.py  # 🚀 MAIN FILE - Entry point
│   ├── analyze_activity_distribution.py     # 📊 Phân tích phân phối activities
│   ├── validate_accelerometer_with_har.py   # ✅ Validate với HAR model
│   ├── improve_accelerometer_patterns.py    # 🔧 Cải thiện accelerometer patterns
│   │
│   ├── COMPREHENSIVE_README.md        # 📖 Tài liệu tổng quan
│   ├── DEFENSE_QA.md                 # 🛡️ Q&A cho defense
│   └── README.md                     # 📄 Hướng dẫn cơ bản
│
└── data/                             # 📂 Dữ liệu
    ├── WISDM_ar_v1.1_raw.txt        # WISDM accelerometer dataset
    ├── quota_balanced_health_data_30days.csv  # Output chính
    └── realistic_location_health_data_30days.csv  # Output backup
```

---

## 📚 CHI TIẾT TỪNG FILE

### 1. 🚀 `refactored_health_data_generator.py` - MAIN FILE

**Chức năng**: Entry point của toàn bộ hệ thống, điều phối tất cả các module để tạo dataset.

**Các bước xử lý**:

1. **Khởi tạo các components**
   ```python
   user_profile = UserProfile(age=28, gender='Female')
   wisdm_loader = WisdmDataLoader()
   activity_manager = ActivityManager()
   schedule_generator = DailyScheduleGenerator(activity_manager)
   behavioral_tracker = BehavioralTracker()
   metrics_calculator = HealthMetricsCalculator(user_profile)
   ```

2. **Load WISDM data**
   - Đọc file `WISDM_ar_v1.1_raw.txt`
   - Parse accelerometer data cho 6 activities
   - Lưu vào memory để sử dụng

3. **Generate life events**
   - Tạo các sự kiện đặc biệt (sick day, stressful period, vacation)
   - Ảnh hưởng đến stress, energy, mood trong nhiều ngày

4. **Loop qua từng ngày** (30 ngày)
   ```python
   for current_date in date_range:
       daily_schedule = schedule_generator.generate_improved_daily_schedule(
           date=current_date,
           life_events=life_events
       )
   ```

5. **Generate samples mỗi ngày**
   - Từ lịch trình, tạo samples mỗi 30 giây
   - Mỗi sample chứa: activity, location, accelerometer, health metrics, behavioral features

6. **Validate & Save**
   - Kiểm tra HAR sequence consistency
   - Lưu vào CSV file

**Key parameters**:
```python
DAYS_TO_GENERATE = 30
SAMPLES_PER_DAY = 1800  # 30 ngày × 24h × 2 samples/phút
```

---

### 2. 📦 `core/activity_manager.py`

**Chức năng**: Quản lý 6 activities (Sitting, Standing, Walking, Jogging, Upstairs, Downstairs) và chuyển đổi giữa chúng.

**Các thành phần chính**:

#### 2.1 Activity Durations
```python
self.activity_durations = {
    'Sitting': (15, 60),      # 15-60 phút
    'Standing': (3, 20),      # 3-20 phút
    'Walking': (10, 45),      # 10-45 phút
    'Jogging': (15, 90),      # 15-90 phút
    'Upstairs': (1, 5),       # 1-5 phút
    'Downstairs': (1, 5)      # 1-5 phút
}
```

#### 2.2 Activity Transitions
Ma trận xác suất chuyển đổi giữa các activities:
```python
self.activity_transitions = {
    'Sitting': {
        'Standing': 0.4,   # 40% chuyển sang Standing
        'Walking': 0.3,    # 30% chuyển sang Walking
        'Sitting': 0.15    # 15% tiếp tục Sitting
    },
    # ... tương tự cho các activities khác
}
```

#### 2.3 Methods

**`verify_activity_from_accelerometer(x, y, z, intended_activity)`**
- Kiểm tra xem accelerometer data có khớp với activity đã chọn không
- Tính `magnitude = sqrt(x² + y² + z²)`
- Phân loại dựa trên magnitude:
  - `< 9.5`: Sitting
  - `9.5-10.5`: Standing
  - `10.5-12.0`: Walking
  - `12.0-14.0`: Upstairs/Downstairs
  - `>= 14.0`: Jogging

**`validate_har_sequence_consistency(data_sequence)`**
- Kiểm tra 180 samples liên tục có hợp lý không
- Đảm bảo không có quá nhiều activities khác nhau trong 1 sequence
- Validate magnitude trung bình khớp với activity

**`choose_contextual_activity(current_time, is_weekend, day_context, previous_activity)`**
- Chọn activity dựa trên:
  - Thời gian trong ngày (sáng/trưa/chiều/tối)
  - Ngày thường vs cuối tuần
  - Activity trước đó (dùng transition probabilities)

---

### 3. 📦 `core/schedule_generator.py`

**Chức năng**: Tạo lịch trình hoạt động hàng ngày thực tế cho người Việt Nam.

**Các thành phần chính**:

#### 3.1 Daily Patterns
```python
self.daily_patterns = {
    'wake_up_time': (6, 8),      # Thức dậy 6-8h
    'work_start': (8, 9),        # Đi làm 8-9h
    'lunch_time': (12, 13),      # Ăn trưa 12-13h
    'work_end': (17, 18),        # Tan làm 17-18h
    'dinner_time': (18, 20),     # Ăn tối 18-20h
    'sleep_time': (22, 24),      # Ngủ 22-24h
    'exercise_days': [1, 3, 5, 6]  # Thứ 2, 4, 6, 7
}
```

#### 3.2 Daily Activity Quotas
Để đạt 85-95% HAR accuracy, mỗi ngày cần:
```python
target_quotas = {
    'Sitting': 4.8,      # 4.8 giờ (30%)
    'Walking': 4.0,      # 4.0 giờ (25%)
    'Standing': 3.2,     # 3.2 giờ (20%)
    'Jogging': 1.6,      # 1.6 giờ (10%)
    'Upstairs': 1.3,     # 1.3 giờ (8%)
    'Downstairs': 1.1    # 1.1 giờ (7%)
}
```

#### 3.3 Methods

**`get_daily_noise_factor(date)`**
- Tạo các yếu tố ngẫu nhiên cho từng ngày
- Bao gồm: sleep quality, stress, mood, energy, weather
- Seed theo ngày để reproducible

**`generate_life_events(start_date, end_date)`**
- Tạo các sự kiện đặc biệt:
  - `sick_day`: Ốm (tăng stress, giảm energy)
  - `stressful_period`: Deadline công việc (tăng stress)
  - `vacation_day`: Nghỉ phép (giảm stress, tăng mood)

**`generate_improved_daily_schedule(date, life_events)`**
- Tạo lịch trình chi tiết cho 1 ngày
- Anti-sitting logic: Bắt buộc nghỉ sau 45-90 phút ngồi
- Quota-aware: Ưu tiên activities còn thiếu quota
- Return: List of (activity, duration, location, start_time)

**`_choose_quota_aware_activity(...)`**
- Chọn activity dựa trên quota còn thiếu
- Ưu tiên activities có urgency cao nhất
- Ngăn sitting vượt quá quota

**`_determine_enhanced_location(...)`**
- Xác định location hợp lý cho activity:
  - `6-9h`: home (sáng ở nhà)
  - `9-17h`: work (giờ làm)
  - `17-19h`: commute, gym, outdoor
  - `19-22h`: home, social

---

### 4. 📦 `core/behavioral_tracker.py`

**Chức năng**: Theo dõi behavioral patterns để tạo sequential data cho LSTM.

**Behavioral State**:
```python
self.behavioral_state = {
    'recent_screen_usage': [],      # 30 phút gần nhất
    'phone_interaction_history': [], # 2 giờ gần nhất
    'social_activity_timeline': [],  # 4 giờ gần nhất
    'stress_accumulation': [],       # 6 giờ gần nhất
    'activity_transitions': [],
    'environmental_changes': []
}
```

#### Methods

**`update_behavioral_state(timestamp, current_data, activity, location)`**
- Cập nhật tất cả behavioral states
- Tự động xóa dữ liệu cũ (giữ trong time window)

**`calculate_screen_intensity(activity, location, stress_level)`**
- Tính screen usage intensity [0-1]
- Base: `Sitting=0.7, Standing=0.4, Walking=0.2, Jogging=0.05`
- Modifier: location (home=1.3, work=1.1), stress

**`generate_phone_interactions(timestamp, activity, stress_level)`**
- Tạo phone unlock/notification events
- Tần suất dựa trên activity và stress
- Return: List of events với type, duration, urgency

**`calculate_social_interaction(timestamp, activity, location)`**
- Tính social interaction level [0-1]
- Cao nhất: `social location, 17-22h, weekend`
- Thấp nhất: `home, đêm khuya, Jogging`

**`get_behavioral_features(timestamp)`**
- Trích xuất features từ behavioral state:
  - `avg_screen_last_30min`, `max_screen_last_30min`
  - `phone_unlocks_last_30min`, `avg_phone_duration`
  - `social_interaction_avg_2h`, `social_interaction_trend`
  - `stress_trend`, `stress_velocity`

---

### 5. 📦 `core/metrics_calculator.py`

**Chức năng**: Tính toán các health metrics (calories, steps, heart rate, mood, stress).

#### Methods

**`calculate_hourly_calories(activity, duration_hours, stress_level, base_metabolic_rate)`**
- Tính calories tiêu thụ theo giờ
- BMR (Base Metabolic Rate):
  - Female: `(10×52) + (6.25×155) - (5×age) - 161`
  - Male: `(10×62) + (6.25×168) - (5×age) + 5`
- Activity multipliers:
  ```python
  'Sitting': BMR × 1.0
  'Standing': BMR × 1.2
  'Walking': BMR × 3.0
  'Jogging': BMR × 8.0
  'Upstairs': BMR × 5.0
  'Downstairs': BMR × 3.5
  ```
- Stress modifier: `1 + (stress_level - 4) × 0.05`

**`calculate_hourly_steps(activity, duration_hours, energy_level)`**
- Tính bước chân theo giờ
- Steps per hour:
  ```python
  'Sitting': 2
  'Standing': 10
  'Walking': 1500
  'Jogging': 4800
  'Upstairs': 400
  'Downstairs': 300
  ```
- Target: 8,000-15,000 steps/ngày

**`calculate_heart_rate(activity, stress_level, base_hr, energy_level)`**
- Tính nhịp tim
- Base HR = Resting HR (từ user profile)
- Activity modifiers:
  ```python
  'Sitting': +0
  'Standing': +6
  'Walking': +18
  'Jogging': +40
  'Upstairs': +28
  'Downstairs': +22
  ```
- Stress effect: `+3 bpm per stress level`
- Energy effect: Low energy → higher HR

**`calculate_mood_score(base_mood_factor, hour, activity, location, stress_level)`**
- Tính mood [1-10]
- Daily rhythm:
  - `< 7h`: -0.2 (buồn ngủ)
  - `7-12h`: +0.5 (tăng dần)
  - `12-14h`: +0.8 (peak sau ăn trưa)
  - `14-16h`: +0.2 (afternoon dip)
  - `16-19h`: +0.5 (recovery)
  - `19-22h`: +0.3 (relaxed)
- Activity effects: `Jogging=+1.2, Walking=+0.5, Sitting=-0.2`
- Location effects: `social=+1.0, outdoor=+0.8, work=-0.3`
- Stress effect: `-0.3 per stress level`

**`calculate_realistic_stress_level(...)`**
- Tính stress level [1-9] thực tế
- Daily rhythm (office worker):
  - `< 7h`: -1.0 (thoải mái buổi sáng)
  - `9-12h`: +0.5 (bắt đầu căng thẳng)
  - `12-13h`: -0.3 (nghỉ trưa)
  - `13-17h`: +1.0 (peak stress)
  - `17-18h`: +0.5 (rush hour)
  - `19-22h`: -1.0 (thư giãn tối)
- Activity effects: `Jogging=-0.8, Walking=-0.3, Sitting=+0.2`
- Location effects: `work=+1.5, commute=+1.0, home=-0.5`
- Heart rate correlation: `>85bpm → +0.5 stress`
- Sleep quality: Poor sleep → +2 stress
- Stress momentum: Stress có xu hướng duy trì

---

### 6. 📦 `core/user_profile.py`

**Chức năng**: Quản lý thông tin cá nhân và tính các metrics sinh lý cơ bản.

**Profile Structure**:
```python
self.profile = {
    'Age': 28,
    'Gender': 'Female',
    'base_sleep_duration': 7.5,
    'base_screen_time': 8.0,
    'base_stress_level': 4,
    'base_reaction_time': 380.0
}
```

#### Methods

**`calculate_bmr()`**
- Tính Base Metabolic Rate (cal/hour)
- Mifflin-St Jeor Equation:
  - Female: `(10×52) + (6.25×155) - (5×age) - 161`
  - Male: `(10×62) + (6.25×168) - (5×age) + 5`
- Giả định cân nặng/chiều cao trung bình người Việt
- Return: 60-90 cal/hour

**`calculate_max_heart_rate()`**
- Max HR = `220 - age`
- Ví dụ: age=28 → Max HR = 192 bpm

**`calculate_resting_heart_rate()`**
- Resting HR dựa trên age và gender:
  - Female < 30: 68 bpm
  - Female 30-50: 70 bpm
  - Female > 50: 72 bpm
  - Male thấp hơn ~3 bpm

---

### 7. 📦 `core/wisdm_loader.py`

**Chức năng**: Load và quản lý dữ liệu accelerometer thật từ WISDM dataset.

**WISDM Dataset**:
- File: `data/WISDM_ar_v1.1_raw.txt`
- Format: `user,activity,timestamp,x,y,z;`
- Khoảng 1.1 triệu samples từ 36 users
- 6 activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing

#### Methods

**`load_wisdm_data()`**
- Đọc file WISDM
- Parse từng dòng thành `[x, y, z]`
- Group theo activity
- Return: Dictionary `{activity: [[x,y,z], ...]}`

**`get_real_accelerometer_sample(activity, add_noise=True)`**
- Lấy accelerometer sample thật từ WISDM
- **Sequential sampling** (không random):
  - Dùng index để lấy sample tuần tự
  - Đảm bảo temporal consistency trong activity segments
  - Reset index khi hết data
- Add minimal noise (0.05) để tránh lặp hoàn toàn
- Fallback: Nếu không có data → synthetic

**`_generate_synthetic_accelerometer(activity)`**
- Tạo accelerometer synthetic nếu không có WISDM data
- Dựa trên pattern phân tích từ WISDM:
  ```python
  'Sitting': {
      'x_base': 1.856, 'x_var': 4.759,
      'y_base': 1.853, 'y_var': 3.258,
      'z_base': 6.560, 'z_var': 3.736
  }
  'Jogging': {
      'x_base': -0.219, 'x_var': 9.168,
      'y_base': 5.434, 'y_var': 9.217,
      'z_base': -0.150, 'z_var': 5.847
  }
  # ... tương tự cho các activities khác
  ```
- Dùng Gaussian distribution: `x = N(x_base, x_var)`
- Clip vào range `[-20, 20]`

---

### 8. 📊 `analyze_activity_distribution.py`

**Chức năng**: Phân tích phân phối activities trong dataset đã generate, tìm vấn đề imbalance.

**Phân tích**:

1. **Current Distribution**
   - Đếm số samples mỗi activity
   - Tính phần trăm
   - Tính giờ/ngày tương ứng

2. **HAR Segment Analysis**
   - HAR cần segments (180 samples/segment)
   - Minimum 50 segments/activity cho training tốt
   - Check từng activity có đủ segments không

3. **Problem Identification**
   - Sitting dominance: Quá nhiều Sitting (>50%)
   - Insufficient sequences: Thiếu Jogging, Upstairs, Downstairs
   - Poor distribution: Chênh lệch quá lớn giữa activities

4. **Target vs Current Comparison**
   ```python
   target_distribution = {
       'Sitting': 30,      # Target 30%
       'Walking': 25,
       'Standing': 20,
       'Jogging': 10,
       'Upstairs': 8,
       'Downstairs': 7
   }
   ```

5. **Required Changes**
   - Daily time allocation changes
   - HAR segment targets
   - Generation improvements needed

**Usage**:
```bash
python analyze_activity_distribution.py
```

**Output**:
- Báo cáo chi tiết về phân phối hiện tại
- So sánh với target
- Đề xuất thay đổi cụ thể

---

### 9. ✅ `validate_accelerometer_with_har.py`

**Chức năng**: Validate dữ liệu accelerometer đã generate với HAR model đã train (96% accuracy).

**Process**:

1. **Load HAR Model**
   ```python
   model = tf.keras.models.load_model('classificator_model.keras')
   ```

2. **Prepare Data**
   - Load generated CSV
   - Chuyển sang HAR format:
     ```python
     {
         'user': 1,
         'activity': activity,
         'timestamp': index,
         'x-axis': Accelerometer_X,
         'y-axis': Accelerometer_Y,
         'z-axis': Accelerometer_Z
     }
     ```

3. **Create Segments**
   - Sliding window: 180 samples
   - Step: TIME_STEP
   - Group by activity để tạo continuous segments

4. **Normalize Data**
   - StandardScaler fit_transform
   - Reshape: `(n_segments, 180, 3)`

5. **Validate with HAR Model**
   - Predict: `predictions = model.predict(X)`
   - Calculate overall accuracy
   - Per-activity accuracy
   - Confusion matrix

6. **Analyze Quality**
   - Accelerometer statistics (mean, std, min, max)
   - Per-activity accelerometer patterns
   - Find problematic cases (most confident wrong predictions)

**Expected Accuracy**:
- Overall: 85-95%
- Walking: 80%+
- Jogging: 85%+
- Sitting/Standing: 90%+
- Upstairs/Downstairs: 70%+

**Usage**:
```bash
python validate_accelerometer_with_har.py
```

---

### 10. 🔧 `improve_accelerometer_patterns.py`

**Chức năng**: Cải thiện accelerometer patterns dựa trên phân tích WISDM dataset.

**Process**:

1. **Analyze WISDM Statistics**
   - Tính mean, std cho từng activity
   - Tìm magnitude ranges
   - Phân tích variance patterns

2. **Generate Improved Patterns**
   - Tạo patterns mới dựa trên stats thực
   - Test với synthetic generation
   - So sánh với WISDM original

3. **Validate Improvements**
   - So sánh magnitude distributions
   - Check activity separability
   - Verify với HAR model

**Output**:
- Updated patterns cho `_generate_synthetic_accelerometer()`
- Statistics report
- Recommendations

---

## 🔄 LUỒNG XỬ LÝ DỮ LIỆU

### Flow Chart

```
START
  ↓
Initialize Components
  ├─ UserProfile(age=28, gender='Female')
  ├─ WisdmDataLoader()
  ├─ ActivityManager()
  ├─ ScheduleGenerator(activity_manager)
  ├─ BehavioralTracker()
  └─ MetricsCalculator(user_profile)
  ↓
Load WISDM Data
  └─ wisdm_loader.load_wisdm_data()
  ↓
Generate Life Events (30 days)
  └─ schedule_generator.generate_life_events(start, end)
  ↓
FOR each day (1-30):
  ↓
  Get Daily Noise Factor
    └─ schedule_generator.get_daily_noise_factor(date)
  ↓
  Generate Daily Schedule
    ├─ Determine wake/sleep times
    ├─ Set daily quotas (Sitting, Walking, ...)
    ├─ Initialize day_context (sleep_quality, energy, stress, ...)
    └─ Generate activity segments (quota-aware)
  ↓
  FOR each time slot (00:00-23:59):
    ↓
    Choose Activity (quota-aware)
      ├─ Check quota urgency
      ├─ Anti-sitting logic
      ├─ Contextual selection
      └─ Return: activity, duration, location
    ↓
    Generate Samples (every 30 seconds)
      ↓
      Get Accelerometer Data
        └─ wisdm_loader.get_real_accelerometer_sample(activity)
      ↓
      Verify Activity from Accelerometer
        └─ activity_manager.verify_activity_from_accelerometer(x, y, z, activity)
      ↓
      Calculate Health Metrics
        ├─ Calories (metrics_calculator.calculate_hourly_calories)
        ├─ Steps (metrics_calculator.calculate_hourly_steps)
        ├─ Heart Rate (metrics_calculator.calculate_heart_rate)
        ├─ Mood (metrics_calculator.calculate_mood_score)
        └─ Stress (metrics_calculator.calculate_realistic_stress_level)
      ↓
      Update Behavioral State
        └─ behavioral_tracker.update_behavioral_state(...)
      ↓
      Get Behavioral Features
        └─ behavioral_tracker.get_behavioral_features(timestamp)
      ↓
      Create Sample Row
        └─ {Timestamp, Activity, Location, Accelerometer_X/Y/Z, 
            Calories, Steps, Heart_Rate, Sleep_Hours, Mood_Score, 
            Stress_Level, Screen_Time, Social_Interaction, ...}
      ↓
      Append to Dataset
  ↓
  END FOR (time slots)
↓
END FOR (days)
  ↓
Validate HAR Consistency
  └─ activity_manager.validate_har_sequence_consistency(dataset)
  ↓
Save to CSV
  └─ df.to_csv('quota_balanced_health_data_30days.csv')
  ↓
END
```

---

## ⚙️ CẤU HÌNH VÀ TÙY CHỈNH

### 1. Thay đổi số ngày generate

**File**: `refactored_health_data_generator.py`
```python
DAYS_TO_GENERATE = 30  # Thay đổi số ngày ở đây
```

### 2. Thay đổi user profile

**File**: `refactored_health_data_generator.py`
```python
user_profile = UserProfile(
    age=28,           # Thay đổi tuổi
    gender='Female'   # 'Female' hoặc 'Male'
)
```

### 3. Thay đổi daily patterns

**File**: `core/schedule_generator.py`
```python
self.daily_patterns = {
    'wake_up_time': (6, 8),      # (min_hour, max_hour)
    'work_start': (8, 9),
    'lunch_time': (12, 13),
    'work_end': (17, 18),
    'dinner_time': (18, 20),
    'sleep_time': (22, 24),
    'exercise_days': [1, 3, 5, 6]  # 0=Monday, 6=Sunday
}
```

### 4. Thay đổi activity quotas

**File**: `core/schedule_generator.py`
```python
target_quotas = {
    'Sitting': 4.8,      # hours/day
    'Walking': 4.0,
    'Standing': 3.2,
    'Jogging': 1.6,
    'Upstairs': 1.3,
    'Downstairs': 1.1
}
```

### 5. Thay đổi activity durations

**File**: `core/activity_manager.py`
```python
self.activity_durations = {
    'Sitting': (15, 60),      # (min_minutes, max_minutes)
    'Standing': (3, 20),
    'Walking': (10, 45),
    'Jogging': (15, 90),
    'Upstairs': (1, 5),
    'Downstairs': (1, 5)
}
```

### 6. Thay đổi activity transitions

**File**: `core/activity_manager.py`
```python
self.activity_transitions = {
    'Sitting': {
        'Standing': 0.4,   # Probability từ Sitting → Standing
        'Walking': 0.3,
        # ... thay đổi probabilities
    },
    # ...
}
```

### 7. Thay đổi calories multipliers

**File**: `core/metrics_calculator.py`
```python
activity_calories = {
    'Sitting': base_metabolic_rate * 1.0,    # multiplier
    'Standing': base_metabolic_rate * 1.2,
    'Walking': base_metabolic_rate * 3.0,
    'Jogging': base_metabolic_rate * 8.0,
    'Upstairs': base_metabolic_rate * 5.0,
    'Downstairs': base_metabolic_rate * 3.5
}
```

### 8. Thay đổi steps per hour

**File**: `core/metrics_calculator.py`
```python
activity_steps_per_hour = {
    'Sitting': 2,
    'Standing': 10,
    'Walking': 1500,
    'Jogging': 4800,
    'Upstairs': 400,
    'Downstairs': 300
}
```

### 9. Enable/Disable life events

**File**: `refactored_health_data_generator.py`
```python
# Để tắt life events:
life_events = {}  # Empty dictionary

# Hoặc giữ nguyên để bật:
life_events = schedule_generator.generate_life_events(start_date, end_date)
```

### 10. Thay đổi sampling frequency

**File**: `refactored_health_data_generator.py`
```python
sample_interval = timedelta(seconds=30)  # Thay đổi interval ở đây
# Ví dụ: 60 seconds → 1 sample/phút
```

---

## 📊 OUTPUT DATA SCHEMA

### CSV Columns

| Column | Type | Range/Values | Description |
|--------|------|--------------|-------------|
| `Timestamp` | datetime | `2024-01-01 00:00:00` | Thời điểm sample |
| `Activity` | string | `Sitting, Standing, Walking, Jogging, Upstairs, Downstairs` | Hoạt động hiện tại |
| `Location` | string | `home, work, commute, gym, outdoor, social` | Địa điểm |
| `Accelerometer_X` | float | `-20 to 20` | Gia tốc trục X (m/s²) |
| `Accelerometer_Y` | float | `-20 to 20` | Gia tốc trục Y (m/s²) |
| `Accelerometer_Z` | float | `-20 to 20` | Gia tốc trục Z (m/s²) |
| `Calories_Burned` | float | `0.5 to 200` | Calories tiêu thụ (cal) |
| `Steps_Count` | int | `0 to 100` | Số bước chân |
| `Heart_Rate` | int | `55 to 192` | Nhịp tim (bpm) |
| `Sleep_Hours` | float | `0 to 24` | Số giờ ngủ tích lũy |
| `Mood_Score` | float | `1.0 to 10.0` | Điểm mood |
| `Stress_Level` | float | `1.0 to 9.0` | Mức độ stress |
| `Screen_Time_Minutes` | float | `0 to 60` | Thời gian xem màn hình (phút) |
| `Social_Interaction_Level` | float | `0.0 to 1.0` | Mức độ tương tác xã hội |
| `Reaction_Time_ms` | float | `250 to 650` | Thời gian phản ứng (ms) |
| `Phone_Unlocks` | int | `0 to 20` | Số lần mở khóa điện thoại |
| `Notifications_Count` | int | `0 to 50` | Số thông báo |
| `Age` | int | `18 to 100` | Tuổi |
| `Gender` | string | `Male, Female` | Giới tính |

### Sample Data

```csv
Timestamp,Activity,Location,Accelerometer_X,Accelerometer_Y,Accelerometer_Z,Calories_Burned,Steps_Count,Heart_Rate,Sleep_Hours,Mood_Score,Stress_Level,Screen_Time_Minutes,Social_Interaction_Level,Reaction_Time_ms,Phone_Unlocks,Notifications_Count,Age,Gender
2024-01-01 06:15:00,Sitting,home,1.82,1.95,6.43,3.2,1,68,7.5,5.3,3.2,15.2,0.12,385.3,2,5,28,Female
2024-01-01 06:15:30,Sitting,home,1.91,1.87,6.51,3.1,0,68,7.5,5.3,3.2,15.5,0.11,386.1,2,5,28,Female
2024-01-01 06:16:00,Standing,home,-1.15,8.99,0.62,4.5,3,74,7.5,5.4,3.1,8.2,0.15,382.7,2,5,28,Female
...
```

---

## 🎯 KẾT QUẢ MONG ĐỢI

### HAR Model Accuracy
- **Overall**: 85-95%
- **Sitting**: 90%+
- **Standing**: 88%+
- **Walking**: 80%+
- **Jogging**: 85%+
- **Upstairs**: 70%+
- **Downstairs**: 70%+

### Activity Distribution
- **Sitting**: ~30% (4.8h/day)
- **Walking**: ~25% (4.0h/day)
- **Standing**: ~20% (3.2h/day)
- **Jogging**: ~10% (1.6h/day)
- **Upstairs**: ~8% (1.3h/day)
- **Downstairs**: ~7% (1.1h/day)

### Data Quality
- ✅ Temporal consistency: Activities thay đổi tự nhiên
- ✅ Accelerometer realistic: Từ WISDM dataset thật
- ✅ Health metrics correlation: HR ↑ khi Jogging, Stress ↓ khi休息
- ✅ Behavioral patterns: Screen time cao khi Sitting, thấp khi Jogging
- ✅ Sequential data: Phù hợp cho LSTM/RNN

---

## 🐛 TROUBLESHOOTING

### Problem 1: WISDM data not found
**Error**: `⚠️ WISDM data not found: ...`

**Solution**: 
```bash
# Download WISDM dataset
# Place WISDM_ar_v1.1_raw.txt in data/ folder
```

### Problem 2: Low HAR accuracy
**Possible causes**:
- Sitting quá nhiều (>50%)
- Thiếu Upstairs/Downstairs samples
- Accelerometer patterns không realistic

**Solution**:
```bash
# 1. Phân tích distribution
python analyze_activity_distribution.py

# 2. Validate với HAR model
python validate_accelerometer_with_har.py

# 3. Điều chỉnh quotas trong schedule_generator.py
```

### Problem 3: Memory error
**Error**: `MemoryError` khi generate nhiều ngày

**Solution**:
```python
# Giảm số ngày hoặc process theo batches
DAYS_TO_GENERATE = 7  # Thay vì 30

# Hoặc save incremental
if len(dataset) > 10000:
    df = pd.DataFrame(dataset)
    df.to_csv(f'batch_{batch_num}.csv')
    dataset = []
```

### Problem 4: Inconsistent accelerometer
**Symptom**: HAR accuracy thấp, activities bị misclassify

**Solution**:
```python
# Enable verification
verified_activity = activity_manager.verify_activity_from_accelerometer(
    x, y, z, intended_activity
)

# Check sequence consistency
is_valid = activity_manager.validate_har_sequence_consistency(dataset)
```

---

## 📖 REFERENCES

1. **WISDM Dataset**: [http://www.cis.fordham.edu/wisdm/dataset.php](http://www.cis.fordham.edu/wisdm/dataset.php)
2. **HAR with Deep Learning**: [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
3. **Mifflin-St Jeor Equation**: BMR calculation
4. **Heart Rate Zones**: [American Heart Association](https://www.heart.org/)

---

## 📝 NOTES

- **Reproducibility**: Sử dụng seed theo date để reproducible
- **Performance**: ~5-10 phút để generate 30 ngày (~54,000 samples)
- **Storage**: CSV file ~15-20 MB cho 30 ngày
- **Validation**: Luôn validate với HAR model sau khi generate
- **Customization**: Có thể customize mọi thứ (patterns, quotas, transitions, metrics)

---

## 🚀 QUICK START

```bash
# 1. Install dependencies
pip install pandas numpy tensorflow scikit-learn

# 2. Prepare WISDM data
# Download và đặt vào data/WISDM_ar_v1.1_raw.txt

# 3. Generate data
cd "generate_and_verify_data/Data generator"
python refactored_health_data_generator.py

# 4. Analyze distribution
python analyze_activity_distribution.py

# 5. Validate với HAR model
python validate_accelerometer_with_har.py
```

---

**Tài liệu được tạo bởi**: AI Assistant  
**Ngày cập nhật**: 2025-12-05  
**Version**: 1.0
