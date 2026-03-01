# Refactored Health Data Generator

## 🔧 Refactoring Overview

Original file `health_data_generator.py` (1852 lines) đã được tách thành **6 core modules** để dễ quản lý và maintain:

## 📁 Architecture

```
Data generator/
├── core/
│   ├── __init__.py                   # Core modules package
│   ├── user_profile.py              # User profile & physiological calculations
│   ├── wisdm_loader.py               # WISDM data loading & management
│   ├── activity_manager.py          # Activity transitions & validation
│   ├── metrics_calculator.py        # Health metrics calculations
│   ├── behavioral_tracker.py        # Behavioral patterns & sequences
│   └── schedule_generator.py        # Daily schedule generation
├── refactored_health_data_generator.py  # Main orchestrator
├── health_data_generator.py         # Original file (for reference)
└── README.md                        # This file
```

## 🎯 Module Responsibilities

### 1. `UserProfile` (113 lines)
- **Responsibility**: Quản lý thông tin cá nhân và tính toán sinh lý
- **Methods**:
  - `calculate_bmr()` - Base Metabolic Rate
  - `calculate_max_heart_rate()` - Maximum HR dựa trên tuổi
  - `calculate_resting_heart_rate()` - Resting HR theo giới tính/tuổi

### 2. `WisdmDataLoader` (115 lines)
- **Responsibility**: Load và quản lý dữ liệu accelerometer từ WISDM
- **Methods**:
  - `load_wisdm_data()` - Load real accelerometer data
  - `get_real_accelerometer_sample()` - Sequential sampling với temporal coherence
  - `_generate_synthetic_accelerometer()` - Fallback synthetic generation

### 3. `ActivityManager` (224 lines)
- **Responsibility**: Quản lý activities, transitions và HAR validation
- **Methods**:
  - `verify_activity_from_accelerometer()` - Validate activity vs accelerometer
  - `validate_har_sequence_consistency()` - HAR model compatibility check
  - `choose_contextual_activity()` - Smart activity selection
  - `get_improved_activity_duration()` - HAR-optimized durations

### 4. `HealthMetricsCalculator` (102 lines)
- **Responsibility**: Tính toán calories, steps, heart rate
- **Methods**:
  - `calculate_hourly_calories()` - Activity-based calorie calculation
  - `calculate_hourly_steps()` - Step counting logic
  - `calculate_heart_rate()` - Dynamic HR based on context
  - `calculate_reaction_time()` - Cognitive performance metrics

### 5. `BehavioralTracker` (284 lines)
- **Responsibility**: Theo dõi behavioral patterns cho LSTM sequences
- **Methods**:
  - `update_behavioral_state()` - Update sequential patterns
  - `calculate_screen_intensity()` - Screen usage modeling
  - `generate_phone_interactions()` - Phone event simulation
  - `get_behavioral_features()` - Extract LSTM features

### 6. `DailyScheduleGenerator` (246 lines)
- **Responsibility**: Tạo lịch trình hàng ngày thực tế
- **Methods**:
  - `generate_improved_daily_schedule()` - HAR-optimized schedules
  - `get_daily_noise_factor()` - Daily variation modeling
  - `generate_life_events()` - Special events simulation

### 7. `RefactoredHealthDataGenerator` (268 lines)
- **Responsibility**: Main orchestrator, coordinates all modules
- **Methods**:
  - `generate_enhanced_dataset()` - Main dataset generation
  - `calculate_enhanced_daily_metrics()` - Aggregate daily metrics
  - `generate_accelerometer_with_variations()` - Coordinated data generation

## ✅ Benefits of Refactoring

### 1. **Maintainability**
- **Before**: 1852 lines trong 1 file → khó tìm và sửa code
- **After**: Max 284 lines per module → dễ dàng navigate và maintain

### 2. **Separation of Concerns**
- **Before**: Tất cả logic trộn lẫn trong 1 class
- **After**: Mỗi module có responsibility rõ ràng và độc lập

### 3. **Testability**
- **Before**: Khó test individual components
- **After**: Có thể unit test từng module riêng biệt

### 4. **Reusability**
- **Before**: Phải copy toàn bộ code để tái sử dụng
- **After**: Có thể import và sử dụng individual modules

### 5. **Extensibility**
- **Before**: Thêm feature mới phải modify large file
- **After**: Chỉ cần extend relevant module hoặc add new module

### 6. **Code Organization**
- **Before**: Methods liên quan scatter khắp file
- **After**: Grouped logically theo functionality

## 🚀 Usage

### Basic Usage
```python
from refactored_health_data_generator import RefactoredHealthDataGenerator

# Initialize với user profile
generator = RefactoredHealthDataGenerator(age=28, gender='Female')

# Generate dataset
df = generator.generate_enhanced_dataset("2024-01-01", 30)
```

### Advanced Usage - Individual Modules
```python
from core.user_profile import UserProfile
from core.wisdm_loader import WisdmDataLoader
from core.activity_manager import ActivityManager

# Use individual components
user = UserProfile(25, 'Male')
bmr = user.calculate_bmr()

wisdm = WisdmDataLoader()
wisdm.load_wisdm_data()
sample = wisdm.get_real_accelerometer_sample('Walking')

activity_mgr = ActivityManager()
next_activity = activity_mgr.choose_contextual_activity(14.5, False, {})
```

## 🔄 Migration Guide

### From Original to Refactored
```python
# OLD WAY (Original)
from health_data_generator import HealthMonitoringDataGenerator
generator = HealthMonitoringDataGenerator()

# NEW WAY (Refactored)
from refactored_health_data_generator import RefactoredHealthDataGenerator
generator = RefactoredHealthDataGenerator()

# API remains the same
df = generator.generate_enhanced_dataset("2024-01-01", 30)
```

## 📊 Performance Impact

- **Memory**: Tương đương (chỉ khác về organization)
- **Speed**: Tương đương với slight overhead từ module calls
- **Features**: 100% preserved, no functionality lost
- **Data Quality**: Identical output với improved code organization

## 🧪 Testing Strategy

### Unit Testing Examples
```python
# Test individual modules
def test_user_profile():
    user = UserProfile(30, 'Female')
    assert 60 <= user.calculate_bmr() <= 90
    assert user.calculate_max_heart_rate() == 190

def test_activity_manager():
    mgr = ActivityManager()
    activity = mgr.choose_contextual_activity(9.0, False, {})
    assert activity in mgr.har_activities

def test_metrics_calculator():
    user = UserProfile(25, 'Male')
    calc = HealthMetricsCalculator(user)
    calories = calc.calculate_hourly_calories('Walking', 1.0, 5)
    assert calories > 100  # Should burn calories walking
```

## 🎯 Future Extensions

### Easy to Add New Features
```python
# Add new module for additional functionality
from core.nutrition_tracker import NutritionTracker
from core.sleep_analyzer import SleepAnalyzer

# Extend existing modules
class EnhancedActivityManager(ActivityManager):
    def add_new_activity(self, activity_name):
        # Easy to extend without modifying core
        pass
```

## 📝 Code Quality Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines per file | 1852 | Max 284 | 85% reduction |
| Cyclomatic complexity | High | Low | Better readability |
| Coupling | Tight | Loose | Better modularity |
| Cohesion | Low | High | Related code grouped |
| Testability | Poor | Good | Easy unit testing |

## 🏆 Summary

Refactoring đã transform một **monolithic 1852-line file** thành **7 focused modules** với:

- ✅ **100% functionality preserved**
- ✅ **Dramatically improved maintainability**
- ✅ **Better code organization**
- ✅ **Enhanced testability**
- ✅ **Future-proof architecture**

Code giờ đây **dễ hiểu, dễ maintain, và dễ extend** cho future requirements!
