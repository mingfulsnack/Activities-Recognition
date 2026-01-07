# Dataset Selection Guide - UPDATED

## 🎯 Recommended: Use 23-Feature Dataset

Sau khi thảo luận, chúng ta quyết định **GIỮ LẠI Accelerometer X, Y, Z** để có flexibility hơn.

---

## 📊 Available Datasets

### **1. Full Dataset (44 features)** - Archive
- File: `quota_balanced_health_data_30days_v2.csv`
- Use: Backup, full data for future research
- ❌ Not recommended for training (too many features)

### **2. Optimized Dataset (23 features)** ⭐ **RECOMMENDED**
- File: `optimized_health_data_23features.csv`
- Features: Accelerometer (3) + Core (9) + Behavioral (7) + Environmental (4)
- ✅ **USE THIS for research and model development**

### **3. Minimal Dataset (20 features)** - Alternative
- File: `optimized_health_data_20features_v2.csv`
- No accelerometer data
- Use: If you only want to predict from high-level features

---

## 🎯 Why 23 Features? (20 + X,Y,Z)

### **Advantages:**

1. **Single Dataset** - No need to maintain multiple versions
2. **Flexibility** - Can experiment with different architectures:
   - End-to-end: Sensor → Stress
   - Multi-task: Sensor → Activity + Stress
   - Two-stage: Sensor → Activity → Stress
3. **Realistic** - Real deployment will have sensor data
4. **Still Optimized** - Reduced 47.7% features (44 → 23)

### **Trade-offs:**

- +3 features compared to 20-feature version
- Still achieves goal of reducing complexity
- Better for research comparison

---

## 🏗️ Model Architectures You Can Try

### **Architecture 1: Two-Stage (Baseline)**
```python
# Stage 1: HAR
X,Y,Z → LSTM → Activity (6 classes)

# Stage 2: Stress
Activity + Context (20) → LSTM → Stress
```

### **Architecture 2: Multi-Task Learning** ⭐
```python
Input: [X,Y,Z] + Context (20)
  ↓
Shared LSTM Encoder
  ↓
├─→ Activity Head (classification)
└─→ Stress Head (regression)
```

### **Architecture 3: End-to-End**
```python
Input: [X,Y,Z] + Context (20)
  ↓
Deep LSTM/Transformer
  ↓
Output: Stress Level
(Activity as intermediate representation)
```

---

## 📋 Feature List (23)

### **Sensor Data (3)**
1. Accelerometer_X
2. Accelerometer_Y
3. Accelerometer_Z

### **Core Features (9)**
4. Timestamp
5. Activity
6. Location
7. Stress_Level (target)
8. Heart_Rate
9. Sleep_Duration
10. Sleep_Quality
11. Energy_Level
12. Mood_Score

### **Behavioral Sequences (7)**
13. Screen_Usage_Current
14. Screen_Usage_15min_Avg
15. Screen_Usage_Trend
16. Phone_Usage_Intensity
17. Phone_Event_Frequency
18. Social_Current_Level
19. Social_1hour_Avg

### **Environmental Context (4)**
20. Ambient_Light
21. Noise_Level
22. Weather_Condition
23. Exercise_Minutes

---

## 🚀 Quick Start

### **Load Dataset**
```python
import pandas as pd

# RECOMMENDED
df = pd.read_csv('data/optimized_health_data_23features.csv')

print(f"Shape: {df.shape}")
print(f"Features: {list(df.columns)}")
```

### **Prepare for Training**
```python
# Sensor features
sensor_cols = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']

# Context features
context_cols = [
    'Heart_Rate', 'Sleep_Duration', 'Sleep_Quality', 'Energy_Level',
    'Mood_Score', 'Screen_Usage_Current', 'Screen_Usage_15min_Avg',
    'Screen_Usage_Trend', 'Phone_Usage_Intensity', 'Phone_Event_Frequency',
    'Social_Current_Level', 'Social_1hour_Avg', 'Ambient_Light',
    'Noise_Level', 'Weather_Condition', 'Exercise_Minutes'
]

# Target
target_col = 'Stress_Level'

# Optional: Activity label
activity_col = 'Activity'
```

---

## 📝 Summary

| Dataset | Features | Use Case |
|---------|----------|----------|
| 44-field | 44 | Archive/Backup |
| **23-field** ⭐ | 23 | **Training & Research** |
| 20-field | 20 | Alternative (no sensor) |

**Final Recommendation: Use 23-feature dataset for maximum flexibility and research value!**
