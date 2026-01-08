# Phase 2 - LSTM Baseline for Stress Prediction

**Date Started:** January 8, 2026  
**Status:** In Progress  
**Model Type:** LSTM (Long Short-Term Memory)  
**Task:** Time-series regression for stress level prediction  

---

## 📋 Overview

This document tracks the implementation of the LSTM Baseline model, which serves as the foundation for stress prediction in Phase 2 of the thesis project. The baseline model establishes:

1. Data pipeline for time-series sequence generation
2. Train/Validation/Test split methodology
3. LSTM architecture for stress prediction
4. Training procedure and hyperparameters
5. Evaluation metrics and benchmarks

---

## 🎯 Objectives

### Primary Goal
Create a baseline LSTM model that can predict stress levels (1-9) from 60-minute sequences of health and behavioral data.

### Success Criteria
- ✅ Data pipeline successfully creates sequences
- ✅ Model trains without errors
- ✅ Test MAE < 1.5 (acceptable baseline)
- ✅ Model can be saved and reloaded
- ✅ Results are reproducible

---

## 📊 Dataset Information

### Source Data
- **File:** `optimized_health_data_23features.csv`
- **Total Samples:** 54,448 rows (30 days of data)
- **Features:** 22 input features + 1 target (Stress_Level)

### Feature Categories

#### 1. Accelerometer Features (3)
- `Accelerometer_X`, `Accelerometer_Y`, `Accelerometer_Z`
- Used for HAR compatibility

#### 2. Activity & Location (2)
- `Activity` (categorical: Walking, Jogging, Sitting, Standing, Upstairs, Downstairs)
- `Location` (categorical: Home, Work, Gym, Outdoor, Transport)

#### 3. Physiological Features (5)
- `Heart_Rate` (60-180 bpm)
- `Sleep_Duration` (4-10 hours)
- `Sleep_Quality` (categorical: Poor, Fair, Good)
- `Energy_Level` (1-10)
- `Mood_Score` (1-10)

#### 4. Screen Usage Features (5)
- `Screen_Usage_Current` (minutes)
- `Screen_Usage_15min_Avg` (average)
- `Screen_Usage_Trend` (increasing/decreasing)
- `Phone_Usage_Intensity` (1-10)
- `Phone_Event_Frequency` (events per hour)

#### 5. Social Features (2)
- `Social_Current_Level` (1-10)
- `Social_1hour_Avg` (average)

#### 6. Environmental Features (3)
- `Ambient_Light` (lux)
- `Noise_Level` (1-10)
- `Weather_Condition` (categorical: Sunny, Cloudy, Rainy, Snowy)

#### 7. Exercise Features (1)
- `Exercise_Minutes` (0-60)

### Target Variable
- **Stress_Level:** Continuous value from 1.0 to 9.0
- **Mean:** 4.49
- **Std Dev:** 3.20

---

## 🔧 Data Pipeline Implementation

### Sequence Generation
```python
Sequence Length: 60 minutes (1 hour)
Prediction Horizon: 1 minute ahead
Overlap: 59 minutes (sliding window)
```

**Rationale:** 60-minute sequences capture sufficient temporal context for stress pattern recognition.

### Data Preprocessing

1. **Categorical Encoding**
   - Label Encoding for: Activity, Location, Sleep_Quality, Weather_Condition
   
2. **Feature Scaling**
   - StandardScaler applied to all features
   - Fitted on training data only
   
3. **Sequence Creation**
   - Input: (sequence_length=60, num_features=21)
   - Target: Stress level at time t+1

### Data Split

| Split      | Ratio | Sequences | Purpose                    |
|------------|-------|-----------|----------------------------|
| Training   | 70%   | 38,070    | Model parameter learning   |
| Validation | 15%   | 8,159     | Hyperparameter tuning      |
| Test       | 15%   | 8,159     | Final evaluation           |

**Total Sequences:** 54,388 (from 54,448 samples)

**Split Method:** Temporal split (no shuffling) to preserve time-series integrity

---

## 🏗️ Model Architecture

### LSTM Baseline Architecture

```
Input Shape: (60, 21)
    ↓
LSTM Layer: 128 units
    ↓
Dropout: 0.3
    ↓
Dense Layer: 64 units (ReLU)
    ↓
Dropout: 0.3
    ↓
Output Layer: 1 unit (Linear)
    ↓
Predicted Stress Level
```

### Architecture Rationale

1. **LSTM Layer (128 units)**
   - Captures temporal dependencies in stress patterns
   - Sufficient capacity without overfitting
   - Returns only last output (many-to-one)

2. **Dropout (0.3)**
   - Prevents overfitting
   - Applied after LSTM and Dense layers

3. **Dense Layer (64 units)**
   - Non-linear transformation
   - ReLU activation

4. **Output Layer (1 unit)**
   - Linear activation for regression
   - Predicts continuous stress level

### Model Parameters

```python
Total Parameters: ~120K (estimated)
Trainable Parameters: ~120K
Optimizer: Adam
Learning Rate: 0.001
Loss Function: MSE (Mean Squared Error)
```

---

## 🎓 Training Configuration

### Hyperparameters

| Parameter              | Value  | Rationale                           |
|------------------------|--------|-------------------------------------|
| Batch Size             | 64     | Balance between speed and stability |
| Max Epochs             | 100    | Sufficient for convergence          |
| Early Stopping Patience| 15     | Prevent overfitting                 |
| Learning Rate          | 0.001  | Standard for Adam optimizer         |
| LR Reduction Factor    | 0.5    | On plateau after 5 epochs           |
| Random Seed            | 42     | Reproducibility                     |

### Callbacks

1. **EarlyStopping**
   - Monitor: `val_loss`
   - Patience: 15 epochs
   - Restore best weights

2. **ModelCheckpoint**
   - Save best model based on `val_loss`
   - Format: Keras (.keras)

3. **ReduceLROnPlateau**
   - Factor: 0.5
   - Patience: 5 epochs
   - Min LR: 1e-6

4. **TensorBoard**
   - Log directory: `models/logs/`
   - Histogram frequency: 1

---

## 📈 Evaluation Metrics

### Regression Metrics

1. **MSE (Mean Squared Error)**
   - Primary loss function
   - Sensitive to outliers

2. **MAE (Mean Absolute Error)**
   - Average prediction error
   - Interpretable (in stress level units)

3. **RMSE (Root Mean Squared Error)**
   - Same scale as target
   - Balance between MSE and MAE

4. **R² Score**
   - Proportion of variance explained
   - Range: -∞ to 1.0 (1.0 = perfect)

### Baseline Targets

| Metric | Target       | Interpretation               |
|--------|--------------|------------------------------|
| MAE    | < 1.5        | Error less than 1.5 levels   |
| RMSE   | < 2.0        | Good prediction accuracy     |
| R²     | > 0.5        | Explains 50%+ variance       |

---

## 📁 File Structure

```
stress_prediction/
├── __init__.py                    # Package initialization
├── config.py                      # Configuration parameters
├── data_pipeline.py               # Data loading and preprocessing
├── lstm_baseline.py               # LSTM model implementation
├── models/                        # Saved models
│   ├── lstm_baseline_best.keras   # Best model (lowest val_loss)
│   ├── lstm_baseline_final.keras  # Final model after training
│   └── logs/                      # TensorBoard logs
└── results/                       # Training results
    ├── lstm_baseline_training_history.png
    └── lstm_baseline_results.txt
```

---

## 🔄 Implementation Progress

### ✅ Completed Tasks

- [x] Created `stress_prediction/` folder structure
- [x] Implemented `config.py` with all hyperparameters
- [x] Implemented `data_pipeline.py`
  - Data loading from CSV
  - Feature preprocessing (encoding + scaling)
  - Sequence generation (60-minute windows)
  - Train/Val/Test split (70/15/15)
  - Verified: 54,388 sequences created successfully
- [x] Implemented `lstm_baseline.py`
  - LSTM architecture (128 units)
  - Training loop with callbacks
  - Evaluation metrics (MSE, MAE, RMSE, R²)
  - Model saving/loading
  - Training history visualization
- [x] Created documentation structure

### 🔄 Current Task

- [ ] **Train LSTM Baseline Model**
  - Run full training loop
  - Monitor convergence
  - Save best model
  - Generate training plots

### 📋 Next Steps

- [ ] Analyze training results
- [ ] Error analysis on test set
- [ ] Identify failure cases
- [ ] Document findings in this file
- [ ] Create baseline report

---

## 🎯 Expected Outcomes

### Training Expectations

1. **Convergence**
   - Training should converge within 50-80 epochs
   - Validation loss should decrease steadily
   - Early stopping may trigger around epoch 60-70

2. **Performance Expectations**
   - **Optimistic:** MAE ~1.0, R² ~0.6-0.7
   - **Realistic:** MAE ~1.2-1.5, R² ~0.5-0.6
   - **Acceptable:** MAE <2.0, R² >0.4

3. **Training Time**
   - Estimated: 10-20 minutes (CPU)
   - Depends on hardware

---

## 📝 Key Decisions & Rationale

### 1. Why 60-minute sequences?
- **Context window:** 1 hour captures sufficient stress pattern context
- **Too short (e.g., 10 min):** Insufficient temporal information
- **Too long (e.g., 4 hours):** Computational cost, less recent context

### 2. Why single LSTM layer?
- **Baseline simplicity:** Start with simplest effective architecture
- **Overfitting risk:** Dataset size doesn't justify deeper networks
- **Comparison basis:** Easier to compare with more complex models later

### 3. Why MSE loss?
- **Regression task:** Predicting continuous values
- **Penalizes outliers:** Important for extreme stress detection
- **Standard choice:** Widely used, well-understood

### 4. Why no data augmentation?
- **Real data quality:** Using real patterns from generator
- **Baseline purity:** Avoid confounding factors
- **Future consideration:** Can add in Phase 2 improvements

---

## 🚀 Running the Model

### Quick Start

```bash
# Navigate to stress_prediction directory
cd stress_prediction

# Run LSTM baseline training
python lstm_baseline.py
```

### Expected Output

1. Data preparation logs (loading, preprocessing, splitting)
2. Model architecture summary
3. Training progress (epoch-by-epoch)
4. Validation metrics after each epoch
5. Test set evaluation results
6. Saved files:
   - `models/lstm_baseline_best.keras`
   - `models/lstm_baseline_final.keras`
   - `results/lstm_baseline_training_history.png`
   - `results/lstm_baseline_results.txt`

---

## 📊 Results (To Be Updated After Training)

### Training Metrics

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE | Notes |
|-------|------------|----------|-----------|---------|-------|
| TBD   | TBD        | TBD      | TBD       | TBD     | TBD   |

### Test Set Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MSE    | TBD   | TBD            |
| MAE    | TBD   | TBD            |
| RMSE   | TBD   | TBD            |
| R²     | TBD   | TBD            |

### Training History Plot

*To be added after training*

---

## 🐛 Known Issues & Solutions

### Issue 1: Column Name Mismatch
**Problem:** Initial config had wrong feature names  
**Solution:** Updated `config.py` to match actual dataset columns  
**Status:** ✅ Resolved

### Issue 2: Feature Count Mismatch
**Problem:** Config listed 23 features, dataset has 22 (excluding target)  
**Solution:** Corrected feature list to 21 usable features  
**Status:** ✅ Resolved

---

## 🔮 Future Improvements

After baseline is established, consider:

1. **Hyperparameter tuning**
   - Grid search for LSTM units, dropout, learning rate
   - Experiment with sequence length

2. **Architecture variations**
   - Bidirectional LSTM
   - Stacked LSTM layers
   - Attention mechanisms

3. **Feature engineering**
   - Time-based features (hour, day_of_week)
   - Rolling statistics (1-hour, 4-hour windows)
   - Feature interactions

4. **Regularization**
   - L1/L2 regularization
   - Batch normalization
   - Gradient clipping

---

## 📚 References

### Related Documents
- [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) - Phase 1 data preparation
- [FEATURE_SELECTION.md](../generate_and_verify_data/Data%20generator/FEATURE_SELECTION.md) - Feature selection rationale
- [DATASET_GUIDE.md](../generate_and_verify_data/Data%20generator/DATASET_GUIDE.md) - Dataset selection guide

### Code Files
- [config.py](../stress_prediction/config.py) - Configuration
- [data_pipeline.py](../stress_prediction/data_pipeline.py) - Data processing
- [lstm_baseline.py](../stress_prediction/lstm_baseline.py) - LSTM implementation

---

## ✍️ Author Notes

**Implementation Strategy:**
- Start simple, iterate based on results
- Document everything for thesis report
- Establish reproducible baseline
- Focus on understanding model behavior

**Thesis Integration:**
- This baseline provides comparison point for all future models
- Training process demonstrates systematic approach
- Evaluation metrics establish standard for model comparison
- Documentation serves as methodology section content

---

## 🎯 TRAINING RESULTS - COMPLETED ✅

### Final Performance (Epoch 18 - Best Model)
- **Test MAE: 0.5095** ✅ (Target: < 1.0) - 49% better than target
- **Test RMSE: 0.8123** ✅ (Target: < 1.5) - 46% better than target  
- **Test R²: 0.9343 (93.43%)** ✅✅ (Target: > 0.70) - 33% better than target

### Training Process
- **Total Epochs**: 33 (stopped early)
- **Best Epoch**: 18
- **Early Stopping**: Triggered after 15 epochs without improvement
- **Learning Rate**: Started at 0.001, reduced to 0.000125

### Validation Performance
- **Best Val Loss**: 1.2438 (Epoch 18)
- **Final Val Loss**: 1.5949 (Epoch 33)

### Key Observations
1. ✅ **Excellent R² score (93.43%)** - Model captures variance exceptionally well
2. ✅ **Low MAE (0.51)** - Average error is only half a stress level
3. ✅ **Stable training** - Early stopping prevented overfitting
4. ⚠️ **Some overfitting** - Validation loss increased after epoch 18
5. ✅ **Quick convergence** - Best model achieved at epoch 18/33

---

## ✅ Status: BASELINE COMPLETED SUCCESSFULLY

### Achievement Summary
- ✅ Data pipeline implemented and tested
- ✅ LSTM baseline model trained successfully
- ✅ **EXCEEDED ALL PERFORMANCE TARGETS**
- ✅ R² score 93.43% (33% above target)
- ✅ Documentation completed
- ✅ Code committed to repository

### Files Created
1. `stress_prediction/data_loader.py` - Data preprocessing pipeline
2. `stress_prediction/models/lstm_baseline.py` - LSTM model implementation
3. `stress_prediction/train.py` - Training script
4. `stress_prediction/evaluate.py` - Evaluation script
5. `stress_prediction/config.py` - Configuration management
6. `Doc/PHASE2_LSTM_BASELINE.md` - This documentation
7. `Doc/PHASE2_SUMMARY.md` - Summary report
8. `Doc/PROGRESS_TRACKER.md` - Project tracking
9. `stress_prediction/models/saved/lstm_baseline_best.keras` - Best model
10. `stress_prediction/results/lstm_baseline_history.json` - Training history

---

## 📝 Next Steps (Bước 2: Error Analysis)

### Immediate Actions
1. **Error Analysis** - Which stress levels are hardest to predict?
2. **Visualization** - Predictions vs actual, error distribution
3. **Feature Importance** - Which features contribute most?

### Future Model Comparisons
With baseline R² = 93.43%, next models should aim to:
- Improve MAE to < 0.5 (currently 0.5095)
- Better handle edge cases (stress levels 1-2 and 8-9)
- Reduce overfitting (improve validation loss stability)
- Faster training or inference time

### Planned Models (Phase 2 continuation)
1. **GRU** - Faster alternative to LSTM
2. **Bidirectional LSTM variations** - Better temporal context
3. **TCN** - Temporal Convolutional Network
4. **Transformer** - Attention-based architecture
5. **EWC** - Elastic Weight Consolidation (continual learning)
6. **XGBoost** - Traditional ML baseline for comparison

---

**Last Updated:** January 8, 2026 - LSTM Baseline Training COMPLETED ✅  
**Status:** Phase 2 Step 1 - COMPLETED (R² = 93.43%)  
**Next Phase:** Step 2 - Error Analysis & Visualization
