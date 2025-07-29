# PHASE 1: MULTI-MODAL STRESS PREDICTION - PROJECT README

This repository contains the implementation and documentation for Phase 1 of the Multi-Modal Stress Prediction research project.

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### Run the Research
```bash
python phase1_multimodal_stress_prediction.py
```

**Expected Runtime:** ~7 minutes  
**Output:** Results, visualizations, and trained model

---

## 📁 Project Structure

```
├── phase1_multimodal_stress_prediction.py    # Main research implementation
├── health_data_generator.py                  # Dataset generation tool
├── data/
│   └── enhanced_health_monitoring_data_30days.csv  # Health dataset
├── PHASE1_RESEARCH_DOCUMENTATION.md          # Complete research documentation
├── PHASE1_EXECUTIVE_SUMMARY.md               # Executive summary
├── best_multimodal_stress_model.h5           # Trained model
├── multimodal_stress_prediction_results.png  # Result visualizations
└── README.md                                 # This file
```

---

## 🎯 Research Overview

**Objective:** Combine Human Activity Recognition with physiological data for real-time stress prediction.

**Key Achievement:** 95.67% accuracy in stress classification (Low/Medium/High categories)

**Architecture:** Multi-modal deep learning with attention-based fusion

---

## 📊 Results Summary

| Metric | Value | Status |
|--------|-------|---------|
| **Classification Accuracy** | 95.67% | ✅ Excellent |
| **Regression R²** | -0.043 | ⚠️ Needs improvement |
| **Training Time** | 7 minutes | ✅ Optimized |
| **Model Parameters** | 52,660 | ✅ Lightweight |

---

## 🔧 Configuration

### Key Parameters (Optimized)
```python
config = {
    'sequence_length': 180,      # WISDM-validated
    'batch_size': 64,            # Efficiency optimized
    'epochs': 20,                # With early stopping
    'learning_rate': 0.001,      # Adam optimizer
    'hidden_units': 30,          # HAR-proven
    'dropout_rate': 0.3,         # Regularization
    'sample_fraction': 0.3       # 30% of data for speed
}
```

**Project Status:** ✅ Phase 1 Complete - Ready for Phase 2  
**Last Updated:** July 28, 2025


### Dependencies
- matplotlib 1.5.3
- seaborn 0.8.1
- numpy 1.14
- pandas 0.20.3
- scikit-learn 0.19.1
- tensorflow 1.5.0


### Use
1. Run the script with  `python3 HAR_Recognition.py`
