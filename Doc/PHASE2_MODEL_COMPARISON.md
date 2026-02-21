# Phase 2 - Model Comparison Results

## Overview
So sanh 5 kien truc model khac nhau tren cung 13-feature dataset va data pipeline de chung minh Stacked Bi-LSTM (Tuned) la lua chon toi uu.

## Cau hinh chung
- **Dataset**: optimized_health_data_13features.csv (54,448 samples, 13 features)
- **Sequence Length**: 60 timesteps
- **Split**: 70/15/15 (Train/Val/Test)
- **Pipeline**: Split -> Encode(fit train) -> Normalize -> Sequences
- **Max Epochs**: 80 | **Batch Size**: 32
- **Early Stopping**: patience=15 (restore best weights)
- **Loss**: MSE | **Optimizer**: Adam

## Ket qua so sanh

| # | Model | MAE | RMSE | R² | Parameters | Time (s) | Epochs |
|---|-------|-----|------|----|------------|----------|--------|
| 1 | **MLP (Dense)** | 0.9310 | 1.2968 | 0.8331 | 241,153 | 60s | 19 |
| 2 | **Simple LSTM** | 0.5213 | 0.7603 | 0.9426 | 83,073 | 452s | 17 |
| 3 | **Stacked Bi-LSTM (Baseline)** | 0.7159 | 0.9698 | 0.9067 | 320,129 | 1,148s | 16 |
| 4 | **Stacked Bi-GRU** | 0.7551 | 0.9103 | 0.9178 | 243,841 | 2,855s | 46 |
| 5 | **Stacked Bi-LSTM (Tuned)** | **0.4414** | **0.6697** | **0.9555** | 163,585 | 825s | 20 |

## Phan tich ket qua

### 1. Best Model: Stacked Bi-LSTM (Tuned)
- **MAE = 0.4414** (thap nhat) - Sai so trung binh chi 0.44 diem stress
- **R² = 0.9555** (cao nhat) - Giai thich 95.6% phuong sai du lieu
- **RMSE = 0.6697** (thap nhat) - Xu ly tot ca outlier
- **163,585 params** - So tham so hop ly (khong qua lon, khong qua nho)

### 2. Xep hang theo MAE
1. **Bi-LSTM Tuned**: 0.4414 (BEST)
2. **Simple LSTM**: 0.5213 (+18.1%)
3. **Stacked Bi-LSTM Baseline**: 0.7159 (+62.2%)
4. **Stacked Bi-GRU**: 0.7551 (+71.1%)
5. **MLP (Dense)**: 0.9310 (+111.0%)

### 3. Nhan xet chinh

#### MLP (Dense) - Worst
- R² = 0.8331 - Thap nhat do khong capture duoc temporal dependencies
- Nhanh nhat (60s) nhung ket qua kem
- Chung minh time-series models can thiet cho bai toan nay

#### Simple LSTM vs Bi-LSTM Baseline
- Simple LSTM (0.5213) **tot hon** Bi-LSTM Baseline (0.7159)
- Ly do: cung learning rate va config, nhung Bi-LSTM co nhieu params hon (320K vs 83K) nen can HP tuning tot hon
- 1 layer LSTM don gian nhung hieu qua voi default hyperparameters

#### Bi-GRU vs Bi-LSTM
- Bi-GRU (0.7551) va Bi-LSTM Baseline (0.7159) cho ket qua tuong tu
- GRU it params hon (243K vs 320K) nhung cham hon do train nhieu epochs (46 vs 16)
- Hai kien truc tuong duong voi default config

#### Hyperparameter Tuning la yeu to quyet dinh
- Bi-LSTM Baseline (default config): MAE = 0.7159
- Bi-LSTM Tuned (HP optimized): MAE = 0.4414
- **Cai thien 38.3%** chi bang HP tuning (khong doi kien truc)
- Chung minh HP tuning quan trong hon viec chon kien truc

### 4. Trade-off Analysis

| Metric | MLP | LSTM | Bi-LSTM | Bi-GRU | Bi-LSTM Tuned |
|--------|-----|------|---------|--------|---------------|
| Accuracy | Low | High | Medium | Medium | **Highest** |
| Speed | **Fastest** | Fast | Medium | Slow | Medium |
| Complexity | **Lowest** | Low | High | Medium | Medium |
| Overall | * | *** | ** | ** | ***** |

### 5. Ket luan cho bao cao

> **Stacked Bidirectional LSTM voi Bayesian Optimization** la kien truc toi uu cho bai toan du doan stress level tu 13 features. 
> - Vuot troi MLP 111% (chung minh temporal modeling can thiet)
> - Vuot troi Simple LSTM 18% (bidirectional + stacking co gia tri khi tuned dung)
> - HP Tuning cai thien 38.3% (yeu to quyet dinh nhat)
> - R² = 0.9555 - Du doan chinh xac 95.6% phuong sai

## Output Files
- `results/model_comparison/comparison_results.json` - Ket qua chi tiet
- `results/model_comparison/comparison_summary.csv` - Bang tom tat
- `results/model_comparison/model_comparison.png` - 6 bieu do so sanh
- `results/model_comparison/model_comparison_radar.png` - Radar chart
- `results/model_comparison/model_comparison_table.png` - Bang truc quan
- `results/model_comparison/training_histories.json` - Lich su training
