# 30-Day Health Monitoring Dataset – End-to-End Design and Validation

## 1) Project Goal and Outcome
- Build a realistic 30-day, multi-modal health dataset for a thesis on human tracking, Human Activity Recognition (HAR), and real-time stress prediction.
- Achieved HAR-compatibility with an existing Bidirectional LSTM model (`HAR/model/classificator_model.keras`).
- Final balanced dataset: `data/quota_balanced_health_data_30days.csv` with 53,800 records (30 days), validated at 86.2% HAR accuracy (overall) and production-ready for LSTM stress prediction.

## 2) High-Level Architecture (Refactored and Modular)
```
Data generator/
├── core/
│   ├── user_profile.py            # Age, gender, BMR, HR baselines, base sleep/screen habits
│   ├── wisdm_loader.py            # Load WISDM raw, sample real accelerometer, synthetic patterns
│   ├── activity_manager.py        # Activity selection, transitions, sanity checks
│   ├── schedule_generator.py      # Context-aware daily schedule with quota enforcement
│   ├── metrics_calculator.py      # Heart rate, calories, steps, mood, stress
│   └── behavioral_tracker.py      # Sequences: screen/phone/social/stress temporal features
├── refactored_health_data_generator.py  # Orchestrator – generates full dataset
├── validate_accelerometer_with_har.py   # HAR validation on generated accelerometer signals
├── analyze_activity_distribution.py     # Distribution and HAR-readiness analysis
└── improve_accelerometer_patterns.py    # Update synthetic patterns based on real WISDM stats
```

Key design choices:
- Separate concerns into cohesive modules for maintainability and extension.
- Use real WISDM accelerometer data to maximize HAR fidelity, complemented by physics-based patterns derived from WISDM statistics.
- Enforce realistic schedule, locations, and physiology with a daily quota system to fix activity imbalance and maximize HAR performance.

## 3) Data Flow
1. Load WISDM dataset metadata and samples (20Hz, users, 6 labeled activities).
2. For each day: generate realistic schedule (wake→commute→work→commute→evening→sleep skipped 0–8h) with daily noise and life events.
3. For each schedule segment: generate accelerometer (x,y,z) synchronized with activity using real WISDM samples plus small noise; or synthetic patterns when needed.
4. Compute physiological metrics (HR, steps, calories), environmental context (weather, light, noise), and behavioral sequences (screen/phone/social/stress).
5. Calculate mood and stress level dynamically by time of day, activity, location, HR, sleep quality, work intensity, and momentum.
6. Write cumulative metrics (steps, calories) and export dataset.
7. Validate accelerometer data against the trained HAR model.

## 4) HAR Compatibility and Accelerometer Generation
- Uses `HAR/preprocessing.py` parameters: `SEGMENT_TIME_SIZE=180` (≈9s), `TIME_STEP=100`.
- `core/wisdm_loader.py`:
  - `load_wisdm_data()`: loads raw WISDM and indexes samples per activity.
  - `get_real_accelerometer_sample(activity)`: picks a real sample, adds tiny stress/fatigue noise to keep patterns intact for HAR.
  - `_generate_synthetic_accelerometer(activity)`: physics-based fallback informed by real WISDM statistics (means/stds/ranges per axis). Updated using `improve_accelerometer_patterns.py` to tighten Walking/Upstairs/Standing/Sitting/Jogging/Downstairs.
- In `refactored_health_data_generator.py`:
  - `generate_accelerometer_with_variations(...)` produces sequences per schedule slot.
  - Timestamps are consistent 30±2s to preserve temporal assumptions (critical for sliding windows).

Result: High activity–signal alignment verified by HAR model.

## 5) Schedule Generation – Context, Realism, and Quotas
Implemented in `core/schedule_generator.py`:
- Time realism (VN lifestyle):
  - Weekdays: wake 6.5–8.0; work 9–17; commute windows; evening at home; sleep 22.5–23.5.
  - Weekends: wake later; flexible day blocks; exercise and social chances increase.
- Locations: home → commute → work → commute → home, with activity-based overrides (e.g., jog only outdoor/gym; stairs mainly at work/home blocks; no “7:29 AM at work”).
- Anti-sitting logic: enforced breaks after 45–90 minutes (randomized), replacing long sitting with movement activities contextually.
- Jogging rules: only 6–8 AM or 17–19 PM; outdoor/gym; forbidden during work hours.
- NEW Daily Activity Quotas (hours per day):
  - Sitting 4.8 (max)
  - Walking 4.0 (min)
  - Standing 3.2 (min)
  - Jogging 1.6 (min)
  - Upstairs 1.3 (min)
  - Downstairs 1.1 (min)
- Quota-aware selection `_choose_quota_aware_activity(...)`:
  - Prioritizes under-served activities based on remaining time and deficits.
  - Context-aware filters by time-of-day (work vs commute vs evening; weekday/weekend).
  - Preserves realism and transitions while pushing distribution to target.
- Duration tuning for HAR:
  - Reduced sitting durations; increased walking/jogging/standing.
  - Greatly increased stairs durations (Upstairs/Downstairs) to ensure enough contiguous windows for HAR.

Outcome: Sitting reduced to ~28.9%; others raised to targets; enough windows for all activities.

## 6) Metrics and Behavioral Sequences
Implemented in `core/metrics_calculator.py` and `core/behavioral_tracker.py`:
- Steps: realistic per-activity steps/hour; short-duration scaling; energy modifier.
- Calories: realistic low-sample floor; stress-adjusted metabolism; cumulative per-day continuity.
- HR: derived from base HR (UserProfile), activity intensity, stress; ordered expectation: Jogging > (Up/Downstairs) > Walking > Standing > Sitting (verified).
- Mood: daily base + circadian + activity/location/stress effects; bounded 1–10.
- Stress: multi-factor (time-of-day, activity, location, sleep, HR, work intensity) + momentum; bounded 1–9.
- Behavioral sequences: screen usage, phone events, social interaction trends, stress velocity – maintained as short-term windows for LSTM.

## 7) Validation and Analysis
- HAR validation (`validate_accelerometer_with_har.py`):
  - Converts generated data to HAR window format; normalizes; loads `HAR/model/classificator_model.keras`.
  - Reports overall and per-activity accuracy; prints top misclassifications.
- Distribution analysis (`analyze_activity_distribution.py`):
  - Prints counts, % per activity, daily hours, HAR window counts, gaps vs targets.
- Key milestones:
  - Pre-quota dataset: 81.3% accuracy; weak Upstairs/Walking windows.
  - Quota-balanced dataset: 86.2% accuracy overall; 100% for Jogging/Standing/Downstairs; Walking 80.1%; Sitting 87.0%; Upstairs 34.4% (from 0%).

## 8) Reproduce – Windows CMD (cmd.exe)
Run in the folder: `C:\Users\APC\Downloads\har-wisdm-bidirectional-lstm-rnns-stacked_lstm_wihout_BO\generate_and_verify_data\Data generator`

1) Generate quota-balanced dataset
```
python refactored_health_data_generator.py
```
- Output: `data\quota_balanced_health_data_30days.csv`

2) Analyze activity distribution and HAR readiness
```
python analyze_activity_distribution.py
```

3) Validate accelerometer vs HAR model
```
python validate_accelerometer_with_har.py
```

4) Optional – Improve synthetic patterns from real WISDM stats (if you retrain or want deltas)
```
python improve_accelerometer_patterns.py
```

## 9) Results to Present (Key Slides)
- Goal and constraints: realistic 30-day, multi-modal dataset for HAR + stress prediction.
- Modular architecture and data flow (diagram of modules and sequence).
- WISDM integration for accelerometer realism; physics-based fallback tuned by real statistics.
- Daily schedule realism: time blocks, commute, work, evening; location logic; anti-sitting; jogging rules.
- Quota system: how activity deficits are detected and resolved; balanced distribution.
- Physiological + behavioral modeling: HR, steps, calories, mood, stress; screen/phone/social sequences.
- Validation: overall 86.2% HAR accuracy; per-activity details; improved stairs and walking; realistic metrics checks.
- Ready for LSTM stress modeling: feature richness and temporal structure.

## 10) Troubleshooting
- PermissionError on writing CSV: close the file and rerun; or auto-change filename in generator.
- Time continuity gaps: ensure the generator uses the updated timestamp logic (30±2s jitter).
- Paths: `wisdm_loader.py` uses path to `data\WISDM_ar_v1.1_raw.txt`. Confirm dataset exists.
- HAR model missing: check `HAR\model\classificator_model.keras` is present.

## 11) Future Improvements
- Further enhance Upstairs/Walking separability (increase Z-axis periodicity for stairs; add gait cycle templates).
- Extend dataset to 60–90 days for even stronger sequence learning.
- Personalization: multiple user profiles with different baselines and schedules.
- Domain augmentation: introduce device placement variation (pocket/hand/wrist) with proper transforms.

## 12) Key Files and What to Show in the Defense
- `refactored_health_data_generator.py`: orchestrates generation; shows the pipeline.
- `core/schedule_generator.py`: realism + quota-aware balancing (the biggest lever for HAR accuracy).
- `core/wisdm_loader.py`: real-sample usage + pattern derivation from WISDM.
- `core/metrics_calculator.py`: all physiological/psychological computations.
- `validate_accelerometer_with_har.py`: proof of HAR-compatibility (86.2%).
- `analyze_activity_distribution.py`: proof of balanced activity distribution and sufficient HAR windows.

---
Prepared for thesis presentation – demonstrates realistic human behavior synthesis, HAR-compatibility, and readiness for LSTM-based stress prediction.
