"""
Configuration file for stress prediction models
Phase 2 - LSTM Baseline
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'generate_and_verify_data', 'Data generator', 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'stress_prediction', 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'stress_prediction', 'results')

# Data file
DATA_FILE = os.path.join(DATA_DIR, 'optimized_health_data_23features.csv')

# Feature columns (22 features - excluding Stress_Level which is the target)
# Based on actual dataset columns from optimized_health_data_23features.csv
FEATURE_COLUMNS = [
    # Accelerometer features (for HAR compatibility)
    'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z',
    
    # Activity and Location
    'Activity', 'Location',
    
    # Physiological features
    'Heart_Rate', 'Sleep_Duration', 'Sleep_Quality',
    'Energy_Level', 'Mood_Score',
    
    # Screen usage features
    'Screen_Usage_Current', 'Screen_Usage_15min_Avg', 
    'Screen_Usage_Trend', 'Phone_Usage_Intensity', 'Phone_Event_Frequency',
    
    # Social features
    'Social_Current_Level', 'Social_1hour_Avg',
    
    # Environmental features
    'Ambient_Light', 'Noise_Level', 'Weather_Condition',
    
    # Exercise features
    'Exercise_Minutes'
]

TARGET_COLUMN = 'Stress_Level'

# Sequence parameters
SEQUENCE_LENGTH = 60  # 60 minutes (1 hour) of data for prediction
PREDICTION_HORIZON = 1  # Predict next time step

# Train/Val/Test split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# LSTM Model hyperparameters
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15  # Early stopping patience

# Random seed for reproducibility
RANDOM_SEED = 42
