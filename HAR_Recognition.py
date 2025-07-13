import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pickle
from tempfile import TemporaryFile

# Local libraries
from config import * # Global variables
from preprocessing import get_convoluted_data


##################################################
### FUNCTIONS
##################################################

# Returns a TensorFlow/Keras bidirectional LSTM model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(N_HIDDEN_NEURONS, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            input_shape=input_shape
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(N_HIDDEN_NEURONS, dropout=0.2, recurrent_dropout=0.2)
        ),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    
    return model


def train_evaluate_classifier(data_convoluted, labels):
    # Ensure data is in the right format and normalize
    data_convoluted = data_convoluted.astype(np.float32)
    labels = labels.astype(np.float32)
    
    # Normalize the data to prevent gradient explosion
    print("Normalizing data...")
    original_shape = data_convoluted.shape
    data_reshaped = data_convoluted.reshape(-1, data_convoluted.shape[-1])
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_reshaped)
    data_convoluted = data_normalized.reshape(original_shape)
    
    print(f"Data shape: {data_convoluted.shape}")
    print(f"Data range after normalization: {data_convoluted.min():.3f} to {data_convoluted.max():.3f}")
    
    # Check for any NaN or inf values
    if np.any(np.isnan(data_convoluted)) or np.any(np.isinf(data_convoluted)):
        print("WARNING: Found NaN or inf values in data!")
        data_convoluted = np.nan_to_num(data_convoluted, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # SPLIT INTO TRAINING AND TEST SETS
    X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3, random_state=RANDOM_SEED)

    # BUILD A MODEL
    input_shape = (SEGMENT_TIME_SIZE, N_FEATURES)
    model = create_model(input_shape)
    
    # Use lower learning rate and gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # Lower learning rate
        clipnorm=1.0  # Gradient clipping
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the model
    model.save('classificator_model.keras')
    
    # Evaluate the model
    _, acc_final = model.evaluate(X_test, y_test, verbose=0)
    
    return acc_final

##################################################
### MAIN
##################################################
if __name__ == '__main__':

    # LOAD DATA
    print("Loading data...")
    # Handle malformed lines in the CSV file
    data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, 
                       sep=',', on_bad_lines='skip', engine='python')
    
    # Fix the semicolon issue in z-axis column and convert to numeric
    data = data.assign(**{'z-axis': data['z-axis'].astype(str).str.replace(';', '', regex=True)})
    
    # Convert numeric columns explicitly
    for col in ['x-axis', 'y-axis', 'z-axis']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    print(f"Data loaded: {data.shape[0]} samples")
    
    print("Processing data...")
    data_convoluted, labels = get_convoluted_data(data)
    print(f"Processed data: {data_convoluted.shape[0]} windows")
    
    acc_final = train_evaluate_classifier(data_convoluted, labels)
    print(f"Final accuracy: {acc_final:.4f}")
