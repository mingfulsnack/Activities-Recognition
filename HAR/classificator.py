import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Local libraries
from config import * # Global variables
from preprocessing import get_convoluted_data

##################################################
### FUNCTIONS
##################################################
def softmax_to_label(array):
    i = np.argmax(array)
    return LABELS_NAMES[i]

def evaluate(X_test, y_test):
    # Load the saved model
    model = tf.keras.models.load_model('classificator_model.keras')
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Show some predictions
    for i in range(min(10, len(y_test))):
        actual = softmax_to_label(y_test[i])
        predicted = softmax_to_label(predictions[i])
        print(f"Actual: {actual}\t Predicted: {predicted}")
    
    # Calculate accuracy
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

##################################################
### MAIN
##################################################
if __name__ == '__main__':
    # LOAD DATA
    data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, 
                       sep=',', on_bad_lines='skip', engine='python')
    
    # Fix the semicolon issue and convert to numeric
    data = data.assign(**{'z-axis': data['z-axis'].astype(str).str.replace(';', '', regex=True)})
    for col in ['x-axis', 'y-axis', 'z-axis']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    data_convoluted, labels = get_convoluted_data(data)

    # Use all data for evaluation
    X_test = data_convoluted
    y_test = labels

    accuracy = evaluate(X_test, y_test)
    print(f"Final accuracy: {accuracy:.4f}")
