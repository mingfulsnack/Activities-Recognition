import pandas as pd
import numpy as np
from config import *
from preprocessing import get_convoluted_data

# Load and check data
data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, 
                   sep=',', on_bad_lines='skip', engine='python')
data = data.assign(**{'z-axis': data['z-axis'].astype(str).str.replace(';', '', regex=True)})

print('Data shape before dropna:', data.shape)
print('Data dtypes:')
print(data.dtypes)
print('NaN values by column:')
print(data.isnull().sum())

# Convert numeric columns
for col in ['x-axis', 'y-axis', 'z-axis']:
    try:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    except:
        print(f"Error converting {col}")

print('\nAfter numeric conversion:')
print('NaN values by column:')
print(data.isnull().sum())

data = data.dropna()
print('\nData shape after dropna:', data.shape)

# Check for any remaining issues
print('Sample numeric values:')
print(data[['x-axis', 'y-axis', 'z-axis']].head())

# Test preprocessing
data_conv, labels = get_convoluted_data(data)
print('\nConvoluted data shape:', data_conv.shape)
print('Labels shape:', labels.shape)
print('Convoluted data has NaN?', np.isnan(data_conv).sum())
print('Labels has NaN?', np.isnan(labels).sum())
print('Data range - min:', data_conv.min(), 'max:', data_conv.max())
print('Sample label sum:', labels[0].sum()) 