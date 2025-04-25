# test_model.py
import pickle
import pandas as pd
import numpy as np

# Load model and scaler
model = pickle.load(open('models/artifacts/best_model.pkl', 'rb'))
scaler = pickle.load(open('models/artifacts/scaler.pkl', 'rb'))

# Create sample data
data = {
    'PT08.S1(CO)': 1360,
    'NMHC(GT)': 150,
    'C6H6(GT)': 11.9,
    'PT08.S2(NMHC)': 1046,
    'NOx(GT)': 166,
    'PT08.S3(NOx)': 1056,
    'NO2(GT)': 113,
    'PT08.S4(NO2)': 1692,
    'PT08.S5(O3)': 1268,
    'T': 13.6,
    'RH': 48.9,
    'AH': 0.7578,
    'hour': 14,
    'day': 23,
    'month': 4,
    'dayofweek': 2,
    'CO(GT)_lag_1': 2.6,
    'CO(GT)_lag_3': 2.6,
    'CO(GT)_lag_6': 2.6,
    'CO(GT)_lag_12': 2.6,
    'CO(GT)_lag_24': 2.6,
    'CO(GT)_rolling_mean_3': 2.6,
    'CO(GT)_rolling_mean_6': 2.6,
    'CO(GT)_rolling_mean_12': 2.6,
    'CO(GT)_rolling_mean_24': 2.6,
    'CO(GT)_rolling_std_3': 0,
    'CO(GT)_rolling_std_6': 0,
    'CO(GT)_rolling_std_12': 0,
    'CO(GT)_rolling_std_24': 0
}

# Convert to DataFrame
df = pd.DataFrame([data])

# Scale and predict
X_scaled = scaler.transform(df)
prediction = model.predict(X_scaled)[0]

print(f"Prediction: {prediction}")