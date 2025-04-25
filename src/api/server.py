# src/api/server.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

app = Flask(__name__)

# Load model and scaler
model_path = Path(__file__).parent.parent.parent / 'models' / 'artifacts' / 'best_model.pkl'
scaler_path = Path(__file__).parent.parent.parent / 'models' / 'artifacts' / 'scaler.pkl'

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded successfully from {scaler_path}")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                "error": "Model or scaler not loaded",
                "status": "error"
            }), 500
        
        data = request.json
        print(f"Received data: {data}")
        
        # Create DataFrame - use a dictionary to ensure all columns are properly created
        df_dict = {}
        
        # Add temporal features
        if 'DateTime' in data:
            dt = pd.to_datetime(data['DateTime'])
            df_dict['hour'] = dt.hour
            df_dict['day'] = dt.day
            df_dict['month'] = dt.month
            df_dict['dayofweek'] = dt.dayofweek
        else:
            current_time = pd.Timestamp.now()
            df_dict['hour'] = current_time.hour
            df_dict['day'] = current_time.day
            df_dict['month'] = current_time.month
            df_dict['dayofweek'] = current_time.dayofweek
        
        # Add basic features from input data
        for feature in ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
                       'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)',
                       'PT08.S5(O3)', 'T', 'RH', 'AH']:
            df_dict[feature] = data.get(feature, 0)
        
        # Add lag and rolling features
        target_col = 'CO(GT)'
        current_value = data.get(target_col, 2.0)
        
        # Add lag features
        for lag in [1, 3, 6, 12, 24]:
            df_dict[f'{target_col}_lag_{lag}'] = current_value
        
        # Add rolling features
        for window in [3, 6, 12, 24]:
            df_dict[f'{target_col}_rolling_mean_{window}'] = current_value
            df_dict[f'{target_col}_rolling_std_{window}'] = 0
        
        # Create DataFrame from dictionary
        df = pd.DataFrame([df_dict])
        
        # Get features in the exact order expected by the scaler
        feature_columns = scaler.feature_names_in_
        X = df[feature_columns]
        
        print(f"Features prepared: {X.shape}")
        print(f"Feature values: {X.iloc[0].to_dict()}")
        
        # Scale features
        X_scaled = scaler.transform(X)
        print(f"Features scaled: {X_scaled.shape}")
        
        prediction = model.predict(X_scaled)[0]
        print(f"Prediction made: {prediction}")
        
        return jsonify({
            "prediction": float(prediction),
            "status": "success"
        })
    
    except Exception as e:
        import traceback
        print(f"Error in prediction: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "status": "error",
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)