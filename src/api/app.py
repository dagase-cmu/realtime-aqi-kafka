"""
Flask API for air quality prediction service.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ml.predict import AirQualityPredictor
from src.utils.logger import setup_logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize logger
logger = setup_logger("api_service")

# Initialize predictor
predictor = AirQualityPredictor()
predictor.load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for single prediction.
    
    Expected JSON payload:
    {
        "DateTime": "2024-04-23 14:00:00",
        "PT08.S1(CO)": 1100,
        "NMHC(GT)": 200,
        ...
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result = predictor.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Endpoint for batch predictions.
    
    Expected JSON payload:
    {
        "data": [
            {"DateTime": "2024-04-23 14:00:00", "PT08.S1(CO)": 1100, ...},
            {"DateTime": "2024-04-23 15:00:00", "PT08.S1(CO)": 1200, ...}
        ]
    }
    """
    try:
        payload = request.get_json()
        
        if not payload or 'data' not in payload:
            return jsonify({'error': 'No data provided'}), 400
        
        data_list = payload['data']
        results = predictor.predict_batch(data_list)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/future', methods=['POST'])
def predict_future():
    """
    Endpoint for future predictions.
    
    Expected JSON payload:
    {
        "current_data": {"DateTime": "2024-04-23 14:00:00", ...},
        "horizon": 24
    }
    """
    try:
        payload = request.get_json()
        
        if not payload or 'current_data' not in payload:
            return jsonify({'error': 'No current data provided'}), 400
        
        current_data = payload['current_data']
        horizon = payload.get('horizon', 24)
        
        results = predictor.predict_future(current_data, horizon)
        
        return jsonify({'future_predictions': results})
    
    except Exception as e:
        logger.error(f"Future prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    try:
        return jsonify({
            'model_type': type(predictor.model).__name__ if predictor.model else None,
            'feature_count': len(predictor.feature_names) if predictor.feature_names else 0,
            'features': predictor.feature_names,
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the model from MLflow or disk."""
    try:
        predictor.load_model()
        return jsonify({
            'status': 'success',
            'message': 'Model reloaded successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For production, use gunicorn or other WSGI server
    app.run(host='0.0.0.0', port=8080, debug=False)