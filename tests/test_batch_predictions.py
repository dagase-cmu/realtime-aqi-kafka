"""
Test script for batch predictions to address feedback requirements.
"""

import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ml.predict import AirQualityPredictor
from src.utils.config import get_config

def test_batch_predictions_local():
    """Test batch predictions locally without API."""
    print("Testing batch predictions locally...")
    
    # Initialize predictor
    predictor = AirQualityPredictor()
    predictor.load_model()
    
    # Load test data
    data_config = get_config("data")
    test_data_path = data_config["raw_data_path"]
    
    # Load and prepare test data
    df = pd.read_csv(test_data_path, sep=';', decimal=',')
    
    # Take first 100 rows for testing
    test_df = df.head(100)
    
    # Convert to list of dictionaries
    test_data = test_df.to_dict('records')
    
    # Make batch predictions
    results = predictor.predict_batch(test_data)
    
    # Analyze results
    errors = [r['error'] for r in results if 'error' in r and r['error'] is not None]
    successful_predictions = len(results) - len(errors)
    
    print(f"Total predictions: {len(results)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {len(errors)}")
    
    if successful_predictions > 0:
        avg_error = np.mean([r['error'] for r in results if 'error' in r and r['error'] is not None])
        print(f"Average error: {avg_error:.4f}")
    
    return results

def test_batch_predictions_api():
    """Test batch predictions through API."""
    print("Testing batch predictions through API...")
    
    api_url = "http://localhost:8080/predict/batch"
    
    # Create test data
    test_data = []
    base_date = datetime.now()
    
    for i in range(10):
        data_point = {
            'DateTime': (base_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
            'PT08.S1(CO)': 1100 + i * 10,
            'NMHC(GT)': 200 + i * 5,
            'C6H6(GT)': 10.5 + i * 0.1,
            'PT08.S2(NMHC)': 1000 + i * 15,
            'NOx(GT)': 150 + i * 5,
            'PT08.S3(NOx)': 850 + i * 10,
            'NO2(GT)': 100 + i * 3,
            'PT08.S4(NO2)': 1200 + i * 20,
            'PT08.S5(O3)': 1000 + i * 15,
            'T': 22.5 + i * 0.2,
            'RH': 50.5 + i * 0.5,
            'AH': 1.2 + i * 0.01
        }
        test_data.append(data_point)
    
    # Make API request
    try:
        response = requests.post(api_url, json={'data': test_data})
        
        if response.status_code == 200:
            results = response.json()['predictions']
            print(f"Successfully predicted {len(results)} data points")
            
            # Show first prediction
            print("\nFirst prediction result:")
            print(json.dumps(results[0], indent=2))
        else:
            print(f"API error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("API server not running. Please start the API server first.")
    except Exception as e:
        print(f"Error: {e}")

def test_future_predictions():
    """Test future predictions (time series forecasting)."""
    print("\nTesting future predictions...")
    
    predictor = AirQualityPredictor()
    predictor.load_model()
    
    # Create a current data point
    current_data = {
        'DateTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'PT08.S1(CO)': 1100,
        'NMHC(GT)': 200,
        'C6H6(GT)': 10.5,
        'PT08.S2(NMHC)': 1000,
        'NOx(GT)': 150,
        'PT08.S3(NOx)': 850,
        'NO2(GT)': 100,
        'PT08.S4(NO2)': 1200,
        'PT08.S5(O3)': 1000,
        'T': 22.5,
        'RH': 50.5,
        'AH': 1.2
    }
    
    # Predict next 24 hours
    future_predictions = predictor.predict_future(current_data, horizon=24)
    
    print(f"Generated {len(future_predictions)} future predictions")
    print("\nFirst 5 predictions:")
    for i, pred in enumerate(future_predictions[:5]):
        print(f"Hour {i+1}: CO = {pred['predicted_value']:.2f}")

def main():
    """Run all tests."""
    print("Starting batch prediction tests...\n")
    
    # Test local batch predictions
    test_batch_predictions_local()
    
    # Test API batch predictions
    # test_batch_predictions_api()  # Uncomment when API is running
    
    # Test future predictions
    test_future_predictions()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()