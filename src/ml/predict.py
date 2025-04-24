"""
Prediction module for air quality forecasting.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import mlflow
import pickle
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.ml.feature_engineering import AirQualityFeatureEngineer

logger = setup_logger("prediction")

class AirQualityPredictor:
    """Air quality predictor with MLflow model loading and real-time prediction capabilities."""
    
    def __init__(self):
        """Initialize predictor with configuration."""
        self.data_config = get_config("data")
        self.model_config = get_config("model")
        self.mlflow_config = get_config("mlflow")
        
        self.model = None
        self.scaler = None
        self.feature_engineer = AirQualityFeatureEngineer()
        self.feature_names = None
        
    def load_model(self):
        """
        Load the model from a local file in src/ml/best_model.pkl.
        Also optionally loads associated scaler and feature names if available.
        """
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))  # path to src/ml/
            model_path = os.path.join(base_dir, "best_model.pkl")
            scaler_path = os.path.join(base_dir, "scaler.pkl")
            feature_names_path = os.path.join(base_dir, "feature_names.json")

            # Load model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded from local path: {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            # Load optional scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Feature scaler loaded.")

            # Load optional feature names
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")

        except Exception as e:
            logger.error(f"Error loading local model or artifacts: {e}")
            raise


    def prepare_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features from raw data for prediction.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Feature DataFrame
        """
        # Convert single record to DataFrame
        df = pd.DataFrame([data])
        
        # Create DateTime if not present
        if 'DateTime' not in df.columns and 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        elif 'timestamp' in df.columns:
            df['DateTime'] = pd.to_datetime(df['timestamp'])
        
        # Apply feature engineering
        df_features = self.feature_engineer.transform(df)
        
        # Select features in the correct order
        if self.feature_names:
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df_features.columns:
                    # Fill missing features with appropriate values
                    if 'lag' in feature:
                        df_features[feature] = np.nan
                    elif 'rolling' in feature:
                        df_features[feature] = np.nan
                    else:
                        df_features[feature] = 0
            
            df_features = df_features[self.feature_names]
        
        return df_features
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single data point.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Prediction result with additional metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare features
            features_df = self.prepare_features(data)
            
            # Handle any NaN values
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            if features_df.isna().any().any():
                features_df = features_df.fillna(0)
            
            # Scale features if scaler is available
            if self.scaler:
                features_scaled = self.scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Prepare result
            result = {
                'predicted_value': float(prediction),
                'timestamp': datetime.now().isoformat(),
                'model_version': getattr(self.model, 'version', 'unknown'),
                'features_used': len(self.feature_names) if self.feature_names else features_df.shape[1]
            }
            
            # Add confidence interval if model supports it
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(features_scaled)[0]
                    result['confidence'] = float(np.max(proba))
                except Exception:
                    pass
            
            # Add actual value if available for comparison
            if self.data_config["target_column"] in data:
                actual_value = data[self.data_config["target_column"]]
                result['actual_value'] = float(actual_value)
                result['error'] = float(abs(prediction - actual_value))
                result['percentage_error'] = float(abs(prediction - actual_value) / actual_value * 100) if actual_value != 0 else None
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of data points.
        
        Args:
            data_list: List of raw data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for data in data_list:
            try:
                result = self.predict(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for data point: {e}")
                result = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
        
        return results
    
    def predict_future(self, current_data: Dict[str, Any], horizon: int = 24) -> List[Dict[str, Any]]:
        """
        Make predictions for future time steps.
        
        Args:
            current_data: Current data point
            horizon: Number of future steps to predict
            
        Returns:
            List of future predictions
        """
        predictions = []
        
        # Start with current data
        last_data = current_data.copy()
        
        for step in range(1, horizon + 1):
            # Update temporal features for future step
            if 'DateTime' in last_data:
                future_time = pd.to_datetime(last_data['DateTime']) + pd.Timedelta(hours=step)
                last_data['DateTime'] = future_time
                last_data['hour'] = future_time.hour
                last_data['day'] = future_time.day
                last_data['month'] = future_time.month
                last_data['dayofweek'] = future_time.dayofweek
            
            # Make prediction
            result = self.predict(last_data)
            result['forecast_step'] = step
            result['forecast_time'] = last_data['DateTime'].isoformat() if 'DateTime' in last_data else None
            
            predictions.append(result)
            
            # Update last_data with the prediction for next iteration
            if self.data_config["target_column"] in last_data:
                last_data[self.data_config["target_column"]] = result['predicted_value']
        
        return predictions
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test DataFrame with features and target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare features
        X_test = test_data[self.feature_names] if self.feature_names else test_data.drop(columns=[self.data_config["target_column"]])
        y_test = test_data[self.data_config["target_column"]]
        
        # Scale features if scaler is available
        if self.scaler:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test.values
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics

def main():
    """Example usage of the predictor."""
    # Initialize predictor
    predictor = AirQualityPredictor()
    
    # Load model
    predictor.load_model()
    
    # Example prediction
    sample_data = {
        'DateTime': '2024-04-23 14:00:00',
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
    
    result = predictor.predict(sample_data)
    print(f"Prediction result: {result}")
    
    # Forecast future values
    future_predictions = predictor.predict_future(sample_data, horizon=6)
    print(f"Future predictions: {future_predictions}")

if __name__ == "__main__":
    main()