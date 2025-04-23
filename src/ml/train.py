"""
ML training script with MLflow integration for air quality prediction.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("ml_training")

class AirQualityModelTrainer:
    """Main class for training air quality prediction models with MLflow tracking."""
    
    def __init__(self):
        """Initialize the trainer with configuration."""
        self.data_config = get_config("data")
        self.model_config = get_config("model")
        self.mlflow_config = get_config("mlflow")
        
        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        mlflow.set_experiment(self.mlflow_config["experiment_name"])
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the data."""
        logger.info("Loading data...")
        data_path = self.data_config["raw_data_path"]
        
        try:
            df = pd.read_csv(data_path, sep=';', decimal=',')
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by handling missing values and converting types."""
        logger.info("Preprocessing data...")
        
        # Create a copy to avoid warnings
        df_clean = df.copy()
        
        # First, identify numeric columns
        numeric_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 
                          'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
                          'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
                          'T', 'RH', 'AH']
        
        # Convert columns to numeric, replacing -200 with NaN
        for col in numeric_columns:
            if col in df_clean.columns:
                # Replace comma with dot for decimal conversion
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.replace(',', '.')
                
                # Convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Replace -200 with NaN
                df_clean[col] = df_clean[col].replace(-200, np.nan)
        
        # Convert date and time columns
        try:
            df_clean['DateTime'] = pd.to_datetime(
                df_clean['Date'] + ' ' + df_clean['Time'],
                format='%d/%m/%Y %H.%M.%S',
                errors='coerce'
            )
        except Exception as e:
            logger.warning(f"Date parsing error: {e}")
            df_clean['DateTime'] = pd.date_range(
                start='2004-03-10', 
                periods=len(df_clean), 
                freq='H'
            )
        
        # Fill missing values
        for col in numeric_columns:
            if col in df_clean.columns:
                # Forward fill first
                df_clean[col] = df_clean[col].fillna(method='ffill')
                # Then backward fill
                df_clean[col] = df_clean[col].fillna(method='bfill')
                # Fill any remaining with mean
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        # Remove any completely empty columns
        df_clean = df_clean.dropna(axis=1, how='all')
        
        logger.info(f"Preprocessing complete. Shape: {df_clean.shape}")
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features including temporal and lag features."""
        logger.info("Creating features...")
        
        df_features = df.copy()
        
        # Ensure DateTime is in the correct format
        if 'DateTime' not in df_features.columns:
            logger.error("DateTime column missing")
            return df_features
        
        # Temporal features
        df_features['hour'] = df_features['DateTime'].dt.hour
        df_features['day'] = df_features['DateTime'].dt.day
        df_features['month'] = df_features['DateTime'].dt.month
        df_features['dayofweek'] = df_features['DateTime'].dt.dayofweek
        
        # Lag features for the target
        target_col = self.data_config["target_column"]
        
        # Ensure target column is numeric
        if target_col in df_features.columns:
            df_features[target_col] = pd.to_numeric(df_features[target_col], errors='coerce')
            
            for lag in [1, 3, 6, 12, 24]:
                df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
            
            # Rolling features
            windows = [3, 6, 12, 24]
            for window in windows:
                df_features[f'{target_col}_rolling_mean_{window}'] = (
                    df_features[target_col].rolling(window=window, min_periods=1).mean()
                )
                df_features[f'{target_col}_rolling_std_{window}'] = (
                    df_features[target_col].rolling(window=window, min_periods=1).std()
                )
            
            # Fill NaN values in rolling std with 0
            for col in df_features.columns:
                if 'rolling_std' in col:
                    df_features[col] = df_features[col].fillna(0)
        
        # Drop rows with NaN values in lag features
        initial_rows = len(df_features)
        df_features = df_features.dropna()
        
        logger.info(f"Dropped {initial_rows - len(df_features)} rows with NaN values")
        logger.info(f"Features created. Shape: {df_features.shape}")
        
        return df_features
    
    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for training."""
        logger.info("Preparing data for training...")
        
        # Select features
        feature_cols = self.data_config["feature_columns"] + [
            'hour', 'day', 'month', 'dayofweek'
        ] + [
            f'{self.data_config["target_column"]}_lag_{lag}' for lag in [1, 3, 6, 12, 24]
        ] + [
            f'{self.data_config["target_column"]}_rolling_mean_{window}' for window in [3, 6, 12, 24]
        ] + [
            f'{self.data_config["target_column"]}_rolling_std_{window}' for window in [3, 6, 12, 24]
        ]
        
        # Filter only columns that exist
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features]
        y = df[self.data_config["target_column"]]
        
        # Ensure all data is numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Drop any remaining rows with NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.model_config["test_size"], 
            random_state=self.model_config["random_state"],
            shuffle=False  # Time series data
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Data prepared. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model."""
        with mlflow.start_run(run_name="Linear_Regression"):
            logger.info("Training Linear Regression model...")
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Linear Regression - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            return model, mae
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model with hyperparameter tuning."""
        with mlflow.start_run(run_name="Random_Forest"):
            logger.info("Training Random Forest model...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            
            # Perform grid search
            rf = RandomForestRegressor(random_state=self.model_config["random_state"])
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Get best model
            model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Random Forest - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            return model, mae
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost model."""
        with mlflow.start_run(run_name="XGBoost"):
            logger.info("Training XGBoost model...")
            
            model = xgb.XGBRegressor(
                n_estimators=self.model_config["n_estimators_rf"],
                learning_rate=self.model_config["learning_rate_xgb"],
                random_state=self.model_config["random_state"]
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("n_estimators", self.model_config["n_estimators_rf"])
            mlflow.log_param("learning_rate", self.model_config["learning_rate_xgb"])
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"XGBoost - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            return model, mae
    
    def train_lstm(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """Train LSTM model."""
        with mlflow.start_run(run_name="LSTM"):
            logger.info("Training LSTM model...")
            
            # Reshape for LSTM
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            
            # Build model
            model = Sequential([
                LSTM(self.model_config["lstm_units"], activation='relu', input_shape=(1, X_train_scaled.shape[1])),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            history = model.fit(
                X_train_lstm, y_train,
                epochs=self.model_config["epochs"],
                batch_size=self.model_config["batch_size"],
                validation_split=0.2,
                verbose=0
            )
            
            # Make predictions
            y_pred = model.predict(X_test_lstm).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("lstm_units", self.model_config["lstm_units"])
            mlflow.log_param("epochs", self.model_config["epochs"])
            mlflow.log_param("batch_size", self.model_config["batch_size"])
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            logger.info(f"LSTM - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            return model, mae
    
    def save_best_model(self, best_model, scaler):
        """Save the best model locally."""
        logger.info("Saving best model...")
        
        model_dir = self.mlflow_config["artifact_path"]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def train_all_models(self):
        """Train all models and select the best one."""
        # Load and preprocess data
        df = self.load_data()
        df_clean = self.preprocess_data(df)
        df_features = self.create_features(df_clean)
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = self.prepare_data(df_features)
        
        # Train models
        models = {}
        metrics = {}
        
        # Linear Regression
        model_lr, mae_lr = self.train_linear_regression(X_train, X_test, y_train, y_test)
        models['linear_regression'] = model_lr
        metrics['linear_regression'] = mae_lr
        
        # Random Forest
        model_rf, mae_rf = self.train_random_forest(X_train, X_test, y_train, y_test)
        models['random_forest'] = model_rf
        metrics['random_forest'] = mae_rf
        
        # XGBoost
        model_xgb, mae_xgb = self.train_xgboost(X_train, X_test, y_train, y_test)
        models['xgboost'] = model_xgb
        metrics['xgboost'] = mae_xgb
        
        # LSTM
        model_lstm, mae_lstm = self.train_lstm(X_train_scaled, X_test_scaled, y_train, y_test)
        models['lstm'] = model_lstm
        metrics['lstm'] = mae_lstm
        
        # Select best model
        best_model_name = min(metrics, key=metrics.get)
        best_model = models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with MAE: {metrics[best_model_name]:.4f}")
        
        # Save best model
        self.save_best_model(best_model, scaler)
        
        return best_model, best_model_name, metrics

def main():
    """Main entry point."""
    trainer = AirQualityModelTrainer()
    trainer.train_all_models()

if __name__ == "__main__":
    main()