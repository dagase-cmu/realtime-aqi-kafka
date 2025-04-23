"""
Configuration settings for the project.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Kafka configuration
KAFKA_CONFIG = {
    "bootstrap_servers": "localhost:9092",
    "topic": "air-quality-data",
    "consumer_group": "air-quality-consumer-group",
    "producer_acks": "all",
    "producer_retries": 3,
    "auto_offset_reset": "earliest",
    "batch_size": 100,
    "simulation_speed": 0.1  # Delay between messages in seconds
}

# Data configuration
DATA_CONFIG = {
    "raw_data_path": DATA_DIR / "raw" / "AirQualityUCI.csv",
    "processed_data_path": DATA_DIR / "processed" / "processed_air_quality_data.csv",
    "feature_columns": [
        "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
        "PT08.S5(O3)", "T", "RH", "AH"
    ],
    "target_column": "CO(GT)",
    "date_column": "DateTime",
    "time_column": "Time"
}

# MLflow configuration
MLFLOW_CONFIG = {
    "tracking_uri": "sqlite:///mlflow.db",
    "experiment_name": "air_quality_prediction",
    "artifact_path": MODELS_DIR / "artifacts",
    "model_registry_name": "air_quality_model"
}

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators_rf": 100,
    "learning_rate_xgb": 0.1,
    "lstm_units": 50,
    "epochs": 50,
    "batch_size": 32
}

# Monitoring configuration
MONITORING_CONFIG = {
    "dashboard_port": 8050,
    "reference_data_path": DATA_DIR / "processed" / "reference_data.csv",
    "alert_threshold": 0.1,  # Alert if model performance degrades by 10%
    "monitoring_interval": 3600  # Check every hour
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_dir": DATA_DIR / "logs"
}

def get_config(section: str) -> Dict[str, Any]:
    """
    Get configuration for a specific section.
    
    Args:
        section: Configuration section name
    
    Returns:
        Configuration dictionary
    """
    config_mapping = {
        "kafka": KAFKA_CONFIG,
        "data": DATA_CONFIG,
        "mlflow": MLFLOW_CONFIG,
        "model": MODEL_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "logging": LOGGING_CONFIG
    }
    
    if section not in config_mapping:
        raise KeyError(f"Configuration section '{section}' not found")
    
    return config_mapping[section]