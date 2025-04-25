"""
Feature engineering module for air quality prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import get_config

logger = setup_logger("feature_engineering")

class AirQualityFeatureEngineer:
    """Feature engineering for air quality data."""
    
    def __init__(self):
        """Initialize feature engineer with configuration."""
        self.data_config = get_config("data")
        self.model_config = get_config("model")
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from datetime column.
        
        Args:
            df: DataFrame with DateTime column
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features...")
        
        df_features = df.copy()
        
        # Ensure DateTime is in datetime format
        if 'DateTime' in df_features.columns:
            df_features['DateTime'] = pd.to_datetime(df_features['DateTime'], errors='coerce')
            if df_features['DateTime'].isna().any():
                df_features['DateTime'].fillna(pd.Timestamp.now(), inplace=True)
        else:
            raise ValueError("DateTime column not found in DataFrame")
        
        # Extract temporal components
        df_features['hour'] = df_features['DateTime'].dt.hour
        df_features['day'] = df_features['DateTime'].dt.day
        df_features['month'] = df_features['DateTime'].dt.month
        df_features['dayofweek'] = df_features['DateTime'].dt.dayofweek
        df_features['quarter'] = df_features['DateTime'].dt.quarter
        df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
        
        # ... rest of the method remains the same
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create lag features for the target variable.
        
        Args:
            df: DataFrame
            target_col: Target column name
            lags: List of lag values
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Creating lag features...")
        
        df_features = df.copy()
        
        for lag in lags:
            df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
        
        logger.info(f"Created {len(lags)} lag features")
        
        return df_features
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str, windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: DataFrame
            target_col: Target column name
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info("Creating rolling features...")
        
        df_features = df.copy()
        
        for window in windows:
            # Rolling mean
            df_features[f'{target_col}_rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
            
            # Rolling standard deviation
            df_features[f'{target_col}_rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
            
            # Rolling min and max
            df_features[f'{target_col}_rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
            df_features[f'{target_col}_rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
        
        logger.info(f"Created {len(windows) * 4} rolling features")
        
        return df_features
    
    def create_exponential_features(self, df: pd.DataFrame, target_col: str, 
                                   alphas: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> pd.DataFrame:
        """
        Create exponential moving average features.
        
        Args:
            df: DataFrame
            target_col: Target column name
            alphas: List of alpha values for EMA
            
        Returns:
            DataFrame with EMA features
        """
        logger.info("Creating exponential moving average features...")
        
        df_features = df.copy()
        
        for alpha in alphas:
            df_features[f'{target_col}_ema_{alpha}'] = df_features[target_col].ewm(alpha=alpha).mean()
        
        logger.info(f"Created {len(alphas)} EMA features")
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified columns.
        
        Args:
            df: DataFrame
            feature_pairs: List of feature pairs for interaction
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        df_features = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication interaction
                df_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Ratio interaction (with small epsilon to avoid division by zero)
                df_features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        logger.info(f"Created {len(feature_pairs) * 2} interaction features")
        
        return df_features
    
    def handle_multicollinearity(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            df: DataFrame
            threshold: Correlation threshold
            
        Returns:
            DataFrame with reduced features
        """
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        logger.info(f"Dropping {len(to_drop)} correlated features: {to_drop}")
        
        return df.drop(columns=to_drop)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Create datetime column if not exists
        if 'DateTime' not in df.columns and 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
        
        # Create temporal features
        df_features = self.create_temporal_features(df)
        
        # Create lag features
        target_col = self.data_config["target_column"]
        df_features = self.create_lag_features(df_features, target_col)
        
        # Create rolling features
        df_features = self.create_rolling_features(df_features, target_col)
        
        # Create exponential features
        df_features = self.create_exponential_features(df_features, target_col)
        
        # Create interaction features for pollutants
        pollutant_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
                         'PT08.S4(NO2)', 'PT08.S5(O3)']
        
        interaction_pairs = []
        for i, col1 in enumerate(pollutant_cols):
            for col2 in pollutant_cols[i+1:]:
                interaction_pairs.append((col1, col2))
        
        df_features = self.create_interaction_features(df_features, interaction_pairs)
        
        # Drop rows with NaN values created by feature engineering
        original_rows = len(df_features)
        df_features = df_features.dropna()
        
        logger.info(f"Dropped {original_rows - len(df_features)} rows with NaN values")
        
        # Handle multicollinearity
        df_features = self.handle_multicollinearity(df_features)
        
        # Select final features
        self.feature_names = [col for col in df_features.columns if col not in ['Date', 'Time', 'DateTime', target_col]]
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df_features
    
    def fit_scaler(self, X: pd.DataFrame):
        """Fit the scaler on training data."""
        self.scaler.fit(X)
        
    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale features using fitted scaler."""
        return self.scaler.transform(X)
    
    def inverse_scale(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features."""
        return self.scaler.inverse_transform(X_scaled)
    
    # Update the prepare_single_prediction method in AirQualityFeatureEngineer class

    def prepare_single_prediction(self, data: Dict) -> pd.DataFrame:
        """Prepare a single data point for prediction."""
        df = pd.DataFrame([data])
        
        # Ensure DateTime is properly converted
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
            # If conversion fails, use current time
            if df['DateTime'].isna().any():
                df['DateTime'] = pd.Timestamp.now()
        else:
            # If no DateTime column, use current time
            df['DateTime'] = pd.Timestamp.now()
        
        # Now we can safely use .dt accessor
        df = self.create_temporal_features(df)
        
        # For single predictions, we set lag and rolling features to the current value
        target_col = self.data_config["target_column"]
        if target_col in df.columns:
            current_value = df[target_col].iloc[0]
            
            # Add lag features
            for lag in [1, 3, 6, 12, 24]:
                df[f'{target_col}_lag_{lag}'] = current_value
            
            # Add rolling features
            for window in [3, 6, 12, 24]:
                df[f'{target_col}_rolling_mean_{window}'] = current_value
                df[f'{target_col}_rolling_std_{window}'] = 0
                df[f'{target_col}_rolling_min_{window}'] = current_value
                df[f'{target_col}_rolling_max_{window}'] = current_value
            
            # Add EMA features
            for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
                df[f'{target_col}_ema_{alpha}'] = current_value
        
        # Create interaction features if possible
        pollutant_cols = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
                        'PT08.S4(NO2)', 'PT08.S5(O3)']
        
        interaction_pairs = []
        for i, col1 in enumerate(pollutant_cols):
            for col2 in pollutant_cols[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    interaction_pairs.append((col1, col2))
        
        if interaction_pairs:
            df = self.create_interaction_features(df, interaction_pairs)
        
        return df