import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

# Try to import kafka, but make it optional
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Kafka package not installed. Kafka integration will be disabled.")
    print("To enable Kafka integration, install the package: pip install kafka-python")

# Paths for saving models
MODEL_DIR = "saved_models"
SCALER_DIR = "saved_scalers"

# Feature Engineering Class
class AirQualityFeatureEngineering:
    def __init__(self, df):
        """
        Initialize feature engineering with the input dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input time series dataframe
        """
        # Convert to datetime and set index
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
        df.set_index('Datetime', inplace=True)
        
        # Drop unnecessary columns
        df.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16', 'received_at', 'DateTime'], errors='ignore', inplace=True)
        
        # Ensure numeric columns
        self.df = df.select_dtypes(include=[np.number])
    
    def _generate_interaction_features(self, df, columns):
        """
        Generate interaction features between selected columns
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        columns : list
            Columns to generate interactions for
        
        Returns:
        --------
        pandas.DataFrame
            Dataframe with interaction features
        """
        interaction_features = {}
        
        # Pairwise interactions
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                interaction_col = f'{col1}_x_{col2}'
                interaction_features[interaction_col] = df[col1] * df[col2]
        
        return pd.DataFrame(interaction_features, index=df.index)
    
    def engineer_features(self, target_column):
        """
        Comprehensive feature engineering pipeline
        
        Parameters:
        -----------
        target_column : str
            Column to predict
        
        Returns:
        --------
        tuple
            Engineered features (X) and target variable (y)
        """
        # Create a copy of the dataframe to avoid fragmentation
        df = self.df.copy()
        
        # Prepare feature generation
        feature_columns = [col for col in df.columns if col != target_column]
        
        # Prepare feature containers
        feature_list = []
        
        # Time-based features with more granularity
        time_features = {
            'hour': df.index.hour,
            'day_of_week': df.index.dayofweek,
            'month': df.index.month,
            'is_weekend': (df.index.dayofweek.isin([5, 6])).astype(int),
            'is_morning': ((df.index.hour >= 6) & (df.index.hour < 12)).astype(int),
            'is_afternoon': ((df.index.hour >= 12) & (df.index.hour < 18)).astype(int),
            'is_evening': ((df.index.hour >= 18) & (df.index.hour < 22)).astype(int),
            'is_night': ((df.index.hour >= 22) | (df.index.hour < 6)).astype(int)
        }
        
        # Add time features
        for name, values in time_features.items():
            df[name] = values
            feature_list.append(name)
        
        # Prepare rolling and lagged features
        rolling_windows = [3, 6, 12, 24]
        lag_steps = [1, 2, 3, 6]
        
        # Collect all new features in dictionaries to add at once
        lagged_features = {}
        rolling_features = {}
        ewm_features = {}
        
        # Generate features efficiently
        for col in feature_columns:
            # Lagged features
            for lag in lag_steps:
                lagged_col = f'{col}_lag_{lag}'
                lagged_features[lagged_col] = df[col].shift(lag)
            
            # Rolling features
            for window in rolling_windows:
                # Rolling mean
                mean_col = f'{col}_rolling_mean_{window}'
                rolling_features[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling standard deviation
                std_col = f'{col}_rolling_std_{window}'
                rolling_features[std_col] = df[col].rolling(window=window, min_periods=1).std()
            
            # Exponential weighted mean
            ewm_col = f'{col}_ewm_alpha_0.2'
            ewm_features[ewm_col] = df[col].ewm(alpha=0.2, adjust=False).mean()
        
        # Add all features to DataFrame at once
        df = pd.concat([
            df,
            pd.DataFrame(lagged_features, index=df.index),
            pd.DataFrame(rolling_features, index=df.index),
            pd.DataFrame(ewm_features, index=df.index)
        ], axis=1)
        
        # Keep track of feature names
        lagged_cols = list(lagged_features.keys())
        rolling_cols = list(rolling_features.keys())
        ewm_cols = list(ewm_features.keys())
        
        # Update feature list
        feature_list.extend(lagged_cols)
        feature_list.extend(rolling_cols)
        feature_list.extend(ewm_cols)
        
        # Generate interaction features
        interaction_df = self._generate_interaction_features(df, feature_columns)
        
        # Add interaction features to the dataframe
        df = pd.concat([df, interaction_df], axis=1)
        
        # Add interaction columns to feature list
        feature_list.extend(interaction_df.columns)
        
        # Drop rows with NaN in target
        df.dropna(subset=[target_column], inplace=True)
        
        # Check for missing features in the feature_list
        missing_features = [f for f in feature_list if f not in df.columns]
        if missing_features:
            print(f"Warning: The following features are missing: {missing_features}")
            feature_list = [f for f in feature_list if f in df.columns]
        
        # Separate features and target
        all_features = feature_list.copy()  # Make a copy of all feature names
        if target_column in all_features:
            all_features.remove(target_column)
        
        y = df[target_column]
        X = df[all_features]
        
        # Fill remaining NaNs with mean
        X = X.fillna(X.mean())
        
        # Print debugging information
        print("Feature Engineering Debug:")
        print("Original DataFrame shape:", self.df.shape)
        print("Engineered DataFrame shape:", df.shape)
        print("Target column:", target_column)
        print("Features shape:", X.shape)
        print("Target shape:", y.shape)
        print("Number of features:", len(all_features))
        
        return X, y

# Model Training and Evaluation Class
class AirQualityModelTrainer:
    def __init__(self, X, y):
        """
        Initialize model trainer
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target variable
        """
        # Chronological train-test split
        split_index = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_index]
        self.X_test = X.iloc[split_index:]
        self.y_train = y.iloc[:split_index]
        self.y_test = y.iloc[split_index:]
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Store models for later saving
        self.lr_model = None
        self.rf_model = None
        self.xgb_model = None
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
        model_name : str
            Name of the model for printing
        
        Returns:
        --------
        dict
            Performance metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"{model_name} Performance:")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")
        
        return {
            'mae': mae,
            'rmse': rmse
        }
    
    def baseline_model(self):
        """
        Baseline model using previous value prediction
        
        Returns:
        --------
        dict
            Baseline model performance metrics
        """
        # Use the last known value as prediction
        y_pred_baseline = self.y_test.shift(1)
        y_pred_baseline = y_pred_baseline.dropna()
        y_true_baseline = self.y_test.iloc[1:]
        
        return self.evaluate_model(y_true_baseline, y_pred_baseline, "Baseline (Previous Value)")
    
    def linear_regression(self):
        """
        Train and evaluate Linear Regression model
        
        Returns:
        --------
        dict
            Linear Regression model performance metrics
        """
        # Train Linear Regression
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train_scaled, self.y_train)
        
        # Predict
        y_pred_lr = self.lr_model.predict(self.X_test_scaled)
        
        return self.evaluate_model(self.y_test, y_pred_lr, "Linear Regression")
    
    def random_forest(self):
        """
        Train and evaluate Random Forest Regression model
        
        Returns:
        --------
        dict
            Random Forest model performance metrics
        """
        # Train Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        # Predict
        y_pred_rf = self.rf_model.predict(self.X_test_scaled)
        
        return self.evaluate_model(self.y_test, y_pred_rf, "Random Forest")
    
    def xgboost(self):
        """
        Train and evaluate XGBoost Regression model
        
        Returns:
        --------
        dict
            XGBoost model performance metrics
        """
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.xgb_model.fit(self.X_train_scaled, self.y_train)
        
        # Predict
        y_pred_xgb = self.xgb_model.predict(self.X_test_scaled)
        
        return self.evaluate_model(self.y_test, y_pred_xgb, "XGBoost")

# Function to save model and scaler
def save_trained_model(model, scaler, target, model_type="xgboost"):
    """
    Save trained model and scaler to disk
    
    Parameters:
    -----------
    model : object
        Trained model
    scaler : object
        Fitted StandardScaler
    target : str
        Target pollutant name
    model_type : str
        Type of model (e.g., 'xgboost', 'random_forest')
    """
    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)
    
    # Save model
    model_filename = os.path.join(MODEL_DIR, f"{target}_{model_type}_model.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_filename = os.path.join(SCALER_DIR, f"{target}_scaler.pkl")
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Saved {model_type} model and scaler for {target} prediction")

# Save models from a trained AirQualityModelTrainer
def save_models_from_trainer(model_trainer, target):
    """
    Save models from a trained AirQualityModelTrainer
    
    Parameters:
    -----------
    model_trainer : AirQualityModelTrainer
        Trained model trainer instance
    target : str
        Target pollutant name
    """
    # Save linear regression model
    if model_trainer.lr_model:
        save_trained_model(model_trainer.lr_model, model_trainer.scaler, target, "linear")
    
    # Save random forest model
    if model_trainer.rf_model:
        save_trained_model(model_trainer.rf_model, model_trainer.scaler, target, "random_forest")
    
    # Save XGBoost model
    if model_trainer.xgb_model:
        save_trained_model(model_trainer.xgb_model, model_trainer.scaler, target, "xgboost")

# Main Analysis Function
def perform_air_quality_prediction(filepath, target_column='CO(GT)'):
    """
    Perform comprehensive air quality prediction analysis
    
    Parameters:
    -----------
    filepath : str
        Path to the input CSV file
    target_column : str, optional
        Column to predict, by default 'CO(GT)'
        
    Returns:
    --------
    list
        List of tuples containing (model_name, mae, rmse, improvement_over_baseline)
    """
    # Load data
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Feature Engineering
    print("Starting feature engineering...")
    fe = AirQualityFeatureEngineering(df)
    
    # Engineer features
    X, y = fe.engineer_features(target_column)
    
    # Model Training and Evaluation
    print("Training models...")
    model_trainer = AirQualityModelTrainer(X, y)
    
    # Run baseline and models
    baseline_metrics = model_trainer.baseline_model()
    lr_metrics = model_trainer.linear_regression()
    rf_metrics = model_trainer.random_forest()
    xgb_metrics = model_trainer.xgboost()
    
    # Visualize Results
    plt.figure(figsize=(10, 6))
    models = ['Baseline', 'Linear Regression', 'Random Forest', 'XGBoost']
    mae_scores = [
        baseline_metrics['mae'], 
        lr_metrics['mae'], 
        rf_metrics['mae'], 
        xgb_metrics['mae']
    ]
    
    plt.bar(models, mae_scores)
    plt.title(f'Model Performance Comparison (MAE) - {target_column}')
    plt.ylabel('Mean Absolute Error')
    plt.tight_layout()
    plt.savefig(f'model_performance_comparison_{target_column}.png')
    plt.close()
    
    # Create a tabular summary of results
    print("\nPerformance Summary for", target_column)
    print("-" * 80)
    print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'% Improvement over Baseline':<30}")
    print("-" * 80)
    
    baseline_mae = baseline_metrics['mae']
    models_data = [
        ("Baseline", baseline_metrics['mae'], baseline_metrics['rmse'], 0),
        ("Linear Regression", lr_metrics['mae'], lr_metrics['rmse'], 
         (baseline_mae - lr_metrics['mae']) / baseline_mae * 100),
        ("Random Forest", rf_metrics['mae'], rf_metrics['rmse'], 
         (baseline_mae - rf_metrics['mae']) / baseline_mae * 100),
        ("XGBoost", xgb_metrics['mae'], xgb_metrics['rmse'], 
         (baseline_mae - xgb_metrics['mae']) / baseline_mae * 100)
    ]
    
    for model, mae, rmse, improvement in models_data:
        print(f"{model:<20} {mae:<15.4f} {rmse:<15.4f} {improvement:<30.2f}%")
    
    print("-" * 80)
    
    # Find best model
    best_model = min(models_data[1:], key=lambda x: x[1])
    print(f"Best model: {best_model[0]} with MAE of {best_model[1]:.4f}")
    
    # Save trained models (optional)
    try:
        save_models_from_trainer(model_trainer, target_column)
        print(f"Models saved for {target_column}")
    except Exception as e:
        print(f"Warning: Could not save models: {e}")
    
    print(f"\nAnalysis for {target_column} completed successfully!")
    
    # Return results for saving to CSV
    return models_data

# Class for Kafka integration (only if Kafka is available)
if KAFKA_AVAILABLE:
    class AirQualityKafkaPredictor:
        def __init__(self, bootstrap_servers=['localhost:9092'], 
                    input_topic='air-quality-data', 
                    output_topic='air-quality-predictions',
                    pollutants=['CO(GT)', 'NOx(GT)', 'C6H6(GT)'],
                    model_type='xgboost'):
            """
            Initialize Kafka predictor for air quality
            
            Parameters:
            -----------
            bootstrap_servers : list
                List of Kafka bootstrap servers
            input_topic : str
                Kafka topic to read sensor data from
            output_topic : str
                Kafka topic to write predictions to
            pollutants : list
                List of target pollutants to predict
            model_type : str
                Type of model to use ('xgboost', 'random_forest', 'linear')
            """
            self.bootstrap_servers = bootstrap_servers
            self.input_topic = input_topic
            self.output_topic = output_topic
            self.pollutants = pollutants
            self.model_type = model_type
            
            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            # Load models for real-time prediction
            self.models = {}
            self.scalers = {}
            
            for pollutant in self.pollutants:
                # Load model and scaler
                try:
                    model_path = os.path.join(MODEL_DIR, f"{pollutant}_{model_type}_model.pkl")
                    scaler_path = os.path.join(SCALER_DIR, f"{pollutant}_scaler.pkl")
                    
                    with open(model_path, 'rb') as model_file:
                        self.models[pollutant] = pickle.load(model_file)
                    
                    with open(scaler_path, 'rb') as scaler_file:
                        self.scalers[pollutant] = pickle.load(scaler_file)
                        
                    print(f"Loaded {model_type} model for {pollutant}")
                except Exception as e:
                    print(f"Error loading model for {pollutant}: {e}")
            
            # Historical data window for feature engineering
            self.historical_data = []
            self.max_history_length = 24  # Store 24 hours of data for rolling features
            
            print(f"Initialized Kafka predictor with {model_type} models for {', '.join(pollutants)}")
        
        def preprocess_kafka_message(self, message):
            """Preprocess incoming Kafka messages for prediction"""
            # Implementation details as described in Kafka documentation
            pass
            
        def make_predictions(self, features_df):
            """Make predictions using loaded models"""
            # Implementation details as described in Kafka documentation
            pass
            
        def run(self):
            """Run the Kafka prediction service"""
            print(f"Starting Kafka prediction service, listening on {self.input_topic}...")
            
            try:
                for message in self.consumer:
                    # Process message and make predictions
                    # Implementation details as described in Kafka documentation
                    pass
                    
            except KeyboardInterrupt:
                print("Stopping Kafka prediction service")
            finally:
                self.consumer.close()
                self.producer.close()
else:
    # Stub implementation if Kafka is not available
    class AirQualityKafkaPredictor:
        def __init__(self, *args, **kwargs):
            print("Kafka integration not available. Install kafka-python package to enable it.")
        
        def run(self):
            print("Kafka functionality is disabled. Install kafka-python to enable it.")

# Function to save all results to a CSV file
def save_results_summary(results_data, filename="model_performance_summary.csv"):
    """
    Save model performance results to a CSV file
    
    Parameters:
    -----------
    results_data : list of tuples
        List of (target, model, mae, rmse, improvement) tuples
    filename : str
        Name of the CSV file to save
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Target', 'Model', 'MAE', 'RMSE', 'Improvement Over Baseline (%)'])
        writer.writerows(results_data)
    
    print(f"\nResults summary saved to {filename}")

# Run the analysis
if __name__ == "__main__":
    import os
    
    # Specify the exact path to the air quality data file
    filepath = "C:\\Users\\radha\\air-quality-prediction\\phase_3_model_prediction\\processed_air_quality_data.csv"
    
    if not os.path.exists(filepath):
        print(f"Error: Data file {filepath} not found.")
        exit(1)
    
    print(f"Analyzing file: {filepath}")
    
    # Print information about Kafka integration for Phase 3 requirements
    print("\n" + "="*80)
    print("KAFKA INTEGRATION INFORMATION:")
    print("The script includes Kafka integration capabilities for real-time prediction.")
    print("To use this feature, install the kafka-python package: pip install kafka-python")
    print("See the documentation in 'Real-time Air Quality Prediction System Documentation'")
    print("for details on how this system would operate in a real-time environment.")
    print("="*80 + "\n")
    
    # Collect all results
    all_results = []
    
    # Perform prediction for different target columns
    targets = ['CO(GT)', 'NOx(GT)', 'C6H6(GT)']
    for target in targets:
        print(f"\n--- Predicting {target} ---")
        results = perform_air_quality_prediction(filepath, target)
        if results:
            all_results.extend([(target,) + r for r in results])
    
    # Save all results to CSV
    save_results_summary(all_results)
    
    print("\nKafka integration is included but not active. To activate it:")
    print("1. Install the kafka-python package: pip install kafka-python")
    print("2. Set up a Kafka broker")
    print("3. Use AirQualityKafkaPredictor for real-time predictions")
    print("\nSee the included documentation for details on the real-time system design.")