"""
Enhanced Kafka consumer for processing streamed air quality data with MLflow integration.
"""

import json
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import mlflow
import pickle
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import get_config

# Initialize logger
logger = setup_logger("kafka_consumer")

class AirQualityConsumer:
    """Enhanced Kafka consumer for air quality data processing with ML integration."""
    
    def __init__(self):
        """Initialize the consumer with configuration."""
         # Load feature names
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))  # path to src/kafka
            features_path = os.path.join(base_dir, "../ml/feature_names.json")
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load feature names: {e}")
            self.feature_names = []
        self.api_url = "http://127.0.0.1:8080/predict"  # changes with docker
        self.kafka_config = get_config("kafka")
        self.data_config = get_config("data")
        self.mlflow_config = get_config("mlflow")
        self.data_config["api_output_path"] = Path(self.data_config["api_output_path"])
        self.data_config["processed_data_path"] = Path(self.data_config["processed_data_path"])
        self.consumer = None
        self.model = None
        self.data_buffer = []
        
    def create_consumer(self) -> KafkaConsumer:
        """
        Create and configure Kafka consumer instance.
        
        Returns:
            KafkaConsumer: Configured consumer instance
        """
        try:
            consumer = KafkaConsumer(
                self.kafka_config["topic"],
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                group_id=self.kafka_config["consumer_group"],
                auto_offset_reset=self.kafka_config["auto_offset_reset"],
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                max_poll_records=self.kafka_config["batch_size"],
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )
            logger.info(f"Kafka consumer created successfully. Topic: {self.kafka_config['topic']}")
            return consumer
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            raise
    
    def load_model(self):
        """Load the trained model from MLflow."""
        try:
            # Initialize MLflow
            mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
            
            # Load the latest production model
            model_name = self.mlflow_config["model_registry_name"]
            try:
                model_uri = f"models:/{model_name}/Production"
                self.model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded production model: {model_name}")
            except Exception:
                # Fallback to latest model if no production version
                model_uri = f"models:/{model_name}/latest"
                self.model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Loaded latest model: {model_name}")
                
        except Exception as e:
            logger.warning(f"Failed to load MLflow model: {e}")
            # Load fallback model if exists
            fallback_path = self.mlflow_config["artifact_path"] / "best_model.pkl"
            if fallback_path.exists():
                with open(fallback_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded fallback model")
            else:
                logger.error("No model available for predictions")
                self.model = None
    

    def process_message(self, message) -> Dict[str, Any]:
        data = message.value
        data['processed_at'] = datetime.now().isoformat()

        # To safely parse timestamp
        dt_str = data.get("DateTime")
        if not dt_str:
            logger.warning("Missing DateTime field in message.")
            return None

        dt = pd.to_datetime(dt_str)
        data["hour"], data["day"], data["month"] = dt.hour, dt.day, dt.month

        # Build only the features the model/API expects, using .get to avoid KeyError
        features_only = {
            "DateTime": dt_str,
            "CO(GT)"        : data.get("CO(GT)"),
            "PT08.S1(CO)"   : data.get("PT08.S1(CO)"),
            "NMHC(GT)"      : data.get("NMHC(GT)"),
            "C6H6(GT)"      : data.get("C6H6(GT)"),
            "T"             : data.get("T"),
            "RH"            : data.get("RH"),
            "hour": data.get("hour"),
            "day": data.get("day"),
            "month": data.get("month")
        }

        # Call the prediction API
        try:
            resp = requests.post(self.api_url, json=features_only)
            if resp.status_code == 200:
                result = resp.json()
                prediction = result.get("prediction")
                data["predicted_CO"] = prediction
                logger.info(f"Prediction received: {prediction}")
            else:
                logger.error(f"API error {resp.status_code}: {resp.text}")
                prediction = None
                data["predicted_CO"] = None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to reach prediction API: {e}")
            prediction = None
            data["predicted_CO"] = None

        # Saving both features and prediction
        self.save_api_output(features_only, prediction)

        return data

        
    def save_api_output(self, features: Dict[str, Any], prediction: Any):
        try:

            output_path = Path(self.data_config["api_output_path"])
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        # Prepare data to be saved (features + prediction)
            output_data = {**features, "predicted_CO": prediction, "processed_at": datetime.now().isoformat()}
        
        # Determine file path for saving API output
            output_dir = self.data_config["api_output_path"].parent
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        
            output_file = self.data_config["api_output_path"]
        
            # Check if the file exists, append if so, else create a new one
            file_exists = output_file.exists()

            # Append data to the file
            df = pd.DataFrame([output_data])
            if file_exists:
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)

            logger.info(f"Saved API output to {output_file}")

        except Exception as e:
            logger.error(f"Error saving API output: {e}")

    
    def save_data(self, data_list: List[Dict[str, Any]]):
        """
        Save the processed data to a CSV file.
        
        Args:
            data_list: List of processed data dictionaries
        """
        if not data_list:
            logger.warning("No data to save")
            return
        
        try:
            # Create the output directory if it doesn't exist
            output_dir = self.data_config["processed_data_path"].parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data_list)
            
            # Check if the output file already exists
            output_file = self.data_config["processed_data_path"]
            file_exists = output_file.exists()
            
            # Save to CSV
            if file_exists:
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, index=False)
            
            logger.info(f"Saved {len(data_list)} records to {output_file}")
            
            # Also save a separate file for monitoring
            monitoring_file = output_dir / f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(monitoring_file, index=False)
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def consume_data(self):
        """Consume data from Kafka topic and process it."""
        logger.info(f"Starting to consume data from Kafka topic: {self.kafka_config['topic']}")
        
        messages_processed = 0
        prediction_errors = 0
        
        try:
            while True:
                # Poll for messages
                msg_batch = self.consumer.poll(timeout_ms=1000)
                
                if not msg_batch:
                    continue
                
                for tp, messages in msg_batch.items():
                    for message in messages:
                        processed_data = self.process_message(message)
                        
                        if processed_data:
                            self.data_buffer.append(processed_data)
                            messages_processed += 1
                            
                            # Track prediction errors
                            if 'prediction_error' in processed_data and processed_data['prediction_error'] is not None:
                                if processed_data.get('percentage_error', 0) > 10:  # 10% error threshold
                                    prediction_errors += 1
                                    logger.warning(f"High prediction error: {processed_data['percentage_error']:.2f}%")
                            
                            # Save data in batches
                            if len(self.data_buffer) >= self.kafka_config["batch_size"]:
                                self.save_data(self.data_buffer)
                                self.data_buffer = []
                            
                            # Log progress
                            if messages_processed % 100 == 0:
                                error_rate = (prediction_errors / messages_processed) * 100 if messages_processed > 0 else 0
                                logger.info(f"Messages processed: {messages_processed}, Prediction error rate: {error_rate:.2f}%")
                
                # Commit offsets
                self.consumer.commit()
        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Error in consume_data: {e}")
        finally:
            # Save any remaining data
            if self.data_buffer:
                self.save_data(self.data_buffer)
            
            # Log final statistics
            logger.info(f"Consumer stopped. Total messages processed: {messages_processed}")
            if messages_processed > 0:
                final_error_rate = (prediction_errors / messages_processed) * 100
                logger.info(f"Final prediction error rate: {final_error_rate:.2f}%")
    
    def run(self):
        """Run the consumer pipeline."""
        try:
            # Create consumer
            self.consumer = self.create_consumer()
            
            # Load model
            # self.load_model()
            
            # Start consuming
            self.consume_data()
            
        except Exception as e:
            logger.error(f"Consumer failed: {e}")
            raise
        finally:
            # Close consumer
            if self.consumer:
                self.consumer.close()
                logger.info("Consumer closed successfully")


def main():
    """Main entry point."""
    consumer = AirQualityConsumer()
    consumer.run()


if __name__ == "__main__":
    main()