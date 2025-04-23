"""
Enhanced Kafka producer for streaming air quality data with proper error handling and logging.
"""

import pandas as pd
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.config import get_config

# Initialize logger
logger = setup_logger("kafka_producer")

class AirQualityProducer:
    """Enhanced Kafka producer for air quality data streaming."""
    
    def __init__(self):
        """Initialize the producer with configuration."""
        self.kafka_config = get_config("kafka")
        self.data_config = get_config("data")
        self.producer = None
        
    def create_producer(self) -> KafkaProducer:
        """
        Create and configure Kafka producer instance.
        
        Returns:
            KafkaProducer: Configured producer instance
        """
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks=self.kafka_config["producer_acks"],
                retries=self.kafka_config["producer_retries"],
                compression_type='gzip',  # Enable compression
                max_in_flight_requests_per_connection=5,
                linger_ms=5  # Small delay to batch messages
            )
            logger.info(f"Kafka producer created successfully. Bootstrap servers: {self.kafka_config['bootstrap_servers']}")
            return producer
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the air quality dataset for streaming.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("Starting data preprocessing...")
        
        try:
            # Replace -200 with NaN (missing values)
            df_clean = df.replace(-200, pd.NA)
            
            # Convert date and time columns to datetime
            if 'Date' in df.columns and 'Time' in df.columns:
                try:
                    # Handle the date format carefully
                    df_clean['Date'] = df_clean['Date'].astype(str).str.strip()
                    df_clean['Time'] = df_clean['Time'].astype(str).str.strip()
                    
                    # Try primary parsing method
                    try:
                        df_clean['DateTime'] = pd.to_datetime(
                            df_clean['Date'] + ' ' + df_clean['Time'],
                            format='%d/%m/%Y %H.%M.%S',
                            errors='coerce'
                        )
                    except Exception:
                        # Fallback method
                        df_clean['DateTime'] = pd.to_datetime(
                            df_clean['Date'] + ' ' + df_clean['Time'],
                            errors='coerce'
                        )
                    
                    # Check for parsing failures
                    nat_count = df_clean['DateTime'].isna().sum()
                    if nat_count > 0:
                        logger.warning(f"{nat_count} dates could not be parsed")
                        if nat_count == len(df_clean):
                            # All dates failed - use synthetic dates
                            logger.warning("Using synthetic dates")
                            df_clean['DateTime'] = pd.date_range(
                                start='2004-03-10', 
                                periods=len(df_clean), 
                                freq='H'
                            )
                        else:
                            # Fill NaT values with interpolation or forward fill
                            df_clean['DateTime'] = df_clean['DateTime'].fillna(method='ffill')
                except Exception as e:
                    logger.error(f"Failed to parse dates: {e}")
                    df_clean['DateTime'] = pd.date_range(
                        start='2004-03-10', 
                        periods=len(df_clean), 
                        freq='H'
                    )
            
            # Forward fill missing values
            df_clean = df_clean.ffill()
            
            # Backward fill any remaining missing values
            df_clean = df_clean.bfill()
            
            # Fill any remaining missing values with column mean
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    mean_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(mean_val)
                    logger.info(f"Filled {col} missing values with mean: {mean_val:.2f}")
            
            logger.info(f"Preprocessing complete. Shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the air quality dataset.
        
        Returns:
            Loaded dataframe
        """
        filepath = self.data_config["raw_data_path"]
        logger.info(f"Loading dataset from {filepath}")
        
        try:
            # Try different separators if needed
            for sep in [';', ',']:
                try:
                    df = pd.read_csv(filepath, sep=sep, decimal=',')
                    if df.shape[1] > 1:  # Check if parsing was successful
                        logger.info(f"Dataset loaded successfully with separator '{sep}'. Shape: {df.shape}")
                        return df
                except Exception:
                    continue
            
            # If all attempts fail, raise error
            raise ValueError("Could not parse the CSV file with common separators")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def convert_to_serializable(self, val):
        """Convert non-JSON serializable values to serializable formats."""
        if pd.isna(val):
            return None
        elif isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()
        elif isinstance(val, (int, float, str, bool, type(None))):
            return val
        elif isinstance(val, pd.Series):
            return val.to_dict()
        elif isinstance(val, pd.DataFrame):
            return val.to_dict('records')
        else:
            return str(val)
    
    def stream_data(self, data: pd.DataFrame):
        """
        Stream the dataset to Kafka topic.
        
        Args:
            data: Preprocessed dataframe
        """
        logger.info(f"Starting to stream data to Kafka topic: {self.kafka_config['topic']}")
        
        records_sent = 0
        errors = 0
        batch_count = 0
        
        try:
            for idx, row in data.iterrows():
                try:
                    # Convert row to dictionary with serializable values
                    row_dict = {
                        col: self.convert_to_serializable(val) 
                        for col, val in row.items()
                    }
                    
                    # Add metadata
                    row_dict['timestamp'] = datetime.now().isoformat()
                    row_dict['record_id'] = records_sent
                    
                    # Send the message
                    future = self.producer.send(
                        self.kafka_config["topic"], 
                        value=row_dict
                    )
                    
                    # Wait for the message to be sent
                    future.get(timeout=10)
                    
                    records_sent += 1
                    
                    # Log progress
                    if records_sent % 100 == 0:
                        logger.info(f"Records sent: {records_sent}")
                        batch_count += 1
                    
                    # Simulate real-time delay
                    time.sleep(self.kafka_config["simulation_speed"])
                    
                except KafkaError as ke:
                    logger.error(f"Kafka error sending record {records_sent}: {ke}")
                    errors += 1
                except Exception as e:
                    logger.error(f"Error sending record {records_sent}: {e}")
                    errors += 1
                
                # Stop if too many errors
                if errors > 10:
                    logger.error("Too many errors. Stopping producer.")
                    break
            
            logger.info(f"Streaming completed. Records sent: {records_sent}, Errors: {errors}")
            
        except KeyboardInterrupt:
            logger.info("Producer interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in streaming: {e}")
        finally:
            # Flush remaining messages
            if self.producer:
                self.producer.flush()
                logger.info("Producer flushed successfully")
    
    def run(self):
        """Run the producer pipeline."""
        try:
            # Create producer
            self.producer = self.create_producer()
            
            # Load and preprocess data
            df = self.load_data()
            df_clean = self.preprocess_data(df)
            
            # Start streaming
            self.stream_data(df_clean)
            
        except Exception as e:
            logger.error(f"Producer failed: {e}")
            raise
        finally:
            # Close producer
            if self.producer:
                self.producer.close()
                logger.info("Producer closed successfully")


def main():
    """Main entry point."""
    producer = AirQualityProducer()
    producer.run()


if __name__ == "__main__":
    main()