#!/usr/bin/env python3
"""
air_quality_producer.py - Kafka producer for streaming air quality data

This script reads the UCI Air Quality dataset and streams it to a Kafka topic,
simulating real-time data by introducing time delays between records.
"""

import pandas as pd
import json
import time
import logging
from kafka import KafkaProducer
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_TOPIC = 'air-quality-data'
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'

# Dataset configuration
DATASET_PATH = '/mnt/c/Users/radha/air-quality-prediction/data/AirQualityUCI.csv'  # Update this path to your dataset location
SIMULATION_SPEED = 0.1  # Number of seconds to wait between records (60 = 1 minute)

def preprocess_data(df):
    """
    Preprocess the air quality dataset by handling missing values and converting data types.
    """
    logger.info("Preprocessing the dataset...")
    
    # Replace -200 with NaN (missing values)
    df_clean = df.replace(-200, pd.NA)
    
    # Convert date and time columns to datetime
    if 'Date' in df.columns and 'Time' in df.columns:
        try:
            # Try to handle the date format correctly
            df_clean['DateTime'] = pd.to_datetime(
                df_clean['Date'] + ' ' + df_clean['Time'],
                format='%d/%m/%Y %H.%M.%S',
                errors='coerce'  # Convert parsing errors to NaT
            )
        except Exception as e:
            logger.warning(f"Error converting date/time: {e}")
            # Alternative approach if the first method fails
            logger.info("Trying alternative date parsing approach...")
            try:
                # Remove any duplicate dates and handle format more flexibly
                df_clean['Date'] = df_clean['Date'].astype(str).str[:10]  # Take just first 10 chars
                df_clean['DateTime'] = pd.to_datetime(
                    df_clean['Date'] + ' ' + df_clean['Time'],
                    errors='coerce'
                )
            except Exception as e2:
                logger.error(f"Failed alternative date parsing: {e2}")
                # If date parsing fails, create a simple index instead
                df_clean['DateTime'] = pd.date_range(
                    start='2004-03-10', 
                    periods=len(df_clean), 
                    freq='H'
                )
                logger.warning("Using synthetic DateTime index due to parsing errors")
    
    # Forward fill missing values
    df_clean = df_clean.ffill()
    
    # If there are still missing values at the beginning, backward fill
    df_clean = df_clean.bfill()
    
    # If there are any remaining missing values, replace with column mean
    for col in df_clean.select_dtypes(include=['number']).columns:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    logger.info("Preprocessing complete.")
    return df_clean

def load_dataset(filepath):
    """
    Load the UCI Air Quality dataset.
    
    Args:
        filepath (str): Path to the dataset CSV file
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    logger.info(f"Loading dataset from {filepath}")
    try:
        # Read with semicolon separator as per UCI Air Quality dataset format
        df = pd.read_csv(filepath, sep=';', decimal=',')
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def create_kafka_producer():
    """
    Create and return a Kafka producer instance.
    
    Returns:
        KafkaProducer: Configured Kafka producer
    """
    logger.info(f"Creating Kafka producer. Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        logger.info("Kafka producer created successfully")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise

def stream_data_to_kafka(producer, data):
    """
    Stream the dataset to Kafka topic with time delays to simulate real-time data.
    
    Args:
        producer (KafkaProducer): Kafka producer instance
        data (pandas.DataFrame): The preprocessed dataset
    """
    logger.info(f"Starting to stream data to Kafka topic: {KAFKA_TOPIC}")
    
    records_sent = 0
    errors = 0
    
    for _, row in data.iterrows():
        try:
            # Convert row to dictionary, handling non-serializable values
            row_dict = {col: convert_to_serializable(val) for col, val in row.items()}
            
            # Send the row to Kafka
            future = producer.send(KAFKA_TOPIC, value=row_dict)
            future.get(timeout=10)  # Wait for the message to be sent
            
            records_sent += 1
            if records_sent % 100 == 0:
                logger.info(f"Records sent: {records_sent}")
            
            # Simulate real-time data by waiting between records
            time.sleep(SIMULATION_SPEED)
            
        except Exception as e:
            logger.error(f"Error sending record to Kafka: {e}")
            errors += 1
            if errors > 10:
                logger.error("Too many errors. Stopping.")
                break
    
    logger.info(f"Finished streaming data. Records sent: {records_sent}, Errors: {errors}")

def convert_to_serializable(val):
    """
    Convert non-JSON serializable values to serializable formats.
    
    Args:
        val: Value to convert
    
    Returns:
        Serializable version of the value
    """
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

def main():
    """Main function to run the Kafka producer."""
    try:
        # Load and preprocess the dataset
        df = load_dataset(DATASET_PATH)
        df_clean = preprocess_data(df)
        
        # Create the Kafka producer
        producer = create_kafka_producer()
        
        # Stream the data to Kafka
        stream_data_to_kafka(producer, df_clean)
        
        # Flush and close the producer
        producer.flush()
        producer.close()
        
        logger.info("Producer completed successfully")
        
    except Exception as e:
        logger.error(f"Producer failed: {e}")
        raise

if __name__ == "__main__":
    main()
