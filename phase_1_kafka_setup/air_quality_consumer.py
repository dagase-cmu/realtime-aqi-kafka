#!/usr/bin/env python3
"""
air_quality_consumer.py - Kafka consumer for processing streamed air quality data

This script consumes air quality data from a Kafka topic, processes it,
and stores it for further analysis.
"""

import json
import logging
import pandas as pd
import os
from kafka import KafkaConsumer
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
KAFKA_GROUP_ID = 'air-quality-consumer-group'

# Output configuration
OUTPUT_DIR = '/mnt/c/Users/radha/air-quality-prediction/data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'processed_air_quality_data.csv')
CHECKPOINT_INTERVAL = 100  # Save to CSV every N records

def create_kafka_consumer():
    """
    Create and return a Kafka consumer instance.
    """
    logger.info(f"Creating Kafka consumer. Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            group_id=KAFKA_GROUP_ID,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        logger.info("Kafka consumer created successfully")
        return consumer
    except Exception as e:
        logger.error(f"Failed to create Kafka consumer: {e}")
        raise

def process_message(message):
    """
    Process a Kafka message containing air quality data.
    """
    try:
        # Extract the value from the Kafka message
        data = message.value
        
        # Add a timestamp for when we received the message
        data['received_at'] = datetime.now().isoformat()
        
        # Here you could add more processing logic based on your needs
        
        return data
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return None

def save_data(data_list):
    """
    Save the processed data to a CSV file.
    """
    if not data_list:
        logger.warning("No data to save")
        return
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(data_list)
        
        # Check if the output file already exists
        file_exists = os.path.isfile(OUTPUT_FILE)
        
        # Save to CSV, appending if the file exists
        if file_exists:
            df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(OUTPUT_FILE, index=False)
        
        logger.info(f"Saved {len(data_list)} records to {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def consume_data():
    """
    Consume data from Kafka topic, process it, and save it.
    """
    logger.info(f"Starting to consume data from Kafka topic: {KAFKA_TOPIC}")
    
    try:
        # Create the Kafka consumer
        consumer = create_kafka_consumer()
        
        # Process and save data
        data_buffer = []
        messages_processed = 0
        
        for message in consumer:
            processed_data = process_message(message)
            
            if processed_data:
                data_buffer.append(processed_data)
                messages_processed += 1
                
                # Save data in batches
                if len(data_buffer) >= CHECKPOINT_INTERVAL:
                    save_data(data_buffer)
                    data_buffer = []
                
                if messages_processed % 100 == 0:
                    logger.info(f"Messages processed: {messages_processed}")
    
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user")
    except Exception as e:
        logger.error(f"Error in consume_data: {e}")
    finally:
        # Save any remaining data
        if data_buffer:
            save_data(data_buffer)
        
        # Close the consumer
        if 'consumer' in locals():
            consumer.close()
            
        logger.info(f"Consumer stopped. Total messages processed: {messages_processed}")

def main():
    """Main function to run the Kafka consumer."""
    try:
        consume_data()
    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        raise

if __name__ == "__main__":
    main()
