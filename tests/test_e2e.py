# test_e2e.py
import os
import sys
from pathlib import Path
import json
import requests
import pandas as pd
import time
import threading
import docker
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.kafka.producer import AirQualityProducer
from src.kafka.consumer import AirQualityConsumer
from src.ml.predict import AirQualityPredictor

logger = setup_logger("e2e_test")

class EndToEndTest:
    def __init__(self):
        self.api_url = "http://localhost:8080"
        self.kafka_bootstrap_servers = ["localhost:9092"]
        self.producer = None
        self.consumer = None
        self.predictor = None
        
    def setup(self):
        """Set up all components for testing"""
        logger.info("Setting up end-to-end test environment...")
        
        # 1. Start Docker containers
        self.start_docker_services()
        
        # 2. Wait for services to be ready
        self.wait_for_services()
        
        # 3. Initialize components
        self.producer = AirQualityProducer(bootstrap_servers=self.kafka_bootstrap_servers)
        self.consumer = AirQualityConsumer(bootstrap_servers=self.kafka_bootstrap_servers)
        self.predictor = AirQualityPredictor()
        self.predictor.load_model()
        
    def start_docker_services(self):
        """Start Docker services"""
        logger.info("Starting Docker services...")
        os.system("docker-compose -f docker/docker-compose.yml up -d")
        time.sleep(30)  # Wait for services to start
        
    def wait_for_services(self):
        """Wait for all services to be ready"""
        logger.info("Waiting for services to be ready...")
        
        # Wait for API
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    logger.info("API is ready")
                    break
            except:
                time.sleep(2)
        
        # Wait for Kafka
        time.sleep(10)
        logger.info("Kafka should be ready")
        
    def test_model_prediction(self):
        """Test direct model prediction"""
        logger.info("Testing direct model prediction...")
        
        sample_data = {
            "DateTime": datetime.now().isoformat(),
            "CO(GT)": 2.6,
            "PT08.S1(CO)": 1360,
            "NMHC(GT)": 150,
            "C6H6(GT)": 11.9,
            "PT08.S2(NMHC)": 1046,
            "NOx(GT)": 166,
            "PT08.S3(NOx)": 1056,
            "NO2(GT)": 113,
            "PT08.S4(NO2)": 1692,
            "PT08.S5(O3)": 1268,
            "T": 13.6,
            "RH": 48.9,
            "AH": 0.7578
        }
        
        result = self.predictor.predict_single(sample_data)
        logger.info(f"Direct prediction result: {result}")
        
    def test_api_endpoints(self):
        """Test API endpoints"""
        logger.info("Testing API endpoints...")
        
        # Test health check
        response = requests.get(f"{self.api_url}/health")
        logger.info(f"Health check: {response.json()}")
        
        # Test single prediction
        sample_data = {
            "DateTime": datetime.now().isoformat(),
            "CO(GT)": 2.6,
            "PT08.S1(CO)": 1360,
            "NMHC(GT)": 150,
            "C6H6(GT)": 11.9,
            "PT08.S2(NMHC)": 1046,
            "NOx(GT)": 166,
            "PT08.S3(NOx)": 1056,
            "NO2(GT)": 113,
            "PT08.S4(NO2)": 1692,
            "PT08.S5(O3)": 1268,
            "T": 13.6,
            "RH": 48.9,
            "AH": 0.7578
        }
        
        response = requests.post(f"{self.api_url}/predict", json=sample_data)
        logger.info(f"Single prediction: {response.json()}")
        
        # Test batch prediction
        batch_data = [sample_data, sample_data]
        response = requests.post(f"{self.api_url}/predict/batch", json=batch_data)
        logger.info(f"Batch prediction: {response.json()}")
        
        # Test future prediction
        response = requests.post(f"{self.api_url}/predict/future", json={"current_data": sample_data, "horizon": 6})
        logger.info(f"Future prediction: {response.json()}")
        
    def test_kafka_pipeline(self):
        """Test Kafka producer and consumer"""
        logger.info("Testing Kafka pipeline...")
        
        # Start consumer in a separate thread
        consumer_thread = threading.Thread(target=self._run_consumer)
        consumer_thread.daemon = True
        consumer_thread.start()
        
        # Send some test data through producer
        test_data_path = "data/raw/AirQualityUCI.csv"
        df = pd.read_csv(test_data_path, sep=';', decimal=',', nrows=10)
        
        for _, row in df.iterrows():
            data = row.to_dict()
            self.producer.produce(data)
            time.sleep(1)
        
        # Wait for consumer to process
        time.sleep(10)
        
    def _run_consumer(self):
        """Run consumer for testing"""
        self.consumer.consume(lambda msg: logger.info(f"Consumed message: {msg}"))
        
    def test_monitoring(self):
        """Test Evidently monitoring"""
        logger.info("Testing monitoring dashboard...")
        # You can add Evidently dashboard testing here
        
    def test_end_to_end_flow(self):
        """Test complete end-to-end flow"""
        logger.info("Testing end-to-end flow...")
        
        # 1. Produce data
        sample_data = {
            "DateTime": datetime.now().isoformat(),
            "CO(GT)": 2.6,
            "PT08.S1(CO)": 1360,
            # ... other fields
        }
        
        self.producer.produce(sample_data)
        
        # 2. Wait for processing
        time.sleep(5)
        
        # 3. Check API prediction
        response = requests.post(f"{self.api_url}/predict", json=sample_data)
        logger.info(f"End-to-end prediction: {response.json()}")
        
    def teardown(self):
        """Clean up resources"""
        logger.info("Tearing down test environment...")
        os.system("docker-compose -f docker/docker-compose.yml down")
        
    def run_all_tests(self):
        """Run all tests"""
        try:
            self.setup()
            self.test_model_prediction()
            self.test_api_endpoints()
            self.test_kafka_pipeline()
            self.test_end_to_end_flow()
        finally:
            self.teardown()

if __name__ == "__main__":
    tester = EndToEndTest()
    tester.run_all_tests()