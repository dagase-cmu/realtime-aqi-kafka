# Kafka Setup Documentation

## 1. Introduction

This document outlines the process followed to set up Apache Kafka for the Air Quality Prediction project. The setup involves creating a real-time data streaming pipeline using Kafka to process air quality sensor data.

## 2. Setup Approach

Initially, I attempted to install Kafka directly on Windows, but faced several challenges with path length limitations and configuration. After exploring alternatives, I decided to use Docker containers with WSL (Windows Subsystem for Linux) to create a more reliable and reproducible Kafka environment.

## 3. Installation Process

### 3.1 Initial Attempt: Native Windows Installation

I first attempted to install Kafka directly on Windows:
- Downloaded Kafka 3.9.0 from the Apache website
- Extracted the files to `C:\Users\radha\kafka_2.13-3.9.0`
- Attempted to configure the Zookeeper and Kafka properties files

However, I encountered several issues:
- Windows command line errors with "input line too long"
- Path length limitations affecting the batch files
- Issues with starting Zookeeper and Kafka servers

### 3.2 Docker-based Solution

After the initial challenges, I switched to a Docker-based approach:

1. Installed Docker Desktop for Windows
2. Enabled WSL integration
3. Started WSL and ran the following commands:
   ```bash
   # Run Zookeeper container
   docker run -d --name zookeeper -p 2181:2181 wurstmeister/zookeeper
   
   # Run Kafka container, connected to Zookeeper
   docker run -d --name kafka -p 9092:9092 \
     --link zookeeper:zookeeper \
     -e KAFKA_ADVERTISED_HOST_NAME=localhost \
     -e KAFKA_ADVERTISED_PORT=9092 \
     -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
     -e KAFKA_CREATE_TOPICS="air-quality-data:1:1" \
     wurstmeister/kafka
   ```

4. Verified the containers were running:
   ```bash
   docker ps
   ```

This approach resolved the Windows-specific issues and provided a stable Kafka environment.

## 4. Python Environment Setup

To work with Kafka from Python, I set up a virtual environment:

```bash
# Create virtual environment
python3 -m venv ~/kafka-venv

# Activate the environment
source ~/kafka-venv/bin/activate

# Install required packages
pip install kafka-python pandas numpy
```

## 5. Testing the Kafka Setup

To verify the Kafka setup was working correctly:

1. Created a simple test producer and consumer
2. Verified that messages could be sent and received
3. Confirmed the Kafka topic 'air-quality-data' was properly created

## 6. Challenges and Solutions

### 6.1 Windows Path Length Limitations

**Challenge**: Windows has limitations on command line length and path length, which affected Kafka's batch scripts.

**Solution**: Used Docker containers running in WSL to avoid Windows-specific limitations.

### 6.2 Kafka-Python Package Issues

**Challenge**: Initially had difficulties importing the KafkaProducer from the kafka-python package.

**Solution**: Created a fresh Python virtual environment and installed a specific version of kafka-python (2.0.2).

### 6.3 Docker Image Compatibility

**Challenge**: First attempted with johnnypark/kafka-zookeeper image which had platform compatibility issues.

**Solution**: Switched to the wurstmeister Kafka and Zookeeper images which are more widely compatible.

## 7. Final Configuration

The final Kafka setup consists of:

- **Zookeeper Container**: Managing cluster state and configuration
  - Port: 2181
  - Image: wurstmeister/zookeeper

- **Kafka Container**: Providing the message broker functionality
  - Port: 9092
  - Image: wurstmeister/kafka
  - Environment:
    - KAFKA_ADVERTISED_HOST_NAME=localhost
    - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181

- **Kafka Topic**: air-quality-data
  - Partitions: 1
  - Replication Factor: 1




