# docker/Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/raw/.gitkeep ./data/raw/
COPY data/processed/.gitkeep ./data/processed/
COPY data/logs/.gitkeep ./data/logs/
COPY models/saved/.gitkeep ./models/saved/
COPY models/artifacts/.gitkeep ./models/artifacts/

# Create directories
RUN mkdir -p data/raw data/processed data/logs models/saved models/artifacts

# Set environment variables for configuration
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Make ports available
EXPOSE 8080

# Default command (can be overridden)
CMD ["python", "src/kafka/consumer.py"]