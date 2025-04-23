# docker/Dockerfile.api
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

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install flask flask-cors gunicorn

# Copy project files
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose API port
EXPOSE 8080

# Run the API with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "src.api.app:app"]