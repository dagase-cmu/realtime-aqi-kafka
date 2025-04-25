# Use official Python image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

ENV PYTHONPATH="/app"

# Expose the port for the Flask API (matches app.py)
EXPOSE 8080

# Run the Flask app directly (can be swapped for gunicorn if needed)
CMD ["python", "src/api/server.py"]