"""
Complete workflow demonstration script.
"""

import subprocess
import time
import sys
from pathlib import Path

def run_command(command, wait=True):
    """Run a command and optionally wait for it to complete."""
    print(f"Running: {command}")
    if wait:
        subprocess.run(command, shell=True)
    else:
        subprocess.Popen(command, shell=True)
    time.sleep(2)

def main():
    """Demonstrate the complete workflow."""
    print("=== Air Quality Prediction System Demo ===")
    
    # 1. Start MLflow server
    print("\n1. Starting MLflow server...")
    run_command("python -m mlflow ui --backend-store-uri sqlite:///mlflow.db", wait=False)
    time.sleep(5)
    print("MLflow UI available at: http://localhost:5000")
    
    # 2. Train models
    print("\n2. Training models...")
    run_command("python src/ml/train.py")
    
    # 3. Test batch predictions
    print("\n3. Testing batch predictions...")
    run_command("python src/tests/test_batch_predictions.py")
    
    # 4. Start API server
    print("\n4. Starting API server...")
    run_command("python src/api/app.py", wait=False)
    time.sleep(5)
    print("API available at: http://localhost:8080")
    
    # 5. Start monitoring dashboard
    print("\n5. Starting monitoring dashboard...")
    run_command("python src/monitoring/evidently_dashboard.py", wait=False)
    time.sleep(5)
    print("Monitoring dashboard available at: http://localhost:8050")
    
    # 6. Generate training visualizations
    print("\n6. Generating training visualizations...")
    run_command("python src/visualization/training_plots.py")
    
    print("\n=== Demo Complete ===")
    print("Services running:")
    print("- MLflow UI: http://localhost:5000")
    print("- API Server: http://localhost:8080")
    print("- Monitoring Dashboard: http://localhost:8050")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all services...")

if __name__ == "__main__":
    main()