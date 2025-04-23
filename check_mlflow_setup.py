# check_mlflow_setup.py
import mlflow
import sqlite3
import os

# Set the correct tracking URI
tracking_uri = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

print("Current tracking URI:", mlflow.get_tracking_uri())

# Check if the database file exists
if os.path.exists("mlflow.db"):
    print("MLflow database exists")
    
    # Connect to the database
    conn = sqlite3.connect("mlflow.db")
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check experiments
    cursor.execute("SELECT experiment_id, name FROM experiments;")
    experiments = cursor.fetchall()
    print("\nExperiments:")
    for exp in experiments:
        print(f"  - ID: {exp[0]}, Name: {exp[1]}")
    
    # Check runs
    cursor.execute("SELECT run_uuid, experiment_id, status FROM runs;")
    runs = cursor.fetchall()
    print(f"\nTotal runs: {len(runs)}")
    
    conn.close()
else:
    print("MLflow database does not exist")

# Use MLflow client to check
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
print("\nExperiments via MLflow client:")
for exp in experiments:
    print(f"  - Name: {exp.name}, ID: {exp.experiment_id}")
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"    Runs: {len(runs)}")