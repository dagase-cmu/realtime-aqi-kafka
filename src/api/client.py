# src/api/client.py
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8080"

def test_server():
    """Test the server endpoints"""
    
    # Test health endpoint
    try:
        response = requests.get(f"{API_URL}/health")
        print("Health Check:", response.json())
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test prediction endpoint
    data = {
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
    
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        print("Response status:", response.status_code)
        print("Prediction response:", response.json())
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    test_server()