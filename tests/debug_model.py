# debug_model.py
import pickle
from pathlib import Path

# Check model loading
model_path = Path('models/artifacts/best_model.pkl')
scaler_path = Path('models/artifacts/scaler.pkl')

print(f"Model path exists: {model_path.exists()}")
print(f"Scaler path exists: {scaler_path.exists()}")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")
    print(f"Scaler type: {type(scaler)}")
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Features expected: {scaler.feature_names_in_}")
except Exception as e:
    print(f"Error loading scaler: {e}")