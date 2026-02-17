import pickle
import pandas as pd
import numpy as np
import os
import sys

# Mock for preprocessor
class MockPreprocessor:
    def __init__(self):
        self.lstm_features = ['temp_max', 'temp_min', 'temp_avg', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'rainfall', 'solar_radiation', 'cloud_cover']

preprocessor = MockPreprocessor()

def debug_model():
    MODELS_DIR = "models"
    target = "pm10"
    path = os.path.join(MODELS_DIR, f"xgb_chain_{target}.pkl")
    
    with open(path, "rb") as f:
        model = pickle.load(f)
        
    print(f"Model type: {type(model)}")
    if hasattr(model, "feature_names_in_"):
        print(f"Feature names in model: {model.feature_names_in_}")
        print(f"Count: {len(model.feature_names_in_)}")
    
    # Construct exactly like mc_dropout.py
    embedding_dim = 32
    xgb_feature_names = [f'emb_{j}' for j in range(embedding_dim)] + \
                        preprocessor.lstm_features + \
                        ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
    
    X_xgb_np = np.zeros((1, len(xgb_feature_names)), dtype=np.float32)
    X_xgb_df = pd.DataFrame(X_xgb_np, columns=xgb_feature_names)
    
    print("\nAttempting prediction with DataFrame...")
    try:
        pred = model.predict(X_xgb_df)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")
        
    print("\nAttempting prediction with Numpy Array...")
    try:
        pred = model.predict(X_xgb_np)
        print("Success (Numpy)!")
    except Exception as e:
        print(f"Failed (Numpy): {e}")

if __name__ == "__main__":
    debug_model()
