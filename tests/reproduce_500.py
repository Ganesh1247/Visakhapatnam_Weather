import os
import sys
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Mock preprocessor
class MockPreprocessor:
    def __init__(self):
        self.lstm_features = [
            'temp_max', 'temp_min', 'temp_avg', 'humidity', 
            'wind_speed', 'wind_direction', 'pressure', 'rainfall',
            'solar_radiation', 'cloud_cover'
        ]

def test_inference():
    preprocessor = MockPreprocessor()
    XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(32)] + \
                      preprocessor.lstm_features + \
                      ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
    
    # Load one model
    with open("models/xgb_chain_pm10.pkl", "rb") as f:
        model = pickle.load(f)
        
    print(f"Model expected features: {len(model.feature_names_in_)}")
    print(f"Code providing features: {len(XGB_FEATURE_NAMES)}")
    
    # Dummy data
    feat_dict = {col: 0.1 for col in XGB_FEATURE_NAMES}
    X_xgb = pd.DataFrame([feat_dict])[XGB_FEATURE_NAMES].astype('float32')
    
    print("\nAttempting prediction...")
    try:
        pred = model.predict(X_xgb)
        print(f"Success! Prediction: {pred[0]}")
    except Exception as e:
        print(f"Failed: {e}")
        # Try to print differences
        m_names = list(model.feature_names_in_)
        c_names = XGB_FEATURE_NAMES
        
        missing = [x for x in m_names if x not in c_names]
        extra = [x for x in c_names if x not in m_names]
        print(f"Missing in code: {missing}")
        print(f"Extra in code: {extra}")
        
        if len(m_names) == len(c_names):
            for i, (m, c) in enumerate(zip(m_names, c_names)):
                if m != c:
                    print(f"Mismatch at index {i}: model='{m}', code='{c}'")

if __name__ == "__main__":
    test_inference()
