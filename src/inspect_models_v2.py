import pickle
import os
import pandas as pd
import numpy as np
import xgboost as xgb

# Load XGB Models
xgb_models = {}
active_targets = ['pm10', 'pm2_5', 'temp_avg', 'temp_min', 'temp_max', 'humidity', 'rainfall', 'wind_speed']

for target in active_targets:
    try:
        path = f"models/xgb_chain_{target}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
                print(f"\n--- Target: {target} ---")
                print(f"Model type: {type(model)}")
                
                # Try many ways to get feature names
                names = None
                if hasattr(model, 'feature_names_in_'):
                    names = list(model.feature_names_in_)
                    print(f"SKLearn feature_names_in_ ({len(names)})")
                elif hasattr(model, 'feature_names'):
                    names = model.feature_names
                    print(f"Model.feature_names ({len(names)})")
                elif hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    names = booster.feature_names
                    print(f"Booster.feature_names ({len(names)})")
                elif isinstance(model, xgb.Booster):
                    names = model.feature_names
                    print(f"Is Booster, feature_names ({len(names)})")
                
                if names:
                    print(f"First 10: {names[:10]}")
                    print(f"Last 10: {names[-10:]}")
                else:
                    print("Could not retrieve feature names.")
                    
    except Exception as e:
        print(f"Error loading {target}: {e}")

# Check current app.py logic features
lstm_features = ['temp_max', 'temp_min', 'temp_avg', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'rainfall', 'solar_radiation', 'cloud_cover']
XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(32)] + \
                  lstm_features + \
                  ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']

print(f"\n--- Code Logic Features ---")
print(f"Total features: {len(XGB_FEATURE_NAMES)}")
print(f"First 10: {XGB_FEATURE_NAMES[:10]}")
print(f"Last 10: {XGB_FEATURE_NAMES[-10:]}")
