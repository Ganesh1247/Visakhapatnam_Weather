import pickle
import os
import pandas as pd
import numpy as np

# Load XGB Models
xgb_models = {}
active_targets = ['pm10', 'pm2_5', 'temp_avg', 'temp_min', 'temp_max', 'humidity', 'rainfall', 'wind_speed']

for target in active_targets:
    try:
        path = f"models/xgb_chain_{target}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
                xgb_models[target] = model
                print(f"\nTarget: {target}")
                try:
                    # For scikit-learn wrapper
                    print(f"Feature names (sklearn): {model.feature_names_in_[:5]} ... {model.feature_names_in_[-5:]}")
                    print(f"Number of features: {len(model.feature_names_in_)}")
                except:
                    try:
                        # For native Booster
                        booster = model.get_booster()
                        print(f"Feature names (booster): {booster.feature_names[:5]} ... {booster.feature_names[-5:]}")
                        print(f"Number of features: {len(booster.feature_names)}")
                    except:
                        print("Could not retrieve feature names easily.")
    except Exception as e:
        print(f"Error loading {target}: {e}")

# Also check the constructed DataFrame in the actual code
# Sample feat_dict construction
feat_dict = {}
for j in range(32):
    feat_dict[f'emb_{j}'] = 0.0
lstm_features = ['temp_max', 'temp_min', 'temp_avg', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'rainfall', 'solar_radiation', 'cloud_cover']
for col in lstm_features:
    feat_dict[col] = 0.0
feat_dict.update({'month': 2, 'day_of_week': 3, 'day': 11, 'is_weekend': 0, 'wind_dir_sin': 0.0, 'wind_dir_cos': 0.0, 'pressure_delta': 0.0})

XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(32)] + \
                  lstm_features + \
                  ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']

X_xgb = pd.DataFrame([feat_dict])[XGB_FEATURE_NAMES]
print(f"\nConstructed X_xgb columns: {list(X_xgb.columns[:5])} ... {list(X_xgb.columns[-5:])}")
print(f"X_xgb Shape: {X_xgb.shape}")
print(f"X_xgb Types: {X_xgb.dtypes.value_counts()}")
