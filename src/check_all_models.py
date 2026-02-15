import pickle

# Check each XGBoost model to see how many features they expect
models = ['pm10', 'pm2_5', 'temp_avg', 'temp_min', 'temp_max', 'humidity', 'rainfall', 'wind_speed']

print("=" * 60)
print("XGBoost Model Feature Requirements:")
print("=" * 60)

for target in models:
    try:
        with open(f'models/xgb_chain_{target}.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"{target:20s} -> {model.n_features_in_} features")
    except Exception as e:
        print(f"{target:20s} -> ERROR: {e}")
