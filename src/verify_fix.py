import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Mocking the logic in app.py
def get_season(month):
    if month in [1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8, 9]: return 2
    elif month in [10, 11, 12]: return 3
    return 0

SEQ_LENGTH = 14
weather_features = [
    'temp_max', 'temp_min', 'temp_avg', 'humidity', 
    'wind_speed', 'wind_direction', 'pressure', 'rainfall',
    'solar_radiation', 'cloud_cover', 'season'
]

# Simulate data
today = datetime.now().date()
hist_end = today - timedelta(days=2)
hist_start = hist_end - timedelta(days=20)

df_hist = pd.DataFrame({
    'date': pd.date_range(start=hist_start, end=hist_end),
    'temp_max': np.random.uniform(20, 30, 21),
    'temp_avg': np.random.uniform(20, 30, 21),
    'humidity': np.random.uniform(50, 90, 21),
    'pressure': np.random.uniform(1000, 1015, 21)
})
# Fill other features
for col in weather_features:
    if col not in df_hist.columns and col != 'season':
        df_hist[col] = 0

df_fore = pd.DataFrame({
    'date': pd.date_range(start=today, periods=8),
    'temp_max': np.random.uniform(20, 30, 8),
    'temp_avg': np.random.uniform(20, 30, 8),
    'humidity': np.random.uniform(50, 90, 8),
    'pressure': np.random.uniform(1000, 1015, 8)
})
for col in weather_features:
    if col not in df_fore.columns and col != 'season':
        df_fore[col] = 0

# TEST LOGIC FROM APP.PY
feat_dict = {}
for j in range(32):
    feat_dict[f'emb_{j}'] = np.random.random()

for col in ['temp_max', 'temp_min', 'temp_avg', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'rainfall', 'solar_radiation', 'cloud_cover']:
    feat_dict[col] = np.random.random()

feat_dict.update({'month': 2, 'day_of_week': 3, 'day': 11, 'is_weekend': 0, 'wind_dir_sin': 0.5, 'wind_dir_cos': 0.5, 'pressure_delta': 0.1})

XGB_FEATURE_NAMES = [f'emb_{j}' for j in range(32)] + \
                  ['temp_max', 'temp_min', 'temp_avg', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'rainfall', 'solar_radiation', 'cloud_cover'] + \
                  ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']

X_xgb = pd.DataFrame([feat_dict])[XGB_FEATURE_NAMES]

print(f"X_xgb Shape: {X_xgb.shape}")
print(f"X_xgb Columns Match Expected Order: {list(X_xgb.columns) == XGB_FEATURE_NAMES}")
print(f"First 5 Columns: {list(X_xgb.columns[:5])}")
print(f"Last 5 Columns: {list(X_xgb.columns[-5:])}")
