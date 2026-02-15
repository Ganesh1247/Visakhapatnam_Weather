import pickle
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor

# Load one of the XGBoost models
with open('models/xgb_chain_pm10.pkl', 'rb') as f:
    model = pickle.load(f)

# Check what features it expects
print("=" * 60)
print("XGBoost Model Feature Count:")
print("=" * 60)
print(f"Number of features expected: {model.n_features_in_}")

# Let's also check the preprocessing
preprocessor = DataPreprocessor(sequence_length=14)

print("\n" + "=" * 60)
print("Preprocessor Configuration:")
print("=" * 60)
print(f"LSTM features (len={len(preprocessor.lstm_features)}): {preprocessor.lstm_features}")
print(f"\nWeather features (len={len(preprocessor.weather_features)}): {preprocessor.weather_features}")
print(f"\nTarget columns (len={len(preprocessor.target_columns)}): {preprocessor.target_columns}")

# Calculate expected XGBoost features
# LSTM embeddings: 32 (from shape in train_hybrid_model.py, line 64)
# Weather features: 11 (including season)
# Time features: month, day_of_week, day = 3
# Total = 32 + 11 + 3 = 46
print("\n" + "=" * 60)
print("Expected XGBoost Input Features:")
print("=" * 60)
print("LSTM embeddings: 32")
print(f"Weather features: {len(preprocessor.weather_features)}")
print("Time features: 3 (month, day_of_week, day)")
print(f"TOTAL: {32 + len(preprocessor.weather_features) + 3}")
