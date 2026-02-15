import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
from preprocessing import DataPreprocessor

# Config
SEQ_LENGTH = 14
os.makedirs("models", exist_ok=True)

# 1. Initialize & Load
print("Initializing...")
preprocessor = DataPreprocessor(sequence_length=SEQ_LENGTH)
df_weather, df_combined = preprocessor.process_data(
    "final_weather_dataset_2010-2025.csv",
    "final_dataset.csv"
)
preprocessor.fit_scalers(df_weather, df_combined)

# 2. Prepare Data (Same splits as training)
# LSTM Data
X_combined, y_combined = preprocessor.create_sequences(df_combined, is_training_targets=True)
split_idx = int(len(X_combined) * 0.8)
X_test_lstm = X_combined[split_idx:]
y_test_lstm = y_combined[split_idx:]

# XGBoost Data
X_xgb, y_xgb = preprocessor.get_xgb_data(df_combined)
split_xgb = int(len(X_xgb) * 0.8)
X_train_xgb = X_xgb.iloc[:split_xgb]
y_train_xgb = y_xgb.iloc[:split_xgb]
X_test_xgb = X_xgb.iloc[split_xgb:]
y_test_xgb = y_xgb.iloc[split_xgb:]

# 3. Load LSTM Model
print("Loading LSTM Model...")
try:
    lstm_model = load_model("models/lstm_finetuned_hybrid.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback: Rebuild and load weights if you want (omitted for now)
    # Or try loading with different options
    raise e

y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = preprocessor.scaler_targets.inverse_transform(y_pred_lstm_scaled)
y_true_lstm = preprocessor.scaler_targets.inverse_transform(y_test_lstm)

# 4. Train/Load XGBoost & Predict
print("Training XGBoost Models (for Ensemble)...")
xgb_preds = []
xgb_models = {}
targets = preprocessor.target_columns

for col in targets:
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    model.fit(X_train_xgb, y_train_xgb[col])
    xgb_models[col] = model
    
    # Save XGB model
    with open(f"models/xgb_{col}.pkl", "wb") as f:
        pickle.dump(model, f)
        
    pred = model.predict(X_test_xgb)
    xgb_preds.append(pred)

y_pred_xgb = np.column_stack(xgb_preds)
y_true_xgb = y_test_xgb.values

# 5. Ensemble & Evaluate
# Align using the last N samples (intersection of test sets)
n_samples = min(len(y_pred_lstm), len(y_pred_xgb))
y_true_ens = y_true_lstm[-n_samples:] # Should match y_true_xgb mostly

p_lstm = y_pred_lstm[-n_samples:]
p_xgb = y_pred_xgb[-n_samples:]
p_ens = (p_lstm + p_xgb) / 2

results = []

print("\n" + "="*80)
print(f"{'Target':<20} | {'Model':<10} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10}")
print("-" * 80)

for i, target in enumerate(targets):
    # LSTM Metrics
    rmse_l = np.sqrt(mean_squared_error(y_true_ens[:, i], p_lstm[:, i]))
    mae_l = mean_absolute_error(y_true_ens[:, i], p_lstm[:, i])
    r2_l = r2_score(y_true_ens[:, i], p_lstm[:, i])
    
    # XGB Metrics
    rmse_x = np.sqrt(mean_squared_error(y_true_ens[:, i], p_xgb[:, i]))
    mae_x = mean_absolute_error(y_true_ens[:, i], p_xgb[:, i])
    r2_x = r2_score(y_true_ens[:, i], p_xgb[:, i])
    
    # Ensemble Metrics
    rmse_e = np.sqrt(mean_squared_error(y_true_ens[:, i], p_ens[:, i]))
    mae_e = mean_absolute_error(y_true_ens[:, i], p_ens[:, i])
    r2_e = r2_score(y_true_ens[:, i], p_ens[:, i])
    
    print(f"{target:<20} | {'LSTM':<10} | {rmse_l:<10.4f} | {mae_l:<10.4f} | {r2_l:<10.4f}")
    print(f"{'':<20} | {'XGBoost':<10} | {rmse_x:<10.4f} | {mae_x:<10.4f} | {r2_x:<10.4f}")
    print(f"{'':<20} | {'Ensemble':<10} | {rmse_e:<10.4f} | {mae_e:<10.4f} | {r2_e:<10.4f}")
    print("-" * 80)
    
    results.extend([
        {'Target': target, 'Model': 'LSTM', 'RMSE': rmse_l, 'MAE': mae_l, 'R2': r2_l},
        {'Target': target, 'Model': 'XGBoost', 'RMSE': rmse_x, 'MAE': mae_x, 'R2': r2_x},
        {'Target': target, 'Model': 'Ensemble', 'RMSE': rmse_e, 'MAE': mae_e, 'R2': r2_e}
    ])

# Save comparison
pd.DataFrame(results).to_csv("final_model_comparison.csv", index=False)
print("\nDetailed comparison saved to 'final_model_comparison.csv'")
