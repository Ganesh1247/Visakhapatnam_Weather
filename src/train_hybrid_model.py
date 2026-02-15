import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle
from preprocessing import DataPreprocessor

# Config
SEQ_LENGTH = 14
EPOCHS_LSTM = 20
BATCH_SIZE = 32

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Initialize & Load
print("Initializing...")
preprocessor = DataPreprocessor(sequence_length=SEQ_LENGTH)
df_weather, df_combined = preprocessor.process_data(
    "final_weather_dataset_2010-2025.csv",
    "final_dataset.csv"
)

# 2. Fit Scalers (on Log-transformed data logic handled inside create_sequences mostly, but scalers fit on "processed" data?)
# Refactored fit_scalers in preprocessing to use apply_log_transform internally?
# Actually my updated preprocessing.py fit_scalers uses raw data unless I modified it. 
# Let's manually fit scalers on LOG-transformed data for consistency.
print("Fitting Scalers...")
df_weather_log = preprocessor.apply_log_transform(df_weather)
df_combined_log = preprocessor.apply_log_transform(df_combined)
preprocessor.fit_scalers(df_weather_log, df_combined_log) 

# ==========================================
# STAGE 1: Chain-LSTM Training
# ==========================================
print("\n--- STAGE 1: Training LSTM Feature Extractor ---")

# Step 1.1: Pre-training data (Weather -> Weather)
# We can still do weather pretraining if desired, or skip to direct Hybrid training.
# User wants "Chained Architecture". 
# Let's Train LSTM on [Weather + PM] -> [Weather + PM] (Forecast)
# This forces LSTM to learn dynamics.

X_seq, y_seq, meta_df = preprocessor.create_sequences(df_combined, use_log_targets=True)

# Split
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
meta_train, meta_test = meta_df.iloc[:split_idx], meta_df.iloc[split_idx:]

# Define LSTM
input_shape = (X_train.shape[1], X_train.shape[2]) 
inputs = Input(shape=input_shape)
x = LSTM(64, return_sequences=True, name='lstm_1')(inputs)
x = Dropout(0.2)(x)
# LATENT EMBEDDING LAYER
lstm_2 = LSTM(32, return_sequences=False, name='lstm_embeddings')
embeddings = lstm_2(x)
x = Dropout(0.2)(embeddings)
# Regression Head (For supervision)
outputs = Dense(y_train.shape[1], name='output')(x)

lstm_model = Model(inputs=inputs, outputs=outputs)
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train LSTM
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS_LSTM,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)
lstm_model.save("models/lstm_hybrid_chain.h5")

# ==========================================
# STAGE 2: Generate Embeddings & Train XGBoost
# ==========================================
print("\n--- STAGE 2: Training Chained XGBoost ---")

# Feature Extractor Model
feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer('lstm_embeddings').output)

# Generate Embeddings for ALL data
X_all_emb = feature_extractor.predict(X_seq)

# Prepare XGB Data (Concatenate Embeddings + Weather + Time)
X_xgb_full, y_xgb_full_log = preprocessor.prepare_xgb_data(X_all_emb, meta_df)

# Split (Same indices as LSTM)
X_xgb_train = X_xgb_full.iloc[:split_idx]
y_xgb_train = y_xgb_full_log.iloc[:split_idx]

X_xgb_test = X_xgb_full.iloc[split_idx:]
y_xgb_test = y_xgb_full_log.iloc[split_idx:]

# Train one XGB per target
xgb_models = {}
y_pred_xgb_log = []

for target in preprocessor.target_columns:
    print(f"Training XGBoost for {target}...")
    # Per-target hyperparameters for better accuracy
    if target == 'pm2_5':
        params = {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.2,
            "min_child_weight": 3,
            "reg_alpha": 0.5,
            "reg_lambda": 1.2,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
    elif target == 'pm10':
        params = {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.15,
            "min_child_weight": 2,
            "reg_alpha": 0.3,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
    elif target == 'wind_speed':
        params = {
            "n_estimators": 350,
            "learning_rate": 0.06,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.05,
            "min_child_weight": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
    else:
        # Default for other targets
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

    model = xgb.XGBRegressor(n_jobs=-1, **params)
    model.fit(X_xgb_train, y_xgb_train[target])
    xgb_models[target] = model
    
    # Save Model
    with open(f"models/xgb_chain_{target}.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # Predict on Test
    pred_log = model.predict(X_xgb_test)
    y_pred_xgb_log.append(pred_log)

y_pred_xgb_log = np.column_stack(y_pred_xgb_log)

# ==========================================
# STAGE 3: Inverse Transform & Guardrails
# ==========================================
print("\n--- STAGE 3: Evaluation & Guardrails ---")

# 1. Inverse Log Transform (np.expm1)
# Check if target was PM (needs expm1) or Weather (linear).
# In preprocessing, we logged ONLY PM targets.
# We need to know which columns are PM.
pm_targets = preprocessor.pm_targets
all_targets = preprocessor.target_columns

y_pred_final = pd.DataFrame(y_pred_xgb_log, columns=all_targets)
y_true_final = y_xgb_test.reset_index(drop=True)

# Apply Inverse Log to PM columns
for col in pm_targets:
    y_pred_final[col] = np.expm1(y_pred_final[col])
    y_true_final[col] = np.expm1(y_true_final[col])

# 2. Guardrails
# Rule 1: Non-negative
y_pred_final[y_pred_final < 0] = 0

# Rule 2: PM2.5 <= PM10 (Soft constraint or Hard?)
# "pm25 = min(pm25, pm10)" - User requested
if 'pm2_5' in y_pred_final and 'pm10' in y_pred_final:
    y_pred_final['pm2_5'] = np.minimum(y_pred_final['pm2_5'], y_pred_final['pm10'])
    # Rule 3: pm25 = max(pm25, 0.25 * pm10) - User requested (Lower bound check)
    y_pred_final['pm2_5'] = np.maximum(y_pred_final['pm2_5'], 0.25 * y_pred_final['pm10'])

# 3. Calculate Metrics
results = []
for col in all_targets:
    y_t = y_true_final[col]
    y_p = y_pred_final[col]
    
    mse = mean_squared_error(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    
    print(f"{col}: RMSE={np.sqrt(mse):.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    results.append({'Target': col, 'RMSE': np.sqrt(mse), 'MAE': mae, 'R2': r2})

# Save Metrics
pd.DataFrame(results).to_csv("metrics_scientific.csv", index=False)

# ==========================================
# STAGE 3b: Visual Diagnostics (Last 60 days)
# ==========================================
# Time-series shape often matters more than pure metrics,
# so we export quick plots for PM2.5 and PM10 on the test split.

def plot_last_60(y_true: pd.Series, y_pred: pd.Series, title: str, filename: str):
    n_last = min(60, len(y_true))
    if n_last <= 1:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[-n_last:].reset_index(drop=True), label="True", linewidth=2)
    plt.plot(y_pred[-n_last:].reset_index(drop=True), label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Last {} days".format(n_last))
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename))
    plt.close()

# PM2.5 and PM10 true vs predicted (already back-transformed)
if 'pm2_5' in y_true_final.columns:
    plot_last_60(
        y_true_final['pm2_5'],
        y_pred_final['pm2_5'],
        "PM2.5 – True vs Predicted (Last 60 days)",
        "pm25_true_vs_pred_last60.png",
    )

if 'pm10' in y_true_final.columns:
    plot_last_60(
        y_true_final['pm10'],
        y_pred_final['pm10'],
        "PM10 – True vs Predicted (Last 60 days)",
        "pm10_true_vs_pred_last60.png",
    )

# 4. AQI Class Accuracy
# Define bins (Standard India AQI PM2.5 cuts: 0-30, 31-60, 61-90, 91-120... approximate)
# Or US EPA: 0-12, 12-35, 35-55...
# Let's use simple logic: 
# Good: <30, Moderate: 30-60, Poor: 60-90, Very Poor: >90 (Example)
# User asked for "AQI class match (%)".
# Let's define: Good(<30), Satisfactory(30-60), Moderate(60-90), Poor(90-120), Very Poor(120-250)
if 'pm2_5' in y_pred_final:
    bins = [0, 30, 60, 90, 120, 500]
    labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor']
    
    y_true_class = pd.cut(y_true_final['pm2_5'], bins=bins, labels=labels)
    y_pred_class = pd.cut(y_pred_final['pm2_5'], bins=bins, labels=labels)
    
    acc = np.mean(y_true_class == y_pred_class)
    print(f"\nPM2.5 AQI Class Accuracy: {acc*100:.2f}%")
    
    # Save class report
    with open("aqi_accuracy.txt", "w") as f:
        f.write(f"PM2.5 AQI Accuracy: {acc*100:.2f}%\n")

print("Training & Evaluation Complete.")
