import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import os
import pickle
import optuna
import warnings
from preprocessing import DataPreprocessor

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Config
SEQ_LENGTH = 21
EPOCHS_LSTM = 80
BATCH_SIZE = 32

print("Initializing Preprocessor...")
preprocessor = DataPreprocessor(sequence_length=SEQ_LENGTH)
df_weather, df_combined = preprocessor.process_hourly_data(
    "data/vizag_aqi_hourly.csv",
    "data/visakhapatnam_weather_hourly_2015_2025.csv",
    "data/fire_archive_SV-C2_725728.csv",
    "data/vizag_traffic_congestion_proxy.csv"
)

print("Fitting Scalers...")
df_weather_log = preprocessor.apply_log_transform(df_weather)
df_combined_log = preprocessor.apply_log_transform(df_combined)
preprocessor.fit_scalers(df_weather_log, df_combined_log) 

X_seq, y_seq, meta_df = preprocessor.create_sequences(df_combined, use_log_targets=True)

split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
meta_train, meta_test = meta_df.iloc[:split_idx], meta_df.iloc[split_idx:]

print("\n--- STAGE 1: Multi-Task LSTM Learner ---")
input_shape = (X_train.shape[1], X_train.shape[2])
inputs = Input(shape=input_shape)
x = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
x = Dropout(0.2)(x)
# Increased embedding to 64
lstm_2 = LSTM(64, return_sequences=False, name='lstm_embeddings')
embeddings = lstm_2(x)
x = Dropout(0.2)(embeddings)

# Multi-task Dense Heads
outputs = []
loss_dict = {}
y_train_dict = {}
y_test_dict = {}

# We create an explicit named output for each target to force independent gradient flows
for i, target_name in enumerate(preprocessor.target_columns):
    out = Dense(1, name=f'out_{target_name}')(x)
    outputs.append(out)
    loss_dict[f'out_{target_name}'] = 'mse'
    y_train_dict[f'out_{target_name}'] = y_train[:, i]
    y_test_dict[f'out_{target_name}'] = y_test[:, i]

lstm_model = Model(inputs=inputs, outputs=outputs)
lstm_model.compile(optimizer='adam', loss=loss_dict)

print("Training Multi-Task LSTM...")
lstm_model.fit(
    X_train, y_train_dict,
    validation_data=(X_test, y_test_dict),
    epochs=10, # Lower epochs for brevity, ideally 50-80
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)
lstm_model.save("models/lstm_hybrid_chain.h5")

# Extract Embeddings
print("\n--- STAGE 2: Trees & Residuals ---")
feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer('lstm_embeddings').output)
X_all_emb = feature_extractor.predict(X_seq, verbose=0)
X_xgb_full, y_xgb_full_log = preprocessor.prepare_xgb_data(X_all_emb, meta_df)

X_xgb_train = X_xgb_full.iloc[:split_idx]
y_xgb_train = y_xgb_full_log.iloc[:split_idx]
X_xgb_test = X_xgb_full.iloc[split_idx:]
y_xgb_test = y_xgb_full_log.iloc[split_idx:]

# Optuna Objective for LGBM (Only tune PM2.5 to save time)
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    # Target PM2.5 specifically for tuning
    target_y = y_xgb_train['pm2_5'].values
    X_val_np = X_xgb_train.values

    for train_idx, val_idx in tscv.split(X_val_np):
        X_tr, X_val = X_val_np[train_idx], X_val_np[val_idx]
        y_tr, y_val = target_y[train_idx], target_y[val_idx]
        
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        scores.append(mean_squared_error(y_val, preds))
        
    return np.mean(scores)

print("Running Optuna Hyperparameter Search for PM2.5 (LGBM)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5) # 5 trials for speed
best_lgbm_params = study.best_params
print(f"Best LGBM Params for PM2.5: {best_lgbm_params}")

y_pred_final_log = pd.DataFrame(index=y_xgb_test.index, columns=preprocessor.target_columns)
xgb_models = {}

for target in preprocessor.target_columns:
    print(f"\nTraining Models for {target}...")
    
    # 1. Base Model (LightGBM)
    lgbm_params = best_lgbm_params if target == 'pm2_5' else {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 'random_state': 42}
    base_model = lgb.LGBMRegressor(**lgbm_params)
    base_model.fit(X_xgb_train, y_xgb_train[target])
    
    # Predict on Train to calculate residuals
    train_base_preds = base_model.predict(X_xgb_train)
    residuals_train = y_xgb_train[target] - train_base_preds
    
    # 2. Residual Model (XGBoost)
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4, 'random_state': 42}
    resid_model = xgb.XGBRegressor(**xgb_params)
    resid_model.fit(X_xgb_train, residuals_train)
    
    # Save the ensemble
    xgb_models[f"{target}_base"] = base_model
    xgb_models[f"{target}_resid"] = resid_model
    
    # Test Prediction
    test_base_preds = base_model.predict(X_xgb_test)
    test_resid_preds = resid_model.predict(X_xgb_test)
    final_preds = test_base_preds + test_resid_preds
    y_pred_final_log[target] = final_preds

# Save all models
with open("models/lgbm_xgb_ensemble.pkl", "wb") as f:
    pickle.dump(xgb_models, f)

print("\n--- STAGE 3: Evaluation (RMSE, MAPE, Seasonality) ---")
# Inverse Log Transform for PM
y_pred_final = y_pred_final_log.copy()
y_true_final = y_xgb_test.reset_index(drop=True)

for col in preprocessor.pm_targets:
    y_pred_final[col] = np.expm1(y_pred_final[col])
    y_true_final[col] = np.expm1(y_true_final[col])
    
y_pred_final[y_pred_final < 0] = 0

if 'pm2_5' in y_pred_final and 'pm10' in y_pred_final:
    y_pred_final['pm2_5'] = np.minimum(y_pred_final['pm2_5'], y_pred_final['pm10'])
    y_pred_final['pm2_5'] = np.maximum(y_pred_final['pm2_5'], 0.25 * y_pred_final['pm10'])

results = []
meta_test['month'] = pd.to_datetime(meta_test['date']).dt.month.values

for col in preprocessor.target_columns:
    y_t = y_true_final[col]
    y_p = y_pred_final[col]
    
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    mape = mean_absolute_percentage_error(y_t, y_p)
    
    # Seasonal split (Summer vs Winter)
    winter_mask = meta_test['month'].isin([11, 12, 1, 2]).values
    summer_mask = meta_test['month'].isin([4, 5, 6, 7]).values
    
    if winter_mask.sum() > 0 and summer_mask.sum() > 0:
        winter_rmse = np.sqrt(mean_squared_error(y_t[winter_mask], y_p[winter_mask]))
        summer_rmse = np.sqrt(mean_squared_error(y_t[summer_mask], y_p[summer_mask]))
    else:
        winter_rmse, summer_rmse = 0.0, 0.0
    
    print(f"{col}: RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.3f}, R2={r2:.3f} | WinterRMSE={winter_rmse:.3f}, SummerRMSE={summer_rmse:.3f}")
    results.append({
        'Target': col, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
        'Winter_RMSE': winter_rmse, 'Summer_RMSE': summer_rmse
    })

metrics_df = pd.DataFrame(results)
metrics_df.to_csv("data/metrics_scientific.csv", index=False)
print("\nEvaluation Complete and saved.")
