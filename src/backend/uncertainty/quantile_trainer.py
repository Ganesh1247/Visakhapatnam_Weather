import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import os
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocessing import DataPreprocessor
from sklearn.metrics import mean_absolute_error

# Config
SEQ_LENGTH = 14
ALPHAS = [0.05, 0.5, 0.95] # 90% Confidence Interval (0.05 to 0.95)

os.makedirs("models/quantile", exist_ok=True)

def train_quantile_models():
    print("Initializing Data...")
    preprocessor = DataPreprocessor(sequence_length=SEQ_LENGTH)
    df_weather, df_combined = preprocessor.process_data(
        "final_weather_dataset_2010-2025.csv",
        "final_dataset.csv"
    )

    # Fit Scalers
    print("Fitting Scalers...")
    df_weather_log = preprocessor.apply_log_transform(df_weather)
    df_combined_log = preprocessor.apply_log_transform(df_combined)
    preprocessor.fit_scalers(df_weather_log, df_combined_log)

    # Create Sequences
    print("Creating Sequences...")
    X_seq, y_seq, meta_df = preprocessor.create_sequences(df_combined, use_log_targets=True)

    # Load LSTM Feature Extractor
    print("Loading LSTM Feature Extractor...")
    try:
        lstm_full = load_model("models/lstm_hybrid_chain.h5", compile=False)
        feature_extractor = Model(inputs=lstm_full.input, outputs=lstm_full.get_layer('lstm_embeddings').output)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Generate Embeddings
    print("Generating Embeddings (this may take a moment)...")
    X_emb = feature_extractor.predict(X_seq, verbose=1)

    # Prepare XGB Data
    print("Preparing XGB Data...")
    X_xgb_full, y_xgb_full_log = preprocessor.prepare_xgb_data(X_emb, meta_df)

    # Train/Test Split (Last 20% for validation/testing)
    split_idx = int(len(X_xgb_full) * 0.8)
    X_train = X_xgb_full.iloc[:split_idx]
    y_train = y_xgb_full_log.iloc[:split_idx]
    X_test = X_xgb_full.iloc[split_idx:]
    y_test = y_xgb_full_log.iloc[split_idx:]

    targets = ['pm2_5', 'pm10'] # Quantile regression mainly for PM
    
    for target in targets:
        print(f"\nTraining Quantile Models for {target}...")
        
        for alpha in ALPHAS:
            print(f"  - Training alpha={alpha}...")
            
            # XGBoost Quantile Configuration
            # Note: 'reg:quantileerror' is available in recent XGBoost versions.
            # If not, we fall back to 'reg:linear' with custom objective, but standard XGBoost >1.0 supports it?
            # Actually, standard sklearn API uses `objective='reg:quantileerror', quantile_alpha=alpha`
            
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=alpha,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                n_jobs=-1,
                random_state=42
            )
            
            model.fit(X_train, y_train[target])
            
            # Save
            save_path = f"models/quantile/xgb_{target}_{alpha}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
                
            # Evaluate
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test[target], preds)
            print(f"    -> MAE: {mae:.4f}")

    print("\nQuantile Training Complete.")

if __name__ == "__main__":
    train_quantile_models()
