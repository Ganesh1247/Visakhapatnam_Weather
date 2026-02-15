import numpy as np
import pandas as pd
import pickle
import json
import os
import sys

# Add parent dir to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocessing import DataPreprocessor
from tensorflow.keras.models import load_model, Model

def calibrate_conformal():
    print("Initializing Data for Calibration...")
    preprocessor = DataPreprocessor(sequence_length=14)
    df_weather, df_combined = preprocessor.process_data(
        "final_weather_dataset_2010-2025.csv",
        "final_dataset.csv"
    )

    # Fit Scalers
    df_weather_log = preprocessor.apply_log_transform(df_weather)
    df_combined_log = preprocessor.apply_log_transform(df_combined)
    preprocessor.fit_scalers(df_weather_log, df_combined_log)

    # Create Sequences
    X_seq, y_seq, meta_df = preprocessor.create_sequences(df_combined, use_log_targets=True)

    # Load Feature Extractor
    print("Loading LSTM Feature Extractor...")
    lstm_full = load_model("models/lstm_hybrid_chain.h5", compile=False)
    feature_extractor = Model(inputs=lstm_full.input, outputs=lstm_full.get_layer('lstm_embeddings').output)

    # Generate Embeddings
    print("Generating Embeddings...")
    X_emb = feature_extractor.predict(X_seq, verbose=0)

    # Prepare XGB Data
    X_xgb_full, y_xgb_full_log = preprocessor.prepare_xgb_data(X_emb, meta_df)

    # Load Standard XGBoost Models
    xgb_models = {}
    for target in ['pm2_5', 'pm10']:
        with open(f"models/xgb_chain_{target}.pkl", "rb") as f:
            xgb_models[target] = pickle.load(f)

    # Calibration Split (Use last 10% separate from training used for XGB if possible, 
    # but since models are already trained on 80%, we use the 20% validation set for calibration to avoid data leakage from train set)
    # Train/Test split was 80/20 in original training logic likely.
    # Let's use the last 20% as calibration set.
    val_split_idx = int(len(X_xgb_full) * 0.8)
    X_cal = X_xgb_full.iloc[val_split_idx:]
    y_cal = y_xgb_full_log.iloc[val_split_idx:]

    print(f"Calibration Set Size: {len(X_cal)} samples")

    calibration_params = {}

    for target in ['pm2_5', 'pm10']:
        print(f"Calibrating {target}...")
        # 1. Predict on Calibration Set
        y_pred_log = xgb_models[target].predict(X_cal)
        
        # In Log Space? Or Real Space?
        # Conformal prediction guarantees coverage in the space we calibrate.
        # If we want coverage on real values, we should transform back.
        # Metric: Absolute Error.
        
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_cal[target].values) # y_cal is log transformed
        
        # 2. Compute Non-Conformity Scores (Absolute Error)
        scores = np.abs(y_true - y_pred)
        
        # 3. Compute Quantile (90% Confidence -> alpha=0.1 -> 0.9 quantile)
        alpha = 0.1
        q_val = np.quantile(scores, 1 - alpha)
        
        calibration_params[target] = float(q_val)
        print(f"  -> q_hat (90%): {q_val:.4f}")

    # Save Params
    with open("models/conformal_params.json", "w") as f:
        json.dump(calibration_params, f)
        
    print("Calibration Complete. Parameters saved to models/conformal_params.json")

if __name__ == "__main__":
    calibrate_conformal()
