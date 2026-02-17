import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from preprocessing import DataPreprocessor
from backend.uncertainty.mc_dropout import MCDropoutPredictor
import pickle
import time

def test_vectorized_mc():
    print("Testing Vectorized MC Dropout Predictor...")
    
    # Setup
    preprocessor = DataPreprocessor(sequence_length=14)
    # Load data for scalers
    df_weather = pd.read_csv("data/final_weather_dataset_2010-2025.csv")
    df_combined = pd.read_csv("data/final_dataset.csv")
    df_weather_log = preprocessor.apply_log_transform(df_weather)
    df_combined_log = preprocessor.apply_log_transform(df_combined)
    preprocessor.fit_scalers(df_weather_log, df_combined_log)
    
    # Load Models
    lstm_full = load_model("models/lstm_hybrid_chain.h5", compile=False)
    feature_extractor = Model(inputs=lstm_full.input, outputs=lstm_full.get_layer('lstm_embeddings').output)
    
    xgb_models = {}
    active_targets = ['pm10', 'pm2_5']
    for target in active_targets:
        with open(f"models/xgb_chain_{target}.pkl", "rb") as f:
            xgb_models[target] = pickle.load(f)
            
    mc_predictor = MCDropoutPredictor(
        lstm_model=lstm_full,
        feature_extractor=feature_extractor,
        xgb_models=xgb_models,
        preprocessor=preprocessor,
        n_iter=50
    )
    
    # Mock Batch Input (7 days)
    batch_size = 7
    X_lstm_batch = np.random.rand(batch_size, 14, 10).astype('float32')
    
    base_feat_list = []
    for i in range(batch_size):
        feat = {col: 0.0 for col in preprocessor.lstm_features}
        feat.update({
            'month': 2, 'day_of_week': 3, 'day': 17, 'is_weekend': 0,
            'wind_dir_sin': 0.0, 'wind_dir_cos': 1.0, 'pressure_delta': 0.0
        })
        base_feat_list.append(feat)
        
    # Run Prediction
    start_time = time.time()
    results = mc_predictor.predict_with_uncertainty(X_lstm_batch, base_feat_list)
    end_time = time.time()
    
    print(f"Prediction for {batch_size} days took {end_time - start_time:.4f} seconds.")
    print(f"Result count: {len(results)}")
    
    if len(results) == batch_size:
        print("PASS: Vectorized Batch Prediction Successful.")
        # Check one result structure
        res = results[0]
        # Raw mc_predictor output has targets as top-level keys
        if 'pm2_5' in res and 'prediction' in res['pm2_5'] and 'uncertainty' in res['pm2_5']:
             print("PASS: Result structure is correct.")
        else:
             print(f"FAIL: Result structure is incorrect. Keys: {res.keys()}")
             if 'pm2_5' in res: print(f"pm2_5 keys: {res['pm2_5'].keys()}")
    else:
        print(f"FAIL: Expected {batch_size} results, got {len(results)}")

if __name__ == "__main__":
    test_vectorized_mc()
