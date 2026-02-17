import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Any
from datetime import datetime
import time

class UncertaintyValidator:
    """Validates uncertainty predictions against statistical properties."""
    
    @staticmethod
    def validate_prediction(prediction: float, uncertainty: Dict[str, Any]) -> List[str]:
        errors = []
        
        # 1. Std Dev > 0 (Stochasticity Check)
        if uncertainty['std'] <= 0:
            errors.append(f"Standard deviation must be positive, got {uncertainty['std']}")
            
        # 2. Prediction inside 90% CI
        ci_90 = uncertainty['confidence_90']
        if not (ci_90[0] <= prediction <= ci_90[1]):
            errors.append(f"Prediction {prediction} outside 90% CI {ci_90}")
            
        # 3. 95% CI wider than 90% CI
        ci_95 = uncertainty['confidence_95']
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        if width_95 < width_90:
            errors.append(f"95% CI width ({width_95}) smaller than 90% CI width ({width_90})")
            
        # 4. No Negative Lower Bounds (unless physically possible, but PM2.5 can't be negative)
        if ci_95[0] < 0:
            errors.append(f"95% CI lower bound is negative: {ci_95[0]}")
            
        return errors

class MCDropoutPredictor:
    def __init__(self, lstm_model: tf.keras.Model, 
                 feature_extractor: tf.keras.Model,
                 xgb_models: Dict[str, Any], 
                 preprocessor: Any,
                 n_iter: int = 50):
        self.lstm_model = lstm_model
        # Ensure we use the model that has Dropout layers
        self.feature_extractor = feature_extractor 
        self.xgb_models = xgb_models
        self.preprocessor = preprocessor
        self.n_iter = n_iter
        self.validator = UncertaintyValidator()

    def predict_with_uncertainty(self, X_lstm_batch: np.ndarray, base_feat_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run MC Dropout inference for a BATCH of days.
        
        Args:
            X_lstm_batch: Input tensor for LSTM (Batch_Size, Seq_Len, Features)
            base_feat_list: List of dictionaries of static features for each day in batch
            
        Returns:
            List of Dicts containing mean prediction and uncertainty metrics for each day.
        """
        batch_size = X_lstm_batch.shape[0]
        embedding_dim = 32 # Based on model architecture
        
        # 1. MC Dropout Forward Pass (LSTM) - Vectorized across Batch and Iterations
        # Repeat each day n_iter times: [D1, D1, ..., D2, D2, ..., DN, DN]
        X_tiled = np.repeat(X_lstm_batch, self.n_iter, axis=0) # Shape: (batch_size * n_iter, Seq_Len, Features)
        
        # Force training=True to enable Dropout
        # mc_embeddings shape: (batch_size * n_iter, 32)
        mc_embeddings = self.feature_extractor(X_tiled, training=True).numpy()
        
        # 2. Construct XGBoost Feature Matrix for ALL days and ALL iterations
        # IMPORTANT: Order must match the models exactly: 32 embeddings + 10 weather + 7 time/derived
        xgb_feature_names = [f'emb_{j}' for j in range(embedding_dim)] + \
                            self.preprocessor.lstm_features + \
                            ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
                            
        total_rows = batch_size * self.n_iter
        X_xgb_np = np.zeros((total_rows, len(xgb_feature_names)), dtype=np.float32)
        
        # Fill Embeddings
        X_xgb_np[:, :embedding_dim] = mc_embeddings
        
        # Fill static features for each day in the batch
        # Map each feature name to its column index for reliable assignment
        feat_map = {name: i for i, name in enumerate(xgb_feature_names)}
        
        for i, feat_dict in enumerate(base_feat_list):
            start_row = i * self.n_iter
            end_row = (i + 1) * self.n_iter
            for col, val in feat_dict.items():
                if col in feat_map:
                    # Explicit conversion to float32 to match expectations
                    X_xgb_np[start_row:end_row, feat_map[col]] = np.float32(val)

        # Convert to DataFrame with explicit feature names and float32 dtype
        # This is critical to avoid "data did not contain feature names" error
        X_xgb_df = pd.DataFrame(X_xgb_np, columns=xgb_feature_names).astype('float32')
        
        # 3. Predict for each target across entire batch
        predictions_map = {}
        for target in ['pm2_5', 'pm10']:
            if target in self.xgb_models:
                model = self.xgb_models[target]
                log_preds = model.predict(X_xgb_df)
                preds = np.expm1(log_preds)
                preds = np.maximum(0, preds)
                # Reshape to (batch_size, n_iter)
                predictions_map[target] = preds.reshape(batch_size, self.n_iter)

        # 4. Aggregate results for each day
        batch_results = []
        for i in range(batch_size):
            day_preds = {t: predictions_map[t][i] for t in predictions_map}
            batch_results.append(self._aggregate_uncertainty(day_preds))
            
        return batch_results

    def _aggregate_uncertainty(self, predictions_map: Dict[str, np.ndarray]) -> Dict[str, Any]:
        results = {}
        
        for target, preds in predictions_map.items():
            mean_pred = float(np.mean(preds))
            std_dev = float(np.std(preds))
            
            p05 = np.percentile(preds, 5)
            p95 = np.percentile(preds, 95)
            p025 = np.percentile(preds, 2.5)
            p975 = np.percentile(preds, 97.5)
            
            uncertainty = {
                'prediction': round(mean_pred, 2),
                'uncertainty': {
                    'std': round(std_dev, 2),
                    'confidence_90': [round(p05, 2), round(p95, 2)],
                    'confidence_95': [round(p025, 2), round(p975, 2)],
                    'method': 'monte_carlo_dropout'
                }
            }
            
            validation_errors = self.validator.validate_prediction(mean_pred, uncertainty['uncertainty'])
            if validation_errors:
                uncertainty['validation_errors'] = validation_errors
                print(f"Validation Warning for {target}: {validation_errors}")
                
            results[target] = uncertainty
            
        return results
