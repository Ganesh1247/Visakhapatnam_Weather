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

    def predict_with_uncertainty(self, X_lstm: np.ndarray, base_feat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run MC Dropout inference.
        
        Args:
            X_lstm: Input tensor for LSTM (1, Seq_Len, Features)
            base_feat_dict: Dictionary of static features (Time, Weather) for XGBoost
            
        Returns:
            Dict containing mean prediction and uncertainty metrics.
        """
        
        # 1. MC Dropout Forward Pass (LSTM)
        # We must repeat the input to batch it, or run a loop. Batching is faster.
        # Create a batch of size n_iter
        X_tiled = np.tile(X_lstm, (self.n_iter, 1, 1))
        
        # Force training=True to enable Dropout
        # shape: (n_iter, embedding_dim)
        mc_embeddings = self.feature_extractor(X_tiled, training=True).numpy()
        
        # 2. Process each embedding through XGBoost
        # This is the tricky part: XGBoost is not a tensor model, so we might need to loop 
        # or construct a large DataFrame. DataFrame overhead is high, so let's try to optimize.
        
        # Construct XGBoost Feature Matrix for ALL iterations at once
        # Base features are constant across iterations
        
        # Create DataFrame for all iterations
        # Columns: [emb_0...emb_31] + [weather...] + [time...]
        
        # Feature Names
        xgb_feature_names = [f'emb_{j}' for j in range(mc_embeddings.shape[1])] + \
                            self.preprocessor.lstm_features + \
                            ['month', 'day_of_week', 'day', 'is_weekend', 'wind_dir_sin', 'wind_dir_cos', 'pressure_delta']
                            
        n_rows = self.n_iter
        
        # Embeddings part
        X_xgb_np = np.zeros((n_rows, len(xgb_feature_names)), dtype=np.float32)
        X_xgb_np[:, :mc_embeddings.shape[1]] = mc_embeddings
        
        # Static features part (fill the rest)
        # Map feature name to index
        feat_map = {name: i for i, name in enumerate(xgb_feature_names)}
        
        start_static = mc_embeddings.shape[1]
        
        # Fill static features efficiently
        for col, val in base_feat_dict.items():
            if col in feat_map:
                col_idx = feat_map[col]
                X_xgb_np[:, col_idx] = float(val)

        # Convert to DataFrame (XGBoost expects DMatrix or DataFrame with correct names)
        # Using DataFrame with correct column names is safer for feature mapping
        X_xgb_df = pd.DataFrame(X_xgb_np, columns=xgb_feature_names)
        
        predictions_map = {}
        
        # 3. Predict for each target
        # Calculate uncertainty for PM2.5 and PM10 primarily
        for target in ['pm2_5', 'pm10']:
            if target in self.xgb_models:
                model = self.xgb_models[target]
                
                # Predict
                log_preds = model.predict(X_xgb_df)
                
                # Inverse Transform (Log -> Linear)
                preds = np.expm1(log_preds)
                preds = np.maximum(0, preds) 
                
                # Additional Bias Correction if needed (assumed handled outside or additive)
                # If bias is additive constant, it doesn't affect variance/std, just shifts mean.
                
                predictions_map[target] = preds

        return self._aggregate_uncertainty(predictions_map)

    def _aggregate_uncertainty(self, predictions_map: Dict[str, np.ndarray]) -> Dict[str, Any]:
        results = {}
        
        for target, preds in predictions_map.items():
            # Basic Stats
            mean_pred = float(np.mean(preds))
            std_dev = float(np.std(preds))
            
            # Percentiles for CIs
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
            
            # Validate
            validation_errors = self.validator.validate_prediction(mean_pred, uncertainty['uncertainty'])
            if validation_errors:
                uncertainty['validation_errors'] = validation_errors
                # We log but don't crash, or maybe flag it
                print(f"Validation Warning for {target}: {validation_errors}")
                
            results[target] = uncertainty
            
        return results
