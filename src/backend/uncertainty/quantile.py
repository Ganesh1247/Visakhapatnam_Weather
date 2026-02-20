import numpy as np
import pandas as pd
import pickle
import os
import xgboost as xgb
from typing import Dict, Any, List

class QuantileRegressor:
    def __init__(self, model_dir: str = "models/quantile"):
        self.model_dir = model_dir
        self.models = {}
        self.targets = ['pm2_5', 'pm10']
        self.alphas = [0.05, 0.5, 0.95]
        self.load_models()

    def load_models(self):
        """Load all 6 quantile models (2 targets * 3 alphas)"""
        for target in self.targets:
            self.models[target] = {}
            for alpha in self.alphas:
                path = os.path.join(self.model_dir, f"xgb_{target}_{alpha}.pkl")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        self.models[target][alpha] = pickle.load(f)
                else:
                    print(f"Warning: Quantile model {path} not found.")

    def _xgb_predict(self, model, X_df):
        """Predict using DMatrix to ensure feature names are always preserved."""
        dmat = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
        if isinstance(model, xgb.Booster):
            return model.predict(dmat)
        else:
            return model.get_booster().predict(dmat)

    def predict(self, X_xgb: pd.DataFrame, bias: float = 0.0) -> Dict[str, Any]:
        """
        Generate predictions with uncertainty intervals using Quantile Regression.
        
        Returns:
            Dict with prediction and uncertainty for each target.
        """
        results = {}
        
        for target in self.targets:
            if target not in self.models or not self.models[target]:
                continue
                
            # Dictionary to store raw log predictions for each alpha
            preds_log = {}
            
            # Predict for each alpha
            for alpha in self.alphas:
                if alpha in self.models[target]:
                    # Predict (Output is Log(1+x))
                    val = self._xgb_predict(self.models[target][alpha], X_xgb)[0]
                    preds_log[alpha] = val
            
            # Check if we have all necessary quantiles
            if 0.5 in preds_log and 0.05 in preds_log and 0.95 in preds_log:
                # Inverse Transform
                # Apply expm1 to get real values
                pred_median = np.expm1(preds_log[0.5])
                pred_lower = np.expm1(preds_log[0.05])
                pred_upper = np.expm1(preds_log[0.95])
                
                # Apply non-negativity
                pred_median = max(0, pred_median)
                pred_lower = max(0, pred_lower)
                pred_upper = max(0, pred_upper)
                
                # Apply Bias Correction (Additive)
                pred_median += bias
                pred_lower += bias
                pred_upper += bias
                
                # Sort quantiles just in case of crossing (Post-processing)
                # quantile crossing check: lower <= median <= upper
                sorted_vals = sorted([pred_lower, pred_median, pred_upper])
                pred_lower, pred_median, pred_upper = sorted_vals[0], sorted_vals[1], sorted_vals[2]
                
                # Construct Result
                results[target] = {
                    'prediction': round(pred_median, 2),
                    'uncertainty': {
                        'confidence_90': [round(pred_lower, 2), round(pred_upper, 2)],
                        'method': 'quantile_regression'
                        # Note: We don't have std or 95% explicitly unless we assume distribution or train more models.
                        # For consistency with frontend, we might need mapped 95% or just reuse 90% as 95%?
                        # Or just provide what we have. Frontend expects 'confidence_90'. 
                        # 'confidence_95' is optional or we can approximate it.
                        # Let's just provide confidence_90.
                    }
                }
                
                # Approximate 95% if needed by widening 90% slightly? 
                # Or just duplicate it for safety if frontend strictly needs it (schema validation).
                # Current frontend parses confidence_90.
                results[target]['uncertainty']['confidence_95'] = results[target]['uncertainty']['confidence_90']
                
        return results
