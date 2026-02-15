import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ConformalPredictor:
    def __init__(self, model_path: str = "models/conformal_params.json", alpha: float = 0.1):
        """
        Split Conformal Prediction.
        alpha = 0.1 means 90% confidence.
        """
        self.alpha = alpha
        self.q_hat = {} # Calibration Quantiles per target
        self.load_params(model_path)

    def load_params(self, path: str):
        import json
        import os
        
        # Try as is
        if os.path.exists(path):
            with open(path, "r") as f:
                self.q_hat = json.load(f)
            print(f"Loaded Conformal Params from {path}: {self.q_hat}")
            return

        # Try relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming structure: backend/uncertainty/conformal.py -> ../../models/conformal_params.json
        rel_path = os.path.join(base_dir, "../../models/conformal_params.json")
        rel_path = os.path.abspath(rel_path)
        
        if os.path.exists(rel_path):
            with open(rel_path, "r") as f:
                self.q_hat = json.load(f)
            print(f"Loaded Conformal Params from {rel_path}: {self.q_hat}")
        else:
            print(f"Warning: Conformal params not found at {path} or {rel_path}.")

    def predict(self, y_pred: float, target: str) -> Dict[str, Any]:
        """
        Return prediction interval.
        [y_pred - q_hat, y_pred + q_hat]
        """
        if target not in self.q_hat:
            return None
        
        q = self.q_hat[target]
        lower = y_pred - q
        upper = y_pred + q
        
        # Guardrails
        lower = max(0, lower)
        
        return {
            'confidence_90': [round(lower, 2), round(upper, 2)],
            'method': 'conformal_prediction'
        }
