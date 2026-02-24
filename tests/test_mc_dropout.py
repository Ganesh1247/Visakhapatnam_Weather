
import unittest
import numpy as np
import tensorflow as tf
import time
from backend.uncertainty.mc_dropout import MCDropoutPredictor, UncertaintyValidator
from unittest.mock import MagicMock, Mock

class TestMCDropout(unittest.TestCase):
    
    def setUp(self):
        # Mock LSTM/Feature Extractor
        self.mock_fe = MagicMock()
        # Mock the behavior of the model when called:
        # It should return random embeddings if training=True
        
        def mock_call(inputs, training=False):
            # inputs shape: (batch, seq_len, feat)
            batch_size = inputs.shape[0]
            # Random embeddings (32 dim)
            # If training=True, add noise to simulate dropout
            base = np.zeros((batch_size, 32))
            if training:
                noise = np.random.normal(0, 0.1, (batch_size, 32))
                return tf.convert_to_tensor(base + noise, dtype=tf.float32)
            return tf.convert_to_tensor(base, dtype=tf.float32)
            
        self.mock_fe.side_effect = mock_call
        
        # Mock XGBoost
        self.mock_xgb_pm25 = MagicMock()
        # Predict returns log(pm2.5)
        # If input has noise, output should have noise
        def mock_predict(X):
            # X is DataFrame or array
            # mimic some function of input
            # sum of first embedding feature
            val = X.iloc[:, 0].values if hasattr(X, 'iloc') else X[:, 0]
            # return log(10 + val)
            return np.log1p(10 + val * 10) # 10 is base, val is noise
            
        self.mock_xgb_pm25.predict.side_effect = mock_predict
        
        self.xgb_models = {'pm2_5': self.mock_xgb_pm25}
        
        # Mock Preprocessor
        self.mock_preprocessor = MagicMock()
        self.mock_preprocessor.lstm_features = ['f'+str(i) for i in range(10)]
        
        self.predictor = MCDropoutPredictor(
            lstm_model=None,
            feature_extractor=self.mock_fe,
            xgb_models=self.xgb_models,
            preprocessor=self.mock_preprocessor,
            n_iter=50
        )

    # 1. Stochasticity Check
    def test_stochasticity(self):
        """Ensure that multiple forward passes produce different results."""
        X_input = np.zeros((1, 14, 10)) # 1 sample, 14 steps, 10 features
        base_feat = {'month': 1, 'day_of_week': 0, 'day': 1, 'is_weekend': 0, 
                     'wind_dir_sin': 0, 'wind_dir_cos': 0, 'pressure_delta': 0}
        for f in self.mock_preprocessor.lstm_features:
            base_feat[f] = 0
            
        # The predictor calls feature_extractor(training=True)
        # Our mock adds noise when training=True
        results = self.predictor.predict_with_uncertainty(X_input, [base_feat])[0]
        
        uncertainty = results['pm2_5']['uncertainty']
        self.assertGreater(uncertainty['std'], 0.0, "Standard deviation should be positive with dropout")
        
        # Verify 90% CI width > 0
        ci_90 = uncertainty['confidence_90']
        self.assertGreater(ci_90[1] - ci_90[0], 0.0)

    # 2. Prediction Inside CI
    def test_prediction_coverage(self):
        """Ensure mean prediction is within confidence intervals."""
        X_input = np.zeros((1, 14, 10))
        base_feat = {'month': 1} # Minimal needed for mock checks, but full for code
        # ... (reuse feat setup)
        base_feat.update({k:0 for k in self.mock_preprocessor.lstm_features})
        base_feat.update({'month': 1, 'day_of_week': 0, 'day': 1, 'is_weekend': 0, 
                     'wind_dir_sin': 0, 'wind_dir_cos': 0, 'pressure_delta': 0})

        results = self.predictor.predict_with_uncertainty(X_input, [base_feat])[0]
        pm25 = results['pm2_5']
        
        mean = pm25['prediction']
        ci90 = pm25['uncertainty']['confidence_90']
        ci95 = pm25['uncertainty']['confidence_95']
        
        # Check Mean inside CI
        self.assertTrue(ci90[0] <= mean <= ci90[1], f"Mean {mean} not in 90% CI {ci90}")
        self.assertTrue(ci95[0] <= mean <= ci95[1], f"Mean {mean} not in 95% CI {ci95}")

    # 3. Validation Logic
    def test_validator_logic(self):
        """Test the UncertaintyValidator catches errors."""
        validator = UncertaintyValidator()
        
        # Valid case
        errors = validator.validate_prediction(100, {
            'std': 10, 'confidence_90': [80, 120], 'confidence_95': [70, 130]
        })
        self.assertEqual(len(errors), 0)
        
        # Invalid: Std = 0
        errors = validator.validate_prediction(100, {
            'std': 0, 'confidence_90': [100, 100], 'confidence_95': [100, 100]
        })
        self.assertIn("Standard deviation must be positive, got 0", errors)
        
        # Invalid: 95 CI narrower than 90 CI
        errors = validator.validate_prediction(100, {
            'std': 10, 'confidence_90': [80, 120], 'confidence_95': [90, 110]
        })
        self.assertTrue(any("smaller than 90% CI" in e for e in errors))

    # 4. Performance Benchmark
    def test_performance(self):
        """Benchmark inference time."""
        X_input = np.zeros((1, 14, 10))
        base_feat = {k:0 for k in self.mock_preprocessor.lstm_features}
        base_feat.update({'month': 1, 'day_of_week': 0, 'day': 1, 'is_weekend': 0, 
                     'wind_dir_sin': 0, 'wind_dir_cos': 0, 'pressure_delta': 0})
        
        start = time.time()
        self.predictor.predict_with_uncertainty(X_input, [base_feat])
        duration = time.time() - start
        
        # Should be fast with mocks, but let's set a generous threshold for "local unit test"
        # Real LSTM is heavier, but we test the overhead of LOOP logic here.
        # Overhead should be < 100ms for 50 iters if vectorized
        self.assertLess(duration, 0.5, f"Inference took too long: {duration:.4f}s")
        print(f"\nInference Time (Mocked, 50 iters): {duration*1000:.2f}ms")

    # 5. Integration / End-to-End Structure
    def test_output_structure(self):
        X_input = np.zeros((1, 14, 10))
        base_feat = {k:0 for k in self.mock_preprocessor.lstm_features}
        base_feat.update({'month': 1, 'day_of_week': 0, 'day': 1, 'is_weekend': 0, 
                     'wind_dir_sin': 0, 'wind_dir_cos': 0, 'pressure_delta': 0})
        
        result = self.predictor.predict_with_uncertainty(X_input, [base_feat])[0]
        
        self.assertIn('pm2_5', result)
        self.assertIn('prediction', result['pm2_5'])
        self.assertIn('uncertainty', result['pm2_5'])
        
        u = result['pm2_5']['uncertainty']
        self.assertIn('confidence_90', u)
        self.assertIn('confidence_95', u)
        self.assertIn('std', u)
        self.assertEqual(u['method'], 'monte_carlo_dropout')

if __name__ == '__main__':
    unittest.main()
