# Test model loading for deployment compatibility

from tensorflow.keras.models import load_model
import pickle
import os

# Test LSTM model loading
try:
    lstm_model = load_model('models/lstm_hybrid_chain.h5')
    print('LSTM model loaded successfully.')
except Exception as e:
    print(f'Error loading LSTM model: {e}')

# Test XGBoost model loading
xgb_targets = ['pm10', 'pm2_5', 'temp_avg', 'temp_min', 'temp_max', 'humidity', 'rainfall', 'wind_speed']
for target in xgb_targets:
    try:
        with open(f'models/xgb_chain_{target}.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        print(f'XGB model for {target} loaded successfully.')
    except Exception as e:
        print(f'Error loading XGB model for {target}: {e}')
