# Re-save LSTM model without optimizer and custom metrics
from tensorflow.keras.models import load_model

# Load the model (ignore optimizer and custom metrics)
lstm_model = load_model('models/lstm_hybrid_chain.h5', compile=False)

# Save the model without optimizer
lstm_model.save('models/lstm_hybrid_chain.h5', include_optimizer=False)
print('LSTM model re-saved successfully.')
