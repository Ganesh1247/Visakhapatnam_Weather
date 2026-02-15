import tensorflow as tf

# Load LSTM model
model = tf.keras.models.load_model('models/lstm_hybrid_chain.h5', compile=False)

print("=" * 60)
print("LSTM Model Configuration:")
print("=" * 60)
print("\nModel Input Shape:")
print(model.input_shape)
print("\nModel Summary:")
model.summary()
