import tensorflow as tf
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model('path/to/your/model')

# Function to make predictions
def predict_anomaly(data):
    predictions = model.predict(data)
    return predictions

# Example data for prediction
sample_data = np.random.rand(1, 64, 64, 3)  # shape depends on your model
anomaly_predictions = predict_anomaly(sample_data)
print(anomaly_predictions)
