import tensorflow as tf
import numpy as np
import os

# Define the directory for saved models
model_dir = "saved_models"

# Get the latest model file
saved_models = sorted([f for f in os.listdir(model_dir) if f.endswith('.h5')], reverse=True)
latest_model_path = os.path.join(model_dir, saved_models[0]) if saved_models else None

# Load the latest model if it exists
if latest_model_path:
    q_model = tf.keras.models.load_model(latest_model_path)
    print(f"Loaded model from {latest_model_path}")
else:
    print("No model found in saved_models directory.")
    exit()

# Sample new data
new_data = np.array([[0.03]])  # Replace with actual new data as needed

# Run the model on the new data
predicted_action = q_model.predict(new_data)
print("Predicted action for new data:", predicted_action)
