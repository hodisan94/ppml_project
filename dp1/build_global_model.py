import pickle
import tensorflow as tf
import numpy as np
import os

# === CONFIG ===
weights_path = "results/global_weights_round_5.pkl"
output_model_path = "results/global_model.h5"
input_dim = 20  # שנה את זה לפי מספר הפיצ'רים האמיתי שלך

# === Load aggregated weights ===
with open(weights_path, "rb") as f:
    weights = pickle.load(f)

print(f"[INFO] Loaded weights from {weights_path}")
print(f"[INFO] Weight shapes: {[w.shape for w in weights]}")

# === Build the same model structure as the clients ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.build()

# === Assign weights ===
try:
    model.set_weights(weights)
    print("[INFO] Weights successfully loaded into the model.")
except Exception as e:
    print(f"[ERROR] Failed to set weights: {e}")
    exit(1)

# === Save the unified global model ===
os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
model.save(output_model_path)
print(f"[SUCCESS] Global model saved to: {output_model_path}")
