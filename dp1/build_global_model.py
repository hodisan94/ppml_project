import pickle
import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

# === Load global weights from FL ===
weights_path = "results/global_weights_round_5.pkl"
with open(weights_path, "rb") as f:
    weights = pickle.load(f)

print(f"[INFO] Loaded weights from {weights_path}")
print(f"[INFO] weights[0] shape (coefficients): {weights[0].shape}")
print(f"[INFO] weights[1] shape (intercept): {weights[1].shape}")

# === Build scikit-learn LogisticRegression model ===
model = LogisticRegression()
model.coef_ = weights[0].reshape(1, -1)  # shape: (1, num_features)
model.intercept_ = np.array([weights[1][0]])  # shape: (1,)
model.classes_ = np.array([0, 1])  # must be ndarray, not list

print(f"[INFO] Model ready with coef_.shape = {model.coef_.shape}")
print(f"[INFO] Intercept = {model.intercept_}")
print(f"[INFO] Classes = {model.classes_}")

# === Save model ===
output_path = "results/global_model_sklearn.pkl"
os.makedirs("results", exist_ok=True)
joblib.dump(model, output_path)

print(f"[SUCCESS] Sklearn global model saved to {output_path}")
