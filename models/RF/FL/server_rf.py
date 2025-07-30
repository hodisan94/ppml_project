import os
import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Paths
CLIENT_MODELS_DIR = "models/RF/FL/models"
TEST_PATH = "data/processed/full_preprocessed.csv"
EVAL_DIR = "evaluation"
THRESHOLD = 0.25  # adjustable

# Create evaluation directory if not exists
os.makedirs(EVAL_DIR, exist_ok=True)

print("[SERVER] Loading test set from:", TEST_PATH)
df = pd.read_csv(TEST_PATH)
X = df.drop(columns=["Readmitted"])
y = df["Readmitted"].values
print(f"[DEBUG] Test shape: {X.shape}, Label 1s: {sum(y)}")

# Load client models
client_models = []
for i in range(1, 6):
    model_path = os.path.join(CLIENT_MODELS_DIR, f"client_{i}_model.pkl")
    print(f"[SERVER] Loading model: {model_path}")
    try:
        model = joblib.load(model_path)
        if len(model.classes_) < 2:
            print(f"[WARNING] Skipping model {i}: only one class in training (shape: {X.shape[0]}, 1)")
            continue

        client_models.append(model)

    except FileNotFoundError:
        print(f"[ERROR] Model not found: {model_path}")

if len(client_models) == 0:
    print("[FATAL] No models loaded. Aborting ensemble prediction.")
    exit()

print(f"[SERVER] Loaded {len(client_models)} valid client models.")
print("[SERVER] Performing ensemble prediction (average probs)...")

# Average prediction
probs = np.zeros(len(X))
for model in client_models:
    p = model.predict_proba(X)[:, 1]
    probs += p

probs /= len(client_models)

# Save for MIA
np.save(os.path.join(EVAL_DIR, "ensemble_probs.npy"), probs)
np.save(os.path.join(EVAL_DIR, "ensemble_labels.npy"), y)
print("[SERVER] Saved ensemble probabilities and labels for MIA.")

# Evaluation
y_pred = (probs >= THRESHOLD).astype(int)
auc = roc_auc_score(y, probs)
print("\n[ENSEMBLE] Evaluation Metrics:")
print(f"AUC: {auc:.4f}")
print(classification_report(y, y_pred, digits=4))
