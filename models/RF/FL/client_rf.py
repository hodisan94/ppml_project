import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Paths
DATA_DIR = "data/clients"
MODEL_DIR = "models/RF/FL/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("[CLIENT] Starting federated client training...")

for i in range(1, 6):
    client_file = os.path.join(DATA_DIR, f"client_{i}.csv")
    print(f"\n[CLIENT {i}] Loading {client_file}")

    if not os.path.exists(client_file):
        print(f"[ERROR] File not found: {client_file}")
        continue

    df = pd.read_csv(client_file)
    print(f"[DEBUG] Shape: {df.shape}")

    label_counts = df["Readmitted"].value_counts().to_dict()
    print(f"[DEBUG] Label distribution: {label_counts}")

    X = df.drop("Readmitted", axis=1).values
    y = df["Readmitted"].values

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"client_{i}_model.pkl")
    joblib.dump(model, model_path)
    print(f"[CLIENT {i}] Saved model to {model_path}")

    # Skip evaluation if only one class present
    if len(np.unique(y)) < 2:
        print(f"[WARNING] Client {i} contains only one class ({np.unique(y)[0]}). Skipping AUC and report.\n")
        continue

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"[CLIENT {i}] AUC on train set: {auc:.4f}")
    print(f"[CLIENT {i}] Classification Report:\n{classification_report(y, y_pred)}")

print("\n[CLIENT] All client models trained and saved.")
