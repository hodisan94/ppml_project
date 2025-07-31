# fl+dp.py with Debug Logs
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import json

def add_laplace_noise(X, epsilon=1.0, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=X.shape)
    print(f"[DEBUG] Laplace noise shape: {noise.shape}, scale: {scale:.4f}")
    return X + noise

def load_client_data(client_files):
    clients_data = {}
    for i, file_path in enumerate(client_files, 1):
        client_id = f"client_{i}"
        print(f"[INFO] Loading {client_id} data from {file_path}")
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        df_nodup = df.drop_duplicates()
        print(f"[INFO] {client_id}: {len(df)} -> {len(df_nodup)} samples after deduplication")
        X = df_nodup.drop("Readmitted", axis=1).values
        y = df_nodup["Readmitted"].values
        print(f"[DEBUG] {client_id} feature shape: {X.shape}, label shape: {y.shape}")
        clients_data[client_id] = {'X': X, 'y': y, 'size': len(X)}
    return clients_data

def federated_training_with_dp(clients_data, n_estimators=100, test_size=0.2,
                                random_state=42, use_dp=True, epsilon=1.0):
    print(f"\n{'=' * 60}")
    print("FEDERATED TRAINING WITH DIFFERENTIAL PRIVACY")
    print(f"{'=' * 60}")

    n_clients = len(clients_data)
    trees_per_client = n_estimators // n_clients
    extra_trees = n_estimators % n_clients

    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    all_trees = []

    for i, (client_id, data) in enumerate(clients_data.items()):
        client_num = i + 1
        client_trees = trees_per_client + (1 if i < extra_trees else 0)

        print(f"[DEBUG] Splitting data for {client_id} with {client_trees} trees")
        X_train, X_test, y_train, y_test = train_test_split(
            data['X'], data['y'], test_size=test_size, random_state=random_state,
            stratify=data['y'] if len(np.unique(data['y'])) > 1 else None
        )

        print(f"[DEBUG] {client_id} - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        if use_dp:
            print(f"[DP] Adding noise to {client_id} (epsilon={epsilon})")
            X_train = add_laplace_noise(X_train, epsilon=epsilon)

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_test.append(X_test)
        all_y_test.append(y_test)

        local_model = RandomForestClassifier(
            n_estimators=client_trees,
            random_state=random_state + client_num,
            class_weight="balanced"
        )
        local_model.fit(X_train, y_train)
        all_trees.extend(local_model.estimators_)

        print(f"[INFO] {client_id}: trained {client_trees} trees")

    X_train_combined = np.vstack(all_X_train)
    y_train_combined = np.hstack(all_y_train)
    X_test_combined = np.vstack(all_X_test)
    y_test_combined = np.hstack(all_y_test)

    print(f"[DEBUG] Combined train shape: {X_train_combined.shape}, test shape: {X_test_combined.shape}")

    train_set = {tuple(x) for x in X_train_combined}
    test_set = {tuple(x) for x in X_test_combined}
    overlap = train_set.intersection(test_set)
    if overlap:
        X_test_combined, y_test_combined = zip(*[
            (x, y) for x, y in zip(X_test_combined, y_test_combined)
            if tuple(x) not in overlap
        ])
        X_test_combined = np.array(X_test_combined)
        y_test_combined = np.array(y_test_combined)
        print(f"[INFO] Removed {len(overlap)} overlapping samples")
    else:
        print("[INFO] No overlap between train and test")

    print("[INFO] Building global model...")
    global_model = RandomForestClassifier(n_estimators=len(all_trees), random_state=random_state)
    dummy_X = X_train_combined[:100]
    dummy_y = y_train_combined[:100]
    global_model.fit(dummy_X, dummy_y)
    global_model.estimators_ = all_trees

    y_pred = global_model.predict(X_test_combined)
    y_proba = global_model.predict_proba(X_test_combined)[:, 1]
    accuracy = accuracy_score(y_test_combined, y_pred)
    auc = roc_auc_score(y_test_combined, y_proba)

    print(f"\n[RESULTS] Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")

    fl_metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "n_clients": n_clients,
        "n_estimators": len(all_trees),
        "use_dp": use_dp,
        "epsilon": epsilon
    }
    with open('fl_dp_metrics.json', 'w') as f:
        json.dump(fl_metrics, f, indent=2)

    print("[DEBUG] Saving model and data...")
    joblib.dump(global_model, "federated_model_dp.pkl")
    np.save("federated_X_train.npy", X_train_combined)
    np.save("federated_y_train.npy", y_train_combined)
    np.save("federated_X_test.npy", X_test_combined)
    np.save("federated_y_test.npy", y_test_combined)
    print("[INFO] Saved model and datasets with DP for MIA attack")

    return global_model

if __name__ == "__main__":
    client_files = [
        "../../../data/clients/client_1.csv",
        "../../../data/clients/client_2.csv",
        "../../../data/clients/client_3.csv",
        "../../../data/clients/client_4.csv",
        "../../../data/clients/client_5.csv"
    ]
    clients_data = load_client_data(client_files)
    print(f"[DEBUG] Loaded data for {len(clients_data)} clients")
    federated_training_with_dp(clients_data, use_dp=True, epsilon=1.0)