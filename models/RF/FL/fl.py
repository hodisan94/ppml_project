# simple_federated_learning.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import json


def load_client_data(client_files):
    """Load data for all clients"""
    clients_data = {}

    for i, file_path in enumerate(client_files, 1):
        client_id = f"client_{i}"
        print(f"[INFO] Loading {client_id} data from {file_path}")

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Remove duplicates within client
        df_nodup = df.drop_duplicates()
        print(f"[INFO] {client_id}: {len(df)} -> {len(df_nodup)} samples after deduplication")

        # Split features and labels
        X = df_nodup.drop("Readmitted", axis=1).values
        y = df_nodup["Readmitted"].values

        clients_data[client_id] = {
            'X': X,
            'y': y,
            'size': len(X)
        }

    return clients_data


def simple_federated_training(clients_data, n_estimators=100, test_size=0.2, random_state=42):
    """
    Simple federated training: each client trains trees, then combine them
    """
    print(f"\n{'=' * 60}")
    print("SIMPLE FEDERATED LEARNING TRAINING")
    print(f"{'=' * 60}")

    # Calculate trees per client
    n_clients = len(clients_data)
    trees_per_client = n_estimators // n_clients
    extra_trees = n_estimators % n_clients

    print(f"[INFO] Training with {n_clients} clients")
    print(f"[INFO] Base trees per client: {trees_per_client}")
    print(f"[INFO] Extra trees for first {extra_trees} clients")

    # Store all training/test data
    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []

    # Store all trained trees
    all_trees = []

    for i, (client_id, data) in enumerate(clients_data.items()):
        client_num = i + 1

        # Determine number of trees for this client
        client_trees = trees_per_client + (1 if i < extra_trees else 0)

        print(f"[INFO] Training {client_id} with {client_trees} trees")

        # Split client data
        X_train, X_test, y_train, y_test = train_test_split(
            data['X'], data['y'],
            test_size=test_size,
            random_state=random_state,
            stratify=data['y'] if len(np.unique(data['y'])) > 1 else None
        )

        # Store for later combination
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_test.append(X_test)
        all_y_test.append(y_test)

        # Train local model
        local_model = RandomForestClassifier(
            n_estimators=client_trees,
            random_state=random_state + client_num,
            class_weight="balanced"
        )

        local_model.fit(X_train, y_train)

        # Extract trees from local model
        for tree in local_model.estimators_:
            all_trees.append(tree)

        print(f"[INFO] {client_id}: {len(X_train)} train, {len(X_test)} test samples")

    # Combine all client data
    X_train_combined = np.vstack(all_X_train)
    y_train_combined = np.hstack(all_y_train)
    X_test_combined = np.vstack(all_X_test)
    y_test_combined = np.hstack(all_y_test)

    print(f"[INFO] Combined training data: {X_train_combined.shape}")
    print(f"[INFO] Combined test data: {X_test_combined.shape}")

    # Remove overlap between train and test sets to ensure MIA validity
    print("[INFO] Checking for and removing overlapping samples between train and test sets...")
    train_set = {tuple(row) for row in X_train_combined}
    test_set = {tuple(row) for row in X_test_combined}

    overlap = train_set.intersection(test_set)
    print(f"[INFO] Found {len(overlap)} overlapping samples")

    if overlap:
        # Mask out overlapping rows
        X_test_clean = []
        y_test_clean = []

        for x, y in zip(X_test_combined, y_test_combined):
            if tuple(x) not in overlap:
                X_test_clean.append(x)
                y_test_clean.append(y)

        X_test_combined = np.array(X_test_clean)
        y_test_combined = np.array(y_test_clean)

        print(f"[INFO] Test set after cleaning: {X_test_combined.shape}")
    else:
        print("[INFO] No overlap found. No cleaning necessary.")

    # Create global model by combining trees
    print("[INFO] Creating global federated model...")
    global_model = RandomForestClassifier(
        n_estimators=len(all_trees),
        random_state=random_state
    )

    # Fit with dummy data to initialize structure
    dummy_X = X_train_combined[:100]  # Use real data subset
    dummy_y = y_train_combined[:100]
    global_model.fit(dummy_X, dummy_y)

    # Replace estimators with federated trees
    global_model.estimators_ = all_trees
    global_model.n_estimators = len(all_trees)

    # Evaluate global model
    y_pred = global_model.predict(X_test_combined)
    y_proba = global_model.predict_proba(X_test_combined)[:, 1]

    accuracy = accuracy_score(y_test_combined, y_pred)
    auc = roc_auc_score(y_test_combined, y_proba)

    print(f"\n[RESULTS] Federated Model Performance:")
    print(f"[RESULTS] Accuracy: {accuracy:.4f}")
    print(f"[RESULTS] AUC: {auc:.4f}")
    print(f"[RESULTS] Total trees: {len(all_trees)}")

    # Save metrics
    fl_metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'n_clients': n_clients,
        'n_estimators': len(all_trees),
        'trees_per_client': trees_per_client
    }

    with open('fl_metrics.json', 'w') as f:
        json.dump(fl_metrics, f, indent=2)

    return global_model, X_train_combined, y_train_combined, X_test_combined, y_test_combined


def train_centralized_model(full_data_path, test_size=0.2, random_state=42):
    """Train centralized model for comparison"""
    print(f"\n{'=' * 60}")
    print("CENTRALIZED MODEL TRAINING")
    print(f"{'=' * 60}")

    # Load full dataset
    print(f"[INFO] Loading full dataset from {full_data_path}")
    df_full = pd.read_csv(full_data_path)
    df_full_nodup = df_full.drop_duplicates()

    print(f"[INFO] Dataset: {len(df_full)} -> {len(df_full_nodup)} samples after deduplication")

    X_full = df_full_nodup.drop("Readmitted", axis=1).values
    y_full = df_full_nodup["Readmitted"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full
    )

    print(f"[INFO] Training set: {X_train.shape}")
    print(f"[INFO] Test set: {X_test.shape}")

    # Train centralized model
    centralized_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced"
    )

    centralized_model.fit(X_train, y_train)

    # Evaluate
    y_pred = centralized_model.predict(X_test)
    y_proba = centralized_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n[RESULTS] Centralized Model Performance:")
    print(f"[RESULTS] Accuracy: {accuracy:.4f}")
    print(f"[RESULTS] AUC: {auc:.4f}")

    # Save metrics
    cent_metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'n_estimators': 100
    }

    with open('../Naive/centralized_metrics.json', 'w') as f:
        json.dump(cent_metrics, f, indent=2)

    return centralized_model, X_train, y_train, X_test, y_test


def compare_and_save_models(full_data_path, client_files):
    """Compare centralized vs federated and save everything for MIA"""
    print(f"\n{'=' * 80}")
    print("COMPARING CENTRALIZED VS FEDERATED LEARNING")
    print(f"{'=' * 80}")

    # Train centralized model
    cent_model, X_train_cent, y_train_cent, X_test_cent, y_test_cent = train_centralized_model(full_data_path)

    # Train federated model
    clients_data = load_client_data(client_files)
    if not clients_data:
        print("[ERROR] No client data loaded!")
        return None

    fl_model, X_train_fl, y_train_fl, X_test_fl, y_test_fl = simple_federated_training(clients_data)

    # Save models
    joblib.dump(cent_model, '../Naive/centralized_model.pkl')
    joblib.dump(fl_model, 'federated_model.pkl')
    print("[INFO] Models saved!")

    # Save data for MIA attacks
    np.save('../Naive/centralized_X_train.npy', X_train_cent)
    np.save('../Naive/centralized_y_train.npy', y_train_cent)
    np.save('../Naive/centralized_X_test.npy', X_test_cent)
    np.save('../Naive/centralized_y_test.npy', y_test_cent)

    np.save('federated_X_train.npy', X_train_fl)
    np.save('federated_y_train.npy', y_train_fl)
    np.save('federated_X_test.npy', X_test_fl)
    np.save('federated_y_test.npy', y_test_fl)
    print("[INFO] Data saved for MIA attacks!")

    # Create comparison plot
    with open('../Naive/centralized_metrics.json', 'r') as f:
        cent_metrics = json.load(f)
    with open('fl_metrics.json', 'r') as f:
        fl_metrics = json.load(f)

    plt.figure(figsize=(12, 5))

    # Accuracy comparison
    plt.subplot(1, 2, 1)
    models = ['Centralized', 'Federated']
    accuracies = [cent_metrics['accuracy'], fl_metrics['accuracy']]
    bars = plt.bar(models, accuracies, color=['blue', 'green'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f'{acc:.4f}',
                 ha='center', va='bottom')

    # AUC comparison
    plt.subplot(1, 2, 2)
    aucs = [cent_metrics['auc'], fl_metrics['auc']]
    bars = plt.bar(models, aucs, color=['blue', 'green'], alpha=0.7)
    plt.ylabel('AUC')
    plt.title('Model AUC Comparison')
    plt.ylim(0, 1)
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width() / 2, auc + 0.01, f'{auc:.4f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE - READY FOR MIA ATTACKS")
    print(f"{'=' * 80}")

    return {
        'centralized_metrics': cent_metrics,
        'federated_metrics': fl_metrics
    }


if __name__ == "__main__":
    # File paths - adjust as needed
    client_files = [
        "../../../data/clients/client_1.csv",
        "../../../data/clients/client_2.csv",
        "../../../data/clients/client_3.csv",
        "../../../data/clients/client_4.csv",
        "../../../data/clients/client_5.csv"
    ]

    full_data_path = "../../../data/processed/full_preprocessed.csv"

    # Run comparison
    results = compare_and_save_models(full_data_path, client_files)