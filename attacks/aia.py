import os
import numpy as np
import json
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def run_attribute_inference(model_path, X_path, sensitive_col, model_name, output_dir):
    print(f"\n{'=' * 60}")
    print(f"[ATTACK] Running Attribute Inference Attack on: {model_name}")
    print(f"{'=' * 60}")

    # Load model and data
    model = load(model_path)
    X = np.load(X_path)

    print(f"[DEBUG] X shape before removing sensitive column: {X.shape}")

    # Extract sensitive attribute (gender in this case)
    sensitive_attr = X[:, sensitive_col]

    print(f"[DEBUG] First 5 values of sensitive attribute (ground truth):", sensitive_attr[:5])

    # Convert to binary if values are continuous (like in FL+DP)
    if "DP" in model_name:
        median = np.median(sensitive_attr)
        sensitive_attr = (sensitive_attr > median).astype(int)
        print(f"[INFO] Converted sensitive attribute to binary using median threshold: {median:.4f}")

    # Check for class balance
    unique, counts = np.unique(sensitive_attr, return_counts=True)
    print(f"[DEBUG] Class distribution: {dict(zip(unique, counts))}")

    # Calculate baseline accuracy (majority class)
    majority_class_acc = max(counts) / len(sensitive_attr)
    print(f"[DEBUG] Baseline accuracy (majority class): {majority_class_acc:.4f}")

    # CRITICAL FIX: Split data into auxiliary (for training attack) and target (for testing attack)
    X_aux, X_target, sensitive_aux, sensitive_target = train_test_split(
        X, sensitive_attr, test_size=0.5, random_state=42, stratify=sensitive_attr
    )

    print(f"[DEBUG] Auxiliary data shape: {X_aux.shape}")
    print(f"[DEBUG] Target data shape: {X_target.shape}")

    # Get model predictions for both auxiliary and target data (using full data including sensitive attr)
    try:
        # Try to get prediction probabilities
        y_pred_aux = model.predict_proba(X_aux)
        confidence_aux = np.max(y_pred_aux, axis=1).reshape(-1, 1)

        y_pred_target = model.predict_proba(X_target)
        confidence_target = np.max(y_pred_target, axis=1).reshape(-1, 1)

        # Also include the predicted class
        pred_class_aux = model.predict(X_aux).reshape(-1, 1)
        pred_class_target = model.predict(X_target).reshape(-1, 1)

    except:
        # Fallback to regular predictions
        pred_class_aux = model.predict(X_aux).reshape(-1, 1)
        pred_class_target = model.predict(X_target).reshape(-1, 1)

        # Use prediction as confidence (not ideal but works)
        confidence_aux = pred_class_aux.astype(float)
        confidence_target = pred_class_target.astype(float)

    # NOW remove sensitive attribute from the data used for attack training
    # The attacker doesn't have access to the sensitive attribute directly
    X_aux_no_sensitive = np.delete(X_aux, sensitive_col, axis=1)
    X_target_no_sensitive = np.delete(X_target, sensitive_col, axis=1)

    print(f"[DEBUG] Attack data shape (without sensitive attr): {X_aux_no_sensitive.shape}")

    # Create attack features: original features (without sensitive attr) + model outputs
    X_attack_aux = np.hstack([X_aux_no_sensitive, confidence_aux, pred_class_aux])
    X_attack_target = np.hstack([X_target_no_sensitive, confidence_target, pred_class_target])

    print(f"[DEBUG] Attack features shape (auxiliary): {X_attack_aux.shape}")
    print(f"[DEBUG] Attack features shape (target): {X_attack_target.shape}")

    # Train attacker model on auxiliary data
    attack_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    attack_model.fit(X_attack_aux, sensitive_aux)

    # Test attack model on target data (the actual attack)
    inferred_attr = attack_model.predict(X_attack_target)

    # Evaluate attack on target data
    acc = accuracy_score(sensitive_target, inferred_attr)
    print(f"[RESULT] AIA Accuracy for {model_name}: {acc:.4f}")

    # Calculate improvement over baseline
    improvement = acc - majority_class_acc
    print(f"[RESULT] Improvement over baseline: {improvement:.4f}")

    # Feature importance analysis
    feature_names = [f'feat_{i}' for i in range(X_aux_no_sensitive.shape[1])] + ['confidence', 'prediction']
    importances = attack_model.feature_importances_

    # Sort and show top features
    indices = np.argsort(importances)[::-1][:5]
    print("[DEBUG] Top 5 most important features for attack:")
    for i, idx in enumerate(indices):
        print(f"  {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Plot result
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"aia_{model_name.replace(' ', '_').lower()}.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot predictions vs ground truth
    sample_size = min(100, len(sensitive_target))
    ax1.plot(sensitive_target[:sample_size], label='True Attribute', linestyle='--', marker='o', markersize=3)
    ax1.plot(inferred_attr[:sample_size], label='Inferred Attribute', linestyle=':', marker='x', markersize=3)
    ax1.set_title(f'Attribute Inference – {model_name}\nAccuracy={acc:.4f}')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Attribute Value')
    ax1.legend()
    ax1.grid(True)

    # Plot feature importance
    top_features = indices[:10]
    ax2.barh(range(len(top_features)), importances[top_features])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels([feature_names[i] for i in top_features])
    ax2.set_xlabel('Feature Importance')
    ax2.set_title('Top Features for Attack')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] AIA plot saved to: {plot_path}")
    plt.close()

    # Save results as JSON
    json_results = {
        "model": model_name,
        "attack_type": "attribute_inference",
        "results": {
            "accuracy": acc,
            "improvement_over_baseline": improvement,
            "baseline_accuracy": majority_class_acc,
            "plot_file": os.path.basename(plot_path)
        },
        "feature_importance": {
            "top_features": [
                {"feature": feature_names[i], "importance": float(importances[i])} 
                for i in indices[:5]
            ]
        },
        "metadata": {
            "samples_used": len(X),
            "sensitive_column": sensitive_col,
            "timestamp": str(np.datetime64('now'))
        }
    }
    
    json_path = os.path.join(output_dir, f"aia_results_{model_name.replace(' ', '_').lower()}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    print(f"[SAVED] Results saved to: {json_path}")

    return acc, improvement


def main():
    sensitive_col = 1  # Gender
    models = {
        "Naive RF": {
            "model": "models/RF/Naive/rf_naive_model.pkl",
            "X": "models/RF/Naive/X_member.npy",
            "output_dir": "output/results/naive"
        },
        "Federated": {
            "model": "models/RF/FL/federated_model.pkl",
            "X": "models/RF/FL/federated_X_train.npy",
            "output_dir": "output/results/federated"
        },
        "Federated + DP": {
            "model": "models/RF/FL+DP/federated_model_dp.pkl",
            "X": "models/RF/FL+DP/federated_X_train.npy",
            "output_dir": "output/results/federated_dp"
        }
    }

    all_results = {}
    for model_name, paths in models.items():
        # Create output directory for this model
        os.makedirs(paths["output_dir"], exist_ok=True)
        
        acc, improvement = run_attribute_inference(paths["model"], paths["X"], sensitive_col, model_name, paths["output_dir"])
        all_results[model_name] = {"accuracy": acc, "improvement": improvement}

    print(f"\n{'=' * 60}")
    print("[SUMMARY] Attribute Inference Attack Results")
    print(f"{'=' * 60}")
    for model_name, results in all_results.items():
        print(f"{model_name:20} | Accuracy: {results['accuracy']:.4f} | Improvement: {results['improvement']:.4f}")



if __name__ == "__main__":
    main()