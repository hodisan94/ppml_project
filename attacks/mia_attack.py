import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve
)
from keras.models import load_model
import joblib
import json


def get_loss_scores(model, X, y, framework="keras"):
    """Get per-sample loss scores (better than confidence for MIA)"""
    print(f"[DEBUG] Getting loss scores using framework: {framework}")

    if framework == "keras":
        # Use model's loss function to get per-sample loss
        import tensorflow as tf

        # Convert to tf tensors
        X_tf = tf.constant(X, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)

        # Get predictions
        with tf.GradientTape() as tape:
            predictions = model(X_tf, training=False)
            # Use categorical crossentropy for multi-class
            loss = tf.keras.losses.categorical_crossentropy(y_tf, predictions)

        return loss.numpy()

    elif framework == "sklearn":
        # For sklearn, use negative log-likelihood
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # Convert y to class indices if needed
            if len(y.shape) > 1:
                y_indices = np.argmax(y, axis=1)
            else:
                y_indices = y.astype(int)

            # Get probability for true class
            true_class_probs = probs[np.arange(len(y_indices)), y_indices]
            # Return negative log-likelihood
            return -np.log(true_class_probs + 1e-10)
        else:
            raise ValueError("Sklearn model does not support predict_proba")

    else:
        raise ValueError("Unknown framework")


def get_confidence_scores(model, X, framework="keras"):
    """Original confidence-based approach"""
    print(f"[DEBUG] Getting confidence scores using framework: {framework}")
    if framework == "keras":
        probs = model.predict(X)
        return np.max(probs, axis=1)
    elif framework == "sklearn":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return np.max(probs, axis=1)
        else:
            raise ValueError("Sklearn model does not support predict_proba")
    else:
        raise ValueError("Unknown framework")


def get_entropy_scores(model, X, framework="keras"):
    """Get prediction entropy (lower entropy = more confident = more likely member)"""
    print(f"[DEBUG] Getting entropy scores using framework: {framework}")

    if framework == "keras":
        probs = model.predict(X)
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return entropy

    elif framework == "sklearn":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            return entropy
        else:
            raise ValueError("Sklearn model does not support predict_proba")

    else:
        raise ValueError("Unknown framework")


def evaluate_mia_comprehensive(model, X_member, X_nonmember, y_member, y_nonmember, framework="keras"):
    """Comprehensive MIA evaluation with multiple attack strategies"""
    print("[DEBUG] Evaluating MIA with multiple strategies...")

    results = {}

    # Strategy 1: Confidence-based attack
    print("[DEBUG] Strategy 1: Confidence-based attack")
    try:
        member_conf = get_confidence_scores(model, X_member, framework)
        nonmember_conf = get_confidence_scores(model, X_nonmember, framework)

        y_true = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])
        y_scores = np.concatenate([member_conf, nonmember_conf])

        results['confidence'] = evaluate_attack_performance(y_true, y_scores, "Confidence")
    except Exception as e:
        print(f"[ERROR] Confidence attack failed: {e}")
        results['confidence'] = None

    # Strategy 2: Entropy-based attack (lower entropy = more likely member)
    print("[DEBUG] Strategy 2: Entropy-based attack")
    try:
        member_entropy = get_entropy_scores(model, X_member, framework)
        nonmember_entropy = get_entropy_scores(model, X_nonmember, framework)

        # For entropy, we want LOWER scores for members, so we negate
        y_scores = np.concatenate([-member_entropy, -nonmember_entropy])

        results['entropy'] = evaluate_attack_performance(y_true, y_scores, "Entropy")
    except Exception as e:
        print(f"[ERROR] Entropy attack failed: {e}")
        results['entropy'] = None

    # Strategy 3: Loss-based attack (only if we have labels)
    print("[DEBUG] Strategy 3: Loss-based attack")
    if y_member is not None and y_nonmember is not None:
        try:
            member_loss = get_loss_scores(model, X_member, y_member, framework)
            nonmember_loss = get_loss_scores(model, X_nonmember, y_nonmember, framework)

            # For loss, we want LOWER scores for members, so we negate
            y_scores = np.concatenate([-member_loss, -nonmember_loss])

            results['loss'] = evaluate_attack_performance(y_true, y_scores, "Loss")
        except Exception as e:
            print(f"[ERROR] Loss attack failed: {e}")
            results['loss'] = None
    else:
        print("[DEBUG] No labels provided, skipping loss-based attack")
        results['loss'] = None

    return results


def evaluate_attack_performance(y_true, y_scores, attack_name):
    """Evaluate attack performance with threshold sweeping"""
    print(f"[DEBUG] Evaluating {attack_name} attack performance...")

    # Sweep thresholds
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 101)
    best_acc, best_thresh = 0, 0

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # Final prediction with best threshold
    y_pred = (y_scores > best_thresh).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_true, y_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    print(f"[DEBUG] {attack_name} - Best threshold: {best_thresh:.4f}, Accuracy: {best_acc:.4f}")

    return {
        "attack_name": attack_name,
        "best_thresh": best_thresh,
        "accuracy": best_acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc": (fpr, tpr),
        "scores": y_scores
    }


def validate_data_split(X_member, X_nonmember, y_member=None, y_nonmember=None):
    """Validate that member and non-member data are properly split"""
    print("[DEBUG] Validating data split...")

    # Check shapes
    print(f"[DEBUG] Member data shape: {X_member.shape}")
    print(f"[DEBUG] Non-member data shape: {X_nonmember.shape}")

    if y_member is not None:
        print(f"[DEBUG] Member labels shape: {y_member.shape}")
    if y_nonmember is not None:
        print(f"[DEBUG] Non-member labels shape: {y_nonmember.shape}")

    # Check for exact duplicates (this shouldn't happen in proper split)
    if X_member.shape[1] == X_nonmember.shape[1]:
        # Convert to strings for comparison (handles floating point issues)
        member_strings = set(str(row) for row in X_member)
        nonmember_strings = set(str(row) for row in X_nonmember)

        overlap = member_strings.intersection(nonmember_strings)
        if overlap:
            print(f"[WARNING] Found {len(overlap)} exact duplicates between member and non-member data!")
            print("[WARNING] This will make MIA evaluation invalid!")
            return False
        else:
            print("[DEBUG] No exact duplicates found - good!")

    return True


def run_comprehensive_attack(model_path, X_member, X_nonmember, y_member=None, y_nonmember=None, framework="keras"):
    """Run comprehensive MIA attack on a single model"""
    print(f"\n[DEBUG] Running comprehensive MIA on: {model_path}")
    
    # Load model
    try:
        if framework == "keras":
            model = load_model(model_path)
        elif framework == "sklearn":
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unknown framework: {framework}")
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return None

    # Run comprehensive evaluation
    results = evaluate_mia_comprehensive(model, X_member, X_nonmember, y_member, y_nonmember, framework)
    
    if not results:
        print(f"[ERROR] No results obtained for {model_path}")
        return None

    # Print results
    name = os.path.basename(model_path)
    print(f"\n[RESULTS] MIA Results for {name}:")
    print("-" * 50)
    
    for attack_type, result in results.items():
        if result is not None:
            print(f"{attack_type:10} | "
                  f"Acc: {result['accuracy']:.4f} | "
                  f"AUC: {result['auc']:.4f} | "
                  f"F1: {result['f1']:.4f} | "
                  f"Prec: {result['precision']:.4f} | "
                  f"Rec: {result['recall']:.4f}")
        else:
            print(f"{attack_type:10} | FAILED")

    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    for attack_type, result in results.items():
        if result is not None:
            fpr, tpr = result["roc"]
            plt.plot(fpr, tpr, label=f"{result['attack_name']} (AUC={result['auc']:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"MIA ROC Curves - {name}")
    plt.legend()
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    # Determine output directory based on model type
    if "naive" in model_path.lower():
        output_dir = "output/results/naive"
    elif "federated" in model_path.lower() and "dp" in model_path.lower():
        output_dir = "output/results/federated_dp"
    elif "federated" in model_path.lower():
        output_dir = "output/results/federated"
    else:
        output_dir = "output/results/naive"  # Default
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mia_roc_{name.replace('.', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return results


# Example usage with better data validation
if __name__ == "__main__":
    print("[DEBUG] Loading attack data...")

    # Load the data
    X_member = np.load("attack_data/X_member.npy")
    X_nonmember = np.load("attack_data/X_nonmember_clean.npy")  # Use cleaned data

    # Try to load labels if available
    try:
        y_member = np.load("attack_data/y_member.npy")
        y_nonmember = np.load("attack_data/y_nonmember.npy")
        print("[DEBUG] Labels loaded successfully")
    except:
        print("[DEBUG] No labels found, will skip loss-based attack")
        y_member = None
        y_nonmember = None

    # Validate data first
    if not validate_data_split(X_member, X_nonmember, y_member, y_nonmember):
        print("[ERROR] Data validation failed! Please check your data split.")
        exit(1)

    # Run comprehensive attacks on our models
    models_to_test = [
        ("models/RF/Naive/rf_naive_model.pkl", "sklearn", "naive"),
        ("models/RF/FL/federated_model.pkl", "sklearn", "federated"),
        ("models/RF/FL+DP/federated_model_dp.pkl", "sklearn", "federated_dp"),
    ]

    all_results = {}

    for model_path, framework, model_type in models_to_test:
        if os.path.exists(model_path):
            results = run_comprehensive_attack(
                model_path, X_member, X_nonmember, y_member, y_nonmember, framework
            )
            all_results[model_path] = results
            
            # Save JSON results
            if results:
                output_dir = f"output/results/{model_type}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Find the best attack result
                best_result = None
                best_auc = 0
                for attack_type, result in results.items():
                    if result is not None and result['auc'] > best_auc:
                        best_auc = result['auc']
                        best_result = result
                
                if best_result:
                    json_results = {
                        "model": model_type,
                        "attack_type": "membership_inference",
                        "best_attack": best_result['attack_name'],
                        "results": {
                            "accuracy": best_result['accuracy'],
                            "auc": best_result['auc'],
                            "precision": best_result['precision'],
                            "recall": best_result['recall'],
                            "f1_score": best_result['f1']
                        },
                        "all_attacks": {
                            attack_type: {
                                "accuracy": result['accuracy'],
                                "auc": result['auc'],
                                "precision": result['precision'],
                                "recall": result['recall'],
                                "f1_score": result['f1']
                            } for attack_type, result in results.items() if result is not None
                        },
                        "metadata": {
                            "samples_used": len(X_member) + len(X_nonmember),
                            "timestamp": str(np.datetime64('now'))
                        }
                    }
                    
                    json_path = os.path.join(output_dir, f"mia_results_{model_type}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_results, f, indent=2)
                    print(f"[SAVED] MIA results saved to: {json_path}")
        else:
            print(f"[DEBUG] Model not found: {model_path}")

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Model':<30} | {'Best Attack':<12} | {'Accuracy':<8} | {'AUC':<8}")
    print("-" * 80)

    for model_path, results in all_results.items():
        if results:
            name = os.path.basename(model_path)
            best_attack = None
            best_acc = 0

            for attack_type, result in results.items():
                if result is not None and result['accuracy'] > best_acc:
                    best_acc = result['accuracy']
                    best_attack = result['attack_name']

            if best_attack:
                best_result = results[best_attack.lower()]
                print(f"{name:<30} | {best_attack:<12} | {best_result['accuracy']:<8.4f} | {best_result['auc']:<8.4f}")
            else:
                print(f"{name:<30} | {'FAILED':<12} | {'N/A':<8} | {'N/A':<8}")


