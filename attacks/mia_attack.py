import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support, roc_curve
)
from keras.models import load_model
import joblib  # for sklearn models


def get_confidence_scores(model, X, framework="keras"):
    print(f"[DEBUG] Getting confidence scores using framework: {framework}")
    if framework == "keras":
        print(f"[DEBUG] Predicting probabilities for {X.shape[0]} samples...")
        probs = model.predict(X)
        print(f"[DEBUG] Predicted shape: {probs.shape}")
        return np.max(probs, axis=1)
    elif framework == "sklearn":
        if hasattr(model, "predict_proba"):
            print(f"[DEBUG] Predicting probabilities for {X.shape[0]} samples (sklearn)...")
            if isinstance(model.classes_, list):
                model.classes_ = np.array(model.classes_)
            probs = model.predict_proba(X)
            print(f"[DEBUG] Predicted shape: {probs.shape}")
            return np.max(probs, axis=1)
        else:
            raise ValueError("Sklearn model does not support predict_proba")
    else:
        raise ValueError("Unknown framework")


def evaluate_mia(model, X_member, X_nonmember, framework="keras"):
    print("[DEBUG] Evaluating MIA...")

    member_scores = get_confidence_scores(model, X_member, framework)
    nonmember_scores = get_confidence_scores(model, X_nonmember, framework)

    y_true = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])
    y_scores = np.concatenate([member_scores, nonmember_scores])

    print(f"[DEBUG] Member scores: mean={member_scores.mean():.4f}, std={member_scores.std():.4f}")
    print(f"[DEBUG] Non-member scores: mean={nonmember_scores.mean():.4f}, std={nonmember_scores.std():.4f}")

    # Sweep thresholds
    print("[DEBUG] Sweeping thresholds to find best attack accuracy...")
    thresholds = np.linspace(0, 1, 101)
    best_acc, best_thresh = 0, 0
    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    print(f"[DEBUG] Best threshold found: {best_thresh:.2f} with accuracy: {best_acc:.4f}")

    # Final prediction
    y_pred = (y_scores > best_thresh).astype(int)

    # Metrics
    auc = roc_auc_score(y_true, y_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return {
        "best_thresh": best_thresh,
        "accuracy": best_acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc": (fpr, tpr)
    }


def run_and_report(model_path, X_member, X_nonmember, framework="keras"):
    name = os.path.basename(model_path)
    print(f"\n[DEBUG] Loading model: {model_path}")
    if framework == "keras":
        model = load_model(model_path, compile=False)
    elif framework == "sklearn":
        model = joblib.load(model_path)
    else:
        raise ValueError("Unknown framework")

    print(f"[DEBUG] Running MIA on model: {name}")
    result = evaluate_mia(model, X_member, X_nonmember, framework)

    print(f"\n[MIA] Results for {name}")
    print(f"  • Best threshold : {result['best_thresh']:.2f}")
    print(f"  • Accuracy        : {result['accuracy']:.4f}")
    print(f"  • AUC             : {result['auc']:.4f}")
    print(f"  • Precision       : {result['precision']:.4f}")
    print(f"  • Recall          : {result['recall']:.4f}")
    print(f"  • F1-Score        : {result['f1']:.4f}")

    # Optional: save ROC plot
    fpr, tpr = result["roc"]
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC={result['auc']:.2f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.grid()
    os.makedirs("attack_results", exist_ok=True)
    plt.savefig(f"attack_results/roc_{name}.png")
    plt.close()
    print(f"[DEBUG] Saved ROC curve to attack_results/roc_{name}.png")


if __name__ == "__main__":
    print("[DEBUG] Loading member and non-member data...")
    X_member = np.load("attack_data/X_member.npy")
    X_nonmember = np.load("attack_data/X_nonmember.npy")
    print(f"[DEBUG] Loaded X_member shape: {X_member.shape}")
    print(f"[DEBUG] Loaded X_nonmember shape: {X_nonmember.shape}")

    # Attack each FL client
    for i in range(1, 6):
        model_path = f"../dp1/results/fl_dp_model_client_{i}.h5"
        run_and_report(model_path, X_member, X_nonmember, framework="keras")

    # Attack the aggregated keras model (if applicable)
    aggregated_model_path = "../dp1/results/fl_dp_global_model_aggregated.h5"
    if os.path.exists(aggregated_model_path):
        run_and_report(aggregated_model_path, X_member, X_nonmember, framework="keras")
    else:
        print(f"[DEBUG] Aggregated Keras model not found at: {aggregated_model_path}")

    # Attack the naive full-data model
    naive_model_path = "../dp1/results/naive_model.pkl"
    if os.path.exists(naive_model_path):
        run_and_report(naive_model_path, X_member, X_nonmember, framework="sklearn")
    else:
        print(f"[DEBUG] Naive model not found at: {naive_model_path}")