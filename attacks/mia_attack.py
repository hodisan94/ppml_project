import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

def get_confidence_scores(model, X):
    probs = model.predict(X)
    return np.max(probs, axis=1)

def run_mia(X_member, X_nonmember, model_path):
    model = load_model(model_path)

    member_scores = get_confidence_scores(model, X_member)
    nonmember_scores = get_confidence_scores(model, X_nonmember)

    threshold = 0.5
    member_preds = member_scores > threshold
    nonmember_preds = nonmember_scores > threshold

    y_true = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])
    y_pred = np.concatenate([member_preds, nonmember_preds])

    acc = accuracy_score(y_true, y_pred)
    print(f"[MIA] Attack accuracy on {model_path}: {acc:.4f}")

if __name__ == "__main__":
    X_member = np.load("results/X_member.npy")
    X_nonmember = np.load("results/X_nonmember.npy")
    for model_path in [
        "results/naive_model.h5",
        "results/fl_model.h5",
        "results/fl_dp_model.h5"
    ]:
        run_mia(X_member, X_nonmember, model_path)
