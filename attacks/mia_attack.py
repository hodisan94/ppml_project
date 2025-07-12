import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model


def get_confidence_scores(model, X):
    preds = model.predict(X)
    return np.max(preds, axis=1)


def run_mia(member_data, non_member_data, model_path):
    model = load_model(model_path)

    member_scores = get_confidence_scores(model, member_data)
    non_member_scores = get_confidence_scores(model, non_member_data)

    threshold = 0.5  # Simplified threshold
    member_preds = (member_scores > threshold).astype(int)
    non_member_preds = (non_member_scores > threshold).astype(int)

    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
    y_pred = np.concatenate([member_preds, non_member_preds])

    acc = accuracy_score(y_true, y_pred)
    print(f"[MIA] Attack Accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Replace these with real test data splits
    member_data = np.load("evaluation/member_data.npy")
    non_member_data = np.load("evaluation/non_member_data.npy")
    run_mia(member_data, non_member_data, "evaluation/fl_model.h5")
