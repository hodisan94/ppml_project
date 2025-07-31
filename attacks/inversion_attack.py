import os
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import load
import matplotlib.pyplot as plt

def run_inversion_attack(model_path, X_samples, y_true, model_name, output_dir):
    print(f"\n{'='*60}")
    print(f"[ATTACK] Running Inversion Attack on: {model_name}")
    print(f"{'='*60}")

    model = load(model_path)

    try:
        y_pred = model.predict(X_samples)
    except:
        y_pred = model.predict_proba(X_samples)
        y_pred = np.argmax(y_pred, axis=1)

    mse = mean_squared_error(y_true, y_pred)
    print(f"[RESULT] MSE for {model_name}: {mse:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"inversion_{model_name.replace(' ', '_').lower()}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(y_true[:100], label='True Labels', linestyle='--')
    plt.plot(y_pred[:100], label='Predicted Labels', linestyle='-')
    plt.title(f'Model Inversion – {model_name}\nMSE={mse:.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[SAVED] Inversion plot saved to: {plot_path}")
    plt.close()

    return mse


def main():
    output_dir = "results/inversion_attack"
    models = {
        "Naive RF": {
            "model": "models/RF/Naive/rf_naive_model.pkl",
            "X": "models/RF/Naive/X_member.npy",
            "y": "models/RF/Naive/y_member.npy"
        },
        "Federated": {
            "model": "models/RF/FL/federated_model.pkl",
            "X": "models/RF/FL/federated_X_train.npy",
            "y": "models/RF/FL/federated_y_train.npy"
        },
        "Federated + DP": {
            "model": "models/RF/FL+DP/federated_model_dp.pkl",
            "X": "models/RF/FL+DP/federated_X_train.npy",
            "y": "models/RF/FL+DP/federated_y_train.npy"
        }
    }

    results = {}
    
    for model_name, paths in models.items():
        try:
            print(f"\n[INFO] Processing {model_name}...")
            
            # Check if files exist
            for path_type, path in paths.items():
                if not os.path.exists(path):
                    print(f"[ERROR] {path_type} file not found: {path}")
                    continue
            
            X = np.load(paths["X"])
            y = np.load(paths["y"])
            
            print(f"[INFO] Loaded data: X={X.shape}, y={y.shape}")
            
            mse = run_inversion_attack(paths["model"], X, y, model_name, output_dir)
            results[model_name] = mse
            
        except Exception as e:
            print(f"[ERROR] Failed to process {model_name}: {e}")
            results[model_name] = None

    # Print summary
    print(f"\n{'='*60}")
    print("INVERSION ATTACK SUMMARY")
    print(f"{'='*60}")
    for model_name, mse in results.items():
        if mse is not None:
            print(f"{model_name}: MSE = {mse:.4f}")
        else:
            print(f"{model_name}: FAILED")


if __name__ == "__main__":
    main()
