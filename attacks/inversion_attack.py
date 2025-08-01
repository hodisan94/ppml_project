import os
import numpy as np
import json
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

    # Save results as JSON
    json_results = {
        "model": model_name,
        "attack_type": "model_inversion",
        "results": {
            "mse": mse,
            "plot_file": os.path.basename(plot_path)
        },
        "metadata": {
            "samples_used": len(X_samples),
            "timestamp": str(np.datetime64('now'))
        }
    }
    
    json_path = os.path.join(output_dir, f"inversion_results_{model_name.replace(' ', '_').lower()}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    print(f"[SAVED] Results saved to: {json_path}")

    return mse


def main():
    models = {
        "Naive RF": {
            "model": "models/RF/Naive/rf_naive_model.pkl",
            "X": "models/RF/Naive/X_member.npy",
            "y": "models/RF/Naive/y_member.npy",
            "output_dir": "output/results/naive"
        },
        "Federated": {
            "model": "models/RF/FL/federated_model.pkl",
            "X": "models/RF/FL/federated_X_train.npy",
            "y": "models/RF/FL/federated_y_train.npy",
            "output_dir": "output/results/federated"
        },
        "Federated + DP": {
            "model": "models/RF/FL+DP/federated_model_dp.pkl",
            "X": "models/RF/FL+DP/federated_X_train.npy",
            "y": "models/RF/FL+DP/federated_y_train.npy",
            "output_dir": "output/results/federated_dp"
        }
    }

    results = {}
    
    for model_name, paths in models.items():
        try:
            print(f"\n[INFO] Processing {model_name}...")
            
            # Check if files exist
            for path_type, path in paths.items():
                if path_type != "output_dir" and not os.path.exists(path):
                    print(f"[ERROR] {path_type} file not found: {path}")
                    continue
            
            # Create output directory for this model
            os.makedirs(paths["output_dir"], exist_ok=True)
            
            X = np.load(paths["X"])
            y = np.load(paths["y"])
            
            print(f"[INFO] Loaded data: X={X.shape}, y={y.shape}")
            
            mse = run_inversion_attack(paths["model"], X, y, model_name, paths["output_dir"])
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
