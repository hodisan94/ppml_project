# mia_attack_federated+dp.py with DP-aware testing and ROC saving
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from models.RF.Naive.mia_attack_rf_naive import run_comprehensive_attack

def add_laplace_noise(X, epsilon=0.5, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=X.shape)
    print(f"[DP] Added Laplace noise to test set: shape={noise.shape}, scale={scale:.4f}")
    return X + noise

def main():
    print("=" * 60)
    print("MIA ATTACK ON FEDERATED MODEL")
    print("=" * 60)

    try:
        X_train = np.load("federated_X_train.npy")
        y_train = np.load("federated_y_train.npy")
        X_test = np.load("federated_X_test.npy")
        y_test = np.load("federated_y_test.npy")

        print(f"[INFO] Loaded federated data:")
        print(f"[INFO] Training set: {X_train.shape}")
        print(f"[INFO] Test set: {X_test.shape}")

        use_dp = True
        epsilon = 0.5
        if use_dp:
            print(f"[INFO] Applying DP noise to X_test for symmetry (ε={epsilon})")
            X_test = add_laplace_noise(X_test, epsilon=epsilon)

        results = run_comprehensive_attack(
            model_path="federated_model_dp.pkl",
            X_member=X_train,
            X_nonmember=X_test,
            y_member=y_train,
            y_nonmember=y_test,
            framework="sklearn"
        )

        if results:
            print(f"\n[SUMMARY] Federated Model+DP MIA Results:")
            print("-" * 50)
            for attack_type, result in results.items():
                if result is not None:
                    print(f"{attack_type:10} | AUC: {result['auc']:.4f} | Acc: {result['accuracy']:.4f}")

            summary = {
                k: {
                    'auc': v['auc'],
                    'accuracy': v['accuracy'],
                    'precision': v['precision'],
                    'recall': v['recall'],
                    'f1': v['f1']
                } for k, v in results.items() if v is not None
            }

            os.makedirs("attack_results", exist_ok=True)
            with open('federated_mia_results.json', 'w') as f:
                json.dump(summary, f, indent=2)

            # Save ROC plot
            roc_path = "attack_results/comprehensive_roc_federated_model_dp_pkl.png"
            if 'roc_data' in results:
                fpr = results['roc_data']['fpr']
                tpr = results['roc_data']['tpr']
                label = results['roc_data']['label']
                plt.figure()
                for l, f, t in zip(label, fpr, tpr):
                    plt.plot(f, t, label=l)
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('MIA ROC Curves - federated_model_dp.pkl')
                plt.legend()
                plt.grid(True)
                plt.savefig(roc_path)
                print(f"[INFO] ROC curve saved to {roc_path}")

    except FileNotFoundError as e:
        print(f"[ERROR] Required files not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()