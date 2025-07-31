# mia_attack_federated.py
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt

# Import the shared MIA utility
sys.path.append('../../../')
from utils.mia_utils import run_comprehensive_attack

def main():
    print("=" * 60)
    print("MIA ATTACK ON FEDERATED MODEL")
    print("=" * 60)

    # Load federated model data
    try:
        X_train = np.load("federated_X_train.npy")
        y_train = np.load("federated_y_train.npy")
        X_test = np.load("federated_X_test.npy")
        y_test = np.load("federated_y_test.npy")

        print(f"[INFO] Loaded federated data:")
        print(f"[INFO] Training set: {X_train.shape}")
        print(f"[INFO] Test set: {X_test.shape}")

        # Run MIA attack
        results = run_comprehensive_attack(
            model_path="federated_model.pkl",
            X_member=X_train,
            X_nonmember=X_test,
            y_member=y_train,
            y_nonmember=y_test,
            framework="sklearn"
        )

        if results:
            print(f"\n[SUMMARY] Federated Model MIA Results:")
            print("-" * 50)
            for attack_type, result in results.items():
                if result is not None:
                    print(f"{attack_type:10} | AUC: {result['auc']:.4f} | Acc: {result['accuracy']:.4f}")

        # Save results
        if results:
            summary = {}
            for attack_type, result in results.items():
                if result is not None:
                    summary[attack_type] = {
                        'auc': result['auc'],
                        'accuracy': result['accuracy'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'f1': result['f1']
                    }

            os.makedirs("attack_results", exist_ok=True)
            with open('attack_results/federated_mia_results.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            print(f"\n[INFO] Results saved to attack_results/federated_mia_results.json")

    except FileNotFoundError as e:
        print(f"[ERROR] Required files not found: {e}")
        print("[ERROR] Please run federated_learning.py first to generate the data files")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()