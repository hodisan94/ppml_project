from __future__ import annotations
import flwr as flower
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Scalar, FitRes, Parameters
from typing import List, Tuple, Dict, Optional, Union
import sys
import logging
import os
import numpy as np
import tensorflow as tf
import pickle

NUM_CLIENTS = 5


class DPFedAvg(FedAvg):
    """Custom FedAvg strategy that tracks privacy metrics and saves global model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_metrics = {}
        self.round_num = 0
        self.latest_weights = None  # Store final aggregated weights
        self.use_dp = True  # Will be set from main

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_fit = super().aggregate_fit(server_round, results, failures)

        if aggregated_fit is not None:
            self.latest_weights = aggregated_fit[0]
            self.round_num = server_round
            print(f"[SERVER] Round {server_round} - Aggregated weights updated")

            # Save weights after each round (optional)
            self._save_round_weights(server_round)

        return aggregated_fit

    def _save_round_weights(self, round_num):
        """Save weights after each round"""
        if self.latest_weights is not None:
            try:
                os.makedirs("results", exist_ok=True)

                # Convert Parameters to numpy arrays
                if hasattr(self.latest_weights, 'tensors'):
                    weights_list = [np.frombuffer(tensor, dtype=np.float32) for tensor in self.latest_weights.tensors]
                else:
                    # If it's already a list of arrays
                    weights_list = self.latest_weights

                # Save as pickle for easier loading later
                with open(f"results/global_weights_round_{round_num}.pkl", 'wb') as f:
                    pickle.dump(weights_list, f)

                print(f"[SERVER] Round {round_num} weights saved")

            except Exception as e:
                print(f"[SERVER] Error saving round weights: {e}")

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)

        if aggregated_result is None:
            return None

        loss, metrics = aggregated_result
        total_epsilon = 0.0
        total_delta = 0.0
        accuracy_sum = 0.0
        num_clients = len(results)

        for _, evaluate_res in results:
            client_metrics = evaluate_res.metrics
            if "epsilon" in client_metrics:
                total_epsilon += client_metrics["epsilon"]
            if "delta" in client_metrics:
                total_delta = max(total_delta, client_metrics["delta"])
            if "accuracy" in client_metrics:
                accuracy_sum += client_metrics["accuracy"]

        if total_epsilon > 0:
            metrics["total_epsilon"] = total_epsilon
            metrics["max_delta"] = total_delta
            metrics["avg_accuracy"] = accuracy_sum / num_clients if num_clients > 0 else 0.0

            print(
                f"[SERVER] Round {server_round} - Privacy Budget: ε={total_epsilon:.4f}, δ={total_delta:.2e}, Avg Accuracy: {metrics['avg_accuracy']:.4f}")
            self.privacy_metrics[server_round] = {
                "epsilon": total_epsilon,
                "delta": total_delta,
                "accuracy": metrics["avg_accuracy"]
            }

        return loss, metrics

    def save_final_model(self, input_dim):
        """Save the final global model"""
        if self.latest_weights is None:
            print("[SERVER] No weights to save!")
            return False

        try:
            os.makedirs("results", exist_ok=True)

            # Convert Parameters to numpy arrays
            if hasattr(self.latest_weights, 'tensors'):
                weights_np = []
                for tensor in self.latest_weights.tensors:
                    # Convert bytes to numpy array
                    weight_array = np.frombuffer(tensor, dtype=np.float32)
                    weights_np.append(weight_array)
            else:
                # If it's already a list of arrays
                weights_np = [np.array(w, dtype=np.float32) for w in self.latest_weights]

            print(f"[SERVER] Extracted {len(weights_np)} weight arrays")
            for i, w in enumerate(weights_np):
                print(f"[SERVER] Weight {i} shape: {w.shape}")

            # Reshape weights to match logistic regression structure
            if len(weights_np) >= 2:
                # Assume first weight is coefficients, second is intercept
                coef = weights_np[0].reshape(-1, 1)  # Shape: (features, 1)
                intercept = weights_np[1]  # Shape: (1,) or scalar

                # Create TensorFlow model
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(1,
                                          activation='sigmoid',
                                          input_shape=(input_dim,),
                                          use_bias=True)
                ])

                # Build the model with dummy data
                dummy_input = np.zeros((1, input_dim), dtype=np.float32)
                _ = model(dummy_input)

                # Set weights
                model.set_weights([coef, intercept])

                # Save model
                model.save("results/global_model.h5")
                print("[SERVER] Global model saved to results/global_model.h5")

                # Also save weights as numpy arrays for easier access
                np.save("results/global_coef.npy", coef)
                np.save("results/global_intercept.npy", intercept)

                # Save as pickle for MIA attack
                model_weights = {
                    'coef_': coef.flatten(),
                    'intercept_': intercept,
                    'round_num': self.round_num,
                    'use_dp': self.use_dp
                }

                with open("results/global_model_weights.pkl", 'wb') as f:
                    pickle.dump(model_weights, f)

                print("[SERVER] Global model weights saved in multiple formats")
                print(f"[SERVER] Model coefficient shape: {coef.shape}")
                print(f"[SERVER] Model intercept shape: {intercept.shape}")

                return True

            else:
                print(f"[SERVER] Unexpected number of weight arrays: {len(weights_np)}")
                return False

        except Exception as e:
            print(f"[SERVER] Error saving final model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_privacy_summary(self):
        if not self.privacy_metrics:
            return "No privacy metrics collected"

        final_round = max(self.privacy_metrics.keys())
        final_metrics = self.privacy_metrics[final_round]

        return f"""
Privacy Budget Summary:
- Total ε (epsilon): {final_metrics['epsilon']:.4f}
- Total δ (delta): {final_metrics['delta']:.2e}
- Final Accuracy: {final_metrics['accuracy']:.4f}
- Total Rounds: {final_round}
"""


def main():
    use_dp = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else True

    strategy = DPFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    strategy.use_dp = use_dp  # Set DP flag for saving

    print(f"[SERVER] Starting Flower server with {'DP' if use_dp else 'Standard'} configuration...")
    print(f"[SERVER] Waiting for {NUM_CLIENTS} clients to connect...")

    try:
        logging.basicConfig(level=logging.INFO)
        flower.server.start_server(
            server_address="127.0.0.1:8086",
            config=flower.server.ServerConfig(num_rounds=5),
            strategy=strategy,
        )

        print("[SERVER] Training completed successfully!")

        if use_dp:
            print(strategy.get_privacy_summary())

        # Save global model - need to get input dimension from client data
        # You might need to adjust this based on your actual feature count
        try:
            # Try to infer input dimension from a sample client
            import pandas as pd
            from pathlib import Path

            project_root = Path(__file__).resolve().parent.parent
            sample_data_path = project_root / "data" / "clients" / "client_1.csv"

            if sample_data_path.exists():
                df = pd.read_csv(sample_data_path)
                input_dim = len(df.columns) - 1  # Subtract 1 for target column
                print(f"[SERVER] Inferred input dimension: {input_dim}")
            else:
                # Fallback - adjust this to match your actual feature count
                input_dim = 10  # Change this to your actual feature count
                print(f"[SERVER] Using default input dimension: {input_dim}")

            if strategy.save_final_model(input_dim):
                print("[SERVER] Final global model saved successfully!")
            else:
                print("[SERVER] Failed to save final global model")

        except Exception as e:
            print(f"[SERVER] Error determining input dimension: {e}")
            # Try with default dimension
            if strategy.save_final_model(10):  # Adjust default as needed
                print("[SERVER] Final global model saved with default dimension!")

    except Exception as e:
        print(f"[SERVER] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[SERVER] Server finished.")


if __name__ == "__main__":
    main()