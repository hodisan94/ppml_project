
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

NUM_CLIENTS = 5

class DPFedAvg(FedAvg):
    """Custom FedAvg strategy that tracks privacy metrics and saves global model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_metrics = {}
        self.round_num = 0
        self.latest_weights = None  # Store final aggregated weights

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_fit = super().aggregate_fit(server_round, results, failures)
        if aggregated_fit is not None:
            self.latest_weights = aggregated_fit[0]
        return aggregated_fit

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

            print(f"[SERVER] Round {server_round} - Privacy Budget: ε={total_epsilon:.4f}, δ={total_delta:.2e}, Avg Accuracy: {metrics['avg_accuracy']:.4f}")
            self.privacy_metrics[server_round] = {
                "epsilon": total_epsilon,
                "delta": total_delta,
                "accuracy": metrics["avg_accuracy"]
            }

        return loss, metrics

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

    print(f"[SERVER] Starting Flower server with {'DP' if use_dp else 'Standard'} configuration...")
    print(f"[SERVER] Waiting for {NUM_CLIENTS} clients to connect...")

    try:
        logging.basicConfig(level=logging.DEBUG)
        flower.server.start_server(
            server_address="127.0.0.1:8086",
            config=flower.server.ServerConfig(num_rounds=5),
            strategy=strategy,
        )

        print("[SERVER] Training completed successfully!")

        if use_dp:
            print(strategy.get_privacy_summary())

        # Save global model if available
        if strategy.latest_weights is not None:
            weights_np = [np.array(w, dtype=np.float32) for w in strategy.latest_weights.tensors]

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(weights_np[0].shape[0],)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.set_weights(weights_np)

            os.makedirs("results", exist_ok=True)
            model.save("results/global_model.h5")
            print("[SERVER] Global model saved to results/global_model.h5")

    except Exception as e:
        print(f"[SERVER] Error: {e}")
    finally:
        print("[SERVER] Server finished.")


if __name__ == "__main__":
    main()
