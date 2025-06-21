from __future__ import annotations    # (optional, defers all annotation evaluation)
import flwr as flower
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
import sys
import logging
import flwr as flower
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, Scalar
from typing import List, Tuple, Dict, Optional
import sys

NUM_CLIENTS = 5

class DPFedAvg(FedAvg):
    """Custom FedAvg strategy that tracks privacy metrics"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_metrics = {}
        self.round_num = 0

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        """Aggregate evaluation results and track privacy metrics"""
        
        # Standard aggregation
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_result is None:
            return None
        
        loss, metrics = aggregated_result
        
        # Track privacy metrics
        total_epsilon = 0.0
        total_delta = 0.0
        accuracy_sum = 0.0
        num_clients = len(results)
        
        for _, evaluate_res in results:
            client_metrics = evaluate_res.metrics
            if "epsilon" in client_metrics:
                total_epsilon += client_metrics["epsilon"]
            if "delta" in client_metrics:
                total_delta = max(total_delta, client_metrics["delta"])  # Use max for delta
            if "accuracy" in client_metrics:
                accuracy_sum += client_metrics["accuracy"]
        
        # Add privacy metrics to aggregated results
        if total_epsilon > 0:
            metrics["total_epsilon"] = total_epsilon
            metrics["max_delta"] = total_delta
            metrics["avg_accuracy"] = accuracy_sum / num_clients if num_clients > 0 else 0.0
            
            print(f"[SERVER] Round {server_round} - Privacy Budget: ε={total_epsilon:.4f}, δ={total_delta:.2e}, Avg Accuracy: {metrics['avg_accuracy']:.4f}")
            
            # Store for final report
            self.privacy_metrics[server_round] = {
                "epsilon": total_epsilon,
                "delta": total_delta,
                "accuracy": metrics["avg_accuracy"]
            }
        
        return loss, metrics
    
    def get_privacy_summary(self):
        """Get summary of privacy expenditure across all rounds"""
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
    # Parse command line arguments for DP configuration
    use_dp = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else True
    
    # Define strategy
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
        # Start server
        flower.server.start_server(
            server_address="127.0.0.1:8086",
            config=flower.server.ServerConfig(num_rounds=5),
            strategy=strategy,
        )
        
        print("[SERVER] Training completed successfully!")
        
        # Print privacy summary if DP was used
        if use_dp:
            print(strategy.get_privacy_summary())
            
    except Exception as e:
        print(f"[SERVER] Error: {e}")
    finally:
        print("[SERVER] Server finished.")


if __name__ == "__main__":
    main()