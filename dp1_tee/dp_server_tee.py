"""
Enhanced DP Server with TEE Support
"""
from __future__ import annotations
import flwr as flower
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
import sys
import logging
import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Scalar, parameters_to_ndarrays
from tee_config import TEEConfig, SGX_TEE_CONFIG
from sgx_utils import SGXEnclave, secure_aggregate_weights

NUM_CLIENTS = 5


class DPFedAvgTEE(FedAvg):
    """Custom FedAvg strategy with TEE support that tracks privacy metrics"""
    
    def __init__(self, tee_config: TEEConfig = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_metrics = {}
        self.tee_metrics = {}
        self.round_num = 0
        self.tee_config = tee_config or TEEConfig()
        
        # Initialize SGX enclave for server-side operations
        if self.tee_config.use_tee:
            self.sgx_enclave = SGXEnclave(self.tee_config)
            if self.sgx_enclave.initialize():
                print(f"[SERVER] SGX enclave initialized for secure aggregation")
            else:
                print(f"[SERVER] SGX initialization failed, falling back to standard aggregation")
                self.tee_config.use_tee = False
        else:
            self.sgx_enclave = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Scalar]]:
        """Aggregate fit results with TEE-enhanced security"""
        
        print(f"[SERVER] aggregate_fit called for round {server_round} with {len(results)} results")
        
        if not results:
            return None, {}
        
        # Extract weights and perform secure aggregation
        weights_list = []
        num_examples_list = []
        
        for _, fit_res in results:
            # Extract tensors from Flower Parameters object using proper deserialization
            try:
                # Use Flower's official utility to convert Parameters to numpy arrays
                weights = parameters_to_ndarrays(fit_res.parameters)
                weights_list.append(weights)
            except Exception as e:
                print(f"[SERVER] Error extracting parameters: {e}")
                print(f"[SERVER] Parameters type: {type(fit_res.parameters)}")
                raise
            num_examples_list.append(fit_res.num_examples)
        
        # Perform secure aggregation if TEE is enabled
        if self.tee_config.use_tee and self.tee_config.enable_secure_aggregation:
            print(f"[SERVER] Round {server_round} - Performing secure aggregation within SGX")
            
            # Aggregate each weight array separately
            aggregated_weights = []
            for i in range(len(weights_list[0])):  # For each weight matrix/vector
                weight_arrays = [weights[i] for weights in weights_list]
                aggregated_weight = secure_aggregate_weights(weight_arrays, self.tee_config)
                aggregated_weights.append(aggregated_weight)
        else:
            # Standard weighted aggregation
            print(f"[SERVER] Round {server_round} - Performing standard aggregation")
            total_examples = sum(num_examples_list)
            
            aggregated_weights = []
            for i in range(len(weights_list[0])):
                weighted_weights = [
                    weights[i] * num_examples / total_examples
                    for weights, num_examples in zip(weights_list, num_examples_list)
                ]
                aggregated_weights.append(np.sum(weighted_weights, axis=0))
        
        metrics = {
            "aggregation_method": "secure_tee" if self.tee_config.use_tee else "standard"
        }
        
        return aggregated_weights, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and track privacy/TEE metrics"""
        
        print(f"[SERVER] aggregate_evaluate called for round {server_round} with {len(results)} results")
        
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
        
        # Track TEE metrics
        tee_enabled_clients = 0
        enclave_measurements = []
        
        for _, evaluate_res in results:
            client_metrics = evaluate_res.metrics
            
            # Privacy metrics
            if "epsilon" in client_metrics:
                total_epsilon += client_metrics["epsilon"]
            if "delta" in client_metrics:
                total_delta = max(total_delta, client_metrics["delta"])
            if "accuracy" in client_metrics:
                accuracy_sum += client_metrics["accuracy"]
            
            # TEE metrics
            if "tee_enabled" in client_metrics and client_metrics["tee_enabled"]:
                tee_enabled_clients += 1
            if "enclave_measurement" in client_metrics:
                enclave_measurements.append(client_metrics["enclave_measurement"])
        
        # Add privacy metrics to aggregated results
        if total_epsilon > 0:
            metrics["total_epsilon"] = total_epsilon
            metrics["max_delta"] = total_delta
            metrics["avg_accuracy"] = accuracy_sum / num_clients if num_clients > 0 else 0.0
            
            privacy_info = f"Privacy Budget: ε={total_epsilon:.4f}, δ={total_delta:.2e}, Avg Accuracy: {metrics['avg_accuracy']:.4f}"
        else:
            privacy_info = f"Avg Accuracy: {accuracy_sum / num_clients:.4f}" if num_clients > 0 else "No metrics"
        
        # Add TEE metrics
        metrics["tee_enabled_clients"] = tee_enabled_clients
        metrics["total_clients"] = num_clients
        
        tee_info = ""
        if tee_enabled_clients > 0:
            metrics["tee_coverage"] = tee_enabled_clients / num_clients
            tee_info = f", TEE Coverage: {tee_enabled_clients}/{num_clients} clients"
            
            # Verify enclave measurements for consistency
            if len(set(enclave_measurements)) == 1:
                metrics["enclave_consistency"] = True
                tee_info += " (consistent enclaves)"
            else:
                metrics["enclave_consistency"] = False
                tee_info += " (WARNING: inconsistent enclaves)"
        
        print(f"[SERVER] Round {server_round} - {privacy_info}{tee_info}")
        
        # Store metrics for final report
        self.privacy_metrics[server_round] = {
            "epsilon": total_epsilon,
            "delta": total_delta,
            "accuracy": metrics.get("avg_accuracy", 0.0)
        }
        
        self.tee_metrics[server_round] = {
            "tee_enabled_clients": tee_enabled_clients,
            "total_clients": num_clients,
            "tee_coverage": tee_enabled_clients / num_clients if num_clients > 0 else 0.0,
            "enclave_consistency": metrics.get("enclave_consistency", False)
        }
        
        return loss, metrics
    
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: List[np.ndarray], 
        client_manager
    ):
        """Configure the next round of evaluation"""
        print(f"[SERVER] configure_evaluate called for round {server_round}")
        
        # Force evaluation on all available clients
        config = {}
        return [(client, config) for client in client_manager.all().values()]
    
    def get_privacy_summary(self):
        """Get summary of privacy expenditure across all rounds"""
        if not self.privacy_metrics:
            return "No privacy metrics collected"
        
        final_round = max(self.privacy_metrics.keys())
        final_metrics = self.privacy_metrics[final_round]
        
        summary = f"""
Privacy Budget Summary:
- Total ε (epsilon): {final_metrics['epsilon']:.4f}
- Total δ (delta): {final_metrics['delta']:.2e}
- Final Accuracy: {final_metrics['accuracy']:.4f}
- Total Rounds: {final_round}
"""
        
        # Add TEE summary if applicable
        if self.tee_metrics:
            final_tee = self.tee_metrics[final_round]
            summary += f"""
TEE Summary:
- TEE Coverage: {final_tee['tee_coverage']:.1%} ({final_tee['tee_enabled_clients']}/{final_tee['total_clients']} clients)
- Enclave Consistency: {'✓' if final_tee['enclave_consistency'] else '✗'}
- Secure Aggregation: {'✓' if self.tee_config.enable_secure_aggregation else '✗'}
"""
        
        return summary

    def cleanup(self):
        """Cleanup TEE resources"""
        if self.sgx_enclave:
            self.sgx_enclave.cleanup()


def main():
    # Parse command line arguments for DP and TEE configuration
    use_dp = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else True
    use_tee = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False
    
    # Configure TEE
    if use_tee:
        tee_config = SGX_TEE_CONFIG
        tee_config.use_tee = True
        print(f"[SERVER] TEE enabled with {tee_config.enclave_type}")
    else:
        tee_config = TEEConfig(use_tee=False)
        print(f"[SERVER] TEE disabled")
    
    # Define strategy with TEE support
    strategy = DPFedAvgTEE(
        tee_config=tee_config,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    server_description = f"{'TEE-enabled ' if use_tee else ''}{'DP' if use_dp else 'Standard'}"
    print(f"[SERVER] Starting Flower server with {server_description} configuration...")
    print(f"[SERVER] Waiting for {NUM_CLIENTS} clients to connect...")

    try:
        logging.basicConfig(level=logging.DEBUG if tee_config.debug_mode else logging.INFO)
        
        # Start server with explicit evaluation configuration
        config = flower.server.ServerConfig(
            num_rounds=5,
            round_timeout=60.0,  # 60 seconds timeout per round
        )
        
        print(f"[SERVER] Server config: {config}")
        print(f"[SERVER] Strategy evaluation settings: fraction_evaluate={strategy.fraction_evaluate}, min_evaluate_clients={strategy.min_evaluate_clients}")
        
        flower.server.start_server(
            server_address="127.0.0.1:8086",
            config=config,
            strategy=strategy,
        )
        
        print("[SERVER] Training completed successfully!")
        
        # Print privacy and TEE summary
        print(strategy.get_privacy_summary())
            
    except Exception as e:
        print(f"[SERVER] Error: {e}")
    finally:
        print("[SERVER] Performing cleanup...")
        strategy.cleanup()
        print("[SERVER] Server finished.")


if __name__ == "__main__":
    main() 