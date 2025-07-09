"""
Enhanced DP Client with TEE Support
"""
import numpy as np
import pandas as pd
import sys
import flwr as flower
from sklearn.metrics import log_loss, accuracy_score
from dp_utils_tee import load_client_data, get_model, DPConfig
from tee_config import TEEConfig, SGX_TEE_CONFIG
from sgx_utils import SGXEnclave
import tensorflow as tf
import logging

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python dp_client_tee.py <client_id> [use_dp] [noise_multiplier] [use_tee]")
    sys.exit(1)

client_id = int(sys.argv[1])
use_dp = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True
noise_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
use_tee = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False

# Configure TEE
if use_tee:
    tee_config = SGX_TEE_CONFIG
    tee_config.use_tee = True
    print(f"[CLIENT {client_id}] TEE enabled with {tee_config.enclave_type}")
else:
    tee_config = TEEConfig(use_tee=False)
    print(f"[CLIENT {client_id}] TEE disabled")

# Initialize SGX enclave if TEE is enabled
sgx_enclave = None
if tee_config.use_tee:
    sgx_enclave = SGXEnclave(tee_config)
    if not sgx_enclave.initialize():
        print(f"[CLIENT {client_id}] SGX initialization failed, falling back to non-TEE mode")
        tee_config.use_tee = False
    else:
        print(f"[CLIENT {client_id}] SGX enclave initialized successfully")

# Load client data
print(f"[CLIENT {client_id}] Loading data...")
X_train, X_test, y_train, y_test = load_client_data(client_id)

# Configure DP parameters
dp_config = DPConfig(
    noise_multiplier=noise_multiplier,
    l2_norm_clip=1.0,
    microbatches=1,
    learning_rate=0.01,
    target_epsilon=1.0,
    target_delta=1e-5
)

# Create model with TEE support
print(f"[CLIENT {client_id}] Creating model...")
model = get_model(use_dp=use_dp, dp_config=dp_config, tee_config=tee_config)

# For DP model, we need to do an initial fit to build the TensorFlow model
if use_dp:
    # Initialize with a small subset to build the model structure
    batch_size = min(32, len(X_train))
    model.fit(X_train[:2], y_train[:2], epochs=1, batch_size=2, verbose=0)
    
    privacy_info = ""
    if tee_config.use_tee:
        privacy_info = " with SGX protection"
    
    print(f"[CLIENT {client_id}] DP Model initialized (noise_multiplier={noise_multiplier}){privacy_info}")
else:
    # For sklearn model, do dummy fit
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError("Training data must contain at least two classes.")
    
    first_idx = np.where(y_train == unique_classes[0])[0][0]
    second_idx = np.where(y_train == unique_classes[1])[0][0]
    dummy_X = np.vstack([X_train[first_idx], X_train[second_idx]])
    dummy_y = np.array([y_train[first_idx], y_train[second_idx]])
    model.fit(dummy_X, dummy_y)
    
    tee_info = " with TEE awareness" if tee_config.use_tee else ""
    print(f"[CLIENT {client_id}] Standard Model initialized{tee_info}")


class DPPPMLClientTEE(flower.client.NumPyClient):
    """Enhanced Flower client with TEE support"""
    
    def __init__(self):
        self.round_num = 0
        self.enclave_measurement = None
        
        if tee_config.use_tee and hasattr(model, 'get_enclave_measurement'):
            self.enclave_measurement = model.get_enclave_measurement()
            print(f"[CLIENT {client_id}] Enclave measurement: {self.enclave_measurement}")
        
    def get_parameters(self, config=None):
        print(f"[CLIENT {client_id}] get_parameters called (round {self.round_num})", flush=True)
        
        if tee_config.use_tee:
            print(f"[CLIENT {client_id}] Extracting parameters from SGX enclave", flush=True)
        
        try:
            if use_dp:
                print(f"[CLIENT {client_id}] Getting DP model weights...", flush=True)
                weights = model.get_weights()
                if weights is None:
                    print(f"[CLIENT {client_id}] No weights found, returning zeros", flush=True)
                    return [np.zeros(X_train.shape[1]), np.array([0.0])]
                print(f"[CLIENT {client_id}] DP weights extracted successfully", flush=True)
                return [weights['coef_'], weights['intercept_']]
            else:
                print(f"[CLIENT {client_id}] Getting sklearn model weights...", flush=True)
                result = [model.coef_.flatten(), model.intercept_]
                print(f"[CLIENT {client_id}] Sklearn weights extracted successfully", flush=True)
                return result
        except Exception as e:
            print(f"[CLIENT {client_id}] ERROR in get_parameters: {e}", flush=True)
            raise

    def fit(self, parameters, config=None):
        self.round_num += 1
        
        if tee_config.use_tee:
            print(f"[CLIENT {client_id}] Round {self.round_num} - Training within SGX enclave", flush=True)
        
        # Set model parameters
        if use_dp:
            weights_dict = {'coef_': parameters[0], 'intercept_': parameters[1]}
            model.set_weights(weights_dict)
            
            # Train with DP (and TEE if enabled) - reduced epochs for performance
            batch_size = min(32, len(X_train))
            epochs = 1  # Reduced from 5 to 1 for faster training
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Get privacy spent
            privacy_spent = model.get_privacy_spent()
            
            tee_suffix = " (SGX protected)" if tee_config.use_tee else ""
            print(f"[CLIENT {client_id}] Round {self.round_num} - Privacy spent: ε={privacy_spent['epsilon']:.4f}, δ={privacy_spent['delta']:.2e}{tee_suffix}", flush=True)
            
        else:
            # Standard sklearn training
            model.coef_ = np.array(parameters[0]).reshape(1, -1)
            model.intercept_ = np.array(parameters[1])
            model.classes_ = np.array([0, 1])
            model.fit(X_train, y_train)
            
            if tee_config.use_tee:
                print(f"[CLIENT {client_id}] Round {self.round_num} - Training completed with TEE awareness", flush=True)
        
        print(f"[CLIENT {client_id}] Round {self.round_num} - Extracting parameters for return...", flush=True)
        
        try:
            parameters_to_return = self.get_parameters()
            print(f"[CLIENT {client_id}] Round {self.round_num} - Parameters extracted, returning to server...", flush=True)
            return parameters_to_return, len(X_train), {}
        except Exception as e:
            print(f"[CLIENT {client_id}] Round {self.round_num} - ERROR extracting parameters: {e}", flush=True)
            raise

    def evaluate(self, parameters, config=None):
        if tee_config.use_tee:
            logging.debug(f"[CLIENT {client_id}] Evaluating within SGX enclave")
        
        # Set model parameters
        if use_dp:
            weights_dict = {'coef_': parameters[0], 'intercept_': parameters[1]}
            model.set_weights(weights_dict)
            preds = model.predict_proba(X_test)
        else:
            model.coef_ = np.array(parameters[0]).reshape(1, -1)
            model.intercept_ = np.array(parameters[1])
            model.classes_ = np.array([0, 1])
            preds = model.predict_proba(X_test)
        
        # Calculate metrics
        loss = log_loss(y_test, preds)
        accuracy = accuracy_score(y_test, np.argmax(preds, axis=1))
        
        tee_suffix = " (TEE)" if tee_config.use_tee else ""
        print(f"[CLIENT {client_id}] Round {self.round_num} - Accuracy: {accuracy:.4f}{tee_suffix}", flush=True)
        
        metrics = {"accuracy": float(accuracy)}
        
        # Add privacy metrics for DP
        if use_dp:
            privacy_spent = model.get_privacy_spent()
            metrics.update({
                "epsilon": privacy_spent['epsilon'],
                "delta": privacy_spent['delta']
            })
        
        # Add TEE metrics
        if tee_config.use_tee:
            metrics.update({
                "tee_enabled": True,
                "enclave_type": tee_config.enclave_type,
                "secure_aggregation": tee_config.enable_secure_aggregation
            })
            
            if self.enclave_measurement:
                metrics["enclave_measurement"] = self.enclave_measurement
        
        return float(loss), len(X_test), metrics


# Start client
tee_description = f"{'TEE-enabled ' if tee_config.use_tee else ''}{'DP' if use_dp else 'Standard'}"
print(f"[CLIENT {client_id}] Starting {tee_description} client...")

try:
    flower.client.start_numpy_client(
        server_address="127.0.0.1:8086", 
        client=DPPPMLClientTEE()
    )
finally:
    # Cleanup TEE resources
    if sgx_enclave:
        sgx_enclave.cleanup()
    
    if hasattr(model, 'cleanup'):
        model.cleanup()
    
    print(f"[CLIENT {client_id}] Cleanup completed") 