import numpy as np
import pandas as pd
import sys
import flwr as flower
from sklearn.metrics import log_loss, accuracy_score
from dp_utils import load_client_data, get_model, DPConfig
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python dp_client.py <client_id> [use_dp] [noise_multiplier]")
    sys.exit(1)

client_id = int(sys.argv[1])
use_dp = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True
noise_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

# Load client data
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

# Create model
model = get_model(use_dp=use_dp, dp_config=dp_config)

# For DP model, we need to do an initial fit to build the TensorFlow model
if use_dp:
    # Initialize with a small subset to build the model structure
    batch_size = min(32, len(X_train))
    model.fit(X_train[:2], y_train[:2], epochs=1, batch_size=2, verbose=0)
    print(f"[CLIENT {client_id}] DP Model initialized (noise_multiplier={noise_multiplier})")
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
    print(f"[CLIENT {client_id}] Standard Model initialized")


class DPPPMLClient(flower.client.NumPyClient):
    def __init__(self):
        self.round_num = 0
        
    def get_parameters(self, config=None):
        print(f"[CLIENT {client_id}] connected to server", flush=True)
        
        if use_dp:
            weights = model.get_weights()
            if weights is None:
                return [np.zeros(X_train.shape[1]), np.array([0.0])]
            return [weights['coef_'], weights['intercept_']]
        else:
            return [model.coef_.flatten(), model.intercept_]

    def fit(self, parameters, config=None):
        self.round_num += 1
        
        # Set model parameters
        if use_dp:
            weights_dict = {'coef_': parameters[0], 'intercept_': parameters[1]}
            model.set_weights(weights_dict)
            
            # Train with DP
            batch_size = min(32, len(X_train))
            epochs = 5
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Get privacy spent
            privacy_spent = model.get_privacy_spent()
            print(f"[CLIENT {client_id}] Round {self.round_num} - Privacy spent: ε={privacy_spent['epsilon']:.4f}, δ={privacy_spent['delta']:.2e}", flush=True)
            
        else:
            # Standard sklearn training
            model.coef_ = np.array(parameters[0]).reshape(1, -1)
            model.intercept_ = np.array(parameters[1])
            model.classes_ = np.array([0, 1])
            model.fit(X_train, y_train)
        
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config=None):
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
        
        print(f"[CLIENT {client_id}] Round {self.round_num} - Accuracy: {accuracy:.4f}", flush=True)
        
        metrics = {"accuracy": float(accuracy)}
        if use_dp:
            privacy_spent = model.get_privacy_spent()
            metrics.update({
                "epsilon": privacy_spent['epsilon'],
                "delta": privacy_spent['delta']
            })
        
        return float(loss), len(X_test), metrics


# Start client
print(f"[CLIENT {client_id}] Starting {'DP' if use_dp else 'Standard'} client...")
flower.client.start_numpy_client(
    server_address="127.0.0.1:8086", 
    client=DPPPMLClient()
)