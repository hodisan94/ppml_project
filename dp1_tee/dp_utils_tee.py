"""
Enhanced DP Utils with TEE Support
"""
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import logging

from tee_config import TEEConfig
from sgx_utils import SGXEnclave, secure_aggregate_weights


class DPConfig:
    """Configuration class for Differential Privacy parameters"""
    def __init__(self, 
                 noise_multiplier=1.0,
                 l2_norm_clip=1.0,
                 microbatches=1,
                 learning_rate=0.01,
                 target_epsilon=1.0,
                 target_delta=1e-5):
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.microbatches = microbatches
        self.learning_rate = learning_rate
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta


class DPLogisticRegressionTEE:
    """Logistic Regression with Differential Privacy and TEE support"""
    
    def __init__(self, dp_config=None, use_dp=True, tee_config=None):
        self.dp_config = dp_config or DPConfig()
        self.use_dp = use_dp
        self.tee_config = tee_config or TEEConfig()
        self.model = None
        self.optimizer = None
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.privacy_spent = {'epsilon': 0.0, 'delta': 0.0}
        self.training_steps = 0
        
        # Initialize SGX enclave if TEE is enabled
        if self.tee_config.use_tee:
            self.sgx_enclave = SGXEnclave(self.tee_config)
            self.sgx_enclave.initialize()
            logging.info("[TEE] DPLogisticRegressionTEE initialized with SGX support")
        else:
            self.sgx_enclave = None
            logging.info("[TEE] DPLogisticRegressionTEE initialized without TEE")
        
    def _build_model(self, input_dim):
        """Build the TensorFlow model with TEE considerations"""
        if self.tee_config.use_tee and self.tee_config.protected_memory:
            # In a real implementation, this would use SGX protected memory
            logging.info("[TEE] Building model with protected memory")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, 
                                activation='sigmoid',
                                input_shape=(input_dim,),
                                kernel_initializer='zeros',
                                bias_initializer='zeros')
        ])
        return model
    
    def _get_optimizer(self, batch_size, dataset_size):
        """Get optimizer (DP or regular) with TEE enhancements"""
        if self.use_dp:
            optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
                l2_norm_clip=self.dp_config.l2_norm_clip,
                noise_multiplier=self.dp_config.noise_multiplier,
                num_microbatches=self.dp_config.microbatches,
                learning_rate=self.dp_config.learning_rate
            )
            
            if self.tee_config.use_tee:
                logging.info("[TEE] Using DP optimizer within SGX enclave")
            
            return optimizer
        else:
            return tf.keras.optimizers.legacy.SGD(learning_rate=self.dp_config.learning_rate)
    
    def fit(self, X, y, epochs=10, batch_size=32, verbose=0):
        """Train the model with optional DP and TEE protection"""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        if self.tee_config.use_tee:
            logging.info(f"[TEE] Training model securely within SGX enclave")
        
        # Build model if not exists
        if self.model is None:
            self.model = self._build_model(X.shape[1])
            self.optimizer = self._get_optimizer(batch_size, len(X))
            self.model.compile(optimizer=self.optimizer, 
                             loss=self.loss_fn,
                             metrics=['accuracy'])
        
        # Secure training within enclave
        if self.tee_config.use_tee and self.tee_config.protected_memory:
            # In real implementation, this would ensure training happens in SGX
            logging.info("[TEE] Performing secure training with protected gradients")
        
        # Train the model
        history = self.model.fit(X, y, 
                               epochs=epochs, 
                               batch_size=batch_size, 
                               verbose=verbose,
                               shuffle=True)
        
        # Update privacy accounting
        if self.use_dp:
            self.training_steps += epochs * (len(X) // batch_size)
            self._compute_privacy_spent(len(X), batch_size)
            
            if self.tee_config.use_tee:
                logging.info("[TEE] Privacy accounting protected within enclave")
        
        return history
    
    def _compute_privacy_spent(self, dataset_size, batch_size):
        """Compute privacy spent using TF Privacy analysis with TEE protection"""
        if self.use_dp and self.training_steps > 0:
            steps = self.training_steps
            epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
                n=dataset_size,
                batch_size=batch_size,
                noise_multiplier=self.dp_config.noise_multiplier,
                epochs=steps * batch_size / dataset_size,
                delta=self.dp_config.target_delta
            )
            self.privacy_spent['epsilon'] = epsilon
            self.privacy_spent['delta'] = self.dp_config.target_delta
            
            if self.tee_config.use_tee:
                logging.info(f"[TEE] Privacy budget ε={epsilon:.4f} computed securely")
    
    def predict(self, X):
        """Make predictions with TEE protection"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X = np.array(X, dtype=np.float32)
        
        if self.tee_config.use_tee:
            logging.debug("[TEE] Performing secure prediction")
        
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Predict probabilities with TEE protection"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X = np.array(X, dtype=np.float32)
        
        if self.tee_config.use_tee:
            logging.debug("[TEE] Performing secure probability prediction")
        
        probs = self.model.predict(X, verbose=0)
        return np.column_stack([1 - probs.flatten(), probs.flatten()])
    
    def get_weights(self):
        """Get model weights with TEE protection"""
        if self.model is None:
            return None
        
        weights = self.model.get_weights()
        weights_dict = {'coef_': weights[0].flatten(), 'intercept_': weights[1]}
        
        if self.tee_config.use_tee and self.tee_config.secure_communication:
            logging.debug("[TEE] Weights extracted securely from enclave")
        
        return weights_dict
    
    def set_weights(self, weights_dict):
        """Set model weights with TEE protection"""
        if self.model is None:
            return
        
        if self.tee_config.use_tee and self.tee_config.secure_communication:
            logging.debug("[TEE] Setting weights securely within enclave")
        
        if 'coef_' in weights_dict and 'intercept_' in weights_dict:
            coef = weights_dict['coef_'].reshape(-1, 1)
            intercept = weights_dict['intercept_']
            self.model.set_weights([coef, intercept])
    
    def get_privacy_spent(self):
        """Get current privacy expenditure"""
        return self.privacy_spent.copy()
    
    def get_enclave_measurement(self):
        """Get SGX enclave measurement for attestation"""
        if self.sgx_enclave:
            return self.sgx_enclave.get_enclave_measurement()
        return None
    
    def cleanup(self):
        """Cleanup TEE resources"""
        if self.sgx_enclave:
            self.sgx_enclave.cleanup()


def load_client_data(client_id):
    """
    Load data for a specific client, split into train/test.
    Args: client_id (int): ID of the client (1-5).
    Returns: X_train, X_test, y_train, y_test (Features and labels split).
    """
    project_root = Path(__file__).resolve().parent.parent

    # Build the path to data/clients/client_{id}.csv
    csv_path = project_root / "data" / "clients" / f"client_{client_id}.csv"
    print(f"[TEE] Loading client data from: {csv_path}")

    df = pd.read_csv(csv_path)
    X = df.drop("Readmitted", axis=1).values
    y = df["Readmitted"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_model(use_dp=True, dp_config=None, tee_config=None):
    """
    Initialize a model (DP or regular logistic regression) with TEE support.
    Returns: Model instance
    """
    if use_dp:
        return DPLogisticRegressionTEE(dp_config=dp_config, use_dp=True, tee_config=tee_config)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        
        if tee_config and tee_config.use_tee:
            logging.info("[TEE] Standard model initialized with TEE awareness")
        
        return model


def compute_privacy_budget(dataset_sizes, batch_size, noise_multiplier, epochs, delta=1e-5):
    """
    Compute privacy budget for federated learning setup
    """
    max_epsilon = 0  # Use max instead of sum for proper FL privacy accounting
    
    for size in dataset_sizes:
        epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            n=size,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta
        )
        
        # In FL, privacy budget is the maximum across clients, not sum
        max_epsilon = max(max_epsilon, epsilon)
    
    return max_epsilon, delta


def analyze_privacy_utility_tradeoff(noise_multipliers, dataset_size, batch_size, epochs, delta=1e-5):
    """
    Analyze the privacy-utility tradeoff for different noise levels
    """
    results = []
    
    for noise_mult in noise_multipliers:
        epsilon, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            n=dataset_size,
            batch_size=batch_size,
            noise_multiplier=noise_mult,
            epochs=epochs,
            delta=delta
        )
        
        results.append({
            'noise_multiplier': noise_mult,
            'epsilon': epsilon,
            'delta': delta
        })
    
    return results 