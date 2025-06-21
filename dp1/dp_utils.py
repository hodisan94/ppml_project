import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path


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


class DPLogisticRegression:
    """Logistic Regression with Differential Privacy using TensorFlow"""
    
    def __init__(self, dp_config=None, use_dp=True):
        self.dp_config = dp_config or DPConfig()
        self.use_dp = use_dp
        self.model = None
        self.optimizer = None
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.privacy_spent = {'epsilon': 0.0, 'delta': 0.0}
        self.training_steps = 0
        
    def _build_model(self, input_dim):
        """Build the TensorFlow model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, 
                                activation='sigmoid',
                                input_shape=(input_dim,),
                                kernel_initializer='zeros',
                                bias_initializer='zeros')
        ])
        return model
    
    def _get_optimizer(self, batch_size, dataset_size):
        """Get optimizer (DP or regular)"""
        if self.use_dp:
            return dp_optimizer_keras.DPKerasSGDOptimizer(
                l2_norm_clip=self.dp_config.l2_norm_clip,
                noise_multiplier=self.dp_config.noise_multiplier,
                num_microbatches=self.dp_config.microbatches,
                learning_rate=self.dp_config.learning_rate
            )
        else:
            return tf.keras.optimizers.legacy.SGD(learning_rate=self.dp_config.learning_rate)
    
    def fit(self, X, y, epochs=10, batch_size=32, verbose=0):
        """Train the model with optional DP"""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Build model if not exists
        if self.model is None:
            self.model = self._build_model(X.shape[1])
            self.optimizer = self._get_optimizer(batch_size, len(X))
            self.model.compile(optimizer=self.optimizer, 
                             loss=self.loss_fn,
                             metrics=['accuracy'])
        
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
        
        return history
    
    def _compute_privacy_spent(self, dataset_size, batch_size):
        """Compute privacy spent using TF Privacy analysis"""
        if self.use_dp and self.training_steps > 0:
            steps = self.training_steps
            epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                n=dataset_size,
                batch_size=batch_size,
                noise_multiplier=self.dp_config.noise_multiplier,
                epochs=steps * batch_size / dataset_size,
                delta=self.dp_config.target_delta
            )
            self.privacy_spent['epsilon'] = epsilon
            self.privacy_spent['delta'] = self.dp_config.target_delta
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X = np.array(X, dtype=np.float32)
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X = np.array(X, dtype=np.float32)
        probs = self.model.predict(X, verbose=0)
        return np.column_stack([1 - probs.flatten(), probs.flatten()])
    
    def get_weights(self):
        """Get model weights"""
        if self.model is None:
            return None
        weights = self.model.get_weights()
        return {'coef_': weights[0].flatten(), 'intercept_': weights[1]}
    
    def set_weights(self, weights_dict):
        """Set model weights"""
        if self.model is None:
            # Need to build model first with dummy data
            return
        
        if 'coef_' in weights_dict and 'intercept_' in weights_dict:
            coef = weights_dict['coef_'].reshape(-1, 1)
            intercept = weights_dict['intercept_']
            self.model.set_weights([coef, intercept])
    
    def get_privacy_spent(self):
        """Get current privacy expenditure"""
        return self.privacy_spent.copy()


def load_client_data(client_id):
    """
    Load data for a specific client, split into train/test.
    Args: client_id (int): ID of the client (1-5).
    Returns: X_train, X_test, y_train, y_test (Features and labels split).
    """
    project_root = Path(__file__).resolve().parent.parent

    # Build the path to data/clients/client_{id}.csv
    csv_path = project_root / "data" / "clients" / f"client_{client_id}.csv"
    print(f"Loading client data from: {csv_path}")

    df = pd.read_csv(csv_path)
    # df = pd.read_csv(f"./data/clients/client_{client_id}.csv")
    X = df.drop("Readmitted", axis=1).values
    y = df["Readmitted"].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_model(use_dp=True, dp_config=None):
    """
    Initialize a model (DP or regular logistic regression).
    Returns: Model instance
    """
    if use_dp:
        return DPLogisticRegression(dp_config=dp_config, use_dp=True)
    else:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver="lbfgs", max_iter=1000)


def compute_privacy_budget(dataset_sizes, batch_size, noise_multiplier, epochs, delta=1e-5):
    """
    Compute privacy budget for federated learning setup
    """
    total_epsilon = 0
    for size in dataset_sizes:
        epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=size,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta
        )
        total_epsilon += epsilon
    
    return total_epsilon, delta


def analyze_privacy_utility_tradeoff(noise_multipliers, dataset_size, batch_size, epochs, delta=1e-5):
    """
    Analyze the privacy-utility tradeoff for different noise levels
    """
    results = []
    
    for noise_mult in noise_multipliers:
        epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
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