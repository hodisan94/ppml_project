import numpy as np
import sys
import flwr as flower
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss
from utils import load_client_data

# Try different import paths for tensorflow-privacy based on version
try:
    # TF-Privacy ≥0.7.0
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
except ImportError:
    # fallback to the older optimizer class name
    from tensorflow_privacy.privacy.optimizers.dp_optimizer import (
        DPGradientDescentGaussianOptimizer as DPKerasSGDOptimizer
    )

# — privacy accounting import —
try:
    # direct import of the function
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
except ImportError:
    # if not available, disable privacy accounting
    compute_dp_sgd_privacy = None
    print("Warning: compute_dp_sgd_privacy not found; privacy accounting disabled.")
# DP-SGD configuration
noise_multiplier = 1.1
l2_norm_clip = 1.0
learning_rate = 0.15
batch_size = 64
epochs = 1
delta = 1e-5

# Get client ID from command line
client_id = int(sys.argv[1])

# Load client-specific dataset
X_train, X_test, y_train, y_test = load_client_data(client_id)

# Convert to float32 for better compatibility
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Prepare TF dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

# Define logistic regression model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile with DP optimizer if available, otherwise use regular optimizer
if DPKerasSGDOptimizer:
    try:
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=1,
            learning_rate=learning_rate
        )
        print(f"[CLIENT {client_id}] Using DP-SGD optimizer", flush=True)
    except Exception as e:
        print(f"[CLIENT {client_id}] DP optimizer failed, using regular SGD: {e}", flush=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
else:
    print(f"[CLIENT {client_id}] Using regular SGD optimizer", flush=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])


# Flower client definition
class DPClient(flower.client.NumPyClient):
    def get_parameters(self, config=None):
        print(f"[CLIENT {client_id}] Sending initial model weights", flush=True)
        return model.get_weights()

    def fit(self, parameters, config=None):
        print(f"[CLIENT {client_id}] Training with {'DP-SGD' if DPKerasSGDOptimizer else 'regular SGD'}...", flush=True)
        model.set_weights(parameters)

        # Train the model
        history = model.fit(train_dataset, epochs=epochs, verbose=0)

        # Compute privacy loss (epsilon) if DP is available
        eps = 0.0
        if compute_dp_sgd_privacy and DPKerasSGDOptimizer:
            try:
                if hasattr(compute_dp_sgd_privacy, 'compute_dp_sgd_privacy'):
                    eps, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                        n=len(X_train),
                        batch_size=batch_size,
                        noise_multiplier=noise_multiplier,
                        epochs=epochs,
                        delta=delta
                    )
                else:
                    eps, _ = compute_dp_sgd_privacy(
                        n=len(X_train),
                        batch_size=batch_size,
                        noise_multiplier=noise_multiplier,
                        epochs=epochs,
                        delta=delta
                    )
                print(f"[CLIENT {client_id}] ε after round: {eps:.4f}", flush=True)
            except Exception as e:
                print(f"[CLIENT {client_id}] Privacy computation failed: {e}", flush=True)
                eps = 0.0
        else:
            print(f"[CLIENT {client_id}] Privacy computation not available", flush=True)

        return model.get_weights(), len(X_train), {"epsilon": eps}

    def evaluate(self, parameters, config=None):
        model.set_weights(parameters)
        preds = model.predict(X_test, verbose=0)
        preds = np.clip(preds, 1e-7, 1 - 1e-7)  # Avoid log(0) issues

        loss = log_loss(y_test, preds)
        acc = accuracy_score(y_test, np.round(preds))
        print(f"[CLIENT {client_id}] Accuracy: {acc:.4f}", flush=True)
        return loss, len(X_test), {"accuracy": acc}


# Start the Flower client
print(f"[CLIENT {client_id}] Starting client...", flush=True)
flower.client.start_numpy_client(server_address="127.0.0.1:8086", client=DPClient())