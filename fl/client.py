# import numpy as np
# import pandas as pd
# import sys
# import flwr as flower
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import log_loss, accuracy_score
# from utils import load_client_data, get_model
#
# # Parse client_id from command line
# client_id = int(sys.argv[1])
# X_train, X_test, y_train, y_test = load_client_data(client_id)
#
# # Create and fit model
# model = get_model()
#
# # Dummy fit with at least two different classes
# unique_classes = np.unique(y_train)
# if len(unique_classes) < 2:
#     raise ValueError("Training data must contain at least two classes for dummy fit.")
#
# # Get indices of two samples with different classes
# first_idx = np.where(y_train == unique_classes[0])[0][0]
# second_idx = np.where(y_train == unique_classes[1])[0][0]
# dummy_X = np.vstack([X_train[first_idx], X_train[second_idx]])
# dummy_y = np.array([y_train[first_idx], y_train[second_idx]])
# model.fit(dummy_X, dummy_y)
#
# # Define Flower client
# class PPMLClient(flower.client.NumPyClient):
#     def get_parameters(self, config=None):
#         print(f"[CLIENT {client_id}] connected to server", flush=True)
#         return [param.copy() for param in model.coef_] + [model.intercept_.copy()]
#
#     def fit(self, parameters, config=None):
#         model.coef_ = np.array(parameters[:-1])
#         model.intercept_ = np.array(parameters[-1])
#         model.classes_ = np.array([0, 1])  # Ensure binary classes
#         model.fit(X_train, y_train)
#         return self.get_parameters(), len(X_train), {}
#
#     def evaluate(self, parameters, config=None):
#         model.coef_ = np.array(parameters[:-1])
#         model.intercept_ = np.array(parameters[-1])
#         model.classes_ = np.array([0, 1])
#         preds = model.predict_proba(X_test)
#         loss = log_loss(y_test, preds)
#         accuracy = accuracy_score(y_test, np.argmax(preds, axis=1))
#         print(f"[CLIENT {client_id}] accuracy: {accuracy:.4f}", flush=True)
#         return float(loss), len(X_test), {"accuracy": float(accuracy)}
#
# # Start client
# flower.client.start_numpy_client(server_address="127.0.0.1:8086", client=PPMLClient())
import numpy as np
import pandas as pd
import sys
import flwr as flower
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from utils import load_client_data, get_model

# Parse client_id from command line
client_id = int(sys.argv[1])
X_train, X_test, y_train, y_test = load_client_data(client_id)

# Create and fit model
model = get_model()

# Dummy fit with at least two different classes
unique_classes = np.unique(y_train)
if len(unique_classes) < 2:
    raise ValueError("Training data must contain at least two classes for dummy fit.")

# Get indices of two samples with different classes
first_idx = np.where(y_train == unique_classes[0])[0][0]
second_idx = np.where(y_train == unique_classes[1])[0][0]
dummy_X = np.vstack([X_train[first_idx], X_train[second_idx]])
dummy_y = np.array([y_train[first_idx], y_train[second_idx]])
model.fit(dummy_X, dummy_y)

# Define Flower client
class PPMLClient(flower.client.NumPyClient):
    def get_parameters(self, config=None):
        print(f"[CLIENT {client_id}] connected to server", flush=True)
        # Return flattened coefficients and intercept
        return [model.coef_.flatten(), model.intercept_]

    def fit(self, parameters, config=None):
        # Reshape parameters back to original shape
        model.coef_ = np.array(parameters[0]).reshape(1, -1)
        model.intercept_ = np.array(parameters[1])
        model.classes_ = np.array([0, 1])  # Ensure binary classes
        model.fit(X_train, y_train)
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config=None):
        # Reshape parameters back to original shape
        model.coef_ = np.array(parameters[0]).reshape(1, -1)
        model.intercept_ = np.array(parameters[1])
        model.classes_ = np.array([0, 1])
        preds = model.predict_proba(X_test)
        loss = log_loss(y_test, preds)
        accuracy = accuracy_score(y_test, np.argmax(preds, axis=1))
        print(f"[CLIENT {client_id}] accuracy: {accuracy:.4f}", flush=True)
        return float(loss), len(X_test), {"accuracy": float(accuracy)}

# Start client
flower.client.start_numpy_client(server_address="127.0.0.1:8086", client=PPMLClient())