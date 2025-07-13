import numpy as np
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Optimizer

# Dummy optimizer for loading DP models
class DummyDPOptimizer(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(name="DummyDPOptimizer")

    def get_config(self):
        return {"name": "DummyDPOptimizer"}

CLIENT_MODEL_PATHS = [
    "fl_dp_model_client_1.h5",
    "fl_dp_model_client_2.h5",
    "fl_dp_model_client_3.h5",
    "fl_dp_model_client_4.h5",
    "fl_dp_model_client_5.h5"
]

OUTPUT_MODEL_PATH = "fl_dp_global_model_aggregated.h5"

def average_weights(weight_list):
    return [np.mean(w, axis=0) for w in zip(*weight_list)]

def aggregate_models(model_paths, output_path):
    print("[INFO] Loading client models...")
    models = [
        load_model(path, custom_objects={"DPOptimizerClass": DummyDPOptimizer})
        for path in model_paths
    ]
    weights = [model.get_weights() for model in models]
    avg_weights = average_weights(weights)

    print("[INFO] Applying averaged weights to base model...")
    global_model = models[0]
    global_model.set_weights(avg_weights)

    print(f"[SUCCESS] Saving aggregated model to {output_path}")
    save_model(global_model, output_path)

if __name__ == "__main__":
    aggregate_models(CLIENT_MODEL_PATHS, OUTPUT_MODEL_PATH)
