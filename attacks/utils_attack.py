import os
import numpy as np
from tensorflow.keras.models import load_model

def load_data_split(data_dir="attack_data"):
    """Returns (X_member, X_nonmember)."""
    Xm = np.load(os.path.join(data_dir, "X_member.npy"))
    Xn = np.load(os.path.join(data_dir, "X_nonmember.npy"))
    return Xm, Xn

def load_model_by_path(path):
    """Loads a Keras model from .h5 file."""
    return load_model(path, compile=False)
