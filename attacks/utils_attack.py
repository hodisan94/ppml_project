import numpy as np
from tensorflow.keras.models import load_model

def load_data_split():
    X_member = np.load("results/X_member.npy")
    X_nonmember = np.load("results/X_nonmember.npy")
    return X_member, X_nonmember

def load_model_by_name(name):
    return load_model(f"results/{name}.h5")
