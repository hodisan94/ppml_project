
import os
import pandas as pd
import numpy as np

# Path to client CSV
client_path = "../data/clients/client_1.csv"
output_dir = "attack_data"
os.makedirs(output_dir, exist_ok=True)

print(f"[INFO] Loading client data from: {client_path}")
df = pd.read_csv(client_path)

# Assuming last column is the label
X = df.iloc[:, :-1].values

# Split: first 50% as member, last 50% as non-member
split_index = len(X) // 2
X_member = X[:split_index]
X_nonmember = X[split_index:]

# Save to .npy files
np.save(os.path.join(output_dir, "X_member.npy"), X_member)
np.save(os.path.join(output_dir, "X_nonmember.npy"), X_nonmember)

print(f"[SUCCESS] Saved {len(X_member)} member samples to attack_data/X_member.npy")
print(f"[SUCCESS] Saved {len(X_nonmember)} non-member samples to attack_data/X_nonmember.npy")
