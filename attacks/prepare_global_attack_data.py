import os
import pandas as pd
import numpy as np

output_dir = "attack_data"
os.makedirs(output_dir, exist_ok=True)

# Load all 5 clients and concatenate
dfs = []
for i in range(1, 6):
    path = f"../data/clients/client_{i}.csv"
    print(f"[INFO] Loading {path}")
    df = pd.read_csv(path)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
X = full_df.iloc[:, :-1].values  # exclude label

print(f"[INFO] Combined shape: {X.shape}")  # should be (?, 52)

# Split evenly
split = len(X) // 2
X_member = X[:split]
X_nonmember = X[split:]

# Save
np.save(f"{output_dir}/X_member_global.npy", X_member)
np.save(f"{output_dir}/X_nonmember_global.npy", X_nonmember)
print(f"[SUCCESS] Saved global attack data: {len(X_member)} members, {len(X_nonmember)} non-members")
