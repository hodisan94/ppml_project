import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("../Naive/attack_results/comprehensive_mia_results_rf_naive_model_pkl1.json") as f:
    naive = json.load(f)
with open("../FL/attack_results/federated_mia_results.json") as f:
    fl = json.load(f)
with open("../FL+DP/attack_results/federated+dp_mia_results.json") as f:
    fl_dp = json.load(f)

attack_types = ["confidence", "entropy", "loss"]
techniques = ["Naive RF", "Federated", "Federated + DP"]
colors = ["blue", "cyan", "green"]

auc_data = {
    "Naive RF": [naive[t]["auc"] for t in attack_types],
    "Federated": [fl[t]["auc"] for t in attack_types],
    "Federated + DP": [fl_dp[t]["auc"] for t in attack_types],
}

x = np.arange(len(attack_types))
width = 0.25

plt.figure(figsize=(10, 6))
for i, (technique, aucs) in enumerate(auc_data.items()):
    plt.bar(x + i * width, aucs, width, label=technique, color=colors[i])

plt.axhline(0.5, color="gray", linestyle="--", label="Random Guess (AUC = 0.5)")
plt.xticks(x + width, [t.capitalize() for t in attack_types])
plt.ylabel("AUC Score")
plt.ylim(0.45, 0.7)
plt.title("MIA AUC Comparison Across Privacy Techniques")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("mia_auc_privacy_comparison.png")
plt.show()