import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv("../fl_metrics.csv")

# Set plot style
sns.set(style="whitegrid", font_scale=1.2)

# Add slight jitter to accuracy values for visual separation
df["jittered_accuracy"] = df["accuracy"] + np.random.normal(0, 0.0003, size=len(df))

# Set a colorblind-friendly palette and distinct line styles
palette = sns.color_palette("colorblind", n_colors=df["client_id"].nunique())
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Create the plot
plt.figure(figsize=(12, 6))
for idx, (client_id, group) in enumerate(df.groupby("client_id")):
    plt.plot(group["round"], group["jittered_accuracy"],
             label=f"Client {client_id}",
             marker="o",
             linestyle=linestyles[idx % len(linestyles)],
             color=palette[idx % len(palette)])

plt.title("Client Accuracy per Round")
plt.xlabel("Federated Learning Round")
plt.ylabel("Accuracy")
plt.legend(title="Client ID")
plt.grid(True)
plt.tight_layout()
plt.savefig("lineplot_jittered_enhanced.png")
plt.close()

# 2. Boxplot per round
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="round", y="accuracy", palette="Set2")
plt.title("Accuracy Distribution per Round Across Clients")
plt.ylabel("Accuracy")
plt.xlabel("Round")
plt.tight_layout()
plt.savefig("boxplot_rounds.png")
plt.close()

# # 3. Heatmap of accuracies per client per round
pivot = df.pivot(index="client_id", columns="round", values="accuracy")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".4f")
plt.title("Client Accuracies Across Rounds (Heatmap)")
plt.ylabel("Client ID")
plt.xlabel("Round")
plt.tight_layout()
plt.savefig("heatmap_clients_rounds.png")
plt.close()

