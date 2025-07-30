import matplotlib.pyplot as plt

models = ["Naive RF", "Federated", "Federated + DP"]
mse_scores = [0.0001, 0.0992, 0.0992]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, mse_scores, color=['blue', 'cyan', 'green'])
plt.axhline(y=0.0, linestyle='--', color='gray')
plt.ylabel("MSE")
plt.title("Model Inversion Attack – MSE by Privacy Technique")

# Annotate values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f"{yval:.4f}", ha='center', va='bottom')

plt.grid(axis='y')
plt.tight_layout()
plt.savefig("inversion_comparison_mse.png")
plt.show()
