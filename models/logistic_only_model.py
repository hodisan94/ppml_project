import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data from full_preprocessed.csv (not combined clients)
df = pd.read_csv("../data/processed/full_preprocessed.csv")
X = df.drop("Readmitted", axis=1).values
y = df["Readmitted"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and modeling pipeline
numeric_cols = [0, 1]  # Assuming columns 0 and 1 are numeric
preprocessor = ColumnTransformer(transformers=[
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), numeric_cols),
    ('scale', StandardScaler(), numeric_cols)
], remainder='passthrough')

pipeline = ImbPipeline(steps=[
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('logreg', LogisticRegression(solver='liblinear', max_iter=10000, random_state=42, C=0.1))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Save model
os.makedirs("results", exist_ok=True)
joblib.dump(pipeline, "results/naive_logistic_model.pkl")

# Evaluation
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_proba)
}

# Save metrics
with open("results/naive_logistic_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save ROC plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label="ROC (AUC = {:.2f})".format(metrics["auc"]))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Naive Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("results/naive_logistic_roc.png")
plt.close()