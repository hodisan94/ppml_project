# train_rf_and_save.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("../../../data/processed/full_preprocessed.csv")
print(f"[INFO] Original dataset size: {len(df)}")

# Remove exact duplicates
df_nodup = df.drop_duplicates()
print(f"[INFO] After duplicate removal: {len(df_nodup)} rows")

# Split features/labels
X = df_nodup.drop("Readmitted", axis=1).values
y = df_nodup["Readmitted"].values

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[INFO] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "rf_naive_model.pkl")
print("[INFO] Model saved to rf_naive_model.pkl")

# Save data for MIA
np.save("X_member.npy", X_train)
np.save("y_member.npy", y_train)
np.save("X_nonmember.npy", X_test)
np.save("y_nonmember.npy", y_test)
print("[INFO] Member/Non-member data saved.")

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n[INFO] Evaluation:")
print("AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
