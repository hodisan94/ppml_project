#!/usr/bin/env python3
"""
Train a healthcare readmission prediction model using the project's existing data.
Creates a simple logistic regression model that predicts readmission risk.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def load_and_prepare_data():
    """Load and prepare the healthcare dataset."""
    print("[+] Loading healthcare dataset...")
    
    # Try to load preprocessed data first
    preprocessed_path = "../data/processed/full_preprocessed.csv"
    if os.path.exists(preprocessed_path):
        print(f"[+] Using preprocessed data: {preprocessed_path}")
        df = pd.read_csv(preprocessed_path)
    else:
        # Fallback to raw data processing (simplified version)
        print("[!] Preprocessed data not found, using simplified preprocessing...")
        raw_path = "../data/raw/healthcare_dataset.csv"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Healthcare dataset not found at {raw_path}")
        
        df = pd.read_csv(raw_path)
        
        # Simplified preprocessing
        df = df.drop(columns=["Name", "Doctor", "Room Number", "Hospital"], errors='ignore')
        
        # Convert dates and create readmission label
        df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
        df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
        df["Stay Length"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
        df["Readmitted"] = (df["Stay Length"] < 4).astype(int)
        
        # Drop date columns
        df = df.drop(columns=["Date of Admission", "Discharge Date", "Stay Length"])
        
        # Encode gender
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
        
        # One-hot encode key categorical variables
        categorical_cols = ["Blood Type", "Medical Condition", "Admission Type"]
        df = pd.get_dummies(df, columns=categorical_cols)
        
        # Drop remaining string columns
        string_cols = df.select_dtypes(include=['object']).columns
        df = df.drop(columns=string_cols, errors='ignore')
        
        # Normalize numeric columns
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        numeric_cols = ["Age", "Billing Amount"]
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print(f"[+] Dataset shape: {df.shape}")
    print(f"[+] Features: {list(df.columns[:-1])}")
    
    # Separate features and target
    X = df.drop(columns=["Readmitted"]).values
    y = df["Readmitted"].values
    
    print(f"[+] Readmission distribution: {np.bincount(y)}")
    return X, y, list(df.columns[:-1])

def train_model(X, y, feature_names):
    """Train a logistic regression model."""
    print("[+] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("[+] Training logistic regression model...")
    # Use a simple logistic regression for interpretability
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"[+] Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"[+] Test accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"[+] Training AUC: {roc_auc_score(y_train, train_prob):.4f}")
    print(f"[+] Test AUC: {roc_auc_score(y_test, test_prob):.4f}")
    
    print("\n[+] Test set classification report:")
    print(classification_report(y_test, test_pred))
    
    return model, X_test, y_test, feature_names

def save_model_and_sample(model, X_test, y_test, feature_names):
    """Save the trained model and create sample input."""
    print("[+] Saving model...")
    
    # Save the model
    with open("healthcare_model.pkl", "wb") as f:
        pickle.dump({
            'model': model, 
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }, f)
    
    # Create sample input for demo (use a positive case)
    positive_indices = np.where(y_test == 1)[0]
    if len(positive_indices) > 0:
        sample_idx = positive_indices[0]
        sample_input = X_test[sample_idx]
        sample_label = y_test[sample_idx]
        
        # Create sample data directory
        os.makedirs("sample_data", exist_ok=True)
        
        # Save sample input
        with open("sample_data/patient_input.pkl", "wb") as f:
            pickle.dump({
                'features': sample_input,
                'feature_names': feature_names,
                'true_label': sample_label,
                'description': 'High-risk patient for readmission prediction demo'
            }, f)
        
        print(f"[+] Saved sample patient data (true label: {sample_label})")
        print(f"[+] Sample features: {dict(zip(feature_names[:5], sample_input[:5]))}")
    
    print("[+] Model training completed successfully!")
    print("[+] Files created:")
    print("    - healthcare_model.pkl (trained model)")
    print("    - sample_data/patient_input.pkl (sample patient data)")

def main():
    """Main training pipeline."""
    try:
        # Load and prepare data
        X, y, feature_names = load_and_prepare_data()
        
        # Train model
        model, X_test, y_test, feature_names = train_model(X, y, feature_names)
        
        # Save results
        save_model_and_sample(model, X_test, y_test, feature_names)
        
    except Exception as e:
        print(f"[!] Error during training: {e}")
        print("[!] Make sure the healthcare dataset is available in ../data/")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())