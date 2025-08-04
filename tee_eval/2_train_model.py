#!/usr/bin/env python3
"""
SGX Healthcare Demo - Model Training
Trains healthcare readmission model and creates sample data
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def print_header(title):
    """Print training section header."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def load_healthcare_data():
    """Load and prepare healthcare dataset."""
    print("[+] Loading healthcare dataset...")
    
    # Try to load from main project first
    data_paths = [
        "../data/processed/full_preprocessed.csv",
        "../data/raw/healthcare_dataset.csv"
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            print(f"[+] Found dataset: {path}")
            try:
                df = pd.read_csv(path)
                
                # If raw data, do basic preprocessing
                if 'healthcare_dataset.csv' in path:
                    df = preprocess_raw_data(df)
                
                break
            except Exception as e:
                print(f"[!] Failed to load {path}: {e}")
                continue
    
    if df is None:
        print("[!] No healthcare dataset found - creating synthetic data")
        df = create_synthetic_data()
    
    print(f"[+] Dataset loaded: {df.shape}")
    return df

def preprocess_raw_data(df):
    """Basic preprocessing for raw healthcare data."""
    print("[+] Preprocessing raw healthcare data...")
    
    # Drop irrelevant columns
    cols_to_drop = ["Name", "Doctor", "Room Number", "Hospital"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Handle dates if present
    date_cols = ["Date of Admission", "Discharge Date"]
    if all(col in df.columns for col in date_cols):
        df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
        df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])
        df["Stay Length"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
        df["Readmitted"] = (df["Stay Length"] < 4).astype(int)
        df = df.drop(columns=date_cols + ["Stay Length"])
    
    # Encode categorical variables
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    
    # One-hot encode key categorical variables
    categorical_cols = ["Blood Type", "Medical Condition", "Admission Type"]
    existing_cats = [col for col in categorical_cols if col in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats)
    
    # Drop remaining string columns
    string_cols = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=string_cols, errors='ignore')
    
    # Normalize numeric columns
    numeric_cols = ["Age", "Billing Amount"]
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    if existing_numeric:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])
    
    # Ensure we have a target variable
    if "Readmitted" not in df.columns:
        # Create synthetic target based on available features
        if "Age" in df.columns:
            df["Readmitted"] = (df["Age"] > 0.6).astype(int)  # Older patients higher risk
        else:
            df["Readmitted"] = np.random.binomial(1, 0.3, len(df))  # 30% readmission rate
    
    return df

def create_synthetic_data():
    """Create synthetic healthcare data for demo."""
    print("[+] Creating synthetic healthcare dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'Age': np.random.normal(0.5, 0.2, n_samples).clip(0, 1),
        'Gender': np.random.binomial(1, 0.5, n_samples),
        'Billing_Amount': np.random.normal(0.4, 0.3, n_samples).clip(0, 1),
        'Blood_Type_A': np.random.binomial(1, 0.4, n_samples),
        'Blood_Type_B': np.random.binomial(1, 0.1, n_samples),
        'Blood_Type_O': np.random.binomial(1, 0.4, n_samples),
        'Condition_Diabetes': np.random.binomial(1, 0.2, n_samples),
        'Condition_Hypertension': np.random.binomial(1, 0.3, n_samples),
        'Condition_Heart_Disease': np.random.binomial(1, 0.15, n_samples),
        'Admission_Emergency': np.random.binomial(1, 0.3, n_samples),
        'Admission_Elective': np.random.binomial(1, 0.4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic readmission target
    # Higher risk: older, diabetes, heart disease, emergency admission
    risk_score = (
        df['Age'] * 0.3 +
        df['Condition_Diabetes'] * 0.2 +
        df['Condition_Heart_Disease'] * 0.2 +
        df['Admission_Emergency'] * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['Readmitted'] = (risk_score > 0.3).astype(int)
    
    print(f"[+] Synthetic data created: {df.shape}")
    print(f"[+] Readmission rate: {df['Readmitted'].mean():.2%}")
    
    return df

def train_healthcare_model(df):
    """Train logistic regression model for readmission prediction."""
    print_header("Training Healthcare ML Model")
    
    # Separate features and target
    if 'Readmitted' not in df.columns:
        raise ValueError("Dataset must have 'Readmitted' target column")
    
    X = df.drop(columns=['Readmitted']).values
    y = df['Readmitted'].values
    feature_names = list(df.columns[:-1])
    
    print(f"[+] Features: {len(feature_names)}")
    print(f"[+] Samples: {len(X)}")
    print(f"[+] Readmission distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model (LogisticRegression for interpretability)
    print("[+] Training logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n[+] Training accuracy: {accuracy_score(y_train, train_pred):.3f}")
    print(f"[+] Test accuracy: {accuracy_score(y_test, test_pred):.3f}")
    print(f"[+] Training AUC: {roc_auc_score(y_train, train_prob):.3f}")
    print(f"[+] Test AUC: {roc_auc_score(y_test, test_prob):.3f}")
    
    # Save model
    os.makedirs("data", exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'training_stats': {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'train_auc': roc_auc_score(y_train, train_prob),
            'test_auc': roc_auc_score(y_test, test_prob)
        }
    }
    
    with open("data/healthcare_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("\n[+] Model saved to: data/healthcare_model.pkl")
    
    return model, X_test, y_test, feature_names

def create_sample_patient_data(X_test, y_test, feature_names):
    """Create sample patient data for demo."""
    print_header("Creating Sample Patient Data")
    
    # Find a high-risk patient (readmission = 1)
    high_risk_indices = np.where(y_test == 1)[0]
    
    if len(high_risk_indices) > 0:
        sample_idx = high_risk_indices[0]
        sample_features = X_test[sample_idx]
        sample_label = y_test[sample_idx]
        description = "High-risk patient for readmission prediction demo"
    else:
        # Fallback: use first patient
        sample_idx = 0
        sample_features = X_test[sample_idx]
        sample_label = y_test[sample_idx]
        description = "Sample patient for readmission prediction demo"
    
    patient_data = {
        'features': sample_features,
        'feature_names': feature_names,
        'true_label': sample_label,
        'description': description,
        'patient_id': f"DEMO_{sample_idx:04d}"
    }
    
    with open("data/patient_sample.pkl", "wb") as f:
        pickle.dump(patient_data, f)
    
    print(f"[+] Sample patient saved: {description}")
    print(f"[+] True readmission status: {sample_label}")
    print(f"[+] Sample features (first 5): {sample_features[:5]}")
    print("[+] Saved to: data/patient_sample.pkl")

def main():
    """Main training pipeline."""
    print_header("Healthcare ML Model Training")
    print("Training readmission prediction model for SGX demo")
    
    try:
        # Load healthcare data
        df = load_healthcare_data()
        
        # Train model
        model, X_test, y_test, feature_names = train_healthcare_model(df)
        
        # Create sample data
        create_sample_patient_data(X_test, y_test, feature_names)
        
        print_header("Training Completed Successfully")
        print("Files created:")
        print("  - data/healthcare_model.pkl (trained model)")
        print("  - data/patient_sample.pkl (sample patient)")
        print("\nNext: python 3_run_demo.py")
        
        return 0
        
    except Exception as e:
        print(f"\n[!] Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())