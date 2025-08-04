#!/usr/bin/env python3
"""
Healthcare ML inference service for SGX demo.
Loads a trained model and performs readmission risk prediction on patient data.
"""

import pickle
import argparse
import numpy as np
import os
import sys
import time

def load_model():
    """Load the trained healthcare model."""
    try:
        with open("healthcare_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_names = model_data['feature_names']
        n_features = model_data['n_features']
        
        print(f"[+] Loaded model with {n_features} features")
        print(f"[+] Model type: {type(model).__name__}")
        return model, feature_names, n_features
    
    except FileNotFoundError:
        print("[!] Model file 'healthcare_model.pkl' not found!")
        print("[!] Run 'python train_healthcare_model.py' first to train the model.")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        sys.exit(1)

def load_patient_data(input_path):
    """Load patient data from file."""
    try:
        with open(input_path, "rb") as f:
            patient_data = pickle.load(f)
        
        features = patient_data['features']
        feature_names = patient_data.get('feature_names', [])
        true_label = patient_data.get('true_label', None)
        description = patient_data.get('description', 'Unknown patient')
        
        print(f"[+] Loaded patient data: {description}")
        if true_label is not None:
            print(f"[+] True readmission status: {true_label}")
        
        return features, feature_names, true_label
    
    except FileNotFoundError:
        print(f"[!] Patient data file '{input_path}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Error loading patient data: {e}")
        sys.exit(1)

def secure_inference(model, patient_features, feature_names):
    """
    Perform secure inference with simulated confidential processing.
    In a real SGX environment, this would happen inside the enclave.
    """
    print("[+] Starting secure inference...")
    
    # Simulate secure processing time
    time.sleep(0.1)
    
    # Reshape input for sklearn
    patient_input = np.array(patient_features).reshape(1, -1)
    
    # Get prediction and confidence
    prediction = model.predict(patient_input)[0]
    confidence = model.predict_proba(patient_input)[0]
    readmission_prob = confidence[1]  # Probability of readmission
    
    # In SGX, we would only return the final result, not intermediates
    result = {
        'prediction': int(prediction),
        'readmission_probability': float(readmission_prob),
        'risk_level': 'HIGH' if readmission_prob > 0.7 else 'MEDIUM' if readmission_prob > 0.3 else 'LOW'
    }
    
    return result

def simulate_memory_exposure(model, patient_features, feature_names):
    """
    Simulate what an attacker could observe in non-SGX execution.
    This represents the information leak that SGX prevents.
    """
    print("\n[!] SECURITY WARNING: In non-SGX execution, the following sensitive data is exposed:")
    
    # Model coefficients (would be visible in memory)
    coefficients = model.coef_[0] if hasattr(model, 'coef_') else None
    if coefficients is not None:
        print(f"[!] Model coefficients exposed: {coefficients[:5]}... (showing first 5)")
    
    # Patient features (would be visible in process memory)
    print(f"[!] Patient features exposed: {patient_features[:5]}... (showing first 5)")
    
    # Intermediate calculations (would be visible during computation)
    if coefficients is not None and len(patient_features) == len(coefficients):
        linear_combination = np.dot(patient_features, coefficients)
        print(f"[!] Intermediate calculation exposed: linear_combination = {linear_combination:.4f}")
    
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Healthcare ML Inference Service")
    parser.add_argument(
        "--input", 
        default="sample_data/patient_input.pkl",
        help="Path to patient data file (pickle format)"
    )
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Run in secure mode (simulating SGX enclave)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Enable verbose output"
    )
    return parser.parse_args()

def main():
    """Main inference pipeline."""
    args = parse_args()
    
    print("="*60)
    print("Healthcare ML Inference Service")
    print("="*60)
    
    if args.secure:
        print("[+] Running in SECURE mode (simulating SGX enclave)")
    else:
        print("[!] Running in NORMAL mode (vulnerable to memory attacks)")
    
    # Load model and patient data
    model, model_feature_names, n_features = load_model()
    patient_features, patient_feature_names, true_label = load_patient_data(args.input)
    
    # Validate input dimensions
    if len(patient_features) != n_features:
        print(f"[!] Feature dimension mismatch: expected {n_features}, got {len(patient_features)}")
        sys.exit(1)
    
    # Perform inference
    result = secure_inference(model, patient_features, model_feature_names)
    
    # Display results
    print("\n" + "="*40)
    print("INFERENCE RESULTS")
    print("="*40)
    print(f"Prediction: {'WILL BE READMITTED' if result['prediction'] == 1 else 'NO READMISSION'}")
    print(f"Readmission Probability: {result['readmission_probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    
    if true_label is not None:
        accuracy = "CORRECT" if result['prediction'] == true_label else "INCORRECT"
        print(f"Prediction Accuracy: {accuracy}")
    
    # Simulate memory exposure in non-secure mode
    if not args.secure:
        simulate_memory_exposure(model, patient_features, model_feature_names)
        print("\n[!] In SGX mode, this sensitive information would be protected!")
    else:
        print("\n[+] Patient data and model parameters protected by SGX enclave")
    
    print("\n[+] Inference completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())