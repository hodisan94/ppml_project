#!/usr/bin/env python3
"""
Healthcare ML Inference Service
Core component for SGX demo - loads model and processes patient data
"""

import pickle
import argparse
import numpy as np
import sys
import time
import os

class HealthcareInference:
    """Healthcare ML inference service."""
    
    def __init__(self, model_path="data/healthcare_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.n_features = None
        
    def load_model(self):
        """Load trained healthcare model."""
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.n_features = model_data['n_features']
            
            print(f"[+] Loaded healthcare model: {self.n_features} features")
            return True
            
        except FileNotFoundError:
            print(f"[!] Model not found: {self.model_path}")
            print("[!] Run: python 2_train_model.py")
            return False
        except Exception as e:
            print(f"[!] Model loading failed: {e}")
            return False
    
    def load_patient_data(self, data_path="data/patient_sample.pkl"):
        """Load patient data for inference."""
        try:
            with open(data_path, "rb") as f:
                patient_data = pickle.load(f)
            
            features = patient_data['features']
            description = patient_data.get('description', 'Unknown patient')
            true_label = patient_data.get('true_label', None)
            
            print(f"[+] Loaded patient: {description}")
            return features, true_label
            
        except FileNotFoundError:
            print(f"[!] Patient data not found: {data_path}")
            return None, None
        except Exception as e:
            print(f"[!] Patient data loading failed: {e}")
            return None, None
    
    def predict_readmission(self, patient_features):
        """Perform readmission risk prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Reshape for sklearn
        patient_input = np.array(patient_features).reshape(1, -1)
        
        # Get prediction and confidence
        prediction = self.model.predict(patient_input)[0]
        confidence = self.model.predict_proba(patient_input)[0]
        readmission_prob = confidence[1]
        
        risk_level = (
            'HIGH' if readmission_prob > 0.7 else 
            'MEDIUM' if readmission_prob > 0.3 else 
            'LOW'
        )
        
        return {
            'prediction': int(prediction),
            'readmission_probability': float(readmission_prob),
            'risk_level': risk_level
        }
    
    def expose_sensitive_data(self, patient_features):
        """Simulate data exposure in vulnerable execution."""
        print("\n[!] SECURITY WARNING: Sensitive data exposed in memory:")
        
        # Model coefficients (vulnerable to memory extraction)
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_[0]
            print(f"[!] Model coefficients: {coefficients[:5]}... (showing 5/{len(coefficients)})")
        
        # Patient features (visible in process memory)
        print(f"[!] Patient features: {patient_features[:5]}... (showing 5/{len(patient_features)})")
        
        # Intermediate calculations (would be in memory during computation)
        if hasattr(self.model, 'coef_') and len(patient_features) == len(self.model.coef_[0]):
            linear_combination = np.dot(patient_features, self.model.coef_[0])
            print(f"[!] Linear combination: {linear_combination:.4f}")
        
        return True
    
    def run_inference_service(self, mode="vulnerable"):
        """Run inference service in specified mode."""
        print(f"\n{'='*50}")
        print("Healthcare ML Inference Service")
        print(f"{'='*50}")
        
        if mode == "secure":
            print("[+] Running in SECURE mode (SGX enclave simulation)")
        else:
            print("[!] Running in VULNERABLE mode (normal process)")
        
        # Load model and data
        if not self.load_model():
            return False
        
        patient_features, true_label = self.load_patient_data()
        if patient_features is None:
            return False
        
        # Validate dimensions
        if len(patient_features) != self.n_features:
            print(f"[!] Feature mismatch: expected {self.n_features}, got {len(patient_features)}")
            return False
        
        # Perform inference
        print("\n[+] Performing readmission risk assessment...")
        result = self.predict_readmission(patient_features)
        
        # Display results
        print(f"\n{'='*40}")
        print("CLINICAL PREDICTION RESULTS")
        print(f"{'='*40}")
        print(f"Readmission Risk: {'HIGH RISK' if result['prediction'] == 1 else 'LOW RISK'}")
        print(f"Confidence: {result['readmission_probability']:.3f}")
        print(f"Risk Category: {result['risk_level']}")
        
        if true_label is not None:
            accuracy = "CORRECT" if result['prediction'] == true_label else "INCORRECT"
            print(f"Prediction Accuracy: {accuracy}")
        
        # Simulate memory exposure in vulnerable mode
        if mode != "secure":
            self.expose_sensitive_data(patient_features)
            print("\n[!] In SGX mode, this sensitive data would be encrypted and protected!")
        else:
            print("\n[+] Patient data and model protected by SGX enclave")
            print("[+] Sensitive information encrypted in memory")
        
        print(f"\n[+] Inference completed in {mode} mode")
        return True

def run_test():
    """Test the inference component."""
    print("Testing Healthcare Inference Component...")
    
    service = HealthcareInference()
    
    # Test model loading
    if not service.load_model():
        print("[!] Test failed: Cannot load model")
        return False
    
    # Test patient data loading
    features, label = service.load_patient_data()
    if features is None:
        print("[!] Test failed: Cannot load patient data")
        return False
    
    # Test prediction
    try:
        result = service.predict_readmission(features)
        print(f"[+] Test prediction: {result}")
        print("[+] Component test passed")
        return True
    except Exception as e:
        print(f"[!] Test failed: {e}")
        return False

def main():
    """Main inference service entry point."""
    parser = argparse.ArgumentParser(description="Healthcare ML Inference Service")
    parser.add_argument("--mode", choices=["vulnerable", "secure"], 
                       default="vulnerable", help="Execution mode")
    parser.add_argument("--test", action="store_true", 
                       help="Run component test")
    args = parser.parse_args()
    
    if args.test:
        return 0 if run_test() else 1
    
    service = HealthcareInference()
    success = service.run_inference_service(mode=args.mode)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())