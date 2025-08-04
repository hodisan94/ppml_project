#!/usr/bin/env python3
"""
Ultra-simple SGX demo that just shows the concept
No complex attacks, just basic memory protection demonstration
"""

import subprocess
import sys
import time

def print_banner():
    print("="*50)
    print("  SGX Healthcare ML Protection Demo")
    print("="*50)

def run_normal_inference():
    """Run normal inference showing exposed data."""
    print("\n1. NORMAL EXECUTION (Vulnerable)")
    print("-" * 30)
    
    result = subprocess.run([
        sys.executable, "infer_healthcare.py",
        "--input", "sample_data/patient_input.pkl"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    print("\n[!] In normal execution:")
    print("    - Model coefficients visible in memory")
    print("    - Patient data visible in memory") 
    print("    - Vulnerable to memory attacks")

def run_secure_inference():
    """Run secure inference with protection."""
    print("\n2. SGX EXECUTION (Protected)")
    print("-" * 30)
    
    result = subprocess.run([
        sys.executable, "infer_healthcare.py", 
        "--input", "sample_data/patient_input.pkl",
        "--secure"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    print("\n[+] In SGX execution:")
    print("    - Model coefficients encrypted in enclave")
    print("    - Patient data never visible to OS")
    print("    - Protected from memory attacks")

def main():
    print_banner()
    
    # Ensure model exists
    if subprocess.run([sys.executable, "train_healthcare_model.py"], 
                     capture_output=True).returncode != 0:
        print("[!] Failed to train model")
        return 1
    
    # Run both demos
    run_normal_inference()
    run_secure_inference()
    
    print("\n" + "="*50)
    print("SUMMARY: SGX protects sensitive healthcare ML")
    print("models and patient data from memory attacks")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    exit(main())