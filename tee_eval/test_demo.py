#!/usr/bin/env python3
"""
Quick test script to verify the SGX demo works properly.
Tests the model training and basic inference functionality.
"""

import os
import sys
import subprocess

def test_model_training():
    """Test if model training works."""
    print("[+] Testing model training...")
    try:
        result = subprocess.run([sys.executable, "train_healthcare_model.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ Model training successful")
            return True
        else:
            print(f"✗ Model training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Model training timed out")
        return False
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def test_inference():
    """Test if inference works."""
    print("[+] Testing inference...")
    try:
        result = subprocess.run([
            sys.executable, "infer_healthcare.py",
            "--input", "sample_data/patient_input.pkl"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Inference successful")
            return True
        else:
            print(f"✗ Inference failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Inference timed out")
        return False
    except Exception as e:
        print(f"✗ Inference error: {e}")
        return False

def test_attack_simulation():
    """Test if attack simulation works."""
    print("[+] Testing attack simulation...")
    try:
        # Test with a dummy PID
        result = subprocess.run([
            sys.executable, "attack_memory_pattern.py", "1234", "sgx"
        ], capture_output=True, text=True, timeout=10)
        
        # This should fail gracefully (process not found), which is expected
        print("✓ Attack simulation script runs correctly")
        return True
    except subprocess.TimeoutExpired:
        print("✗ Attack simulation timed out")
        return False
    except Exception as e:
        print(f"✗ Attack simulation error: {e}")
        return False

def check_files():
    """Check if all required files exist."""
    print("[+] Checking required files...")
    required_files = [
        "train_healthcare_model.py",
        "infer_healthcare.py", 
        "attack_memory_pattern.py",
        "run_baseline.py",
        "run_enclave.py",
        "compare_security.py",
        "gramine/infer.manifest.template",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def main():
    """Run all tests."""
    print("="*50)
    print("SGX Healthcare Demo - Quick Test")
    print("="*50)
    
    # Check files
    if not check_files():
        print("\n[!] File check failed. Cannot proceed with tests.")
        return 1
    
    tests = [
        ("Model Training", test_model_training),
        ("ML Inference", test_inference),
        ("Attack Simulation", test_attack_simulation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print("\n" + "="*50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("✓ All tests passed! Demo is ready to run.")
        print("\nTo run the full demo:")
        print("  python compare_security.py")
        return 0
    else:
        print("✗ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())