#!/usr/bin/env python3
"""
Baseline demo: Run healthcare ML inference without SGX protection
Shows vulnerability to memory-based attacks
Cross-platform version (Windows/Linux)
"""

import os
import sys
import time
import subprocess
import pickle
import platform

def print_header(title):
    """Print formatted header."""
    print("=" * len(title))
    print(title)
    print("=" * len(title))

def check_prerequisites():
    """Check if required files exist."""
    if not os.path.exists("healthcare_model.pkl"):
        print("[+] Training healthcare model...")
        result = subprocess.run([sys.executable, "train_healthcare_model.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[!] Training failed: {result.stderr}")
            return False
    
    if not os.path.exists("sample_data/patient_input.pkl"):
        print("[!] Sample patient data not found!")
        print("[!] Run 'python train_healthcare_model.py' first")
        return False
    
    return True

def show_patient_data():
    """Display sample patient data information."""
    print("\n[+] Sample patient data being processed:")
    try:
        with open('sample_data/patient_input.pkl', 'rb') as f:
            data = pickle.load(f)
        print(f'Description: {data["description"]}')
        print(f'True label: {data["true_label"]}')
        print(f'First 5 features: {list(data["features"][:5])}')
    except Exception as e:
        print(f"[!] Error reading patient data: {e}")

def run_inference():
    """Run ML inference in background and return process."""
    print("[+] Starting ML inference service...")
    
    # Platform-specific process handling
    if platform.system() == "Windows":
        # Windows
        process = subprocess.Popen([
            sys.executable, "infer_healthcare.py", 
            "--input", "sample_data/patient_input.pkl", 
            "--verbose"
        ], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        # Linux/Unix
        process = subprocess.Popen([
            sys.executable, "infer_healthcare.py",
            "--input", "sample_data/patient_input.pkl", 
            "--verbose"
        ])
    
    # Give process time to start and load model
    time.sleep(3)
    return process

def run_attacks(pid):
    """Run simulated attacks against the process."""
    print("\n===============================================")
    print("SIMULATING MEMORY-BASED ATTACKS")
    print("===============================================")
    
    attacks = []
    
    # Memory access pattern attack
    print("[*] Launching memory access pattern attack...")
    attack1 = subprocess.Popen([
        sys.executable, "attack_memory_pattern.py", str(pid), "pattern"
    ])
    attacks.append(attack1)
    
    time.sleep(1)
    
    # Memory dump attack
    print("[*] Launching memory dump attack...")
    attack2 = subprocess.Popen([
        sys.executable, "attack_memory_pattern.py", str(pid), "dump"
    ])
    attacks.append(attack2)
    
    # Wait for attacks to complete
    for attack in attacks:
        try:
            attack.wait(timeout=10)
        except subprocess.TimeoutExpired:
            attack.terminate()

def cleanup_process(process):
    """Clean up the inference process."""
    try:
        if platform.system() == "Windows":
            process.terminate()
        else:
            process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if platform.system() == "Windows":
            process.kill()
        else:
            process.kill()
    except Exception:
        pass

def main():
    """Main baseline demo."""
    print_header("Healthcare ML Demo - BASELINE (Vulnerable)")
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Show what we're protecting
    show_patient_data()
    
    print("\n===============================================")
    print("BASELINE EXECUTION (NO SGX PROTECTION)")
    print("===============================================")
    
    # Run inference
    inference_process = None
    try:
        inference_process = run_inference()
        
        # Run attacks
        run_attacks(inference_process.pid)
        
    finally:
        # Cleanup
        if inference_process:
            cleanup_process(inference_process)
    
    print("\n===============================================")
    print("BASELINE DEMO SUMMARY")
    print("===============================================")
    print("[!] SECURITY STATUS: VULNERABLE")
    print("[!] Model parameters: EXPOSED to memory attacks")
    print("[!] Patient data: EXPOSED to memory attacks")
    print("[!] Threat level: CRITICAL")
    print("")
    print("[+] Next: Run 'python run_enclave.py' to see SGX protection")
    print("===============================================")
    
    return 0

if __name__ == "__main__":
    exit(main())