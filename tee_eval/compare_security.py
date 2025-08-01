#!/usr/bin/env python3
"""
Security comparison demo: Run both baseline and SGX scenarios
Demonstrates the security difference between normal and enclave execution
Cross-platform version (Windows/Linux)
"""

import sys
import subprocess
import time

def print_banner():
    """Print demo banner."""
    print("###############################################")
    print("#                                             #")
    print("#  Healthcare ML Security Comparison Demo    #")
    print("#                                             #")
    print("###############################################")
    print()
    print("This demo compares the security of healthcare ML inference")
    print("in normal execution vs. SGX enclave protection.")
    print()
    print("Scenario: An attacker with OS-level privileges attempts to")
    print("extract sensitive model parameters and patient data during")
    print("ML inference for readmission risk prediction.")
    print()

def wait_for_user(message):
    """Wait for user input."""
    try:
        input(f"{message}")
    except KeyboardInterrupt:
        print("\n[!] Demo interrupted by user")
        sys.exit(1)

def run_demo_script(script_name, title):
    """Run a demo script and handle errors."""
    print(f"###############################################")
    print(f"{title}")
    print(f"###############################################")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Demo script failed: {e}")
        return False
    except FileNotFoundError:
        print(f"[!] Demo script not found: {script_name}")
        return False

def print_comparison_table():
    """Print security comparison summary."""
    print("###############################################")
    print("SECURITY COMPARISON SUMMARY")
    print("###############################################")
    print()
    
    # Table header
    print(f"{'Security Aspect':<25} {'Baseline':<15} {'SGX Enclave':<15}")
    print("-" * 63)
    
    # Table rows
    comparisons = [
        ("Model Protection", "EXPOSED", "ENCRYPTED"),
        ("Patient Privacy", "EXPOSED", "PROTECTED"),
        ("Memory Attacks", "SUCCESS", "BLOCKED"),
        ("Threat Level", "CRITICAL", "MINIMAL"),
        ("Trust Required", "Full OS", "Only SGX HW")
    ]
    
    for aspect, baseline, sgx in comparisons:
        print(f"{aspect:<25} {baseline:<15} {sgx:<15}")
    
    print("-" * 63)
    print()

def print_insights():
    """Print key insights and real-world impact."""
    print("KEY INSIGHTS:")
    print("✗ Baseline: Sensitive healthcare data fully exposed to memory attacks")
    print("✓ SGX: Hardware-level protection against privileged adversaries")
    print()
    print("REAL-WORLD IMPACT:")
    print("• Healthcare providers can deploy ML models in untrusted cloud environments")
    print("• Patient privacy protected even from cloud provider/OS compromise")
    print("• Model IP protection against sophisticated memory-based attacks")
    print("• Compliance with healthcare privacy regulations (HIPAA, GDPR)")
    print()

def main():
    """Main comparison demo."""
    # Print banner
    print_banner()
    
    # Wait for user to start baseline demo
    wait_for_user("Press Enter to start the baseline (vulnerable) demo...")
    print()
    
    # Run baseline demo
    baseline_success = run_demo_script("run_baseline.py", "PART 1: BASELINE EXECUTION (VULNERABLE)")
    
    if not baseline_success:
        print("[!] Baseline demo failed. Continuing with enclave demo...")
    
    print("\n")
    wait_for_user("Press Enter to start the SGX enclave (protected) demo...")
    print()
    
    # Run enclave demo
    enclave_success = run_demo_script("run_enclave.py", "PART 2: SGX ENCLAVE EXECUTION (PROTECTED)")
    
    if not enclave_success:
        print("[!] SGX enclave demo failed.")
    
    print("\n")
    
    # Print comparison summary
    print_comparison_table()
    print_insights()
    
    print("###############################################")
    print("Demo completed successfully!")
    print("###############################################")
    
    return 0

if __name__ == "__main__":
    exit(main())