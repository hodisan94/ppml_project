#!/bin/bash
# Security comparison demo: Run both baseline and SGX scenarios
# Demonstrates the security difference between normal and enclave execution

set -e

echo "###############################################"
echo "#                                             #"
echo "#  Healthcare ML Security Comparison Demo    #"
echo "#                                             #"  
echo "###############################################"
echo ""
echo "This demo compares the security of healthcare ML inference"
echo "in normal execution vs. SGX enclave protection."
echo ""
echo "Scenario: An attacker with OS-level privileges attempts to"
echo "extract sensitive model parameters and patient data during"
echo "ML inference for readmission risk prediction."
echo ""

read -p "Press Enter to start the baseline (vulnerable) demo..."

echo ""
echo "###############################################"
echo "PART 1: BASELINE EXECUTION (VULNERABLE)"
echo "###############################################"

./run_baseline.sh

echo ""
echo ""
read -p "Press Enter to start the SGX enclave (protected) demo..."

echo ""
echo "###############################################"
echo "PART 2: SGX ENCLAVE EXECUTION (PROTECTED)"  
echo "###############################################"

./run_enclave.sh

echo ""
echo ""
echo "###############################################"
echo "SECURITY COMPARISON SUMMARY"
echo "###############################################"
echo ""
printf "%-25s %-15s %-15s\n" "Security Aspect" "Baseline" "SGX Enclave"
echo "---------------------------------------------------------------"
printf "%-25s %-15s %-15s\n" "Model Protection" "EXPOSED" "ENCRYPTED"
printf "%-25s %-15s %-15s\n" "Patient Privacy" "EXPOSED" "PROTECTED"  
printf "%-25s %-15s %-15s\n" "Memory Attacks" "SUCCESS" "BLOCKED"
printf "%-25s %-15s %-15s\n" "Threat Level" "CRITICAL" "MINIMAL"
printf "%-25s %-15s %-15s\n" "Trust Required" "Full OS" "Only SGX HW"
echo "---------------------------------------------------------------"
echo ""
echo "KEY INSIGHTS:"
echo "✗ Baseline: Sensitive healthcare data fully exposed to memory attacks"
echo "✓ SGX: Hardware-level protection against privileged adversaries"
echo ""
echo "REAL-WORLD IMPACT:"
echo "• Healthcare providers can deploy ML models in untrusted cloud environments"
echo "• Patient privacy protected even from cloud provider/OS compromise"
echo "• Model IP protection against sophisticated memory-based attacks"
echo "• Compliance with healthcare privacy regulations (HIPAA, GDPR)"
echo ""
echo "###############################################"
echo "Demo completed successfully!"
echo "###############################################"