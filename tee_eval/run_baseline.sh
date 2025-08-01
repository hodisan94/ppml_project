#!/bin/bash
# Baseline demo: Run healthcare ML inference without SGX protection
# Shows vulnerability to memory-based attacks

set -e

echo "============================================="
echo "Healthcare ML Demo - BASELINE (Vulnerable)"
echo "============================================="

# Ensure model is trained
if [ ! -f "healthcare_model.pkl" ]; then
    echo "[+] Training healthcare model..."
    python3 train_healthcare_model.py
fi

# Ensure sample data exists
if [ ! -f "sample_data/patient_input.pkl" ]; then
    echo "[!] Sample patient data not found!"
    echo "[!] Run 'python3 train_healthcare_model.py' first"
    exit 1
fi

# Show what we're protecting
echo ""
echo "[+] Sample patient data being processed:"
python3 -c "
import pickle
with open('sample_data/patient_input.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Description: {data[\"description\"]}')
print(f'True label: {data[\"true_label\"]}')
print(f'First 5 features: {data[\"features\"][:5]}')
"

echo ""
echo "==============================================="
echo "BASELINE EXECUTION (NO SGX PROTECTION)"
echo "==============================================="

# Run inference in background
echo "[+] Starting ML inference service..."
python3 infer_healthcare.py --input sample_data/patient_input.pkl --verbose &
INFERENCE_PID=$!

# Give the process time to start and load model
sleep 2

echo ""
echo "==============================================="
echo "SIMULATING MEMORY-BASED ATTACKS"
echo "==============================================="

# Simulate memory access pattern attack
echo "[*] Launching memory access pattern attack..."
python3 attack_memory_pattern.py $INFERENCE_PID pattern &
ATTACK1_PID=$!

sleep 1

# Simulate memory dump attack  
echo "[*] Launching memory dump attack..."
python3 attack_memory_pattern.py $INFERENCE_PID dump &
ATTACK2_PID=$!

# Wait for attacks to complete
wait $ATTACK1_PID 2>/dev/null || true
wait $ATTACK2_PID 2>/dev/null || true

# Clean up inference process
kill $INFERENCE_PID 2>/dev/null || true
wait $INFERENCE_PID 2>/dev/null || true

echo ""
echo "==============================================="
echo "BASELINE DEMO SUMMARY"
echo "==============================================="
echo "[!] SECURITY STATUS: VULNERABLE"
echo "[!] Model parameters: EXPOSED to memory attacks"
echo "[!] Patient data: EXPOSED to memory attacks" 
echo "[!] Threat level: CRITICAL"
echo ""
echo "[+] Next: Run './run_enclave.sh' to see SGX protection"
echo "==============================================="