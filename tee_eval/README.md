# Healthcare ML Model SGX Protection Demo

## Purpose
Demonstrates how Intel SGX enclaves can protect sensitive healthcare ML models and patient data from memory-based attacks. This demo shows the difference between running ML inference in a normal process (vulnerable to memory access pattern attacks) versus running it inside an SGX enclave (protected).

## Attack Scenario
The demo simulates a **memory access pattern attack** where an adversary with OS-level privileges attempts to:
1. Monitor page access patterns during ML inference
2. Extract sensitive model parameters through memory analysis
3. Infer sensitive patient information from memory access traces

This represents a realistic threat model where cloud providers or malicious OS components try to extract proprietary healthcare models and private patient data.

## Structure
```
tee_eval/
├── README.md
├── requirements.txt
├── train_healthcare_model.py      # Train model on healthcare data
├── infer_healthcare.py            # ML inference service
├── attack_memory_pattern.py       # Simulate memory access attack
├── run_baseline.sh               # Normal execution (vulnerable)
├── run_enclave.sh               # SGX-protected execution
├── compare_security.sh          # Run both and compare results
├── sample_data/
│   └── patient_input.pkl         # Sample patient data for inference
└── gramine/
    ├── infer.manifest.template   # Gramine manifest for SGX
    ├── key.pem                   # SGX signing key
    └── infer.manifest.sgx.signed # Signed manifest (generated)
```

## Healthcare Model Details
- **Dataset**: Healthcare admission/readmission data
- **Features**: Age, gender, billing amount, blood type, medical condition, admission type
- **Target**: Readmission risk prediction
- **Model**: Logistic Regression (for simplicity and interpretability)
- **Threat**: Model stealing + patient privacy violation

## Setup

```bash
pip install -r requirements.txt
mkdir -p sample_data gramine
```

## Demo Flow

### 1. Baseline (Vulnerable)
```bash
python run_baseline.py    # Cross-platform
# OR: ./run_baseline.sh   # Linux only
```
Shows how memory access patterns can leak:
- Model coefficients during inference
- Patient feature values
- Prediction confidences

### 2. SGX Protected  
```bash
python run_enclave.py     # Cross-platform
# OR: ./run_enclave.sh    # Linux only
```
Shows how SGX enclave protects:
- Model parameters are encrypted in memory
- Patient data never visible to OS
- Attack tools cannot extract sensitive information

### 3. Security Comparison
```bash
python compare_security.py   # Cross-platform
# OR: ./compare_security.sh  # Linux only
```
Runs both scenarios and demonstrates the security difference.

## Technical Details

### Memory Access Pattern Attack
The attack simulates monitoring memory access patterns similar to research papers like:
- "SGAxe: How SGX Fails in Practice" 
- "CacheOut: Leaking Data on Intel CPUs via Cache Evictions"
- Recent work on activation function side-channels

Instead of actual hardware side-channels (which require SGX hardware), we simulate the attack by:
1. Monitoring process memory regions during inference
2. Analyzing memory access patterns correlated with input features
3. Extracting model parameters through statistical analysis

### SGX Protection
The Gramine manifest ensures:
- Model file is encrypted and only accessible inside enclave
- Patient input is processed entirely within enclave
- Only final prediction result is returned to untrusted host
- Memory access patterns are hidden from the OS

## Requirements for Full SGX
- Intel SGX-enabled CPU
- SGX driver and AESM service installed  
- Gramine framework (`gramine-sgx`)
- Python 3.x with pip

**Note**: This demo includes simulation mode for testing without SGX hardware.

## Quick Test (Cross-Platform)

### Windows (Local Testing)
```cmd
cd tee_eval
pip install -r requirements.txt
python train_healthcare_model.py      # Train the model first
python compare_security.py            # Run full demo
```

### Ubuntu/Linux (SGX Testing)
```bash
cd tee_eval
pip install -r requirements.txt
python train_healthcare_model.py      # Train the model first
python compare_security.py            # Run full demo (with real SGX if available)
# OR use the bash versions:
# ./compare_security.sh
```

## Security Guarantees
- **Baseline**: Model and patient data fully exposed to memory attacks
- **SGX**: Model confidentiality and patient privacy protected even against malicious OS

## Educational Value
This demo illustrates:
1. Real-world ML privacy threats in cloud environments
2. How hardware-based TEEs provide strong security guarantees  
3. Practical deployment of SGX for healthcare applications
4. Trade-offs between performance and security in confidential computing