# dp1_tee - Differential Privacy with Trusted Execution Environment Support

This directory contains the enhanced version of dp1 with Intel SGX (Trusted Execution Environment) support using Gramine.

## Features

- **Original dp1 functionality**: All differential privacy and federated learning features
- **SGX Enclave support**: Run ML training within Intel SGX enclaves
- **Secure aggregation**: Protected weight aggregation on the server
- **Remote attestation**: Verify enclave integrity (configurable)
- **TEE-aware metrics**: Track TEE coverage and enclave consistency

## Quick Start

### 1. Setup SGX Environment
```bash
cd dp1_tee
sudo bash scripts/setup_sgx.sh
```

### 2. Basic Usage (without TEE)
```bash
# Start server
python dp_server_tee.py true false

# Start clients (in separate terminals)
python dp_client_tee.py 1 true 1.0 false
python dp_client_tee.py 2 true 1.0 false
# ... continue for clients 3, 4, 5
```

### 3. With TEE/SGX Enabled
```bash
# Start server with TEE
python dp_server_tee.py true true

# Start clients with TEE (in separate terminals)
python dp_client_tee.py 1 true 1.0 true
python dp_client_tee.py 2 true 1.0 true
# ... continue for clients 3, 4, 5
```

### 4. Automated Experiments
```bash
# Compare TEE vs non-TEE
python dp_main_tee.py --tee-comparison

# Single experiment with TEE
python dp_main_tee.py --use-tee --experiment-name my_tee_experiment

# DP comparison with TEE
python dp_main_tee.py --run-comparison --use-tee
```

## Configuration

### TEE Configuration
Edit `tee_config.py` or use environment variables:

```bash
export USE_TEE=true
export ENCLAVE_TYPE=gramine
export ENCLAVE_HEAP_SIZE=2G
export ATTESTATION_TYPE=dcap
export ENABLE_SECURE_AGGREGATION=true
```

### SGX Scripts
- `scripts/setup_sgx.sh`: Setup SGX environment and dependencies
- `scripts/run_server_sgx.sh`: Run server with SGX
- `scripts/run_client_sgx.sh`: Run client with SGX

## File Structure

```
dp1_tee/
├── tee_config.py              # TEE configuration classes
├── sgx_utils.py               # SGX enclave management utilities
├── dp_utils_tee.py            # Enhanced DP utilities with TEE
├── dp_client_tee.py           # Enhanced client with TEE support
├── dp_server_tee.py           # Enhanced server with TEE support
├── dp_main_tee.py             # Enhanced orchestration with TEE
├── manifests/                 # Gramine manifest templates
│   └── python.manifest.template
├── scripts/                   # SGX setup and run scripts
│   ├── setup_sgx.sh
│   ├── run_server_sgx.sh
│   └── run_client_sgx.sh
├── requirements_sgx.txt       # Additional SGX dependencies
└── results/                   # Experiment results
```

## Arguments

### dp_client_tee.py
```bash
python dp_client_tee.py <client_id> [use_dp] [noise_multiplier] [use_tee]
```
- `client_id`: Client ID (1-5)
- `use_dp`: Enable differential privacy (true/false)
- `noise_multiplier`: DP noise level (float)
- `use_tee`: Enable TEE/SGX (true/false)

### dp_server_tee.py
```bash
python dp_server_tee
``` 