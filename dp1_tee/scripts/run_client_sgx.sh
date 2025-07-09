#!/bin/bash
# Run DP client with SGX support

set -e

# Parse arguments
CLIENT_ID=${1:-1}
USE_DP=${2:-true}
NOISE_MULTIPLIER=${3:-1.0}
USE_TEE=${4:-true}

echo "Starting DP client $CLIENT_ID with TEE support..."
echo "DP: $USE_DP, Noise: $NOISE_MULTIPLIER, TEE: $USE_TEE"

if [ "$USE_TEE" = "true" ]; then
    echo "Running client $CLIENT_ID in SGX enclave..."
    
    # Generate manifest for client
    python3 -c "
import sys
import os
manifest_template = open('manifests/python.manifest.template').read()
manifest = manifest_template.replace('{{ python_executable }}', sys.executable)
manifest = manifest.replace('{{ current_dir }}', os.getcwd())
manifest = manifest.replace('{{ log_level }}', 'error')
manifest = manifest.replace('{{ sgx_debug }}', 'false')
manifest = manifest.replace('{{ enclave_size }}', '1G')
manifest = manifest.replace('{{ thread_num }}', '4')
manifest = manifest.replace('{{ attestation_type }}', 'dcap')
open(f'client_{CLIENT_ID}.manifest', 'w').write(manifest)
"
    
    # Run with Gramine
    gramine-sgx python dp_client_tee.py $CLIENT_ID $USE_DP $NOISE_MULTIPLIER $USE_TEE
else
    echo "Running client $CLIENT_ID without SGX..."
    python dp_client_tee.py $CLIENT_ID $USE_DP $NOISE_MULTIPLIER $USE_TEE
fi
```

```

Finally, let me create the SGX-specific requirements file:

```text:dp1_tee/requirements_sgx.txt
# Additional requirements for SGX/TEE support
# Include all base requirements from parent directory
-r ../requirements.txt

# SGX-specific packages (if available)
# These may need to be installed manually depending on the SGX setup
intel-sgx-driver>=1.0
intel-sgx-sdk>=2.0

# Cryptography for secure communication
cryptography>=3.4.8
pycryptodome>=3.15.0

# Additional monitoring and logging
psutil>=5.8.0
structlog>=21.1.0
```

Now let me create a README for the dp1_tee folder:

```markdown:dp1_tee/README_TEE.md
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
