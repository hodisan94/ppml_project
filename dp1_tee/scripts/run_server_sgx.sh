#!/bin/bash
# Run DP server with SGX support

set -e

# Parse arguments
USE_DP=${1:-true}
USE_TEE=${2:-true}

echo "Starting DP server with TEE support..."
echo "DP: $USE_DP, TEE: $USE_TEE"

if [ "$USE_TEE" = "true" ]; then
    echo "Running server in SGX enclave..."
    
    # Generate manifest
    python3 -c "
import sys
import os
manifest_template = open('manifests/python.manifest.template').read()
manifest = manifest_template.replace('{{ python_executable }}', sys.executable)
manifest = manifest.replace('{{ current_dir }}', os.getcwd())
manifest = manifest.replace('{{ log_level }}', 'error')
manifest = manifest.replace('{{ sgx_debug }}', 'false')
manifest = manifest.replace('{{ enclave_size }}', '2G')
manifest = manifest.replace('{{ thread_num }}', '8')
manifest = manifest.replace('{{ attestation_type }}', 'dcap')
open('server.manifest', 'w').write(manifest)
"
    
    # Run with Gramine
    gramine-sgx python dp_server_tee.py $USE_DP $USE_TEE
else
    echo "Running server without SGX..."
    python dp_server_tee.py $USE_DP $USE_TEE
fi 