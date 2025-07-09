#!/bin/bash
# Setup script for SGX environment

set -e

echo "Setting up SGX environment for dp1_tee..."

# Check if SGX is available
if [ ! -e /dev/sgx_enclave ] && [ ! -e /dev/sgx/enclave ]; then
    echo "WARNING: SGX device files not found. SGX may not be available."
fi

# Install Gramine if not present
if ! command -v gramine-sgx &> /dev/null; then
    echo "Installing Gramine..."
    
    # Add Gramine repository
    curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg https://packages.gramineproject.io/gramine-keyring.gpg
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] https://packages. 