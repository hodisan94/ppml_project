#!/bin/bash
# Setup script for SGX environment on Ubuntu Azure

set -e

echo "Setting up SGX environment for dp1_tee on Ubuntu Azure..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Running as root - good for system setup"
else
   echo "Not running as root - some operations may need sudo"
fi

# Check if SGX is available
if [ ! -e /dev/sgx_enclave ] && [ ! -e /dev/sgx/enclave ]; then
    echo "WARNING: SGX device files not found. SGX may not be available."
    echo "Checking for SGX support..."
    
    # Check CPUID for SGX support
    if command -v cpuid &> /dev/null; then
        cpuid -l 0x7 -s 0x0 | grep SGX && echo "SGX supported by CPU" || echo "SGX not supported by CPU"
    fi
    
    # Check kernel messages
    dmesg | grep -i sgx && echo "SGX messages found in kernel log" || echo "No SGX messages in kernel log"
fi

# Update package list
echo "Updating package list..."
apt update

# Install basic dependencies
echo "Installing basic dependencies..."
apt install -y curl wget gnupg software-properties-common build-essential

# Install Intel SGX driver and SDK (for Azure with SGX)
echo "Installing Intel SGX components..."

# Add Intel's repository
wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -
echo "deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/intel-sgx.list

apt update

# Install SGX driver and SDK
apt install -y libsgx-launch libsgx-urts sgx-aesm-service libsgx-uae-service
apt install -y libsgx-dcap-ql libsgx-dcap-default-qpl

# Start SGX services
systemctl enable aesmd
systemctl start aesmd

echo "SGX driver and services installed."

# Install Gramine if not present
if ! command -v gramine-sgx &> /dev/null; then
    echo "Installing Gramine..."
    
    # Add Gramine repository
    curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg https://packages.gramineproject.io/gramine-keyring.gpg
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] https://packages.gramineproject.io/ stable main' > /etc/apt/sources.list.d/gramine.list

    # Update and install Gramine
    apt update
    apt install -y gramine

    echo "Gramine installation completed."
else
    echo "Gramine already installed."
fi

# Verify SGX services
echo "Checking SGX services..."
systemctl status aesmd || echo "AESMD service not running"

# Check SGX devices
echo "Checking SGX devices..."
ls -la /dev/sgx* || echo "No SGX devices found"

# Set up Python dependencies for SGX
echo "Installing Python dependencies..."
cd "$(dirname "$0")/.."  # Go to dp1_tee directory
pip3 install -r requirements_sgx.txt

# Create results directory
mkdir -p results

# Set permissions for scripts
chmod +x scripts/*.sh

# Set up environment variables
echo "Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'
# SGX Environment Variables
export USE_TEE=true
export ENCLAVE_TYPE=gramine
export ENCLAVE_HEAP_SIZE=2G
export ATTESTATION_TYPE=dcap
export ENABLE_SECURE_AGGREGATION=true
EOF

echo "SGX environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Source your bashrc: source ~/.bashrc"
echo "2. Verify SGX: ls /dev/sgx*"
echo "3. Test Gramine: gramine-sgx --help"
echo "4. Run experiments: python dp_main_tee.py --use-tee"
echo ""
echo "If SGX devices are not found, you may need to:"
echo "- Enable SGX in BIOS/UEFI"
echo "- Install SGX driver manually"
echo "- Check Azure VM SGX support" 