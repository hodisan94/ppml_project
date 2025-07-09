# Deployment Checklist for Ubuntu Azure SGX

## Pre-deployment Verification

### ✅ 1. File Structure
- [ ] All files exist in dp1_tee/
- [ ] Parent requirements.txt exists (✓ confirmed)
- [ ] data/clients/*.csv files exist (✓ confirmed)
- [ ] scripts/ directory with executable permissions
- [ ] manifests/ directory with template files

### ✅ 2. Dependencies
- [ ] Parent requirements.txt has all ML dependencies
- [ ] requirements_sgx.txt references parent correctly
- [ ] Python 3.8+ available
- [ ] pip3 available

### ✅ 3. Azure VM Requirements
- [ ] SGX-capable Azure VM (DCsv2, DCsv3, DCdsv3 series)
- [ ] Ubuntu 18.04 or 20.04 LTS
- [ ] Root/sudo access
- [ ] Internet connectivity for package installation

## Deployment Steps

### Step 1: Basic Setup
```bash
# Navigate to project
cd /path/to/ppml_project/dp1_tee

# Make scripts executable
chmod +x scripts/*.sh

# Install system dependencies
sudo bash scripts/setup_sgx.sh
```

### Step 2: Verify Installation
```bash
# Check SGX devices
ls -la /dev/sgx*

# Check Gramine
gramine-sgx --help

# Check Python dependencies
pip3 list | grep -E "(flwr|tensorflow|numpy)"

# Check SGX services
sudo systemctl status aesmd
```

### Step 3: Network Configuration
```bash
# Open port 8086 for Flower communication
sudo ufw allow 8086

# Or use SSH tunneling (recommended for security)
# ssh -L 8086:localhost:8086 user@azure-vm
```

### Step 4: Test Basic Functionality
```bash
# Test without TEE first
python3 dp_server_tee.py true false &
python3 dp_client_tee.py 1 true 1.0 false

# Kill processes
pkill -f dp_server_tee
pkill -f dp_client_tee
```

### Step 5: Test with TEE
```bash
# Test with TEE enabled
python3 dp_server_tee.py true true &
python3 dp_client_tee.py 1 true 1.0 true

# Kill processes
pkill -f dp_server_tee
pkill -f dp_client_tee
```

### Step 6: Run Full Experiments
```bash
# TEE comparison experiment
python3 dp_main_tee.py --tee-comparison --experiment-name azure_sgx_test

# Check results
ls -la results/
```

## Troubleshooting

### SGX Issues
- **No SGX devices**: Check VM type, enable SGX in Azure portal
- **AESMD not running**: `sudo systemctl start aesmd`
- **Permission denied**: Check user is in sgx group

### Gramine Issues
- **Command not found**: Reinstall Gramine or check PATH
- **Manifest errors**: Check manifest template syntax
- **Memory issues**: Increase enclave heap size

### Python Issues
- **Import errors**: Check all dependencies installed
- **TensorFlow GPU**: May need to disable GPU for SGX
- **Flower connection**: Check port 8086 accessibility

### Network Issues
- **Connection refused**: Check firewall and port 8086
- **Timeout**: Increase client connection timeout
- **SSH tunneling**: Use for remote access

## Performance Expectations

### Without TEE
- Training time: ~2-3 minutes per round
- Memory usage: ~500MB per client
- CPU usage: Moderate

### With TEE (SGX)
- Training time: ~3-5 minutes per round (30-50% slower)
- Memory usage: ~1GB per client (enclave overhead)
- CPU usage: Higher (encryption/attestation)

## Security Notes

### For Testing
- Debug mode enabled
- Local attestation only
- No remote verification

### For Production
- Disable debug mode
- Enable remote attestation
- Use secure communication
- Implement proper key management 