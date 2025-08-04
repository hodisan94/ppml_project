# Gramine SGX Setup Guide

Your system shows: ✅ SGX hardware but ❌ Gramine tools not working

## Problem Diagnosis
Your SGX demo is running in **SIMULATION MODE** instead of real SGX because Gramine is not properly installed.

## Quick Fix - Install Gramine Properly

### Step 1: Add Gramine Repository
```bash
# Add Gramine signing key and repository
sudo curl -fsSLo /usr/share/keyrings/gramine-keyring.gpg \
  https://packages.gramineproject.io/gramine-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/gramine-keyring.gpg] \
  https://packages.gramineproject.io/ $(lsb_release -sc) main" | \
  sudo tee /etc/apt/sources.list.d/gramine.list
```

### Step 2: Add Intel SGX Repository  
```bash
# Add Intel SGX signing key and repository
sudo curl -fsSLo /usr/share/keyrings/intel-sgx-deb.asc \
  https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-sgx-deb.asc] \
  https://download.01.org/intel-sgx/sgx_repo/ubuntu $(lsb_release -sc) main" | \
  sudo tee /etc/apt/sources.list.d/intel-sgx.list
```

### Step 3: Install Everything
```bash
# Update package lists
sudo apt update

# Install Gramine
sudo apt install -y gramine

# Install SGX runtime libraries  
sudo apt install -y \
  libsgx-launch \
  libsgx-urts \
  libsgx-epid \
  libsgx-quote-ex \
  sgx-aesm-service \
  libsgx-aesm-launch-plugin \
  libsgx-aesm-epid-plugin \
  libsgx-dcap-ql \
  libsgx-dcap-quote-verify \
  libsgx-dcap-default-qpl
```

### Step 4: Start SGX Service
```bash
# Start and enable AESM service
sudo systemctl start aesmd
sudo systemctl enable aesmd
sudo systemctl status aesmd
```

### Step 5: Test Installation
```bash
# These should all work now:
gramine-manifest --help
gramine-sgx --help
gramine-sgx-sign --help
is-sgx-available
```

### Step 6: Generate SGX Signing Key (if needed)
```bash
# Only if you haven't done this already
gramine-sgx-gen-private-key
```

## Verification Commands

Run these to check if everything is working:

```bash
# Check SGX hardware
ls -la /dev/sgx* /dev/isgx

# Check AESM service
sudo systemctl status aesmd

# Check Gramine installation
which gramine-sgx
gramine-sgx --help

# Check SGX compatibility
is-sgx-available

# Check for SGX libraries
ldconfig -p | grep sgx
```

## What You Should See After Fix

After proper installation, your diagnostic should show:
- ✅ SGX hardware detected
- ✅ Gramine-SGX tools available
- ✅ SGX manifest template found  
- ✅ Overall SGX readiness: READY

## If Still Having Issues

Try the diagnostic script:
```bash
python3 tee_eval/sgx_diagnostics.py
```

Common issues:
1. **Wrong Ubuntu version**: Make sure `$(lsb_release -sc)` matches your Ubuntu version
2. **Permission issues**: Add user to sgx group: `sudo usermod -a -G sgx_prv $USER`
3. **AESM not running**: `sudo systemctl restart aesmd`
4. **Old kernel**: You need kernel 5.11+ for built-in SGX support

## Expected Demo Behavior After Fix

Once Gramine is properly installed, your SGX demo should show:
- ✅ VERIFICATION SUCCESSFUL  
- Real SGX enclave with actual PID
- Memory attacks BLOCKED (not simulated)
- Clear difference between vulnerable and protected processes

The key difference is it will run **real SGX enclave** instead of simulation mode.