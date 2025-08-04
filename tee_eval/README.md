# SGX Healthcare ML Security Demo

**Single Script Demo** - Shows REAL memory attacks vs SGX protection

## What This Demo Does

1. **REAL Memory Extraction**: Actually reads `/proc/{pid}/mem` to extract model coefficients from a running ML process
2. **SGX Protection**: Shows how Intel SGX blocks these same attacks  
3. **Concrete Evidence**: Displays actual extracted float values that could be model parameters

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete demo (single script)
python sgx_demo.py
```

## What You'll See

### Phase 1: Vulnerable Execution
- ✗ **DETAILED ATTACK SUCCESS**: Shows actual extracted float values
- ✗ **Memory Inspection**: Scans 400+ memory regions, extracts model coefficients  
- ✗ **Data Leaked**: Healthcare model weights + patient data exposed
- ✗ **Impact**: HIPAA/GDPR violations, model IP theft possible

### Phase 2: SGX Protected Execution  
- ✓ **Memory Access Denied**: SGX hardware blocks `/proc/{pid}/mem`
- ✓ **Data Encrypted**: Model runs inside encrypted enclave memory
- ✓ **Real Protection**: Same attacks fail completely with hardware enforcement

## Technical Details

- **Model**: Logistic Regression trained on real healthcare data
- **Attack**: Direct memory inspection to extract float arrays (model coefficients)
- **Protection**: Intel SGX hardware encryption prevents memory access
- **Evidence**: Shows actual extracted values vs blocked attempts

## Files

- `sgx_demo.py` - Complete demo (only file you need)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Requirements

- Linux with SGX hardware (for real SGX demo)
- Python 3.7+
- Root privileges (for memory access attacks)
- Gramine SGX framework installed

## SGX Setup

The demo uses **Gramine** to run Python in SGX enclaves:

```bash
# The gramine/ directory contains:
# - sgx_inference.manifest.template (SGX enclave configuration)

# When you run sgx_demo.py, it will:
# 1. Generate manifest: gramine-manifest gramine/sgx_inference.manifest.template
# 2. Sign for SGX: gramine-sgx-sign --manifest sgx_inference.manifest  
# 3. Run enclave: gramine-sgx sgx_inference
```

Without SGX hardware, demo runs in simulation mode but still shows the attack concepts.