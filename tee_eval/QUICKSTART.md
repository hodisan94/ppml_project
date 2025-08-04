# SGX Healthcare Demo - Quick Start

## Setup (Run Once)
```bash
python setup.py              # Install deps and check environment
python test_demo.py          # Verify everything works
```

## Full Demo
```bash
python compare_security.py   # Run complete security comparison
```

## Individual Components
```bash
python train_healthcare_model.py   # Train the model
python infer_healthcare.py         # Test inference
python run_baseline.py             # Vulnerable execution
python run_enclave.py              # SGX protected execution
```

## Files Created
- `healthcare_model.pkl` - Trained ML model
- `sample_data/patient_input.pkl` - Sample patient data

## Expected Output
The demo will show:
1. **Baseline**: Memory attacks succeed, sensitive data exposed
2. **SGX**: Memory attacks blocked, data protected

## Requirements
- Python 3.x with pip
- Healthcare dataset in `../data/` 
- Optional: SGX hardware + Gramine for real enclave testing