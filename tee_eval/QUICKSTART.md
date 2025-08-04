# SGX Healthcare Demo - Quick Start

## Ultra-Simple Demo
```bash
python simple_demo.py        # Basic concept demonstration
```

## Full Demo with Real Attacks  
```bash
python demo_runner.py        # Complete demo with memory attacks
```

## Setup (if needed)
```bash
python setup.py              # Install deps and check environment
```

## Individual Components
```bash
python train_healthcare_model.py   # Train the model
python infer_healthcare.py         # Test inference
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