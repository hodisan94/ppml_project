# SGX Healthcare ML Demo - Clean Architecture

## 🎯 **Purpose**
Demonstrate how Intel SGX protects healthcare ML models from memory-based attacks that succeed against normal processes.

## 📁 **File Structure (Clean)**
```
tee_eval/
├── README.md                    # This file
├── requirements.txt             # Dependencies
│
├── 1_setup.py                   # One-time environment setup
├── 2_train_model.py             # Train healthcare model
├── 3_run_demo.py                # SINGLE MAIN DEMO RUNNER
│
├── components/                  # Individual testable components
│   ├── inference.py             # ML inference service
│   ├── memory_attack.py         # Memory attack implementation
│   └── attack_analyzer.py       # Attack result analysis
│
├── data/                       # Generated data
│   ├── healthcare_model.pkl    # Trained model
│   └── patient_sample.pkl      # Sample patient data
│
└── gramine/                    # SGX configuration
    └── inference.manifest      # Gramine manifest
```

## 🔥 **Single Demo Runner Usage**
```bash
# Setup (once)
python 1_setup.py

# Train model (once) 
python 2_train_model.py

# Run complete demo
python 3_run_demo.py
```

## 🧩 **Component Testing**
```bash
# Test individual components
python components/inference.py --test
python components/memory_attack.py --test  
python components/attack_analyzer.py --test
```

## 🎭 **Demo Flow**
1. **Baseline**: Normal ML inference → Memory attacks succeed
2. **SGX**: ML inference in enclave → Memory attacks blocked  
3. **Comparison**: Side-by-side security analysis

## 🔬 **Attack Implementation**
- **Real memory inspection**: Uses `/proc/{pid}/mem` when possible
- **Page fault monitoring**: Tracks memory access patterns  
- **Graceful fallback**: Simulation when hardware/privileges unavailable
- **Educational focus**: Shows concepts rather than weaponizing

## 🛡️ **SGX Protection**
- **Gramine framework**: Industry-standard SGX runtime
- **Hardware/simulation**: Auto-detects SGX availability
- **Memory encryption**: Model parameters protected in enclave
- **Attack resistance**: Demonstrates protection effectiveness

## 📊 **Expected Results**
- **Vulnerable execution**: Attacks extract model coefficients and patient data
- **SGX execution**: Attacks blocked, sensitive data protected
- **Clear demonstration**: Healthcare privacy preserved with SGX