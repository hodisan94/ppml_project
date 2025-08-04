#!/usr/bin/env python3
"""
SGX Healthcare Demo - Environment Setup
One-time setup for dependencies and directories
"""

import os
import sys
import subprocess
import platform

def print_header(title):
    """Print setup section header."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def install_python_dependencies():
    """Install required Python packages."""
    print("[+] Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✓ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e.stderr}")
        return False

def create_directories():
    """Create required directories."""
    print("[+] Creating directory structure...")
    
    directories = [
        "components",
        "data", 
        "gramine",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    return True

def check_sgx_environment():
    """Check SGX tools and hardware availability."""
    print("[+] Checking SGX environment...")
    
    sgx_status = {
        'platform': platform.system(),
        'gramine_tools': False,
        'sgx_hardware': False
    }
    
    # Check for Gramine tools
    try:
        result = subprocess.run(["which", "gramine-sgx"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            sgx_status['gramine_tools'] = True
            print("✓ Gramine tools found")
        else:
            print("! Gramine tools not found - will use simulation mode")
    except:
        print("! Gramine tools not available - will use simulation mode")
    
    # Check for SGX hardware  
    if sgx_status['gramine_tools']:
        try:
            result = subprocess.run(["is-sgx-available"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sgx_status['sgx_hardware'] = True
                print("✓ SGX hardware detected")
            else:
                print("! SGX hardware not available - will use simulation mode")
        except:
            print("! Cannot check SGX hardware status")
    
    # Platform-specific notes
    if sgx_status['platform'] == "Windows":
        print("! Windows detected - SGX demo will run in simulation mode")
    elif sgx_status['platform'] == "Linux":
        print("✓ Linux detected - good for SGX development")
    
    return sgx_status

def create_gramine_manifest():
    """Create basic Gramine manifest for SGX."""
    print("[+] Creating Gramine manifest...")
    
    manifest_content = '''# Gramine manifest for healthcare ML inference

loader.entrypoint = "file:{{ gramine.libos }}"
libos.entrypoint = "/usr/bin/python3"

loader.argv = [
    "python3", "../components/inference.py", "--mode", "secure"
]

loader.env.PYTHONDONTWRITEBYTECODE = "1"
loader.env.PYTHONUNBUFFERED = "1"

# SGX configuration
sgx.enclave_size = "256M"
sgx.max_threads = 4
sgx.debug = true

# File system access
fs.mounts = [
    { path = "/usr", uri = "file:/usr" },
    { path = "/lib", uri = "file:/lib" },
    { path = "/lib64", uri = "file:/lib64" },
    { path = "..", uri = "file:.." },
]

# Trusted files
sgx.trusted_files = [
    "file:{{ gramine.libos }}",
    "file:/usr/bin/python3",
    "file:../components/inference.py",
    "file:../data/healthcare_model.pkl",
    "file:../data/patient_sample.pkl",
]

# Allowed files  
sgx.allowed_files = [
    "file:/etc/hosts",
    "file:/etc/resolv.conf",
]
'''
    
    try:
        with open("gramine/inference.manifest", "w") as f:
            f.write(manifest_content)
        print("✓ Gramine manifest created")
        return True
    except Exception as e:
        print(f"✗ Failed to create manifest: {e}")
        return False

def verify_main_project_data():
    """Verify access to main project's healthcare data."""
    print("[+] Checking main project data access...")
    
    data_paths = [
        "../data/raw/healthcare_dataset.csv",
        "../data/processed/full_preprocessed.csv"
    ]
    
    data_available = False
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
            data_available = True
            break
    
    if not data_available:
        print("! Main project data not found")
        print("  Demo will create synthetic data if needed")
    
    return True

def check_system_capabilities():
    """Check system capabilities for memory attacks."""
    print("[+] Checking system attack capabilities...")
    
    capabilities = {
        'proc_mem_access': os.path.exists('/proc'),
        'psutil_available': False
    }
    
    # Check if we have /proc filesystem (Linux)
    if capabilities['proc_mem_access']:
        print("✓ /proc filesystem available for memory analysis")
    else:
        print("! /proc not available - will use simulation mode")
    
    # Check psutil
    try:
        import psutil
        capabilities['psutil_available'] = True
        print("✓ psutil available for process monitoring")
    except ImportError:
        print("! psutil not available - install with: pip install psutil")
    
    return capabilities

def main():
    """Main setup function."""
    print_header("SGX Healthcare Demo - Environment Setup")
    print("Setting up environment for healthcare ML SGX demonstration")
    
    setup_steps = [
        ("Install Python Dependencies", install_python_dependencies),
        ("Create Directories", create_directories),
        ("Check SGX Environment", check_sgx_environment), 
        ("Create Gramine Manifest", create_gramine_manifest),
        ("Verify Project Data", verify_main_project_data),
        ("Check System Capabilities", check_system_capabilities)
    ]
    
    results = {}
    for step_name, step_func in setup_steps:
        print(f"\n--- {step_name} ---")
        results[step_name] = step_func()
    
    # Summary
    print_header("Setup Summary")
    
    success_count = sum(1 for result in results.values() if result)
    
    print(f"Setup steps completed: {success_count}/{len(setup_steps)}")
    
    if success_count == len(setup_steps):
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("  1. python 2_train_model.py    # Train healthcare model")
        print("  2. python 3_run_demo.py       # Run complete demo")
        return 0
    else:
        print("\n! Setup incomplete - some steps failed")
        print("Check error messages above and retry")
        return 1

if __name__ == "__main__":
    exit(main())