#!/usr/bin/env python3
"""
Setup script for the SGX Healthcare Demo
Installs dependencies and verifies the environment
"""

import os
import sys
import subprocess

def install_requirements():
    """Install Python requirements."""
    print("[+] Installing Python requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("[+] Creating directories...")
    directories = ["sample_data", "gramine"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def check_sgx_tools():
    """Check if SGX tools are available."""
    print("[+] Checking SGX tools...")
    
    sgx_tools = ["gramine-sgx", "is-sgx-available", "gramine-manifest"]
    available_tools = []
    
    for tool in sgx_tools:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode == 0:
            available_tools.append(tool)
            print(f"✓ Found: {tool}")
        else:
            print(f"! Not found: {tool}")
    
    if available_tools:
        print(f"✓ SGX tools available: {len(available_tools)}/{len(sgx_tools)}")
    else:
        print("! No SGX tools found - will run in simulation mode")
    
    return True

def main():
    """Main setup function."""
    print("="*50)
    print("SGX Healthcare Demo - Setup")
    print("="*50)
    
    setup_steps = [
        ("Create directories", create_directories),
        ("Install requirements", install_requirements),
        ("Check SGX tools", check_sgx_tools)
    ]
    
    success_count = 0
    
    for step_name, step_func in setup_steps:
        print(f"\n--- {step_name} ---")
        if step_func():
            success_count += 1
        else:
            print(f"✗ {step_name} failed!")
    
    print("\n" + "="*50)
    print(f"Setup complete: {success_count}/{len(setup_steps)} steps successful")
    print("="*50)
    
    if success_count == len(setup_steps):
        print("\n✓ Setup successful! You can now run:")
        print("  python test_demo.py")
        print("  python compare_security.py")
        return 0
    else:
        print("\n✗ Setup incomplete. Check error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())