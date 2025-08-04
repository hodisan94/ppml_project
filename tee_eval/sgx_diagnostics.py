#!/usr/bin/env python3
"""
SGX Environment Diagnostics
Comprehensive check of SGX hardware and Gramine setup
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n--- {description} ---")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")  
            print(result.stderr)
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except FileNotFoundError:
        print("❌ Command not found")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_files_and_permissions():
    """Check SGX device files and permissions."""
    print("\n=== FILE SYSTEM CHECKS ===")
    
    sgx_devices = [
        "/dev/sgx_enclave",
        "/dev/sgx/enclave", 
        "/dev/sgx_provision",
        "/dev/sgx/provision",
        "/dev/isgx"
    ]
    
    for device in sgx_devices:
        if os.path.exists(device):
            stat = os.stat(device)
            permissions = oct(stat.st_mode)[-3:]
            print(f"✅ {device} exists (permissions: {permissions})")
        else:
            print(f"❌ {device} not found")
    
    # Check if current user can access SGX
    for device in sgx_devices:
        if os.path.exists(device):
            try:
                with open(device, 'rb') as f:
                    print(f"✅ Can read {device}")
                break
            except PermissionError:
                print(f"❌ Permission denied for {device}")
            except Exception as e:
                print(f"⚠️  Cannot test {device}: {e}")

def check_gramine_installation():
    """Check Gramine installation and configuration."""
    print("\n=== GRAMINE INSTALLATION CHECKS ===")
    
    # Check Gramine commands
    commands = [
        (["gramine-manifest", "--help"], "Gramine manifest tool"),
        (["gramine-sgx", "--help"], "Gramine SGX runtime"), 
        (["gramine-sgx-sign", "--help"], "Gramine SGX signing tool"),
        (["is-sgx-available"], "SGX availability check"),
    ]
    
    for cmd, desc in commands:
        success = run_command(cmd, desc)
        if not success and cmd[0] == "gramine-sgx":
            # Try alternative installation paths
            alt_commands = [
                ["python3", "-m", "graminelibos.main", "--help"],
                ["/usr/local/bin/gramine-sgx", "--help"],
                ["~/.local/bin/gramine-sgx", "--help"]
            ]
            print("Trying alternative paths...")
            for alt_cmd in alt_commands:
                if run_command(alt_cmd, f"Alternative: {' '.join(alt_cmd)}"):
                    break

def check_python_environment():
    """Check Python and package environment.""" 
    print("\n=== PYTHON ENVIRONMENT ===")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check for required packages
    packages = ["graminelibos", "cryptography", "protobuf", "tomli"]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")

def check_sgx_capabilities():
    """Check detailed SGX capabilities."""
    print("\n=== SGX CAPABILITIES ===")
    
    # Try to read SGX info from proc
    proc_files = [
        "/proc/cpuinfo",
        "/proc/sys/kernel/randomize_va_space"
    ]
    
    for proc_file in proc_files:
        if os.path.exists(proc_file):
            try:
                with open(proc_file, 'r') as f:
                    content = f.read()
                    if "sgx" in content.lower():
                        print(f"✅ SGX mentioned in {proc_file}")
                        # Extract relevant lines
                        for line in content.split('\n'):
                            if 'sgx' in line.lower():
                                print(f"    {line.strip()}")
                    else:
                        print(f"⚠️  No SGX references in {proc_file}")
            except Exception as e:
                print(f"❌ Cannot read {proc_file}: {e}")

def check_gramine_templates():
    """Check manifest templates."""
    print("\n=== TEMPLATE FILES ===")
    
    template_file = "gramine/sgx_inference.manifest.template"
    if os.path.exists(template_file):
        print(f"✅ Template exists: {template_file}")
        try:
            with open(template_file, 'r') as f:
                content = f.read()
                print(f"Template size: {len(content)} bytes")
                # Check for key sections
                if "sgx.enclave_size" in content:
                    print("✅ Contains SGX configuration")
                if "sgx.trusted_files" in content:
                    print("✅ Contains trusted files section")
        except Exception as e:
            print(f"❌ Cannot read template: {e}")
    else:
        print(f"❌ Template missing: {template_file}")

def main():
    print("🔍 SGX ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    check_files_and_permissions()
    check_gramine_installation() 
    check_python_environment()
    check_sgx_capabilities()
    check_gramine_templates()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("\nNext steps:")
    print("1. Check if Gramine is properly installed")
    print("2. Verify SGX driver installation") 
    print("3. Check user permissions for SGX devices")
    print("4. Install missing Python packages if needed")

if __name__ == "__main__":
    main()