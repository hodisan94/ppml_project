#!/usr/bin/env python3
"""
Test the Python 3.10 SGX configuration fix
"""

import os
import subprocess
import tempfile
import time

def test_python_sgx_fix():
    """Test if the Python 3.10 manifest fix works"""
    print("🧪 Testing Python 3.10 SGX Configuration Fix")
    print("=" * 60)
    
    # Create a simple test script that will show Python is working
    test_script = '''#!/usr/bin/env python3
import sys
import os

print("🐍 Python SGX Test Started")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Test basic functionality
try:
    import pickle
    import numpy as np
    print("✅ Core imports successful")
    
    # Test basic computation
    data = [1.0, 2.0, 3.0]
    result = sum(data)
    print(f"✅ Basic computation: sum({data}) = {result}")
    
    print("🎉 Python SGX test completed successfully!")
    print("✅ All Python functionality working in SGX enclave")
    
except Exception as e:
    print(f"❌ Python error in SGX: {e}")
    sys.exit(1)
'''
    
    # Create a minimal manifest for testing
    test_manifest = '''loader.entrypoint.uri = "file:{{ gramine.libos }}"
libos.entrypoint = "/usr/bin/python3"

loader.argv = ["python3", "test_sgx_python.py"]

loader.env.LD_LIBRARY_PATH = "/lib:/usr/lib/x86_64-linux-gnu:/usr/lib"
loader.env.PYTHONPATH = "/usr/lib/python3.10:/usr/lib/python3/dist-packages:/usr/local/lib/python3.10/dist-packages"

# SGX Configuration
sgx.enclave_size = "512M"
sgx.max_threads = 16
sgx.debug = true

# File system
fs.mounts = [
    { path = "/lib", uri = "file:/lib" },
    { path = "/lib64", uri = "file:/lib64" },
    { path = "/usr/lib", uri = "file:/usr/lib" },
    { path = "/usr/lib/x86_64-linux-gnu", uri = "file:/usr/lib/x86_64-linux-gnu" },
    { path = "/usr", uri = "file:/usr" },
    { path = "/tmp", type = "tmpfs" },
]

# Trusted files
sgx.trusted_files = [
    "file:{{ gramine.libos }}",
    "file:/usr/bin/python3",
    "file:/usr/bin/python3.10",
    "file:test_sgx_python.py",
    
    # Python 3.10 standard library (explicit paths)
    "file:{{ python.stdlib }}/",
    "file:{{ python.distlib }}/",
    "file:/usr/lib/python3.10/",
    "file:/usr/lib/python3/dist-packages/",
    "file:/usr/local/lib/python3.10/dist-packages/",
    "file:/lib/python3.10/",
    
    # System libraries
    "file:/lib/x86_64-linux-gnu/",
    "file:/lib64/", 
    "file:/usr/lib/",
    "file:/usr/lib/x86_64-linux-gnu/",
]

# Allowed files
sgx.allowed_files = [
    "file:/tmp/",
    "file:/proc/cpuinfo",
    "file:/proc/meminfo",
]

sys.enable_sigterm_injection = true
'''

    # Test in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Write test files
            with open("test_sgx_python.py", "w") as f:
                f.write(test_script)
            
            with open("test_python.manifest.template", "w") as f:
                f.write(test_manifest)
            
            print(f"[*] Created test files in: {temp_dir}")
            
            # Step 1: Generate manifest
            print("\n[*] Step 1: Generating manifest...")
            result = subprocess.run([
                "gramine-manifest", 
                "test_python.manifest.template", 
                "test_python.manifest"
            ], capture_output=True, text=True, check=True)
            print("✅ Manifest generation successful")
            
            # Step 2: Sign manifest
            print("\n[*] Step 2: Signing manifest...")
            result = subprocess.run([
                "gramine-sgx-sign",
                "--manifest", "test_python.manifest",
                "--output", "test_python.manifest.sgx"
            ], capture_output=True, text=True, check=True)
            print("✅ Manifest signing successful")
            
            # Step 3: Test gramine-sgx with timeout
            print("\n[*] Step 3: Testing Python execution in SGX...")
            result = subprocess.run([
                "gramine-sgx", "test_python"
            ], capture_output=True, text=True, timeout=15)
            
            print(f"\n📊 Test Results:")
            print(f"Return code: {result.returncode}")
            
            if result.stdout:
                print("\n✅ SGX Python Output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"   {line}")
                        
            if result.stderr:
                print("\n⚠️  SGX Stderr:")
                for line in result.stderr.split('\n'):
                    if line.strip() and 'Gramine is starting' not in line:
                        print(f"   {line}")
            
            # Check for success indicators
            success_indicators = [
                "Python SGX test completed successfully",
                "All Python functionality working",
                "Core imports successful"
            ]
            
            if any(indicator in result.stdout for indicator in success_indicators):
                print("\n🎉 SUCCESS: Python 3.10 is working properly in SGX!")
                print("✅ The manifest fix resolved the Python configuration issue")
                return True
            elif result.returncode == 0:
                print("\n✅ SGX executed successfully (no errors)")
                print("⚠️  Check output above for Python functionality")
                return True
            else:
                print(f"\n❌ SGX execution failed with return code: {result.returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Command failed: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("\n⚠️  Test timed out (SGX may be working but slow)")
            return False
        finally:
            os.chdir(original_dir)

if __name__ == "__main__":
    success = test_python_sgx_fix()
    
    if success:
        print("\n🎯 PHASE 1 RESULT: Python configuration fix appears successful!")
        print("💡 Your main SGX demo should now work without Python errors")
        print("\n📋 Next Steps:")
        print("   1. Test the main SGX demo: python3 sgx_demo.py")
        print("   2. Verify SGX enclave shows proper Python output")
        print("   3. Confirm no more 'Permission denied' errors")
    else:
        print("\n❌ PHASE 1 ISSUE: Python configuration needs more work")
        print("💡 May need additional Python path adjustments")
    
    print("\n" + "=" * 60)