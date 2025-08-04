#!/usr/bin/env python3
"""
Simple test to verify Gramine SGX is working
"""

import os
import subprocess
import tempfile

def test_gramine_sgx():
    """Test if Gramine SGX is actually working"""
    print("🧪 Testing Gramine SGX Installation...")
    
    # Create a simple test application
    test_script = '''#!/usr/bin/env python3
print("Hello from SGX!")
import os
print(f"PID: {os.getpid()}")
print("SGX test successful!")
'''
    
    # Create a simple manifest
    manifest_template = '''loader.entrypoint.uri = "file:{{ gramine.libos }}"
libos.entrypoint = "/usr/bin/python3"

loader.argv = ["python3", "test_app.py"]

loader.env.LD_LIBRARY_PATH = "/lib:/usr/lib/x86_64-linux-gnu:/usr/lib"

# SGX Configuration
sgx.enclave_size = "512M"
sgx.max_threads = 16
sgx.debug = true

# File system
fs.mounts = [
    { path = "/lib", uri = "file:/lib" },
    { path = "/lib64", uri = "file:/lib64" },
    { path = "/usr/lib", uri = "file:/usr/lib" },
    { path = "/usr", uri = "file:/usr" },
    { path = "/tmp", type = "tmpfs" },
]

# Trusted files
sgx.trusted_files = [
    "file:{{ gramine.libos }}",
    "file:/usr/bin/python3",
    "file:test_app.py",
    "file:{{ python.stdlib }}/",
    "file:{{ python.distlib }}/", 
    "file:/usr/lib/python3/dist-packages/",
    "file:/lib/x86_64-linux-gnu/",
    "file:/usr/lib/",
]

# Allowed files  
sgx.allowed_files = [
    "file:/proc/cpuinfo",
    "file:/proc/meminfo",
]

sys.enable_sigterm_injection = true
'''

    # Test in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Write test files
        with open("test_app.py", "w") as f:
            f.write(test_script)
        
        with open("test.manifest.template", "w") as f:
            f.write(manifest_template)
        
        print(f"[*] Created test files in: {temp_dir}")
        
        # Step 1: Generate manifest
        print("[*] Step 1: Generating manifest...")
        try:
            result = subprocess.run([
                "gramine-manifest", 
                "test.manifest.template", 
                "test.manifest"
            ], capture_output=True, text=True, check=True)
            print("✅ Manifest generation successful")
        except subprocess.CalledProcessError as e:
            print(f"❌ Manifest generation failed: {e.stderr}")
            return False
        
        # Step 2: Sign manifest
        print("[*] Step 2: Signing manifest...")
        try:
            result = subprocess.run([
                "gramine-sgx-sign",
                "--manifest", "test.manifest",
                "--output", "test.manifest.sgx"
            ], capture_output=True, text=True, check=True)
            print("✅ Manifest signing successful")
        except subprocess.CalledProcessError as e:
            print(f"❌ Manifest signing failed: {e.stderr}")
            return False
        
        # Step 3: Test gramine-sgx
        print("[*] Step 3: Testing gramine-sgx execution...")
        try:
            result = subprocess.run([
                "gramine-sgx", "test"
            ], capture_output=True, text=True, timeout=10)
            
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
                
            if result.returncode == 0 and "Hello from SGX!" in result.stdout:
                print("✅ gramine-sgx is WORKING!")
                return True
            else:
                print("⚠️  gramine-sgx executed but may have issues")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  gramine-sgx timed out")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ gramine-sgx failed: {e}")
            return False

if __name__ == "__main__":
    print("🔍 GRAMINE SGX FUNCTIONALITY TEST")
    print("=" * 50)
    
    if test_gramine_sgx():
        print("\n🎉 SUCCESS: Gramine SGX is fully functional!")
        print("Your SGX demo should work with real SGX protection.")
    else:
        print("\n❌ ISSUE: Gramine SGX is not working properly")
        print("You may need to install additional components.")
    
    print("\n" + "=" * 50)