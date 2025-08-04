#!/usr/bin/env python3
"""
Test the fixed Python 3.10 SGX configuration.
Based on successful Gramine examples from the community.
"""

import tempfile
import subprocess
import os
import shutil

def test_fixed_python_sgx():
    print("🔧 Testing Fixed Python 3.10 SGX Configuration")
    print("=" * 60)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[*] Created test files in: {temp_dir}")
        
        # Copy our fixed manifest template
        fixed_manifest_src = "tee_eval/gramine/sgx_inference_fixed.manifest.template"
        manifest_template = os.path.join(temp_dir, "python.manifest.template")
        shutil.copy(fixed_manifest_src, manifest_template)
        
        # Create a simple Python test script
        test_python_script = os.path.join(temp_dir, "sgx_inference.py")
        with open(test_python_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("Simple calculation: 2 + 2 =", 2 + 2)
print("✅ Python execution successful!")
""")
        
        # Make it executable
        os.chmod(test_python_script, 0o755)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Step 1: Generate manifest
            print("\n[*] Step 1: Generating manifest...")
            manifest_cmd = [
                'gramine-manifest',
                '-Dlog_level=error',
                '-Dentrypoint=sgx_inference.py',
                '-Ddebug=true',
                '-Dnonpie_binary=true',
                '-Denclave_size=256M',
                '-Dmax_threads=4',
                'python.manifest.template',
                'python.manifest'
            ]
            
            result = subprocess.run(manifest_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Manifest generation successful")
            else:
                print("❌ Manifest generation failed:")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
            
            # Step 2: Sign manifest
            print("\n[*] Step 2: Signing manifest...")
            sign_cmd = [
                'gramine-sgx-sign',
                '--manifest', 'python.manifest',
                '--output', 'python.manifest.sgx'
            ]
            
            result = subprocess.run(sign_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Manifest signing successful")
            else:
                print("❌ Manifest signing failed:")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
            
            # Step 3: Test Python execution
            print("\n[*] Step 3: Testing Python execution in SGX...")
            sgx_cmd = ['gramine-sgx', 'python']
            
            result = subprocess.run(sgx_cmd, capture_output=True, text=True, timeout=30)
            
            print("\n📊 Test Results:")
            print(f"Return code: {result.returncode}")
            
            if result.stdout:
                print("\n✅ SGX Python Output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"   {line}")
            
            if result.stderr:
                print("\n⚠️  SGX Stderr:")
                sgx_started = False
                python_error = False
                for line in result.stderr.split('\n'):
                    if line.strip():
                        if "Gramine is starting" in line:
                            print(f"   ✅ {line}")
                            sgx_started = True
                        elif "Fatal Python error" in line:
                            print(f"   ❌ {line}")
                            python_error = True
                        elif "Permission denied" in line and "python" in line:
                            print(f"   ❌ {line}")
                            python_error = True
                        else:
                            print(f"   {line}")
                
                if sgx_started and not python_error:
                    print("\n✅ FIXED: SGX started successfully without Python errors!")
                    return True
                elif sgx_started and python_error:
                    print("\n⚠️  PARTIAL: SGX started but Python has configuration issues")
                    return False
                else:
                    print("\n❌ FAILED: SGX enclave failed to start properly")
                    return False
            
            return result.returncode == 0
            
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_fixed_python_sgx()
    if success:
        print("\n🎉 PHASE 2 SUCCESS: Fixed Python configuration works!")
        print("✅ Ready to proceed with Phase 3: Enhanced demo and reporting")
    else:
        print("\n❌ PHASE 2 INCOMPLETE: Still has Python configuration issues")
        print("💡 Need to investigate further or accept current limitations")
    
    print("\n" + "=" * 60)