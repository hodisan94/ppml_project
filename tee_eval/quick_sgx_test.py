#!/usr/bin/env python3
"""
Quick test to verify Gramine SGX is working with your original demo
"""

import subprocess
import sys
import os

def main():
    print("🚀 QUICK SGX FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Change to the tee_eval directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"[*] Working directory: {os.getcwd()}")
    
    # Run the fixed test
    print("[*] Testing fixed Gramine manifest...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_gramine.py"
        ], capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if "Hello from SGX!" in result.stdout:
            print("\n🎉 SUCCESS: Gramine SGX is working!")
            print("✅ Your original SGX demo should now work with REAL SGX protection")
            print("\n💡 Next step: Run your original demo:")
            print("   python3 sgx_demo.py")
            return True
        else:
            print(f"\n⚠️  Test result unclear (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)