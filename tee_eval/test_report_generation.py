#!/usr/bin/env python3
"""
Test the report generation functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_report import generate_and_save_report

def test_report_generation():
    """Test report generation with sample data"""
    print("🧪 Testing Report Generation...")
    
    # Sample data similar to your actual demo results
    sample_vulnerable = {
        'extracted_data': [0.18318748, 1.4245281, 1.1593513, 1.2029743, 1.1231804, 1.346344, 0.9848051, 1.331398, 1.4242592, 0.8655443],
        'methods': ['direct_memory_read', 'process_analysis'],
        'memory_regions': 425
    }
    
    sample_sgx = {
        'extracted_data': [],  # SGX protection successful - no data extracted
        'blocked_attacks': ['memory_extraction_blocked', 'process_analysis_blocked'],
        'protection_verified': True,
        'memory_regions': 2
    }
    
    sample_model = {
        'model_type': 'Healthcare Risk Prediction Model (Logistic Regression)',
        'feature_count': 20,
        'training_samples': 55500,
        'coefficients_sample': [-0.04765246, -0.04112401, -0.03412162]
    }
    
    sample_hardware = {
        'cpu_model': 'Intel SGX-capable CPU',
        'sgx_version': 'SGX2 with Flexible Launch Control',
        'epc_size': 'Multiple GB EPC available',
        'os': 'Ubuntu Linux with SGX driver support',
        'gramine_version': 'Gramine LibOS for SGX'
    }
    
    try:
        report_file = generate_and_save_report(
            sample_vulnerable, 
            sample_sgx, 
            sample_model, 
            sample_hardware
        )
        
        print(f"✅ Report generated successfully: {report_file}")
        
        # Read and show first few lines
        with open(report_file, 'r') as f:
            lines = f.readlines()
            print(f"📄 Report preview (first 10 lines):")
            for i, line in enumerate(lines[:10]):
                print(f"   {i+1:2d}: {line.rstrip()}")
                
        print(f"\n📊 Report statistics:")
        print(f"   • Total lines: {len(lines)}")
        print(f"   • File size: {os.path.getsize(report_file)} bytes")
        print(f"   • Location: {os.path.abspath(report_file)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 REPORT GENERATION TEST")
    print("=" * 50)
    
    success = test_report_generation()
    
    if success:
        print("\n🎉 Test passed! Report generation is working correctly.")
        print("💡 Your SGX demo will now generate professional reports automatically.")
    else:
        print("\n❌ Test failed! Check the error messages above.")
    
    print("=" * 50)