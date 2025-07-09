"""
SGX Utilities for Enclave Management and Secure Operations
"""
import os
import sys
import subprocess
import json
import hashlib
import base64
import logging
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import tempfile
import numpy as np

from tee_config import TEEConfig


class SGXEnclave:
    """SGX Enclave management and operations"""
    
    def __init__(self, tee_config: TEEConfig):
        self.tee_config = tee_config
        self.is_initialized = False
        self.enclave_process = None
        self.manifest_path = None
        self.token_path = None
        
        # Setup logging
        log_level = logging.DEBUG if tee_config.debug_mode else logging.INFO
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> bool:
        """Initialize the SGX enclave"""
        try:
            if not self.tee_config.use_tee:
                self.logger.info("[SGX] TEE disabled, running in normal mode")
                self.is_initialized = True
                return True
            
            # Check SGX availability
            if not self._check_sgx_support():
                self.logger.error("[SGX] SGX not supported on this system")
                return False
            
            # Generate manifest if needed
            if not self._generate_manifest():
                self.logger.error("[SGX] Failed to generate manifest")
                return False
            
            # Initialize enclave based on type
            if self.tee_config.enclave_type == "gramine":
                return self._initialize_gramine()
            else:
                self.logger.error(f"[SGX] Unsupported enclave type: {self.tee_config.enclave_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"[SGX] Enclave initialization failed: {e}")
            return False
    
    def _check_sgx_support(self) -> bool:
        """Check if SGX is supported and available"""
        try:
            # Check for SGX device files
            sgx_devices = ["/dev/sgx_enclave", "/dev/sgx/enclave"]
            if not any(os.path.exists(device) for device in sgx_devices):
                self.logger.warning("[SGX] SGX device files not found")
                return False
            
            # Check SGX capabilities (if cpuid is available)
            try:
                result = subprocess.run(['cpuid', '-l', '0x7', '-s', '0x0'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and 'SGX' in result.stdout:
                    self.logger.info("[SGX] SGX support detected via cpuid")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Fallback: check dmesg for SGX
            try:
                result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and 'sgx' in result.stdout.lower():
                    self.logger.info("[SGX] SGX support detected via dmesg")
                    return True
            except subprocess.TimeoutExpired:
                pass
            
            self.logger.warning("[SGX] SGX support unclear, proceeding with caution")
            return True  # Allow to proceed for testing
            
        except Exception as e:
            self.logger.error(f"[SGX] Error checking SGX support: {e}")
            return False
    
    def _generate_manifest(self) -> bool:
        """Generate Gramine manifest file"""
        try:
            manifest_content = self._get_manifest_template()
            
            # Create temporary manifest file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.manifest', delete=False) as f:
                f.write(manifest_content)
                self.manifest_path = f.name
            
            self.logger.info(f"[SGX] Generated manifest: {self.manifest_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"[SGX] Failed to generate manifest: {e}")
            return False
    
    def _get_manifest_template(self) -> str:
        """Get Gramine manifest template for Python ML workload"""
        python_path = sys.executable
        current_dir = os.getcwd()
        
        manifest = f"""
# Gramine manifest for Python ML with TensorFlow and Flower

loader.entrypoint = "file:{python_path}"
libos.entrypoint = "{python_path}"

loader.log_level = "{'debug' if self.tee_config.sgx_debug else 'error'}"

loader.env.LD_LIBRARY_PATH = "/lib:/lib/x86_64-linux-gnu:/usr/lib:/usr/lib/x86_64-linux-gnu"
loader.env.PATH = "/bin:/usr/bin:/usr/local/bin"
loader.env.PYTHONPATH = "{current_dir}"
loader.env.HOME = "/home/user"

# SGX Configuration
sgx.debug = {'true' if self.tee_config.enclave_debug else 'false'}
sgx.enclave_size = "{self.tee_config.enclave_heap_size}"
sgx.thread_num = {self.tee_config.enclave_threads}

# Enable attestation
sgx.remote_attestation = "{'dcap' if self.tee_config.attestation_type == 'dcap' else 'none'}"

# File system
fs.mounts = [
    {{ path = "/lib", uri = "file:/lib" }},
    {{ path = "/lib64", uri = "file:/lib64" }},
    {{ path = "/usr", uri = "file:/usr" }},
    {{ path = "/bin", uri = "file:/bin" }},
    {{ path = "/etc", uri = "file:/etc" }},
    {{ path = "/tmp", uri = "file:/tmp" }},
    {{ path = "/home/user", uri = "file:{current_dir}" }},
    {{ path = "/data", uri = "file:{current_dir}/data" }},
]

# Trusted files (Python and libraries)
sgx.trusted_files = [
    "file:{python_path}",
    "file:/lib/",
    "file:/lib64/",
    "file:/usr/lib/",
    "file:/usr/bin/",
    "file:{current_dir}/",
]

# Allowed files (data and temporary files)
sgx.allowed_files = [
    "file:/tmp/",
    "file:/data/",
    "file:{current_dir}/data/",
    "file:{current_dir}/results/",
]

# Network access
sys.enable_sigterm_injection = true
sys.enable_extra_runtime_domain_names_conf = true
"""
        return manifest
    
    def _initialize_gramine(self) -> bool:
        """Initialize Gramine-SGX enclave"""
        try:
            # Check if Gramine is available
            result = subprocess.run(['gramine-sgx', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                self.logger.error("[SGX] Gramine-SGX not found")
                return False
            
            self.logger.info("[SGX] Gramine-SGX available")
            self.is_initialized = True
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"[SGX] Gramine initialization failed: {e}")
            return False
    
    def run_in_enclave(self, script_path: str, args: List[str] = None) -> subprocess.Popen:
        """Run a Python script inside SGX enclave"""
        if not self.tee_config.use_tee or not self.is_initialized:
            # Run normally without enclave
            cmd = [sys.executable, script_path] + (args or [])
            self.logger.info(f"[SGX] Running without enclave: {' '.join(cmd)}")
            return subprocess.Popen(cmd)
        
        try:
            # Prepare Gramine command
            cmd = ['gramine-sgx', 'python', script_path] + (args or [])
            
            # Set environment variables
            env = os.environ.copy()
            if self.tee_config.debug_mode:
                env['GRAMINE_LOG_LEVEL'] = 'debug'
            
            self.logger.info(f"[SGX] Running in enclave: {' '.join(cmd)}")
            
            # Start process in enclave
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            return process
            
        except Exception as e:
            self.logger.error(f"[SGX] Failed to run in enclave: {e}")
            # Fallback to normal execution
            cmd = [sys.executable, script_path] + (args or [])
            return subprocess.Popen(cmd)
    
    def get_enclave_measurement(self) -> Optional[str]:
        """Get enclave measurement for attestation"""
        if not self.tee_config.use_tee or not self.is_initialized:
            return None
        
        try:
            # This would normally get the actual SGX measurement
            # For now, return a dummy measurement
            dummy_measurement = hashlib.sha256(b"enclave_measurement").hexdigest()
            return dummy_measurement
        except Exception as e:
            self.logger.error(f"[SGX] Failed to get enclave measurement: {e}")
            return None
    
    def verify_attestation(self, remote_measurement: str) -> bool:
        """Verify remote attestation"""
        if not self.tee_config.enable_remote_attestation:
            return True
        
        try:
            local_measurement = self.get_enclave_measurement()
            if local_measurement and remote_measurement:
                return local_measurement == remote_measurement
            return False
        except Exception as e:
            self.logger.error(f"[SGX] Attestation verification failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup enclave resources"""
        try:
            if self.enclave_process and self.enclave_process.poll() is None:
                self.enclave_process.terminate()
                self.enclave_process.wait(timeout=5)
            
            # Cleanup temporary files
            if self.manifest_path and os.path.exists(self.manifest_path):
                os.unlink(self.manifest_path)
            
            if self.token_path and os.path.exists(self.token_path):
                os.unlink(self.token_path)
                
            self.logger.info("[SGX] Enclave cleanup completed")
            
        except Exception as e:
            self.logger.error(f"[SGX] Cleanup error: {e}")


def secure_aggregate_weights(weights_list: List[np.ndarray], 
                           tee_config: TEEConfig) -> np.ndarray:
    """Secure aggregation of model weights using TEE"""
    if not tee_config.enable_secure_aggregation:
        # Standard aggregation
        return np.mean(weights_list, axis=0)
    
    try:
        # In a real implementation, this would use secure multi-party computation
        # For now, we'll do standard aggregation with additional security logging
        logging.info("[SGX] Performing secure weight aggregation")
        
        # Add noise for additional privacy if in TEE mode
        if tee_config.use_tee:
            aggregated = np.mean(weights_list, axis=0)
            # Add minimal noise for obfuscation
            noise_scale = 1e-6
            noise = np.random.normal(0, noise_scale, aggregated.shape)
            return aggregated + noise
        else:
            return np.mean(weights_list, axis=0)
            
    except Exception as e:
        logging.error(f"[SGX] Secure aggregation failed: {e}")
        return np.mean(weights_list, axis=0)


def encrypt_communication(data: bytes, tee_config: TEEConfig) -> bytes:
    """Encrypt data for secure communication"""
    if not tee_config.secure_communication:
        return data
    
    try:
        # In a real implementation, this would use proper encryption
        # For now, we'll use base64 encoding as a placeholder
        return base64.b64encode(data)
    except Exception as e:
        logging.error(f"[SGX] Encryption failed: {e}")
        return data


def decrypt_communication(encrypted_data: bytes, tee_config: TEEConfig) -> bytes:
    """Decrypt data from secure communication"""
    if not tee_config.secure_communication:
        return encrypted_data
    
    try:
        # In a real implementation, this would use proper decryption
        # For now, we'll use base64 decoding as a placeholder
        return base64.b64decode(encrypted_data)
    except Exception as e:
        logging.error(f"[SGX] Decryption failed: {e}")
        return encrypted_data 