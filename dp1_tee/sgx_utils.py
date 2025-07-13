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
import shutil
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
            # Check for SGX device files (update to match your system)
            sgx_devices = ["/dev/sgx_enclave", "/dev/sgx_provision", "/dev/sgx/enclave", "/dev/sgx/provision"]
            found = False
            for device in sgx_devices:
                if os.path.exists(device):
                    self.logger.info(f"[SGX] Found SGX device: {device}")
                    found = True
            if not found:
                self.logger.warning("[SGX] SGX device files not found. Will proceed in simulation mode.")
                if self.tee_config.strict_mode:
                    self.logger.error("[SGX] SGX hardware not found and strict mode enabled")
                    return False
                self.logger.warning("[SGX] WARNING: This provides NO real security guarantees!")
                return True  # Allow to proceed for testing/simulation only
            return True
        except Exception as e:
            self.logger.error(f"[SGX] Error checking SGX support: {e}")
            return False
    
    def _generate_manifest(self) -> bool:
        """Generate Gramine manifest file in the project directory"""
        try:
            manifest_content = self._get_manifest_template()
            # Always write manifest to project directory
            manifest_path = os.path.join(os.getcwd(), "python.manifest")
            with open(manifest_path, 'w') as f:
                f.write(manifest_content)
            self.manifest_path = manifest_path
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
# (sgx.thread_num removed—no longer supported)

# Enable attestation
sgx.remote_attestation = "{'dcap' if self.tee_config.attestation_type == 'dcap' else 'none'}"

# File system
fs.mounts = [
    {{ path = "/lib",    uri = "file:/lib" }},
    {{ path = "/lib64",  uri = "file:/lib64" }},
    {{ path = "/usr",    uri = "file:/usr" }},
    {{ path = "/bin",    uri = "file:/bin" }},
    {{ path = "/etc",    uri = "file:/etc" }},
    {{ path = "/tmp",    uri = "file:/tmp" }},
    {{ path = "/home/user", uri = "file:{current_dir}" }},
    {{ path = "/data",   uri = "file:{current_dir}/data" }},
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
            # Find gramine-sgx command with proper PATH
            gramine_cmd = self._find_gramine_command()
            if not gramine_cmd:
                self.logger.error("[SGX] Gramine-SGX not found in PATH")
                return False
            
            # Check if Gramine is available using --version (not --help which exits 2)
            result = subprocess.run([gramine_cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                self.logger.error(f"[SGX] Gramine-SGX check failed (exit {result.returncode})")
                return False
            
            self.logger.info(f"[SGX] Gramine-SGX available: {result.stdout.strip()}")
            self.gramine_path = gramine_cmd
            self.is_initialized = True
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"[SGX] Gramine initialization failed: {e}")
            return False
    
    def _find_gramine_command(self) -> Optional[str]:
        """Find gramine-sgx command in PATH"""
        try:
            # Try which command first
            result = subprocess.run(['which', 'gramine-sgx'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Fallback: check common paths
            common_paths = [
                '/usr/local/bin/gramine-sgx',
                '/usr/bin/gramine-sgx', 
                '/home/azureuser/.local/bin/gramine-sgx',
                shutil.which('gramine-sgx')
            ]
            
            for path in common_paths:
                if path and os.path.isfile(path) and os.access(path, os.X_OK):
                    return path
                    
            return None
            
        except Exception as e:
            self.logger.error(f"[SGX] Error finding gramine command: {e}")
            return None

    def run_in_enclave(self, script_path: str, args: Optional[List[str]] = None) -> Optional[subprocess.Popen]:
        """Run a Python script inside SGX enclave"""
        if not self.tee_config.use_tee or not self.is_initialized:
            cmd = [sys.executable, script_path] + (args or [])
            self.logger.info(f"[SGX] Running without enclave: {' '.join(cmd)}")
            return subprocess.Popen(cmd)

        try:
            gramine_cmd = getattr(self, 'gramine_path', 'gramine-sgx')
            # Ensure manifest exists
            if not self.manifest_path or not os.path.exists(self.manifest_path):
                if not self._generate_manifest():
                    raise RuntimeError("Failed to generate Gramine manifest")

            # At this point, self.manifest_path is guaranteed to be not None
            if not self.manifest_path:
                raise RuntimeError("Manifest path is None after generation")
            
            manifest_dir = os.path.dirname(self.manifest_path)

            # 1) Build the .manifest.sgx if needed
            sgx_manifest_path = self.manifest_path + ".sgx"
            build_cmd = ["gramine-manifest", self.manifest_path, sgx_manifest_path]
            self.logger.info(f"[SGX] Building manifest: {' '.join(build_cmd)}")
            result = subprocess.run(build_cmd, cwd=manifest_dir, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Manifest build failed: {result.stderr}")
            self.logger.info(f"[SGX] Manifest built successfully: {sgx_manifest_path}")

            # 2) Generate signing key if it doesn't exist
            signing_key_path = os.path.expanduser("~/.config/gramine/enclave-key.pem")
            if not os.path.exists(signing_key_path):
                self.logger.info("[SGX] Generating SGX signing key...")
                os.makedirs(os.path.dirname(signing_key_path), exist_ok=True)
                key_gen_result = subprocess.run(["gramine-sgx-gen-private-key"], 
                                              cwd=manifest_dir, capture_output=True, text=True)
                if key_gen_result.returncode != 0:
                    raise RuntimeError(f"Key generation failed: {key_gen_result.stderr}")
                self.logger.info(f"[SGX] Signing key generated: {signing_key_path}")

            # 3) Sign the manifest to create .sig file
            # NOTE: If gramine-sgx-sign fails with "No module named 'pkg_resources'", install:
            # sudo apt-get install python3-setuptools
            # or: pip3 install setuptools
            manifest_base, _ = os.path.splitext(self.manifest_path)  # drops ".manifest"
            sig_file = manifest_base + ".sig"
            
            if not os.path.exists(sig_file):
                self.logger.info("[SGX] Signing manifest...")
                sign_cmd = ["gramine-sgx-sign", "--manifest", self.manifest_path, 
                           "--key", signing_key_path, "--output", sgx_manifest_path]
                sign_result = subprocess.run(sign_cmd, cwd=manifest_dir, capture_output=True, text=True)
                if sign_result.returncode != 0:
                    error_msg = f"Manifest signing failed: {sign_result.stderr}"
                    if "pkg_resources" in sign_result.stderr:
                        error_msg += "\nHint: Install setuptools: sudo apt-get install python3-setuptools"
                    raise RuntimeError(error_msg)
                self.logger.info(f"[SGX] Manifest signed successfully: {sig_file}")

            # 4) **Run** using the base name so Gramine finds python.manifest.sgx and python.sig
            cmd = [gramine_cmd, manifest_base, script_path] + (args or [])
            env = os.environ.copy()
            if self.tee_config.debug_mode:
                env['GRAMINE_LOG_LEVEL'] = 'debug'
            self.logger.info(f"[SGX] Running in enclave: {' '.join(cmd)} (cwd={manifest_dir})")
            return subprocess.Popen(cmd, cwd=manifest_dir, env=env)

        except Exception as e:
            self.logger.error(f"[SGX] Failed to run in enclave: {e}")
            if self.tee_config.strict_mode:
                raise
            # Fall back to normal Python execution with proper stdout handling
            fallback_cmd = [sys.executable, script_path] + (args or [])
            self.logger.info(f"[SGX] Falling back to host execution: {' '.join(fallback_cmd)}")
            return subprocess.Popen(
                fallback_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

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