"""
TEE Configuration for SGX Enclave Support
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TEEConfig:
    """Configuration class for Trusted Execution Environment parameters"""
    
    # Basic TEE settings
    use_tee: bool = False
    enclave_type: str = "gramine"  # gramine, enarx, or native
    
    # SGX-specific settings
    enclave_heap_size: str = "1G"
    enclave_stack_size: str = "4M"
    enclave_threads: int = 4
    
    # Attestation settings
    attestation_type: str = "dcap"  # dcap, epid, or none
    enable_remote_attestation: bool = False
    
    # Security settings
    enable_secure_aggregation: bool = True
    secure_communication: bool = True
    protected_memory: bool = True
    
    # Performance settings
    async_mode: bool = False
    batch_processing: bool = True
    
    # Debug settings
    debug_mode: bool = False
    enclave_debug: bool = False
    sgx_debug: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.use_tee and self.enclave_type not in ["gramine", "enarx", "native"]:
            raise ValueError(f"Unsupported enclave type: {self.enclave_type}")
        
        if self.attestation_type not in ["dcap", "epid", "none"]:
            raise ValueError(f"Unsupported attestation type: {self.attestation_type}")
    
    @classmethod
    def from_env(cls) -> "TEEConfig":
        """Create TEE configuration from environment variables"""
        return cls(
            use_tee=os.getenv("USE_TEE", "false").lower() == "true",
            enclave_type=os.getenv("ENCLAVE_TYPE", "gramine"),
            enclave_heap_size=os.getenv("ENCLAVE_HEAP_SIZE", "1G"),
            enclave_stack_size=os.getenv("ENCLAVE_STACK_SIZE", "4M"),
            enclave_threads=int(os.getenv("ENCLAVE_THREADS", "4")),
            attestation_type=os.getenv("ATTESTATION_TYPE", "dcap"),
            enable_remote_attestation=os.getenv("ENABLE_REMOTE_ATTESTATION", "false").lower() == "true",
            enable_secure_aggregation=os.getenv("ENABLE_SECURE_AGGREGATION", "true").lower() == "true",
            secure_communication=os.getenv("SECURE_COMMUNICATION", "true").lower() == "true",
            protected_memory=os.getenv("PROTECTED_MEMORY", "true").lower() == "true",
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            enclave_debug=os.getenv("ENCLAVE_DEBUG", "false").lower() == "true",
            sgx_debug=os.getenv("SGX_DEBUG", "false").lower() == "true"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "use_tee": self.use_tee,
            "enclave_type": self.enclave_type,
            "enclave_heap_size": self.enclave_heap_size,
            "enclave_stack_size": self.enclave_stack_size,
            "enclave_threads": self.enclave_threads,
            "attestation_type": self.attestation_type,
            "enable_remote_attestation": self.enable_remote_attestation,
            "enable_secure_aggregation": self.enable_secure_aggregation,
            "secure_communication": self.secure_communication,
            "protected_memory": self.protected_memory,
            "debug_mode": self.debug_mode,
            "enclave_debug": self.enclave_debug,
            "sgx_debug": self.sgx_debug
        }


# Default TEE configuration
DEFAULT_TEE_CONFIG = TEEConfig()

# SGX-enabled configuration
SGX_TEE_CONFIG = TEEConfig(
    use_tee=True,
    enclave_type="gramine",
    enclave_heap_size="2G",
    enclave_threads=8,
    attestation_type="dcap",
    enable_remote_attestation=True,
    enable_secure_aggregation=True,
    secure_communication=True,
    protected_memory=True
) 