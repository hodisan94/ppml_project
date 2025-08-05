****PPML Project****
Privacy‑Preserving Machine Learning (PPML) with Random Forests — combining Federated Learning, Differential Privacy, and Intel SGX to defend against inference and memory-leakage threats.

****Overview*****
This project evaluates the effectiveness of privacy-preserving techniques for Random Forest classifiers on sensitive tabular (e.g. healthcare) data.

We compare three training configurations:

Centralized (Naïve RF)

Federated Learning (FL)

Federated Learning with output-level Gaussian Differential Privacy (FL+DP)

We simulate three major privacy attacks:

Membership Inference Attack (MIA)

Model Inversion Attack

Attribute Inference Attack (AIA)

Finally, we protect inference time using Intel SGX enclaves, defending against privileged memory extraction and process introspection adversaries.

**Features**
End-to-end data pipeline: dataset preprocessing, federated training, and evaluation of inference attacks.

Output-level Gaussian DP: noise added to prediction vectors before aggregation; tuned via ε (epsilon).

SGX-based runtime protection: model inference inside secure enclaves shields both model parameters and input features.

**Key Findings**

| Configuration          | Accuracy | AUC (utility) | MIA AUC | Inversion MSE | AIA Accuracy | Privacy Leaks |
| ---------------------- | -------- | ------------- | ------- | ------------- | ------------ | ------------- |
| Naïve RF (centralized) | \~0.895  | \~0.592       | \~0.665 | \~0.0001      | \~0.546      | High          |
| Federated Learning     | \~0.901  | \~0.568       | \~0.561 | \~0.0992      | \~0.604      | Moderate      |
| FL + DP (ε‑controlled) | \~0.901  | \~0.489       | \~0.559 | \~0.0992      | \~0.498      | Low           |

  FL alone slightly improves utility while reducing vulnerability to inference attacks.
  
  Adding DP maintains accuracy but further suppresses inference attack success.
  
  FL+DP brings Attribute Inference Attack accuracy down to ~0.498—close to random guessing.
  
  Hardware-layer defense with SGX (via Gramine-based enclave) completely prevents memory-leakage attacks, blocking extraction of floating-point parameters and input data—even in privileged OS contexts.

**Technical Stack**
  Python for data preprocessing, attack simulation, and RF training.
  
  Federated ensemble using majority voting aggregation.
  
  Gaussian noise addition to prediction vectors with privacy budget management.
  
  Intel SGX + Gramine for enclave-based inference isolation (memory & process protections).

**Why This Matters**
Our approach offers a multi-layered privacy strategy:

  Federated Learning decentralizes training.
  
  Differential Privacy mitigates output-level inference risks.
  
  SGX enclaves provide runtime confidentiality against memory attacks.
  
  Together, these defenses present a robust PPML framework grounded in high real-world applicability—especially in regulated domains like healthcare.


🔐 Intel SGX Protection
To defend against runtime memory-level attacks, we deployed the trained Random Forest model inside a Trusted Execution Environment (TEE) using Intel SGX.

✅ What SGX Adds
While Federated Learning and Differential Privacy protect against inference-time threats, SGX addresses a different attack surface: adversaries with root-level access who attempt to extract model parameters or input data from process memory during inference.

🛡️ Threats Simulated
We tested two common attack vectors:

Direct Memory Extraction via /proc/[pid]/mem

Process Mapping Inspection via /proc/[pid]/maps and tools like psutil

🧪 Results
Attack Type	Without SGX	With SGX
Memory Extraction	10 float values recovered	0 recovered (blocked)
Process Mapping Access	Full memory visible	Enclave hidden
Sensitive Data Leaked	Model weights + inputs	None

💡 Conclusion: SGX completely eliminated memory-level leakage — making it an essential hardware-level defense that complements software-level privacy methods like FL and DP.

🧰 SGX Integration
Built using Gramine, a lightweight LibOS for running unmodified Python inference scripts inside SGX enclaves.

Enclave setup includes:

Manifest definition for memory and I/O policies

Attestation and measurement of model + input files

Runtime isolation from the OS and hypervisor


