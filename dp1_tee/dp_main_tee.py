"""
Enhanced DP Main with TEE Support - Orchestrates federated learning with differential privacy and SGX
"""
import subprocess
import sys
import time
import csv
import os
import signal
import threading
from queue import Queue
import argparse
import json
from pathlib import Path

from tee_config import TEEConfig, SGX_TEE_CONFIG

PYTHON = sys.executable
NUM_CLIENTS = 5
PORT = "8086"


def monitor_client_output(client_id, process, output_queue, use_dp=True, use_tee=False):
    """Monitor client output and extract metrics"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                print(f"[CLIENT {client_id}] {line}")

                # Extract accuracy
                if "accuracy:" in line.lower():
                    try:
                        acc_str = line.split("accuracy:")[-1].strip()
                        # Remove any trailing TEE indicators
                        acc_str = acc_str.split("(")[0].strip()
                        accuracy = float(acc_str)
                        output_queue.put(('accuracy', client_id, accuracy))
                    except (ValueError, IndexError):
                        continue

                # Extract privacy metrics if DP is used
                if use_dp and "privacy spent:" in line.lower():
                    try:
                        # Parse: "Privacy spent: ε=0.1234, δ=1.00e-05"
                        privacy_part = line.split("privacy spent:")[-1].strip()
                        # Remove TEE indicators
                        privacy_part = privacy_part.split("(")[0].strip()
                        
                        epsilon_part = privacy_part.split("ε=")[1].split(",")[0]
                        delta_part = privacy_part.split("δ=")[1]

                        epsilon = float(epsilon_part)
                        delta = float(delta_part)

                        output_queue.put(('privacy', client_id, epsilon, delta))
                    except (ValueError, IndexError):
                        continue
                
                # Extract TEE-specific metrics
                if use_tee and "sgx" in line.lower():
                    try:
                        if "enclave initialized" in line.lower():
                            output_queue.put(('tee_status', client_id, 'initialized'))
                        elif "sgx protected" in line.lower():
                            output_queue.put(('tee_status', client_id, 'protected'))
                    except Exception:
                        continue

    except Exception as e:
        print(f"[MAIN] Error monitoring client {client_id}: {e}")


def cleanup_processes(server_process, client_processes):
    """Clean up all processes"""
    print("[MAIN] Cleaning up processes...")

    # Terminate client processes
    for client_id, proc in client_processes:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception as e:
                print(f"[MAIN] Error terminating client {client_id}: {e}")

    # Terminate server process
    if server_process.poll() is None:
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        except Exception as e:
            print(f"[MAIN] Error terminating server: {e}")


def save_results(results, csv_path, json_path, use_dp=True, use_tee=False):
    """Save results to CSV and JSON files"""
    # Save to CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        if use_dp:
            writer.writerow(["client_id", "round", "accuracy", "epsilon", "delta", "tee_enabled"])
        else:
            writer.writerow(["client_id", "round", "accuracy", "tee_enabled"])

        for result in results:
            writer.writerow(result)

    # Save summary to JSON
    summary = {
        "use_dp": use_dp,
        "use_tee": use_tee,
        "num_clients": NUM_CLIENTS,
        "total_results": len(results),
        "final_metrics": {}
    }

    if results:
        # Calculate final privacy budget and accuracy
        final_epsilons = {}
        final_accuracies = {}
        tee_status = {}

        for result in results:
            client_id = result[0]
            accuracy = result[2]
            final_accuracies[client_id] = accuracy
            
            if use_dp and len(result) > 3:
                epsilon = result[3]
                final_epsilons[client_id] = epsilon
            
            if use_tee and len(result) > 5:
                tee_status[client_id] = result[5]

        summary["final_metrics"]["avg_accuracy"] = sum(final_accuracies.values()) / len(final_accuracies)

        if use_dp and final_epsilons:
            summary["final_metrics"]["total_epsilon"] = sum(final_epsilons.values())
            summary["final_metrics"]["avg_epsilon"] = sum(final_epsilons.values()) / len(final_epsilons)
            summary["final_metrics"]["per_client_epsilon"] = final_epsilons

        if use_tee:
            tee_enabled_count = sum(1 for status in tee_status.values() if status)
            summary["final_metrics"]["tee_coverage"] = tee_enabled_count / len(tee_status) if tee_status else 0
            summary["final_metrics"]["tee_clients"] = tee_enabled_count

        summary["final_metrics"]["per_client_accuracy"] = final_accuracies

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)


def run_experiment(use_dp=True, noise_multiplier=1.0, use_tee=False, experiment_name="default"):
    """Run a single experiment with given parameters"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", experiment_name)

    server_process = None
    client_processes = []

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "fl_metrics_tee.csv")
    json_path = os.path.join(results_dir, "experiment_summary_tee.json")

    try:
        print(f"\n[MAIN] Starting experiment: {experiment_name}")
        print(f"[MAIN] DP: {use_dp}, TEE: {use_tee}, Noise Multiplier: {noise_multiplier}")

        # Start server
        print("[MAIN] Starting Flower server...")
        server_cmd = [PYTHON, "-u", "dp_server_tee.py", str(use_dp).lower(), str(use_tee).lower()]
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Give server time to start
        time.sleep(5)

        # Check if server started successfully
        if server_process.poll() is not None:
            stdout, _ = server_process.communicate()
            print(f"[MAIN] Server failed to start. Output: {stdout}")
            return False

        # Start clients
        output_queue = Queue()
        monitor_threads = []

        for client_id in range(1, NUM_CLIENTS + 1):
            print(f"[MAIN] Starting client {client_id}...")
            try:
                cmd = [PYTHON, "-u", "dp_client_tee.py", str(client_id), str(use_dp).lower()]
                if use_dp:
                    cmd.append(str(noise_multiplier))
                if use_tee:
                    cmd.append(str(use_tee).lower())

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                client_processes.append((client_id, proc))

                # Start monitoring thread
                monitor_thread = threading.Thread(
                    target=monitor_client_output,
                    args=(client_id, proc, output_queue, use_dp, use_tee)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                monitor_threads.append(monitor_thread)

                time.sleep(2)

            except Exception as e:
                print(f"[MAIN] Error starting client {client_id}: {e}")

        print(f"[MAIN] Started {len(client_processes)} clients. Training in progress...")

        # Collect results
        results = []
        round_counts = {i: 0 for i in range(1, NUM_CLIENTS + 1)}
        privacy_metrics = {i: {'epsilon': 0.0, 'delta': 0.0} for i in range(1, NUM_ 