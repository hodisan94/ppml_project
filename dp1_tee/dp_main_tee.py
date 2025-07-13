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

                # Extract accuracy - fixed case sensitivity bug
                if "accuracy:" in line.lower():
                    try:
                        # Use case-insensitive parsing consistently
                        acc_str = line.lower().split("accuracy:")[-1].strip()
                        # Remove any trailing TEE indicators
                        acc_str = acc_str.split("(")[0].strip()
                        accuracy = float(acc_str)
                        output_queue.put(('accuracy', client_id, accuracy))
                    except (ValueError, IndexError):
                        continue

                # Extract privacy metrics if DP is used - fixed case sensitivity bug
                if use_dp and "privacy spent:" in line.lower():
                    try:
                        # Parse: "Privacy spent: ε=0.1234, δ=1.00e-05" - use case-insensitive parsing
                        privacy_part = line.lower().split("privacy spent:")[-1].strip()
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


def monitor_server_output(process, output_queue):
    """Monitor server output and extract federated learning events"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                print(f"[SERVER] {line}")
                
                # Extract server events that indicate FL progress
                if "aggregate_fit called" in line.lower():
                    output_queue.put(('server_event', 'aggregate_fit', line))
                elif "aggregate_evaluate called" in line.lower():
                    output_queue.put(('server_event', 'aggregate_evaluate', line))
                elif "configure_evaluate called" in line.lower():
                    output_queue.put(('server_event', 'configure_evaluate', line))
                elif "round" in line.lower() and ("privacy budget" in line.lower() or "avg accuracy" in line.lower()):
                    output_queue.put(('server_event', 'round_summary', line))

    except Exception as e:
        print(f"[MAIN] Error monitoring server: {e}")


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
        server_args = [str(use_dp).lower(), str(use_tee).lower()]
        
        # Run server inside SGX enclave if TEE is enabled
        if use_tee:
            from sgx_utils import SGXEnclave
            from tee_config import SGX_TEE_CONFIG
            
            # Create enclave instance for the server
            server_enclave = SGXEnclave(SGX_TEE_CONFIG)
            if server_enclave.initialize():
                print("[MAIN] Starting server inside SGX enclave...")
                server_process = server_enclave.run_in_enclave("dp_server_tee.py", server_args)
            else:
                print("[MAIN] SGX enclave initialization failed for server, falling back to normal mode")
                server_cmd = [PYTHON, "-u", "dp_server_tee.py"] + server_args
                server_process = subprocess.Popen(
                    server_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
        else:
            # Run normally without enclave
            server_cmd = [PYTHON, "-u", "dp_server_tee.py"] + server_args
            server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

        # Start server monitoring thread
        output_queue = Queue()
        server_monitor_thread = threading.Thread(
            target=monitor_server_output,
            args=(server_process, output_queue)
        )
        server_monitor_thread.daemon = True
        server_monitor_thread.start()
        print("[MAIN] Server monitoring started")

        # Give server time to start (reduced for performance)
        time.sleep(3)

        # Check if server started successfully
        if server_process.poll() is not None:
            print(f"[MAIN] Server failed to start!")
            return False

        # Start clients
        monitor_threads = [server_monitor_thread]  # Include server monitoring thread

        for client_id in range(1, NUM_CLIENTS + 1):
            print(f"[MAIN] Starting client {client_id}...")
            try:
                args = [str(client_id), str(use_dp).lower()]
                if use_dp:
                    args.append(str(noise_multiplier))
                if use_tee:
                    args.append(str(use_tee).lower())

                # Run client inside SGX enclave if TEE is enabled
                if use_tee:
                    from sgx_utils import SGXEnclave
                    from tee_config import SGX_TEE_CONFIG
                    
                    # Create enclave instance for this client
                    enclave = SGXEnclave(SGX_TEE_CONFIG)
                    if enclave.initialize():
                        print(f"[MAIN] Starting client {client_id} inside SGX enclave...")
                        proc = enclave.run_in_enclave("dp_client_tee.py", args)
                    else:
                        print(f"[MAIN] SGX enclave initialization failed for client {client_id}, falling back to normal mode")
                        cmd = [PYTHON, "-u", "dp_client_tee.py"] + args
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True
                        )
                else:
                    # Run normally without enclave
                    cmd = [PYTHON, "-u", "dp_client_tee.py"] + args
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

                # Reduced sleep for faster startup
                time.sleep(0.5)

            except Exception as e:
                print(f"[MAIN] Error starting client {client_id}: {e}")

        print(f"[MAIN] Started {len(client_processes)} clients. Training in progress...")

        # Collect results
        results = []
        round_counts = {i: 0 for i in range(1, NUM_CLIENTS + 1)}
        privacy_metrics = {i: {'epsilon': 0.0, 'delta': 0.0} for i in range(1, NUM_CLIENTS + 1)}
        tee_status = {i: False for i in range(1, NUM_CLIENTS + 1)}

        timeout_counter = 0
        max_timeout = 300  # 5 minutes

        while any(proc.poll() is None for _, proc in client_processes) and timeout_counter < max_timeout:
            try:
                while not output_queue.empty():
                    item = output_queue.get_nowait()

                    if item[0] == 'accuracy':
                        _, client_id, accuracy = item
                        round_counts[client_id] += 1
                        round_num = round_counts[client_id]

                        if use_dp:
                            epsilon = privacy_metrics[client_id]['epsilon']
                            delta = privacy_metrics[client_id]['delta']
                            results.append([client_id, round_num, accuracy, epsilon, delta, tee_status[client_id]])
                        else:
                            results.append([client_id, round_num, accuracy, tee_status[client_id]])

                        tee_info = " (TEE)" if tee_status[client_id] else ""
                        print(f"[MAIN] Client {client_id}, Round {round_num}, Accuracy: {accuracy:.4f}{tee_info}")

                    elif item[0] == 'privacy' and use_dp:
                        _, client_id, epsilon, delta = item
                        privacy_metrics[client_id]['epsilon'] = epsilon
                        privacy_metrics[client_id]['delta'] = delta
                        print(f"[MAIN] Client {client_id} privacy update: ε={epsilon:.4f}, δ={delta:.2e}")

                    elif item[0] == 'tee_status' and use_tee:
                        _, client_id, status = item
                        if status in ['initialized', 'protected']:
                            tee_status[client_id] = True
                            print(f"[MAIN] Client {client_id} TEE status: {status}")
                    
                    elif item[0] == 'server_event':
                        _, event_type, message = item
                        print(f"[MAIN] Server event ({event_type}): {message}")

                time.sleep(0.1)  # Reduced for better responsiveness
                timeout_counter += 1

            except KeyboardInterrupt:
                print("\n[MAIN] Interrupted by user. Cleaning up...")
                break
            except Exception as e:
                print(f"[MAIN] Error in main loop: {e}")
                break

        # Wait for processes to complete
        print("[MAIN] Waiting for all clients to finish...")
        for client_id, proc in client_processes:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f"[MAIN] Client {client_id} timed out, terminating...")
                proc.terminate()

        # Drain any remaining metrics from the queue
        time.sleep(2)
        while not output_queue.empty():
            try:
                item = output_queue.get_nowait()
                if item[0] == 'accuracy':
                    _, client_id, accuracy = item
                    round_counts[client_id] += 1
                    round_num = round_counts[client_id]

                    if use_dp:
                        epsilon = privacy_metrics[client_id]['epsilon']
                        delta = privacy_metrics[client_id]['delta']
                        results.append([client_id, round_num, accuracy, epsilon, delta, tee_status[client_id]])
                    else:
                        results.append([client_id, round_num, accuracy, tee_status[client_id]])

                elif item[0] == 'privacy' and use_dp:
                    _, client_id, epsilon, delta = item
                    privacy_metrics[client_id]['epsilon'] = epsilon
                    privacy_metrics[client_id]['delta'] = delta
                    
                elif item[0] == 'tee_status' and use_tee:
                    _, client_id, status = item
                    if status in ['initialized', 'protected']:
                        tee_status[client_id] = True
                        
                elif item[0] == 'server_event':
                    _, event_type, message = item
                    print(f"[MAIN] Final server event ({event_type}): {message}")
            except:
                break

        # Save results
        save_results(results, csv_path, json_path, use_dp, use_tee)

        print(f"[MAIN] Experiment completed: {experiment_name}")
        print(f"[MAIN] Results saved to: {results_dir}")
        print(f"[MAIN] Total results collected: {len(results)}")

        # Print final summary
        if use_dp and privacy_metrics:
            total_epsilon = sum(m['epsilon'] for m in privacy_metrics.values())
            avg_epsilon = total_epsilon / len(privacy_metrics)
            print(f"[MAIN] Final Privacy Budget - Total ε: {total_epsilon:.4f}, Average ε: {avg_epsilon:.4f}")

        if use_tee:
            tee_enabled_count = sum(1 for status in tee_status.values() if status)
            print(f"[MAIN] TEE Coverage: {tee_enabled_count}/{NUM_CLIENTS} clients ({tee_enabled_count/NUM_CLIENTS:.1%})")

        return True

    except Exception as e:
        print(f"[MAIN] Error in experiment: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        cleanup_processes(server_process, client_processes)


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\n[MAIN] Received interrupt signal. Shutting down...")
    sys.exit(0)


def main():
    """Main function to run experiments"""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Differential Privacy Federated Learning Experiments with TEE")
    parser.add_argument("--no-dp", action="store_true", help="Run without differential privacy")
    parser.add_argument("--use-tee", action="store_true", help="Enable TEE/SGX support")
    parser.add_argument("--noise-multiplier", type=float, default=1.0, help="Noise multiplier for DP (default: 1.0)")
    parser.add_argument("--experiment-name", type=str, default="default", help="Name for the experiment")
    parser.add_argument("--run-comparison", action="store_true", help="Run comparison experiments")
    parser.add_argument("--tee-comparison", action="store_true", help="Run TEE vs non-TEE comparison")
    parser.add_argument("--noise-sweep", action="store_true", help="Run experiments with different noise levels")

    args = parser.parse_args()

    print("=" * 70)
    print("DIFFERENTIAL PRIVACY FEDERATED LEARNING WITH TEE EXPERIMENT")
    print("=" * 70)

    success_count = 0
    total_experiments = 0

    try:
        if args.tee_comparison:
            # Run TEE comparison experiments
            print("\n[MAIN] Running TEE comparison experiments...")

            # Standard DP experiment
            total_experiments += 1
            print(f"\n--- Experiment {total_experiments}: DP without TEE ---")
            if run_experiment(use_dp=True, noise_multiplier=args.noise_multiplier, 
                              use_tee=False, experiment_name="dp_no_tee"):
                success_count += 1

            # DP + TEE experiment
            total_experiments += 1
            print(f"\n--- Experiment {total_experiments}: DP with TEE ---")
            if run_experiment(use_dp=True, noise_multiplier=args.noise_multiplier, 
                              use_tee=True, experiment_name="dp_with_tee"):
                success_count += 1

        elif args.run_comparison:
            # Run standard comparison between DP and non-DP
            use_tee = args.use_tee
            
            # Non-DP experiment
            total_experiments += 1
            print(f"\n--- Experiment {total_experiments}: Non-DP {'with TEE' if use_tee else 'baseline'} ---")
            if run_experiment(use_dp=False, use_tee=use_tee, 
                              experiment_name=f"non_dp{'_tee' if use_tee else '_baseline'}"):
                success_count += 1

            # DP experiment
            total_experiments += 1
            print(f"\n--- Experiment {total_experiments}: DP {'with TEE' if use_tee else 'standard'} ---")
            if run_experiment(use_dp=True, noise_multiplier=args.noise_multiplier, use_tee=use_tee,
                              experiment_name=f"dp{'_tee' if use_tee else '_standard'}"):
                success_count += 1

        elif args.noise_sweep:
            # Run experiments with different noise levels
            noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
            print(f"\n[MAIN] Running noise sweep experiments with levels: {noise_levels}")
            use_tee = args.use_tee

            for noise in noise_levels:
                total_experiments += 1
                experiment_name = f"noise_sweep_{noise}{'_tee' if use_tee else ''}"
                print(f"\n--- Experiment {total_experiments}: Noise multiplier {noise} {'with TEE' if use_tee else ''} ---")
                if run_experiment(use_dp=True, noise_multiplier=noise, use_tee=use_tee,
                                  experiment_name=experiment_name):
                    success_count += 1

        else:
            # Single experiment
            total_experiments += 1
            use_dp = not args.no_dp
            experiment_name = args.experiment_name
            
            if args.use_tee:
                experiment_name += "_tee"
            
            print(f"\n--- Running single experiment: {experiment_name} ---")
            if run_experiment(use_dp=use_dp, noise_multiplier=args.noise_multiplier, 
                              use_tee=args.use_tee, experiment_name=experiment_name):
                success_count += 1

        # Final summary
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"Total experiments: {total_experiments}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_experiments - success_count}")
        print(f"Success rate: {success_count/total_experiments:.1%}" if total_experiments > 0 else "No experiments run")

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    except Exception as e:
        print(f"[MAIN] Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[MAIN] Main process finished.")


if __name__ == "__main__":
    main() 