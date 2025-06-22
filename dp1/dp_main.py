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

PYTHON = sys.executable
NUM_CLIENTS = 5
PORT = "8086"


def monitor_client_output(client_id, process, output_queue, use_dp=True):
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
                        accuracy = float(acc_str)
                        output_queue.put(('accuracy', client_id, accuracy))
                    except (ValueError, IndexError):
                        continue
                
                # Extract privacy metrics if DP is used
                if use_dp and "privacy spent:" in line.lower():
                    try:
                        # Parse: "Privacy spent: ε=0.1234, δ=1.00e-05"
                        privacy_part = line.split("privacy spent:")[-1].strip()
                        epsilon_part = privacy_part.split("ε=")[1].split(",")[0]
                        delta_part = privacy_part.split("δ=")[1]
                        
                        epsilon = float(epsilon_part)
                        delta = float(delta_part)
                        
                        output_queue.put(('privacy', client_id, epsilon, delta))
                    except (ValueError, IndexError):
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


def save_results(results, csv_path, json_path, use_dp=True):
    """Save results to CSV and JSON files"""
    # Save to CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        if use_dp:
            writer.writerow(["client_id", "round", "accuracy", "epsilon", "delta"])
        else:
            writer.writerow(["client_id", "round", "accuracy"])
        
        for result in results:
            writer.writerow(result)
    
    # Save summary to JSON
    summary = {
        "use_dp": use_dp,
        "num_clients": NUM_CLIENTS,
        "total_results": len(results),
        "final_metrics": {}
    }
    
    if use_dp and results:
        # Calculate final privacy budget
        final_epsilons = {}
        final_accuracies = {}
        
        for result in results:
            client_id = result[0]
            accuracy = result[2]
            if len(result) > 3:
                epsilon = result[3]
                final_epsilons[client_id] = epsilon
            final_accuracies[client_id] = accuracy
        
        if final_epsilons:
            summary["final_metrics"] = {
                "total_epsilon": sum(final_epsilons.values()),
                "avg_epsilon": sum(final_epsilons.values()) / len(final_epsilons),
                "avg_accuracy": sum(final_accuracies.values()) / len(final_accuracies),
                "per_client_epsilon": final_epsilons,
                "per_client_accuracy": final_accuracies
            }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)


def run_experiment(use_dp=True, noise_multiplier=1.0, experiment_name="default"):
    """Run a single experiment with given parameters"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", experiment_name)

    server_process = None
    client_processes = []
    
    # Create results directory
    # results_dir = f"results/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, "fl_metrics.csv")
    json_path = os.path.join(results_dir, "experiment_summary.json")
    
    try:
        print(f"\n[MAIN] Starting experiment: {experiment_name}")
        print(f"[MAIN] DP: {use_dp}, Noise Multiplier: {noise_multiplier}")
        
        # Start server
        print("[MAIN] Starting Flower server...")
        server_process = subprocess.Popen(
            [PYTHON, "-u", "dp_server.py", str(use_dp).lower()],
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
                # cmd = [PYTHON, "-u", "dp_client.py", str(client_id), str(use_dp).lower()] + ([str(noise_multiplier)] if use_dp else [])
                # if use_dp:
                #     cmd.append(str(noise_multiplier))
                cmd = [PYTHON, "-u", "dp_client.py", str(client_id), str(use_dp).lower()]
                if use_dp:
                    cmd.append(str(noise_multiplier))
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
                    args=(client_id, proc, output_queue, use_dp)
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
        privacy_metrics = {i: {'epsilon': 0.0, 'delta': 0.0} for i in range(1, NUM_CLIENTS + 1)}
        
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
                            # Include current privacy metrics
                            epsilon = privacy_metrics[client_id]['epsilon']

                            delta = privacy_metrics[client_id]['delta']
                            results.append([client_id, round_num, accuracy, epsilon, delta])
                        else:
                            results.append([client_id, round_num, accuracy])
                        
                        print(f"[MAIN] Client {client_id}, Round {round_num}, Accuracy: {accuracy:.4f}")
                    
                    elif item[0] == 'privacy' and use_dp:
                        _, client_id, epsilon, delta = item
                        privacy_metrics[client_id]['epsilon'] = item[2]
                        privacy_metrics[client_id]['delta'] = item[3]
                        # privacy_metrics[client_id]['epsilon'] = epsilon
                        # privacy_metrics[client_id]['delta'] = delta

                time.sleep(1)
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

        # Drain any remaining metrics from the queue before we save
        while not output_queue.empty():
            item = output_queue.get_nowait()
            if item[0] == 'accuracy':
                _, client_id, round_num, accuracy = item
                if use_dp:
                    eps = privacy_metrics[client_id]['epsilon']
                    delt = privacy_metrics[client_id]['delta']
                    results.append([client_id, round_num, accuracy, eps, delt])
                else:
                    results.append([client_id, round_num, accuracy])
            elif item[0] == 'privacy' and use_dp:
                _, client_id, epsilon, delta = item
                privacy_metrics[client_id] = {'epsilon': epsilon, 'delta': delta}

        # Save results
        save_results(results, csv_path, json_path, use_dp)
        # # Save results
        # save_results(results, csv_path, json_path, use_dp)
        
        print(f"[MAIN] Experiment completed: {experiment_name}")
        print(f"[MAIN] Results saved to: {results_dir}")
        
        if use_dp and privacy_metrics:
            total_epsilon = sum(m['epsilon'] for m in privacy_metrics.values())
            avg_epsilon = total_epsilon / len(privacy_metrics)
            print(f"[MAIN] Final Privacy Budget - Total ε: {total_epsilon:.4f}, Average ε: {avg_epsilon:.4f}")

        return True

    except Exception as e:
        print(f"[MAIN] Error in experiment: {e}")
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
    parser = argparse.ArgumentParser(description="Run Differential Privacy Federated Learning Experiments")
    parser.add_argument("--no-dp", action="store_true", help="Run without differential privacy")
    parser.add_argument("--noise-multiplier", type=float, default=1.0, help="Noise multiplier for DP (default: 1.0)")
    parser.add_argument("--experiment-name", type=str, default="default", help="Name for the experiment")
    parser.add_argument("--run-comparison", action="store_true", help="Run both DP and non-DP experiments")
    parser.add_argument("--noise-sweep", action="store_true", help="Run experiments with different noise levels")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIFFERENTIAL PRIVACY FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    
    success_count = 0
    total_experiments = 0
    
    try:
        if args.run_comparison:
            # Run comparison between DP and non-DP
            print("\n[MAIN] Running comparison experiments...")
            
            # Non-DP experiment
            total_experiments += 1
            print(f"\n--- Experiment {total_experiments}: Non-DP Baseline ---")
            if run_experiment(use_dp=False, experiment_name="non_dp_baseline"):
                success_count += 1
            
            # DP experiment
            total_experiments += 1
            print(f"\n--- Experiment {total_experiments}: DP with noise multiplier {args.noise_multiplier} ---")
            if run_experiment(use_dp=True, noise_multiplier=args.noise_multiplier, 
                            experiment_name=f"dp_noise_{args.noise_multiplier}"):
                success_count += 1
        
        elif args.noise_sweep:
            # Run experiments with different noise levels
            noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
            print(f"\n[MAIN] Running noise sweep experiments with levels: {noise_levels}")
            
            for noise in noise_levels:
                total_experiments += 1
                print(f"\n--- Experiment {total_experiments}: DP with noise multiplier {noise} ---")
                if run_experiment(use_dp=True, noise_multiplier=noise, 
                                experiment_name=f"dp_noise_sweep_{noise}"):
                    success_count += 1
        
        else:
            # Single experiment
            total_experiments += 1
            use_dp = not args.no_dp
            experiment_type = "DP" if use_dp else "Non-DP"
            print(f"\n--- Single {experiment_type} Experiment ---")
            
            if run_experiment(use_dp=use_dp, noise_multiplier=args.noise_multiplier, 
                            experiment_name=args.experiment_name):
                success_count += 1
    
    except KeyboardInterrupt:
        print("\n[MAIN] Experiments interrupted by user.")
    except Exception as e:
        print(f"[MAIN] Unexpected error: {e}")
    
    finally:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total experiments: {total_experiments}")
        print(f"Successful experiments: {success_count}")
        print(f"Failed experiments: {total_experiments - success_count}")
        
        if success_count > 0:
            print(f"\nResults saved in: results/")
            print("Files generated:")
            print("- fl_metrics.csv: Detailed metrics for each round")
            print("- experiment_summary.json: Experiment summary and final metrics")
        
        print("\nExperiment completed!")


if __name__ == "__main__":
    main()