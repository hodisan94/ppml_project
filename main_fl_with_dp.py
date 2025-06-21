import subprocess
import sys
import time
import csv
import os
import signal
import threading
import re
from queue import Queue

PYTHON = sys.executable
NUM_CLIENTS = 5
PORT = "8086"


def monitor_client_output(client_id, process, output_queue):
    """Monitor client output and extract accuracy and epsilon metrics"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                print(f"[CLIENT {client_id}] {line}")
                if "accuracy:" in line.lower():
                    try:
                        acc_str = line.split("accuracy:")[-1].strip()
                        accuracy = float(acc_str)
                        output_queue.put(("accuracy", client_id, accuracy))
                    except (ValueError, IndexError):
                        continue
                elif "ε after round" in line or "epsilon" in line.lower():
                    try:
                        eps_match = re.search(r"[-+]?\d*\.\d+|\d+", line)
                        if eps_match:
                            epsilon = float(eps_match.group())
                            output_queue.put(("epsilon", client_id, epsilon))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"[MAIN] Error monitoring client {client_id}: {e}")


def cleanup_processes(server_process, client_processes):
    """Clean up all processes"""
    print("[MAIN] Cleaning up processes...")

    for client_id, proc in client_processes:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception as e:
                print(f"[MAIN] Error terminating client {client_id}: {e}")

    if server_process and server_process.poll() is None:
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        except Exception as e:
            print(f"[MAIN] Error terminating server: {e}")


def main():
    server_process = None
    client_processes = []

    try:
        os.makedirs("dp/results", exist_ok=True)
        csv_path = "dp/results/fl_with_dp_metrics.csv"
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["client_id", "round", "accuracy", "epsilon"])

        print("[MAIN] Starting Flower server...")
        server_process = subprocess.Popen(
            [PYTHON, "fl/server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        time.sleep(5)

        if server_process.poll() is not None:
            stdout, _ = server_process.communicate()
            print(f"[MAIN] Server failed to start. Output: {stdout}")
            return

        output_queue = Queue()
        monitor_threads = []

        round_counts = {i: 0 for i in range(1, NUM_CLIENTS + 1)}
        client_epsilons = {i: [] for i in range(1, NUM_CLIENTS + 1)}
        client_accuracies = {i: [] for i in range(1, NUM_CLIENTS + 1)}

        for client_id in range(1, NUM_CLIENTS + 1):
            print(f"[MAIN] Starting client {client_id}...")
            try:
                proc = subprocess.Popen(
                    [PYTHON, "dp/client_dp.py", str(client_id)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                client_processes.append((client_id, proc))

                monitor_thread = threading.Thread(
                    target=monitor_client_output,
                    args=(client_id, proc, output_queue)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                monitor_threads.append(monitor_thread)

                time.sleep(2)

            except Exception as e:
                print(f"[MAIN] Error starting client {client_id}: {e}")

        print("[MAIN] All clients started. Collecting federated learning metrics...")

        timeout_counter = 0
        max_timeout = 300

        while any(proc.poll() is None for _, proc in client_processes) and timeout_counter < max_timeout:
            try:
                while not output_queue.empty():
                    metric_type, client_id, value = output_queue.get_nowait()

                    if metric_type == "accuracy":
                        round_counts[client_id] += 1
                        client_accuracies[client_id].append(value)
                    elif metric_type == "epsilon":
                        client_epsilons[client_id].append(value)

                for client_id in range(1, NUM_CLIENTS + 1):
                    while len(client_accuracies[client_id]) > 0 and len(client_epsilons[client_id]) > 0:
                        acc = client_accuracies[client_id].pop(0)
                        eps = client_epsilons[client_id].pop(0)
                        round_num = round_counts[client_id]

                        with open(csv_path, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([client_id, round_num, acc, eps])

                        print(f"[MAIN] Client {client_id}, Round {round_num}, Accuracy: {acc:.4f}, Epsilon: {eps:.4f}")

                time.sleep(1)
                timeout_counter += 1

            except KeyboardInterrupt:
                print("\n[MAIN] Interrupted by user. Cleaning up...")
                break
            except Exception as e:
                print(f"[MAIN] Error in main loop: {e}")
                break

        print("[MAIN] Waiting for all clients to finish...")
        for client_id, proc in client_processes:
            try:
                proc.wait(timeout=30)
                print(f"[MAIN] Client {client_id} finished")
            except subprocess.TimeoutExpired:
                print(f"[MAIN] Client {client_id} timed out, terminating...")
                proc.terminate()

        for thread in monitor_threads:
            thread.join(timeout=5)

        print(f"[MAIN] Federated learning completed. Results saved to {csv_path}")
        print(f"[MAIN] Total rounds completed per client: {dict(round_counts)}")

    except Exception as e:
        print(f"[MAIN] Unexpected error: {e}")

    finally:
        cleanup_processes(server_process, client_processes)
        print("[MAIN] All processes cleaned up. Exiting.")


if __name__ == "__main__":
    main()
