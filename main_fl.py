# import subprocess
# import sys
# import time
# import csv
# import os
#
# PYTHON = sys.executable
# NUM_CLIENTS = 5
# PORT = "8086"
#
#
# # Step 1: Start server
# print("[MAIN] Starting Flower classic server...")
# server_process = subprocess.Popen([PYTHON, "flower/server.py"])
#
# time.sleep(3)
#
#
# # Step 2: Create results folder
# os.makedirs("flower/results", exist_ok=True)
# csv_path = "flower/results/fl_metrics.csv"
# with open(csv_path, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["client_id", "round", "accuracy"])
#
#
# # Step 3: Start clients
# client_processes = []
# for client_id in range(1, NUM_CLIENTS + 1):
#     print(f"[MAIN] Starting client {client_id}...")
#     proc = subprocess.Popen(
#         [PYTHON, "flower/client.py", str(client_id)],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         text=True
#     )
#     client_processes.append((client_id, proc))
#     time.sleep(5)
#
#
# # Step 4: Wait for clients to finish and collect output
# for client_id, proc in client_processes:
#     print(f"[MAIN] Waiting for client {client_id} to finish...")
#     stdout, _ = proc.communicate()  # Wait and collect output
#     round_num = 0
#     for line in stdout.splitlines():
#         if "accuracy:" in line.lower():
#             print(f"[client {client_id}] {line.strip()}")
#             acc_str = line.strip().split("accuracy:")[-1].strip()
#             try:
#                 accuracy = float(acc_str)
#                 round_num += 1
#                 with open(csv_path, mode="a", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([client_id, round_num, accuracy])
#             except ValueError:
#                 continue
#
#
# # Step 5: Wait for clients to finish
# for _, proc in client_processes:
#     proc.wait()
#
# print("[MAIN] All clients completed. You may close the server manually.")
import subprocess
import sys
import time
import csv
import os
import signal
import threading
from queue import Queue

PYTHON = sys.executable
NUM_CLIENTS = 5
PORT = "8086"


def monitor_client_output(client_id, process, output_queue):
    """Monitor client output and extract accuracy metrics"""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                line = line.strip()
                print(f"[CLIENT {client_id}] {line}")
                if "accuracy:" in line.lower():
                    try:
                        acc_str = line.split("accuracy:")[-1].strip()
                        accuracy = float(acc_str)
                        output_queue.put((client_id, accuracy))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"[MAIN] Error monitoring client {client_id}: {e}")


def cleanup_processes(server_process, client_processes):
    """Clean up all processes"""
    print("[MAIN] Cleaning up processes...")

    # Terminate client processes
    for client_id, proc in client_processes:
        if proc.poll() is None:  # Process is still running
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


def main():
    server_process = None
    client_processes = []

    try:
        # Step 1: Create results folder
        os.makedirs("fl/results", exist_ok=True)
        csv_path = "fl/results/fl_metrics.csv"
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["client_id", "round", "accuracy"])

        # Step 2: Start server
        print("[MAIN] Starting Flower server...")
        server_process = subprocess.Popen(
            [PYTHON, "fl/server.py"],
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
            return

        # Step 3: Start clients
        output_queue = Queue()
        monitor_threads = []

        for client_id in range(1, NUM_CLIENTS + 1):
            print(f"[MAIN] Starting client {client_id}...")
            try:
                proc = subprocess.Popen(
                    [PYTHON, "fl/client.py", str(client_id)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                client_processes.append((client_id, proc))

                # Start monitoring thread for this client
                monitor_thread = threading.Thread(
                    target=monitor_client_output,
                    args=(client_id, proc, output_queue)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                monitor_threads.append(monitor_thread)

                time.sleep(2)  # Stagger client starts

            except Exception as e:
                print(f"[MAIN] Error starting client {client_id}: {e}")

        print(f"[MAIN] Started {len(client_processes)} clients. Waiting for federated learning to complete...")

        # Step 4: Collect metrics from output queue
        round_counts = {i: 0 for i in range(1, NUM_CLIENTS + 1)}
        timeout_counter = 0
        max_timeout = 300  # 5 minutes timeout

        while any(proc.poll() is None for _, proc in client_processes) and timeout_counter < max_timeout:
            try:
                # Check for new accuracy metrics
                while not output_queue.empty():
                    client_id, accuracy = output_queue.get_nowait()
                    round_counts[client_id] += 1
                    round_num = round_counts[client_id]

                    # Save metrics to CSV
                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([client_id, round_num, accuracy])

                    print(f"[MAIN] Client {client_id}, Round {round_num}, Accuracy: {accuracy:.4f}")

                time.sleep(1)
                timeout_counter += 1

            except KeyboardInterrupt:
                print("\n[MAIN] Interrupted by user. Cleaning up...")
                break
            except Exception as e:
                print(f"[MAIN] Error in main loop: {e}")
                break

        # Step 5: Wait for all processes to complete
        print("[MAIN] Waiting for all clients to finish...")
        for client_id, proc in client_processes:
            try:
                proc.wait(timeout=30)
                print(f"[MAIN] Client {client_id} finished")
            except subprocess.TimeoutExpired:
                print(f"[MAIN] Client {client_id} timed out, terminating...")
                proc.terminate()

        # Wait for monitoring threads
        for thread in monitor_threads:
            thread.join(timeout=5)

        print(f"[MAIN] Federated learning completed. Results saved to {csv_path}")
        print(f"[MAIN] Total rounds completed per client: {dict(round_counts)}")

    except Exception as e:
        print(f"[MAIN] Unexpected error: {e}")

    finally:
        # Always cleanup processes
        cleanup_processes(server_process, client_processes)
        print("[MAIN] All processes cleaned up. Exiting.")


if __name__ == "__main__":
    main()