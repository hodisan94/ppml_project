import time
import subprocess

def simulate_timing(run_with_tee=True):
    cmd = ["python3", "dp_client_tee.py", "1", "true", "1.0", "true" if run_with_tee else "false"]

    start = time.time()
    subprocess.run(cmd)
    duration = time.time() - start

    mode = "With TEE" if run_with_tee else "Without TEE"
    print(f"[{mode}] Execution time: {duration:.2f} seconds")

if __name__ == "__main__":
    simulate_timing(run_with_tee=False)
    simulate_timing(run_with_tee=True)
