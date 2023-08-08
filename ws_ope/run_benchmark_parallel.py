#!/usr/bin/env python3

import subprocess
import concurrent.futures
import os

# Configuration
algo_values = ["dr" ] #["mb", "fqe"] #
policy_class = "trifinger_rl_example.example.TorchRandomPolicy"
std_values = ["0.2"]
max_concurrent_processes = 1  # Set the number of concurrent processes here

# Create the logs directory if it doesn't exist
if not os.path.exists('logs_benchmark_parallel'):
    os.makedirs('logs_benchmark_parallel')

# Define a function to run a single job
def run_job(algo, std):
    num_updates = 500000
    
    extra_flags = "--target_policy_noisy" if std != "0.05" else ""
    
    command = (
        f"python -m policy_eval.train_eval_trifinger --logtostderr --trifinger "
        f"--env_name=trifinger-cube-push-real-mixed-v0 "
        f"--trifinger_policy_class={policy_class} "
        f"--target_policy_std={std} --nobootstrap --algo={algo} "
        f"--noise_scale=0.0 --num_updates={num_updates} --discount=0.995 {extra_flags}"
    )
    
    # Log file path
    log_file_path = f"logs_benchmark_parallel/output_algo_{algo}_std_{std}.log"
    
    # Execute the command and redirect stdout and stderr to a file
    with open(log_file_path, 'w') as log_file:
        subprocess.run(command, shell=True, stdout=log_file, stderr=log_file)

# Using concurrent.futures to run multiple processes in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
    # Launch all the jobs
    futures = [
        executor.submit(run_job, algo, std)
        for algo in algo_values
        for std in std_values
    ]
    
    # Wait for all jobs to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Job raised an exception: {e}")