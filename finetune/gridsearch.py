import itertools
import subprocess
import threading
import time

import argparse


import torch


# Parse additional arguments for gridsearch.py
parser = argparse.ArgumentParser(description='Grid Search parameters')
parser.add_argument('-data', '--dataset', type=str, default='pets', help='Dataset to use')
parser.add_argument('-sd', '--seed', type=int, default=0, help='Random seed')
grid_args = parser.parse_args()

# Use these variables later to pass to finetune_lora.py
DATASET = grid_args.dataset
SEED = grid_args.seed


num_devices = torch.cuda.device_count()
if num_devices < 1:
    print("Please select a machine with at least 1 GPU.")
    quit()

    
# rank_values = [1, 2, 4, 8, 16, 32]
# alpha_values = [1, 4, 8, 16, 32, 64]
# dropout_values = [0, 0.05, 0.1]

rank_values = [1]
alpha_values = [1]
dropout_values = [0]

devices = range(num_devices)


lora_query = ["True"]
lora_key = ["True"]
lora_value = ["True"]
lora_projection = ["True"]
lora_mlp = ["True"]
lora_head = ["True"]


# Dictionary to keep track of whether a device is currently in use
device_usage = {device: False for device in devices}

# Set to keep track of used hyperparameter combinations
used_combinations = set()


def run_script(alpha, rank, dropout, device, 
               query, key, value, projection, mlp, head,
               ):

    global device_usage

    command = [
        'python', 'finetune_lora.py',
        '--lora_r', str(rank),
        '--lora_alpha', str(alpha),
        '--lora_dropout', str(dropout),
        '--device', str(device),
        '--lora_query', query,
        '--lora_key', key,
        '--lora_value', value,
        '--lora_projection', projection,
        '--lora_mlp', mlp,
        '--lora_head', head,
        '--verbose', "False",
        '-data', DATASET,  # Add this line
        '-sd', str(SEED),  # Add this line
    ]

    print(f"Starting run with rank = {rank}, alpha = {alpha}, dropout = {dropout} "
          f"lora_query = {query}, lora_key = {key}, "
          f"lora_value = {value}, lora_projection = {projection}, "
          f"lora_mlp = {mlp}, lora_head = {head}, "
          f" on device {device}")
    
    subprocess.run(command)
    
    print(f"Completed run with rank = {rank}, alpha = {alpha}, dropout = {dropout} "
          f"lora_query = {query}, lora_key = {key}, "
          f"lora_value = {value}, lora_projection = {projection}, "
          f"lora_mlp = {mlp}, lora_head = {head}, "
          f" on device {device}")

    # Mark the device as no longer in use
    device_usage[device] = False


def get_available_device():
    while True:
        for device, in_use in device_usage.items():
            if not in_use:
                device_usage[device] = True
                return device
        time.sleep(10)  # Wait before checking again


threads = []

# Using itertools.product to create combinations
for params in itertools.product(rank_values, alpha_values, dropout_values,
                                lora_query, lora_key, lora_value,
                                lora_projection, lora_mlp, lora_head,
                                ):
    
    rank, alpha, dropout, query, key, value, projection, mlp, head = params

    # Check if the combination has already been used
    if ((rank, alpha, dropout,
         query, key, value, projection, mlp, head) in used_combinations):
        continue  # Skip this combination as it's already used

    # Mark the combination as used
    used_combinations.add((rank, alpha, dropout,
                           query, key, value, projection, mlp, head))

    device = get_available_device()
    thread = threading.Thread(target=run_script, args=(rank, alpha, dropout, device,
                                                       query, key, value,
                                                       projection, mlp, head))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All runs completed.")