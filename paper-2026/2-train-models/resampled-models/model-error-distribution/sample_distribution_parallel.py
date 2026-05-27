import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from xthreat import xThreat

time.sleep(10*60)

script_starting_time = time.time()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the process
size = comm.Get_size()  # Total number of processes
print(f"Rank: {rank}, Size: {size}")

# Specify the partition
partition = 0

# Total number of bootstraps
n_bootstraps = 1000 #40 #00 

# Specify the sample sizes
sample_sizes = [
    4000000, 1_300_000,  630_000,  370_000,  240_000,  170_000,  130_000,  100_000,
    ]

# The grid sizes
grid_sizes = [
    # (8, 6), (10, 8), (12, 9),  (14, 11), # New added grids
    # (20, 15), (24, 18), (28, 21),   # New added grids
    (16, 12), (32, 24), (40, 30), (48, 36), (56, 42), (64, 48), 
              ]

# The bootstraps performed in one run: to keep the computation time for the cluster
max_partition_size = 1042

# Create a list with all combinations and assign a unique random_state and partition number
sampling_params = {}
for n_x, n_y in grid_sizes:
    for sample_size in sample_sizes:
        for i_bootstrap in range(n_bootstraps):
            random_state = 42 + (n_x + n_y) * n_bootstraps * 42 + sample_size + i_bootstrap
            i_partition = i_bootstrap // max_partition_size
            sampling_params[(sample_size, n_x, n_y, i_bootstrap)] = (random_state, i_partition)
if rank == 0:
    print(f'Number of individual bootstraps: {len(sampling_params)}')

# Load the data
data_path = '../../../1-data-preparation/data-storage/xThreat_data_v3.parquet'
df_events = pd.read_parquet(data_path, engine='fastparquet')
df_events['shot'] = ~df_events['shot_outcome'].isna()
df_events['goal'] = df_events['shot_outcome'] == 'Goal'
df_events.drop(columns=['id', 'shot_outcome', 'possession'], inplace=True)
df_size_bytes = df_events.memory_usage(deep=True).sum()
df_size_mb = df_size_bytes / (1024 ** 2)  # Convert to MB
if rank == 0:
    print(f"DataFrame size: {df_size_mb:.2f} MB")

# Train an xThreat model and apply _filter_out_of_bounds.
# In this way, the other models won't have to do that during fit.
# Will speed up computations.
xThreat_prefit = xThreat(16, 12)
xThreat_prefit.fit(df_events)
df_events = xThreat_prefit._filter_out_of_bounds(df_events)

cell_begin_time = time.time()

max_size_stored_model = 0
sum_size_stored_model = 0
n = 0

sampled_params = {}

for n_x, n_y in grid_sizes:
    begin_time = time.time()
    
    # Fit the 'true' xThreat model
    xT_true = xThreat(n_x, n_y)
    xT_true.fit(df_events, filter_events=False, convergence_threshold=1e-9)

    for sample_size in sample_sizes:
        for i_bootstrap in range(rank, n_bootstraps, size):
            # Get the random state and partition
            random_state, i_partition = sampling_params[(sample_size, n_x, n_y, i_bootstrap)]

            # Skip if the partition is not the one we want
            if i_partition != partition:
                continue
            
            # Sample from the true model
            df_sample = xT_true.sample(sample_size, random_state=random_state)

            # Fit a model on the sample
            xT_resampled = xThreat(n_x, n_y)
            xT_resampled.fit(df_sample, filter_events=False)

            # Save the resampled method
            file_path = f'/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/resampled-models/xt-N{sample_size}-n_x{n_x}-n_y{n_y}-i_bootstrap{i_bootstrap}-random_state{random_state}.pickle'
            xT_resampled.save_to_pickle(file_path)

            # Keep track of the storage size of the models
            file_size = os.path.getsize(file_path)
            max_size_stored_model = max(max_size_stored_model, file_size)
            n += 1
            sum_size_stored_model += file_size
    
    if rank == 0:
        # Print the time it took to do the samples for this grid size
        hours, remainder = divmod(time.time()-begin_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f'Performed the {n_bootstraps} bootstrap with grid {(n_x, n_y)} within {int(hours)}h, {int(minutes)}m, {int(seconds)}s.\n')

# Synchronize processes and ensure all print statements from processes
comm.Barrier()

# Only the root process (rank 0) prints the final statistics
if rank == 0:
    hours, remainder = divmod(time.time()-cell_begin_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f'The whole resampling process took {int(hours)}h, {int(minutes)}m, {int(seconds)}s')
    print(f'The maximal size of the transition matrix was {max_size_stored_model/1024**2:.2f}MB')
    print(f'The average size of the stored models was {sum_size_stored_model/n/1024**2:.2f}MB for {n} models')
    print(f'The total size of the stored models was {sum_size_stored_model/1024**2:.2f}MB')

    print(f'\nThe whole script took {int((time.time()-script_starting_time)/3600)}h, {int((time.time()-script_starting_time)%3600/60)}m, {int((time.time()-script_starting_time)%60)}s')
