import sys, os
import time

from mpi4py import MPI
import pandas as pd
import numpy as np

sys.path.append("/home/kwvanarem/xthreat-research-v1/run-18-03-2025/")
from xThreat import xThreat

# True model path
true_model_path = "/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/true-models"
resampled_model_path = "/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/resampled-models"


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

# The grid sizes: only consider (16,12) for this part
grid_sizes = [
    (16, 12), #(32, 24), (40, 30), (48, 36), (56, 42), (64, 48), 
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
data_path = '../1-data-preparation/data-storage/xThreat_data_v4_ligue1_1516.parquet' # version 4 has player position added
df_events = pd.read_parquet(data_path, engine='fastparquet')
df_events['shot'] = ~df_events['shot_outcome'].isna()
df_events['goal'] = df_events['shot_outcome'] == 'Goal'
df_events.drop(columns=['id', 'shot_outcome', 'possession'], inplace=True)
df_size_bytes = df_events.memory_usage(deep=True).sum()
df_size_mb = df_size_bytes / (1024 ** 2)  # Convert to MB
if rank == 0:
    print(f"DataFrame size: {df_size_mb:.2f} MB")

# Train an xThreat model and apply _filter_out_of_bounds.
# In this way, any out-of-bounds events are removed from the data beforehand.
xThreat_prefit = xThreat(16, 12)
xThreat_prefit.fit(df_events)
df_events = xThreat_prefit._filter_out_of_bounds(df_events)

cell_begin_time = time.time()

# Initialize what to store
n_x, n_y = 16, 12
results_list = []

# Calculate it for the true model
xT_true = xThreat.load_from_pickle(os.path.join(true_model_path, f"xt-full-data-n_x{n_x}-n_y{n_y}.pickle"))
if rank == 0:
    print('Calculating xT ratings for true model...')
    df_positive_true = xT_true.quantify_action_quality(df_events)
    df_positive_true = df_positive_true[df_positive_true['action_xT'] > 0].copy()
    xT_ratings_true = df_positive_true.groupby(['player_id', 'position']).agg(
        xT_sum_true = ('action_xT', 'sum'),
        n_actions = ('action_xT', 'count'),
    )
    xT_ratings_true[['n_x', 'n_y', 'model_type']] = n_x, n_y, 'true'
    results_list.append(xT_ratings_true.reset_index())


print(f'Rank {rank} starting resampled model calculations...')

# Loop over all combinations of sample params
for sample_size in sample_sizes:
    for i_bootstrap in range(rank, n_bootstraps, size):
        # Get the random state and partition number
        random_state, i_partition = sampling_params[(sample_size, n_x, n_y, i_bootstrap)]

        # Skip if the partition is not the one we want
        if i_partition != partition:
            continue

        # Get the sampled model
        xT_sampled = xThreat.load_from_pickle(
            os.path.join(
                resampled_model_path, 
                f"xt-N{sample_size}-n_x{n_x}-n_y{n_y}-i_bootstrap{i_bootstrap}-random_state{random_state}.pickle",
                ))

        # Calculate the xT ratings for the players
        df_positive_true = xT_sampled.quantify_action_quality(df_events)
        df_positive = df_positive_true[df_positive_true['action_xT'] > 0].copy()

        xT_ratings_resampled = df_positive.groupby(['player_id', 'position']).agg(
            xT_sum_resampled = ('action_xT', 'sum'),
            n_actions = ('action_xT', 'count'),
        )

        xT_ratings_resampled[['n_x', 'n_y', 'sample_size', 'i_bootstrap', 'random_state', 'model_type']] = n_x, n_y, sample_size, i_bootstrap, random_state, 'resampled'

        results_list.append(xT_ratings_resampled.reset_index())


    if rank == 0:
        print(f'Rank {rank} finished sample_size {sample_size}.')
        


all_results = comm.gather(results_list, root=0)
if rank == 0:
    results_list = []
    for res in all_results:
        results_list.extend(res)

    df_results = pd.concat(results_list, ignore_index=True)
    output_path = f'xT_ratings_resampled.parquet'
    df_results.to_parquet(output_path, engine='fastparquet')
    print(f'Saved results to {output_path}')
    print(f'Total script time: {time.time() - script_starting_time:.2f} seconds')






