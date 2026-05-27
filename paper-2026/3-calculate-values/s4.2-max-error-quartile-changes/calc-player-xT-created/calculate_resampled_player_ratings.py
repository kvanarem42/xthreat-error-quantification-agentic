import os
import time

from mpi4py import MPI
import pandas as pd
import numpy as np

from xthreat import xThreat

# True model path
true_model_path = "/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/true-models"
resampled_model_path = "/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/resampled-models"


script_starting_time = time.time()

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the process
size = comm.Get_size()  # Total number of processes
print(f"Rank: {rank}, Size: {size}")

# Batch size for writing away the results
batch_size = 50

# Create parquet directory
output_dir = 'xT_ratings_resampled_10_000.parquet'
os.makedirs(output_dir, exist_ok=True)

# Total number of bootstraps
n_bootstraps = 10_000 #40 #00 

# Specify the sample sizes
sample_sizes = [
    4_000_000, 1_300_000,  630_000,  370_000,  240_000,  170_000,  130_000,  100_000,
    3_500_000, 3_000_000, 2_500_00, 2_000_000, 1_650_000,
    ]

# The grid sizes: only consider (16,12) for this part
grid_sizes = [
    (16, 12), # (32, 24), (40, 30), (48, 36), (56, 42), (64, 48), 
              ]

# Create a list with all combinations and assign a unique random_state
sampling_params = {}
for n_x, n_y in grid_sizes:
    for sample_size in sample_sizes:
        for i_bootstrap in range(n_bootstraps):
            random_state = 42 + (n_x + n_y) * n_bootstraps * 42 + sample_size + i_bootstrap
            sampling_params[(sample_size, n_x, n_y, i_bootstrap)] = random_state
if rank == 0:
    print(f'Number of individual bootstraps: {len(sampling_params)}')

# Load the data
data_path = '../../../1-data-preparation/data-storage/preprocessed_data_ligue1_1516.parquet' 
df_events = pd.read_parquet(data_path, engine='fastparquet')
df_events['shot'] = ~df_events['shot_outcome'].isna()
df_events['goal'] = df_events['shot_outcome'] == 'Goal'
df_events.drop(columns=['id', 'shot_outcome', 'possession', 'type', 'next_type', 'pass_height'], inplace=True)
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
    df_positive = xT_true.quantify_action_quality(df_events)
    df_positive = df_positive[df_positive['action_xT'] > 0]
    xT_ratings_true = df_positive.groupby(['player_id', 'position']).agg(
        xT_sum_true = ('action_xT', 'sum'),
        n_actions = ('action_xT', 'count'),
    )
    xT_ratings_true[['n_x', 'n_y', 'model_type']] = n_x, n_y, 'true'
    results_list.append(xT_ratings_true.reset_index())
    del df_positive, xT_true


print(f'Rank {rank} starting resampled model calculations...')

batch_id = 0
n_in_batch = 0
batch_info = []

# Loop over all combinations of sample params
for sample_size in sample_sizes:
    for i_bootstrap in range(rank, n_bootstraps, size):
        
        # Get the random state
        random_state = sampling_params[(sample_size, n_x, n_y, i_bootstrap)]

        # Get the sampled model
        xT_sampled = xThreat.load_from_pickle(
            os.path.join(
                resampled_model_path, 
                f"xt-N{sample_size}-n_x{n_x}-n_y{n_y}-i_bootstrap{i_bootstrap}-random_state{random_state}.pickle",
                ))

        # Calculate the xT ratings for the players
        df_positive = xT_sampled.quantify_action_quality(df_events)
        df_positive = df_positive[df_positive['action_xT'] > 0]

        xT_ratings_resampled = df_positive.groupby(['player_id', 'position']).agg(
            xT_sum_resampled = ('action_xT', 'sum'),
            n_actions = ('action_xT', 'count'),
        )
        
        del df_positive, xT_sampled

        xT_ratings_resampled[['n_x', 'n_y', 'sample_size', 'i_bootstrap', 'random_state', 'model_type']] = n_x, n_y, sample_size, i_bootstrap, random_state, 'resampled'

        results_list.append(xT_ratings_resampled.reset_index())
        n_in_batch += 1
        
        # Store per batches
        if n_in_batch >= batch_size:
            # Save the batch
            df_batch = pd.concat(results_list, ignore_index=True)
            df_batch.to_parquet(
                os.path.join(output_dir, f"part-rank{rank}-batch{batch_id}.parquet"),
                engine="fastparquet",
                )
            del df_batch
            
            # Reset index and store info
            batch_info.append([rank, batch_id, n_in_batch])
            n_in_batch = 0
            results_list = []
            batch_id += 1

    if rank == 0:
        print(f'Rank {rank} finished sample_size {sample_size}.')
        
# Save remaining iterations
if n_in_batch > 0:
    # Save the batch
    df_batch = pd.concat(results_list, ignore_index=True)
    df_batch.to_parquet(
        os.path.join(output_dir, f"part-rank{rank}-batch{batch_id}.parquet"),
        engine="fastparquet",
        )
    
    # Reset index and store info
    batch_info.append([rank, batch_id, n_in_batch])


# Give overview
all_batches_info = comm.gather(batch_info, root=0)
if rank == 0:
    full_batch_info = []
    n_processed = 0
    for lst in all_batches_info:
        for res in lst:
            n_processed += res[2]
        full_batch_info.extend(lst)
    print(f"The script has been completed. \nProcessed {n_processed} different models.")
    print(f'Saved results to {output_dir}')
    print(f'Total script time: {time.time() - script_starting_time:.2f} seconds')






