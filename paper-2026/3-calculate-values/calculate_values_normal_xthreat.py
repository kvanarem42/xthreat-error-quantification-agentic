########################################################################
### This script calculates the errors of the resampled xThreat models
### models compared to the true xThreat model.
### The errors are calculated for different norms (1 and inf; 
### 2 norm is left out for computational speed).
###
### The script is run on the DelftBlue computing cluster to speed up  
### the calculations.
########################################################################

### Import statements ###
import time
import os
import pandas as pd
import numpy as np
from mpi4py import MPI


from xthreat import xThreat

script_starting_time = time.time()

############## Support functions ##############
def create_error_objects(xT_true, xT):
    # Calculate the error for attributes of the Expected Threat model
    xT_error = xT_true.xT-xT.xT
    g_error = xT_true.xG * xT_true.probability_shooting - xT.xG * xT.probability_shooting
    xG_error = xT_true.xG - xT.xG
    prob_shot_error = xT_true.probability_shooting - xT.probability_shooting
    transition_matrix_error = xT_true.transition_matrix[:xT_true.n_game_states,:xT_true.n_game_states] \
        - xT.transition_matrix[:xT.n_game_states,:xT.n_game_states]
    weighted_transition_matrix_error = transition_matrix_error * xT.xT[np.newaxis, :]
    
    return xT_error, g_error, xG_error, prob_shot_error, transition_matrix_error, weighted_transition_matrix_error

def calculate_error_per_norm(xT_true, xT, norm):
    """
    Calculate the error for attributes of the Expected Threat model.
    Parameters:
    - xT_true (xThreat): The true Expected Threat model
    - xT (xThreat): The Expected Threat model to compare to
    - norm (int): The norm to use in the calculation. 
        See numpy.linalg.norm for more information.

    Returns:
    - xT_error_norm (float): The error in xT
    - g_error_norm (float): The error in g
    - xG_error_norm (float): The error in xG
    - prob_shot_error_norm (float): The error in the probability of shooting
    - transition_matrix_error_norm (float): The error in the transition
        matrix
    """
    xT_error, g_error, xG_error, prob_shot_error, transition_matrix_error, weighted_transition_matrix_error_norm = create_error_objects(xT_true, xT)
    
    xT_error_norm = np.linalg.norm(xT_error, ord=norm)
    g_error_norm = np.linalg.norm(g_error, ord=norm)
    xG_error_norm = np.linalg.norm(xG_error, ord=norm)
    prob_shot_error_norm = np.linalg.norm(prob_shot_error, ord=norm)
    transition_matrix_error_norm = np.linalg.norm(transition_matrix_error, ord=norm)
    weighted_transition_matrix_error_norm = np.linalg.norm(weighted_transition_matrix_error_norm, ord=norm)
    
    return xT_error_norm, g_error_norm, xG_error_norm, prob_shot_error_norm, transition_matrix_error_norm, weighted_transition_matrix_error_norm

def add_errors_to_dict(dict_errors, true_model_path, resampled_model_path, n_x, n_y, sample_size, i_bootstrap, random_state, norms):

    xT_true  = xThreat.load_from_pickle(f'{true_model_path}xt-full-data-n_x{n_x}-n_y{n_y}.pickle')
    xT_sampled = xThreat.load_from_pickle(f'{resampled_model_path}xt-N{sample_size}-n_x{n_x}-n_y{n_y}-i_bootstrap{i_bootstrap}-random_state{random_state}.pickle')

    dict_errors['n_x'].append(n_x)
    dict_errors['n_y'].append(n_y)
    dict_errors['sample_size'].append(sample_size)
    dict_errors['i_bootstrap'].append(i_bootstrap)
    dict_errors['random_state'].append(random_state)
    

    for norm in norms:
        xT_error_norm, g_error_norm, xG_error_norm, prob_shot_error_norm, transition_matrix_error_norm, weighted_transition_matrix_error_norm = calculate_error_per_norm(xT_true, xT_sampled, norm)
        dict_errors[f'xT_error_{norm}_norm'].append(xT_error_norm)
        dict_errors[f'g_error_{norm}_norm'].append(g_error_norm)
        dict_errors[f'xG_error_{norm}_norm'].append(xG_error_norm)
        dict_errors[f'prob_shot_error_{norm}_norm'].append(prob_shot_error_norm)
        dict_errors[f'transition_matrix_error_{norm}_norm'].append(transition_matrix_error_norm)
        dict_errors[f'transition_matrix_true_{norm}_norm'].append(np.linalg.norm(xT_true.transition_matrix, ord=norm))
        dict_errors[f'transition_matrix_resampled_{norm}_norm'].append(np.linalg.norm(xT_sampled.transition_matrix, ord=norm))
        dict_errors[f'weighted_transition_matrix_error_{norm}_norm'].append(weighted_transition_matrix_error_norm)

    return dict_errors

############## Calculations ##############

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get process rank
size = comm.Get_size()  # Get total number of processes

# Total number of bootstraps
n_bootstraps = 1000

# Specify the sample sizes
sample_sizes = [
    4000000, 1_300_000, 630_000, 370_000, 240_000, 170_000, 130_000, 100_000,
]

# The grid sizes
grid_sizes = [
    (16, 12), (32, 24), (40, 30), (48, 36), (56, 42), (64, 48),
]

# The bootstraps performed in one run
max_partition_size = 1042

# Create a list with all combinations and assign a unique random_state and partition number
sampling_params = {}
for n_x, n_y in grid_sizes:
    for sample_size in sample_sizes:
        for i_bootstrap in range(n_bootstraps):
            random_state = 42 + (n_x + n_y) * n_bootstraps * 42 + sample_size + i_bootstrap
            i_partition = i_bootstrap // max_partition_size
            sampling_params[(sample_size, n_x, n_y, i_bootstrap)] = (random_state, i_partition)

# Specify norms of interest
# norms = [1, 2, np.inf]
norms = [1, np.inf]

# Specify parameters of interest
columns = ['n_x', 'n_y', 'sample_size', 'i_bootstrap', 'random_state']
for norm in norms:
    columns += [
        f'xT_error_{norm}_norm',
        f'g_error_{norm}_norm',
        f'xG_error_{norm}_norm',
        f'prob_shot_error_{norm}_norm',
        f'transition_matrix_error_{norm}_norm',
        f'transition_matrix_true_{norm}_norm',
        f'transition_matrix_resampled_{norm}_norm',
        f'weighted_transition_matrix_error_{norm}_norm',
    ]

# Paths to the true and resampled models
true_model_path = '/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/true-models/'
resampled_model_path = '/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/resampled-models/'

# Convert sampling_params to a list for distributing tasks
tasks = list(sampling_params.items())

# Divide tasks among available MPI ranks
chunk_size = len(tasks) // size
start_idx = rank * chunk_size
end_idx = start_idx + chunk_size if rank != size - 1 else len(tasks)

# Local computation for each rank
local_dict_errors = {column: [] for column in columns}

counter = 0
for (sample_size, n_x, n_y, i_bootstrap), (random_state, i_partition) in tasks[start_idx:end_idx]:
    counter += 1
    if counter % 100 == 0 and rank==0:
        print(f'Were currently at {counter} of {len(tasks[start_idx:end_idx])} at one of {size} nodes.')
    if i_partition == 0:
        local_dict_errors = add_errors_to_dict(
            local_dict_errors, true_model_path, resampled_model_path, 
            n_x, n_y, sample_size, i_bootstrap, random_state, norms
        )

# Gather results from all ranks at rank 0
all_results = comm.gather(local_dict_errors, root=0)

# Only rank 0 processes the final results
if rank == 0:
    final_dict_errors = {column: [] for column in columns}
    
    # Merge results from all ranks
    for result in all_results:
        for key in final_dict_errors:
            final_dict_errors[key].extend(result[key])

    # Convert to DataFrame
    df_errors = pd.DataFrame(final_dict_errors)
    
    # Save the results
    df_errors.to_csv("bootstrap_errors_normal_xthreat.csv", index=False)
    print("Final results saved as 'bootstrap_errors_normal_xthreat.csv'")
    print(f'\nThe whole script took {int((time.time()-script_starting_time)/3600)}h, {int((time.time()-script_starting_time)%3600/60)}m, {int((time.time()-script_starting_time)%60)}s')
