import time
import sys
import pandas as pd
import numpy as np

sys.path.append("/home/kwvanarem/xthreat-research-v2/run-06-01-2026/")
from xThreat import xThreat

script_starting_time = time.time()

# The grid sizes
grid_sizes = [
    (16, 12), (32, 24), (40, 30), (48, 36), (56, 42), (64, 48), 
    (8, 6), (10, 8), (12, 9),  (14, 11),
    (20, 15), (24, 18), (28, 21),
              ]

# Load the data
data_path = '/home/kwvanarem/xthreat-research-v2/run-06-01-2026/1-data-preparation/data-storage/preprocessed_data_top5_leagues.parquet'
df_events = pd.read_parquet(data_path, engine='fastparquet')
df_events['shot'] = ~df_events['shot_outcome'].isna()
df_events['goal'] = df_events['shot_outcome'] == 'Goal'
df_events.drop(columns=['id', 'shot_outcome', 'possession'], inplace=True)
df_size_bytes = df_events.memory_usage(deep=True).sum()
df_size_mb = df_size_bytes / (1024 ** 2)  # Convert to MB

print(f"DataFrame size: {df_size_mb:.2f} MB")

# Train an xThreat model and apply _filter_out_of_bounds.
# In this way, the other models won't have to do that during fit.
# Will speed up computations.
xThreat_prefit = xThreat(16, 12)
xThreat_prefit.fit(df_events)
df_events = xThreat_prefit._filter_out_of_bounds(df_events)

for n_x, n_y in grid_sizes: 
    # Fit the 'true' xThreat model
    xT_true = xThreat(n_x, n_y)
    xT_true.fit(df_events, filter_events=False, convergence_threshold=1e-9)

    # Save the resampled method
    file_path = f'/scratch/kwvanarem/xthreat-research-v2/model-storage/run-06-01-2026/true-models/xt-full-data-n_x{n_x}-n_y{n_y}.pickle'
    xT_true.save_to_pickle(file_path)

print(f'\nThe whole script took {int((time.time()-script_starting_time)/3600)}h, {int((time.time()-script_starting_time)%3600/60)}m, {int((time.time()-script_starting_time)%60)}s.')