import pandas as pd
import numpy as np

import time

from mplsoccer import VerticalPitch
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import copy
import pickle

from numba import njit

class xThreat:
    def __init__(self, 
                 n_x: int = 16, 
                 n_y: int = 12, 
                 x_lims: tuple = (0, 105), 
                 y_lims: tuple = (0, 68),
                 dict_attributes = None,
                 ):
        """Initialize the xThreat model.

        Parameters:
        - n_x: The number of states in the x dimension.
        - n_y: The number of states in the y dimension.
        - x_lims: The limits of the x dimension.
        - y_lims: The limits of the y dimension.
        """
        self.n_x = n_x 
        self.n_y = n_y

        self.n_game_states = n_x*n_y
        self.loss_state = n_x*n_y
        self.shot_state  = n_x*n_y+1

        self.x_lims = x_lims
        self.y_lims = y_lims

        self.transition_matrix = None
        self.probability_possession_lost = None
        self.probability_shooting = None

        self.xG = None
        self.xG_std = None

        self.max_errors = None
        self.xT = None

        self.initial_distribution = None
        self.average_distribution = None
        self.n_training_samples = None
        self.n_training_samples_per_state = None

        if dict_attributes is not None:
            self.__dict__.update(dict_attributes)

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self):
        """Convert instance attributes to a dictionary."""
        # If the transition matrix is numpy array, convert to csr_matrix for reducing memory usage
        if not isinstance(self.transition_matrix, csr_matrix):
            self.transition_matrix = csr_matrix(self.transition_matrix)
            
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, dict_attributes):
        """Create an instance from a dictionary."""
        # Get the object from the dictionary
        obj = cls.__new__(cls) 
        obj.__dict__.update(dict_attributes)

        # If the transition matrix is csr_matrix, convert to numpy array
        if isinstance(obj.transition_matrix, csr_matrix):
            obj.transition_matrix = obj.transition_matrix.toarray()

        return obj

    def save_to_pickle(self, file_path):
        """Save object to a binary file using pickle."""
        with open(file_path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load_from_pickle(cls, file_path):
        """Load object from a binary file using pickle."""
        with open(file_path, "rb") as f:
            dict_attributes = pickle.load(f)
        return cls.from_dict(dict_attributes)

    def _filter_out_of_bounds(self, df_events):
        """Filter out all actions that have infeasible locations. These are:
        - actions that start outside of the pitch.
        - actions that end outside the pitch without loss of possession or a shot.
        """
        # First check whether all data points adhere to the desired. If so, do not do anything to save time.
        max_x, max_y = np.max(df_events[['x', 'next_x']].values), np.max(df_events[['y', 'next_y']].values)
        min_x, min_y = np.min(df_events[['x', 'next_x']].values), np.min(df_events[['y', 'next_y']].values)
        if max_x < self.x_lims[1] or max_y < self.y_lims[1] or min_x >= self.x_lims[0] or min_y >= self.y_lims[0]:
            return df_events


        df_events_ = df_events.copy()

        # Filter out events that are out of bounds
        mask_current_state = (df_events_['x'] < self.x_lims[1]) & (df_events_['y'] < self.y_lims[1]) & (df_events_['x'] >= self.x_lims[0]) & (df_events_['y'] >= self.y_lims[0])
        mask_next_state = (df_events_['possession_kept'] == False) | (df_events_['shot'] == True) | ((df_events_['next_x'] < self.x_lims[1]) & (df_events_['next_y'] < self.y_lims[1]) & (df_events_['next_x'] >= self.x_lims[0]) & (df_events_['next_y'] >= self.y_lims[0]))
        
        # Filter out events that are out of bounds
        df_events_ = df_events_[(mask_current_state & mask_next_state)]

        # If events are filtered out, print an update        
        if df_events_.shape[0] != df_events.shape[0]:
            del_df = df_events[~(mask_current_state & mask_next_state)]
            print(f'Filtered out {df_events.shape[0]-df_events_.shape[0]} events that had coordinates out of bounds.')
            print(f"In this, there were {del_df['shot'].sum()} shots and {del_df['goal'].sum()} goals.")

        return df_events_
        
    def _add_state_numbers(self, df_events):
        """Add state numbers to the dataframe. This is a unique identifier for each state."""
        # Add the state numbers with respect to the x dimension to the dataframe
        df_events['x_state'] = np.floor(df_events['x']/(self.x_lims[1]-self.x_lims[0])*self.n_x)
        df_events['next_x_state'] = np.floor(df_events['next_x']/(self.x_lims[1]-self.x_lims[0])*self.n_x)

        # Add the state numbers with respect to the y dimension to the dataframe
        df_events['y_state'] = np.floor(df_events['y']/(self.y_lims[1]-self.y_lims[0])*self.n_y)
        df_events['next_y_state'] = np.floor(df_events['next_y']/(self.y_lims[1]-self.y_lims[0])*self.n_y)

        # Add the state numbers to the dataframe
        df_events['state'] = self.n_x*df_events['y_state'] + df_events['x_state']
        df_events['next_state'] = self.n_x*df_events['next_y_state'] + df_events['next_x_state']

        # Set the state numbers for the loss and shot state
        df_events.loc[df_events['possession_kept'] == False, 'next_state'] = self.loss_state
        df_events.loc[df_events['shot'] == True, 'next_state']  = self.shot_state

        return df_events
    
    def _calculate_transition_matrix(self, df_events):
        """Calculate the transition matrix from the event dataframe."""
        # Initialize the transition matrix
        self.transition_matrix = np.zeros((self.n_game_states+2, self.n_game_states+2), dtype=np.float64)

        # Count the transitions between states
        state_next_state_pairs = df_events[['state', 'next_state']].astype(int).to_numpy()
        np.add.at(self.transition_matrix, (state_next_state_pairs[:, 0], state_next_state_pairs[:, 1]), 1)

        # Normalize rows to get probabilities
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        np.divide(self.transition_matrix, row_sums, out=self.transition_matrix, where=row_sums != 0)

        # Store the probabilities of losing possession and shooting
        self.probability_possession_lost = self.transition_matrix[:self.n_game_states, self.loss_state]
        self.probability_shooting = self.transition_matrix[:self.n_game_states, self.shot_state]   
    
    # def _calculate_transition_matrix(self, df_events):
    #     """Calculate the transition matrix from the event dataframe."""
    #     self.transition_matrix = np.zeros((self.n_game_states+2, self.n_game_states+2))
    #     # Count the transitions between states
    #     transition_counts = df_events.groupby(['state', 'next_state'])['x'].count().reset_index().rename(columns = {'x': 'counts'})    

    #     # Fill the transition matrix
    #     self.transition_matrix[transition_counts['state'].astype(int), transition_counts['next_state'].astype(int)] = transition_counts['counts']

    #     row_sums = np.sum(self.transition_matrix, axis=1, keepdims=True)  # Compute row sums

    #     # Avoid division by zero: If row sum is 0, keep the row as is (or set it to 0)
    #     self.transition_matrix = np.divide(self.transition_matrix, row_sums, out=np.zeros_like(self.transition_matrix), where=row_sums != 0)
    #     # self.transition_matrix = np.where(row_sums != 0, self.transition_matrix / row_sums, 0)

    #     # Store the probabilities of losing possession and shooting
    #     self.probability_possession_lost = self.transition_matrix[:self.n_game_states, self.loss_state]
    #     self.probability_shooting = self.transition_matrix[:self.n_game_states, self.shot_state]


    def _calculate_xG_from_data(self, df_events):
        """Calculate the xG value from the data for each in-game state."""
        # Initialize the xG values
        self.xG = np.zeros(self.n_game_states)

        # Count the number of shots and goals in each state
        xG_counts = df_events.groupby('state')[['goal', 'shot']].sum().reset_index()

        # Limit the values to the number of in-game states
        xG_counts = xG_counts[xG_counts['state'] < self.n_game_states]

        # Calculate the xG values
        self.xG[xG_counts['state'].astype(int)] = np.where(xG_counts['shot'] > 0, xG_counts['goal']/xG_counts['shot'], 0)

        # Calculate the corresponding standard deviation
        self.xG_std = np.sqrt(self.xG * (1 - self.xG))

    def _calculate_xG_from_xG_model(self, df_events, xG_column = 'xG'):
        """Calculate the xG value from the existing xG values for each in-game state."""
        # Initialize the xG values
        self.xG = np.zeros(self.n_game_states)
        self.xG_std = np.zeros(self.n_game_states)

        # Calculate the average probability of scoring in each state
        xG_means = df_events[df_events['shot'] == True].groupby('state')[xG_column].mean().reset_index()
        xG_std = df_events[df_events['shot'] == True].groupby('state')[xG_column].std().reset_index()

        # Limit the values to the number of in-game states
        xG_means = xG_means[xG_means['state'] < self.n_game_states]
        xG_std = xG_std[xG_std['state'] < self.n_game_states]

        # Calculate the xG values and the corresponding standard deviation
        self.xG[xG_means['state'].astype(int)] = xG_means[xG_column]
        self.xG_std[xG_std['state'].astype(int)] = xG_std[xG_column]

    def _calculate_xThreat(self, max_iterations: int = 1000, convergence_threshold: float = 1e-6):
        """Calculate the xThreat value for each in-game state."""

        trans_matr = csr_matrix(self.transition_matrix[:self.n_game_states, :self.n_game_states])

        # Initialize the iterative algorithm
        g = self.xG * self.probability_shooting
        xT = g.copy()
        self.max_errors = []

        for i in range(max_iterations):
            # Calculate the new xT value
            xT_new = g + trans_matr @ xT

            # Check for convergence
            self.max_errors.append(np.max(np.abs(xT_new - xT)))
            if self.max_errors[i] < convergence_threshold:
                break

            # Update the xT value
            xT = xT_new

            # Print a warning if the maximum number of iterations is reached
            if i == max_iterations-1:
                print('Did not converge to a solution within the maximum number of iterations.')

        # Store the xThreat values
        self.xT = xT_new

    def _calculate_initial_distribution(self, df_events):
        """Calculate the initial distribution of the Markov chain."""
        # Initialize the storage of the initial distribution
        self.initial_distribution = np.zeros(self.n_game_states)

        if 'chain_length' in df_events.columns:
            # Count the number of events in each state
            initial_distribution = df_events[df_events['chain_length'] == 0].groupby('state')['x'].count().reset_index(name='counts')

            # Normalize the counts
            initial_distribution['counts'] = initial_distribution['counts']/initial_distribution['counts'].sum()

            # Store the initial distribution
            self.initial_distribution[initial_distribution['state'].astype(int)] = initial_distribution['counts'].values


    def _calculate_average_distribution(self, df_events):
        """Calculate the average distribution of the data points."""
        # Initialize the storage of the initial distribution
        self.average_distribution = np.zeros(self.n_game_states)

        # Calculate the average distribution of the data points
        average_distribution = df_events.groupby('state')['x'].count().reset_index(name='counts')

        # Normalize the counts
        average_distribution['counts'] = average_distribution['counts']/average_distribution['counts'].sum()

        # Store the average distribution
        self.average_distribution[average_distribution['state'].astype(int)] = average_distribution['counts'].values

    def _fill_empty_states(self):
        zero_data_point_states = list(np.argwhere(self.n_training_samples_per_state == 0))
        while len(zero_data_point_states)>0:
            un_imputed_states = []
            for state in zero_data_point_states:
                x_i, y_i = _get_xy_states(state, self.n_x)
                neighbour_states = [
                    (x_i-1, y_i),
                    (x_i, y_i-1),
                    (x_i+1, y_i),
                    (x_i, y_i+1),
                ]
                valid_neighbour_states = [
                    (x,y) for x,y in neighbour_states if (0 <= x and x < self.n_x and 0 <= y and y < self.n_y)
                    ]
                nonzero_neighbour_states = [_get_state_number(x, y, self.n_x) for x,y in valid_neighbour_states if not _get_state_number(x, y, self.n_x) in zero_data_point_states]
                
                if len(nonzero_neighbour_states) > 0:
                    self.transition_matrix[state] = np.mean(self.transition_matrix[nonzero_neighbour_states], axis=0)
                    self.probability_shooting[state] = np.mean(self.probability_shooting[nonzero_neighbour_states])
                    self.probability_possession_lost[state] = np.mean(self.probability_possession_lost[nonzero_neighbour_states])
                    self.xG[state] = np.mean(self.xG[nonzero_neighbour_states])
                    zero_data_point_states.remove(state)
                else:
                    un_imputed_states.append(state)
            zero_data_point_states = un_imputed_states

    def _apply_gaussian_filtering_to_vector(self, flattened_array, sigma: float = 1):
        array_2d = flattened_array.reshape(self.n_y, self.n_x)
        array_2d_smoothed = gaussian_filter(array_2d, sigma=sigma)
        return array_2d_smoothed.flatten()
    
    def _apply_gaussian_filtering(self, sigma: float = 1):
        # Apply Gaussian filtering to the xG values, transition matrix, beginning distribution, and average distribution
        self.xG = self._apply_gaussian_filtering_to_vector(self.xG, sigma)

        for i in range(self.n_game_states):
            self.transition_matrix[i, :self.n_game_states] = self._apply_gaussian_filtering_to_vector(self.transition_matrix[i, :self.n_game_states], sigma)
        self.transition_matrix[:self.n_game_states, self.loss_state] = self._apply_gaussian_filtering_to_vector(self.transition_matrix[:self.n_game_states, self.loss_state], sigma)
        self.transition_matrix[:self.n_game_states, self.shot_state] = self._apply_gaussian_filtering_to_vector(self.transition_matrix[:self.n_game_states, self.shot_state], sigma)

        # Normalise columns of transition matrix
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        np.divide(self.transition_matrix, row_sums, out=self.transition_matrix, where=row_sums != 0)

        # Apply Gaussian filtering to the initial and average distributions
        self.initial_distribution = self._apply_gaussian_filtering_to_vector(self.initial_distribution, sigma)
        self.average_distribution = self._apply_gaussian_filtering_to_vector(self.average_distribution, sigma)

        # Normalise initial and average distributions
        self.initial_distribution = self.initial_distribution / np.sum(self.initial_distribution)
        self.average_distribution = self.average_distribution / np.sum(self.average_distribution)

        # Update the probability of losing possession and shooting
        self.probability_possession_lost = self.transition_matrix[:self.n_game_states, self.loss_state]
        self.probability_shooting = self.transition_matrix[:self.n_game_states, self.shot_state]

    def _apply_gaussian_filtering_boundary_preservation_to_vector(self, flattened_array, sigma: float = 1, pad_size=None):
        """Apply Gaussian filtering with boundary preservation to a 1D array."""
        # If pad_size is not specified, set it to 3*sigma
        if pad_size is None:
            pad_size = int(3*sigma)
        
        # Reshape the array to a 2D array
        array_2d = flattened_array.reshape(self.n_y, self.n_x)

        # Calculate the boundary gradients
        grad_top_lst = [None]*pad_size
        grad_bottom_lst = [None]*pad_size
        grad_left_lst = [None]*pad_size
        grad_right_lst = [None]*pad_size

        for i in range(1, pad_size+1):
            grad_top_lst[i-1] = array_2d[0, :] - array_2d[i, :]
            grad_bottom_lst[i-1] = array_2d[-1, :] - array_2d[-(i+1), :]
            grad_left_lst[i-1] = array_2d[:, 0] - array_2d[:, i]
            grad_right_lst[i-1] = array_2d[:, -1] - array_2d[:, -(i+1)]

        # Pad the array
        array_2d_padded = np.pad(array_2d, pad_size, mode='edge')

        # Adjust the padded points to make them linearly increase with the boundary gradient
        for i in range(1, pad_size+1):
            array_2d_padded[pad_size - i, pad_size:-pad_size] += grad_top_lst[i-1] * i
            array_2d_padded[-(pad_size - i + 1), pad_size:-pad_size] += grad_bottom_lst[i-1] * i
            array_2d_padded[pad_size:-pad_size, pad_size - i] += grad_left_lst[i-1] * i
            array_2d_padded[pad_size:-pad_size, -(pad_size - i + 1)] += grad_right_lst[i-1] * i

        # Apply Gaussian filter
        array_2d_smoothed_padded = gaussian_filter(array_2d_padded, sigma=sigma)

        # Crop back to original size
        array_2d_smoothed = array_2d_smoothed_padded[pad_size:-pad_size, pad_size:-pad_size]
        array_2d_smoothed = np.where(array_2d_smoothed < 0, 0, array_2d_smoothed)

        return array_2d_smoothed.flatten()

    def _apply_gaussian_filtering_boundary_preservation(self, sigma: float = 1, pad_size=None):
        """Apply Gaussian filtering with boundary preservation to the xG values, transition matrix, beginning distribution, and average distribution."""	
        # If pad_size is not specified, set it to 3*sigma
        if pad_size is None:
            pad_size = int(3*sigma)

        # Apply Gaussian filtering with boundary preservation to the xG values, transition matrix, beginning distribution, and average distribution
        self.xG = self._apply_gaussian_filtering_boundary_preservation_to_vector(self.xG, sigma)
        for i in range(self.n_game_states):
            self.transition_matrix[i, :self.n_game_states] = self._apply_gaussian_filtering_boundary_preservation_to_vector(self.transition_matrix[i, :self.n_game_states], sigma)
        self.transition_matrix[:self.n_game_states, self.loss_state] = self._apply_gaussian_filtering_boundary_preservation_to_vector(self.transition_matrix[:self.n_game_states, self.loss_state], sigma)
        self.transition_matrix[:self.n_game_states, self.shot_state] = self._apply_gaussian_filtering_boundary_preservation_to_vector(self.transition_matrix[:self.n_game_states, self.shot_state], sigma)
        
        # Normalise columns of transition matrix
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        np.divide(self.transition_matrix, row_sums, out=self.transition_matrix, where=row_sums != 0)
        
        # Apply Gaussian filtering with boundary preservation to the initial and average distributions
        self.initial_distribution = self._apply_gaussian_filtering_boundary_preservation_to_vector(self.initial_distribution, sigma)
        self.average_distribution = self._apply_gaussian_filtering_boundary_preservation_to_vector(self.average_distribution, sigma)

        # Normalise initial and average distributions
        self.initial_distribution = self.initial_distribution / np.sum(self.initial_distribution)
        self.average_distribution = self.average_distribution / np.sum(self.average_distribution)

        # Update the probability of losing possession and shooting
        self.probability_possession_lost = self.transition_matrix[:self.n_game_states, self.loss_state]
        self.probability_shooting = self.transition_matrix[:self.n_game_states, self.shot_state]

    def fit(self, 
            df_events, 
            max_iterations: int = 1000, 
            convergence_threshold: float = 1e-6, 
            xG_column: str = None,
            filter_events: bool = True,
            apply_symmetry: bool = False,
            fill_empty_states: bool = False,
            apply_gaussian_filtering: bool = False,
            sigma: float = 1,
            ):
        """Fit the model to the data.

        Parameters:
        - df_events: The dataframe with the events, should include the columns 'x', 'y', 'next_x', 'next_y', 'goal', 'shot', 'possession_kept', 'chain_length'.
        - max_iterations: The maximum number of iterations for the xThreat calculation.
        - convergence_threshold: The tolerance for the xThreat calculation.
        - xG_column: The columns with xG values of shots. If None, a goal is taken as 1 and a miss as 0. 
        - filter_events: An indicator whether or not events can be out of bounds and filtering is necessary. Default is to apply filtering. Turning it off might significantly increase computation speed.
        - fill_empty_states: An indicator whether or not empty states should be imputed with the average of their neighbours.
        - apply_gaussian_filtering: An indicator whether or not Gaussian filtering should be applied to the xG values, transition matrix, beginning distribution, and average distribution.
        - sigma: The standard deviation of the Gaussian filter. Default is 1.
        """
        # df_events_ = df_events.copy()
        # Filter out events that are out of bounds
        # begin_time = time.time()
        if filter_events == True:
            df_events = self._filter_out_of_bounds(df_events)
        # print(f'Filtering out events took {time.time()-begin_time:.5f} seconds.')


        # Very greedy and slow way to apply symmetry. Can be optimized.
        if apply_symmetry == True:
            df_events = pd.concat([
                df_events, 
                df_events.copy().assign(
                    # x = self.x_lims[0]+self.x_lims[1]-df_events['x'], 
                    # next_x = self.x_lims[0] + self.x_lims[1]-df_events['next_x'], 
                    y = self.y_lims[0]+self.y_lims[1]-df_events['y'],
                    next_y = self.y_lims[0]+self.y_lims[1]-df_events['next_y'],
                    ignore_index=True,
                    )   
                    ])

        # Add state numbers to the dataframe
        # begin_time = time.time()
        df_events = self._add_state_numbers(df_events)
        # print(f'Adding state information took {time.time()-begin_time:.5f} seconds.')

        # Calculate the transition matrix
        # begin_time = time.time()
        self._calculate_transition_matrix(df_events)
        # print(f'Calculating transition matrix took {time.time()-begin_time:.5f} seconds.')

        # Calculate the xG values
        # begin_time = time.time()
        if xG_column is None:
            self._calculate_xG_from_data(df_events)
        else:
            self._calculate_xG_from_xG_model(df_events, xG_column)
        # print(f'Calculating xG took {time.time()-begin_time:.5f} seconds.')

        # Calculate the initial distribution
        # begin_time = time.time()
        self._calculate_initial_distribution(df_events)
        # print(f'Calculating initial distribution took {time.time()-begin_time:.5f} seconds.')

        # Calculate the average distribution of the data points
        # begin_time = time.time()
        self._calculate_average_distribution(df_events)
        # print(f'Calculate average distribution took {time.time()-begin_time:.5f} seconds.')

        self.n_training_samples = df_events.shape[0]
        self.n_training_samples_per_state = self.n_training_samples*self.average_distribution

        if fill_empty_states == True:
            self._fill_empty_states()

        if apply_gaussian_filtering == True:
            self._apply_gaussian_filtering(sigma)

        if apply_gaussian_filtering == 'boundary preservation':
            self._apply_gaussian_filtering_boundary_preservation(sigma)

        # Calculate the xThreat values
        # begin_time = time.time()
        self._calculate_xThreat(max_iterations, convergence_threshold)
        # print(f'Calculating xT took {time.time()-begin_time:.5f} seconds.')


################################
###### Plotting functions ######
################################

    def plot_pitch_heatmap(self, 
                           values, 
                           figsize = (10.5, 6.8), 
                           cmap='hot', 
                           colorbar_label='',
                           line_color='black',
                           ):
        """Plot a heatmap on a pitch.
        Parameters:
        - values: The values to plot.
        - figsize: The size of the figure.
        - cmap: The colormap to use.
        - colorbar_label: The label of the colorbar.
        - line_color: The color of the lines on the pitch.

        Returns:
        - fig: The figure object.
        - ax: The axis object.
        - cbar: The colorbar object.
        """
        # Create the pitch
        pitch = VerticalPitch(pitch_type='custom', line_zorder=2,
                              pitch_width=68, 
                              pitch_length=105, line_color=line_color)
        fig, ax = pitch.draw(figsize=figsize)

        # Specify the x and y values
        x = np.array([x_i for y_i in range(self.n_y) for x_i in range(self.n_x)])
        y = np.array([y_i for y_i in range(self.n_y) for x_i in range(self.n_x)])
        x_bin_size = (self.x_lims[1]-self.x_lims[0])/self.n_x
        y_bin_size = (self.y_lims[1]-self.y_lims[0])/self.n_y
        x = x_bin_size*(0.5+x)
        y = y_bin_size*(0.5+y)

        # Plot the heatmap
        bin_statistic = pitch.bin_statistic(x, y, values, statistic='min', bins=[self.n_x, self.n_y])
        heatmap = pitch.heatmap(bin_statistic, ax=ax, cmap=cmap)

        # Add colorbar and labels
        cbar = fig.colorbar(heatmap, ax=ax, shrink=0.6)
        cbar.set_label(label=colorbar_label)
        return fig, ax, cbar

    def plot_xThreat(self,
                        cmap = 'hot',
                        figsize=(10.5, 6.8),
                        line_color='black',
                        ):
        """Plot the xThreat values on the pitch.
        Parameters:
        - cmap: The colormap to use.
        - figsize: The size of the figure.
        - line_color: The color of the lines on the pitch.
        """
        # Plot the heatmap
        fig, ax, cbar = self.plot_pitch_heatmap(self.xT[:self.n_game_states], figsize=figsize, cmap=cmap, colorbar_label='xThreat', line_color=line_color)
        plt.show()

    def plot_xG(self,
                    cmap = 'hot',
                    figsize=(10.5, 6.8),
                    line_color='black',
                    ):
        """Plot the xG values on the pitch.
        Parameters:
        - cmap: The colormap to use.
        - figsize: The size of the figure.
        - line_color: The color of the lines on the pitch.
        """
        # Plot the heatmap
        fig, ax, cbar = self.plot_pitch_heatmap(self.xG[:self.n_game_states], figsize=figsize, cmap=cmap, colorbar_label='xG', line_color=line_color)
        plt.show()

    def plot_beginning_distribution(self,
                                    cmap = 'hot',
                                    figsize=(10.5, 6.8),
                                    line_color='black',
                                    ):
        """Plot the initial distribution of the Markov chain.
        Parameters:
        - cmap: The colormap to use.
        - figsize: The size of the figure.
        - line_color: The color of the lines on the pitch.
        """
        # Plot the heatmap
        fig, ax, cbar = self.plot_pitch_heatmap(self.initial_distribution[:self.n_game_states], figsize=figsize, cmap=cmap, colorbar_label='Initial distribution', line_color=line_color)
        plt.show()

    def plot_average_distribution(self,
                                    cmap = 'hot',
                                    figsize=(10.5, 6.8),
                                    line_color='black',
                                    ):
        """Plot the average distribution of the data points.
        Parameters:
        - cmap: The colormap to use.
        - figsize: The size of the figure.
        - line_color: The color of the lines on the pitch.
        """
        # Plot the heatmap
        fig, ax, cbar = self.plot_pitch_heatmap(self.average_distribution[:self.n_game_states], figsize=figsize, cmap=cmap, colorbar_label='Average distribution', line_color=line_color)
        plt.show()

    def plot_transitions_outgoing(self, 
                                  x_i, y_i, 
                                  normalized = False, 
                                  cmap = 'YlGn', 
                                  figsize=(10.5, 6.8), 
                                  line_color='black',
                                  hide_current_state=False,
                                  **square_kwargs,
                                  ):
        """Plot the transition probabilities of the outgoing transitions of a state.
        
        Parameters:
        - x_i: The x state number of the state.
        - y_i: The y state number of the state.
        - normalized: Whether to normalize the values.
        - cmap: The colormap to use.
        - figsize: The size of the figure.
        - line_color: The color of the lines on the pitch.
        - hide_current_state: Whether to hide the current state.
        - **square_kwargs: Additional keyword arguments for the square (a matplotlib.patches.Rectangle object).
        """
        # Calculate game state
        state = self.n_x*y_i + x_i

        # Get the transition probabilities
        values = self.transition_matrix[state,:self.n_game_states]

        # Normalize the values if necessary
        if normalized == True:
            values = values / np.sum(values)

        # Plot the heatmap
        fig, ax, cbar = self.plot_pitch_heatmap(values, figsize=figsize, cmap=cmap, colorbar_label='Transition probabilities', line_color=line_color)
        
        # Add a square to indicate the current state
        if hide_current_state == False:
            x_bin_size = (self.x_lims[1]-self.x_lims[0])/self.n_x
            y_bin_size = (self.y_lims[1]-self.y_lims[0])/self.n_y
            square = patches.Rectangle(
                (y_i*y_bin_size, x_i*x_bin_size),
                width=y_bin_size,
                height=x_bin_size,
                facecolor='none',
                edgecolor='blue',
                **square_kwargs,
                )
            ax.add_patch(square)
        return fig, ax, cbar
        # plt.show()

    def plot_transitions_incoming(self, 
                                  x_i, y_i, 
                                  normalized = False, 
                                  cmap = 'YlGn', 
                                  figsize=(10.5, 6.8), 
                                  line_color='black',
                                  hide_current_state=False,
                                  **square_kwargs,
                                  ):
        # Calculate game state
        state = self.n_x*y_i + x_i

        # Get the transition probabilities
        values = self.transition_matrix[state,:self.n_game_states]

        # Normalize the values if necessary
        if normalized == True:
            values = values / np.sum(values)

        # Plot the heatmap
        fig, ax, cbar = self.plot_pitch_heatmap(values, figsize=figsize, cmap=cmap, colorbar_label='Transition probabilities', line_color=line_color)

        # Add a square to indicate the current state
        if hide_current_state == False:
            x_bin_size = (self.x_lims[1]-self.x_lims[0])/self.n_x
            y_bin_size = (self.y_lims[1]-self.y_lims[0])/self.n_y
            square = patches.Rectangle(
                (y_i*y_bin_size, x_i*x_bin_size),
                width=y_bin_size,
                height=x_bin_size,
                facecolor='none',
                edgecolor='blue',
                **square_kwargs,
                )
            ax.add_patch(square)
        plt.show()

################################
###### Sampling functions ######
################################

    def _sample_goals(self, shot_indicator, states, random_state: int = 42):
        """Sample the goals for the shots in the given sample."""
        # Set the seed
        np.random.seed(random_state)

        # Sample the goals
        uniform_sample = np.random.rand(len(states))
        goals = np.where(shot_indicator, uniform_sample < self.xG[states.astype(int)], False)

        return goals
    
    def _restructure_and_add_goals_sample(self, sample, random_state: int = 42):
        """Restructure the sample and add information about the goals."""
        # Store the results in a dataframe
        df_sample = pd.DataFrame(np.arange(sample.shape[0]), columns=['index'])
        df_sample['state'] = sample[:, 0]
        df_sample['next_state'] = sample[:, 1]
        if sample.shape[1] == 3:
            df_sample['chain_length'] = sample[:, 2]
        df_sample['shot'] = df_sample['next_state'] == self.shot_state
        df_sample['possession_kept'] = df_sample['next_state'] < self.loss_state

        # Sample the goals
        df_sample['goal'] = self._sample_goals(df_sample['shot'], df_sample['state'], random_state)

        return df_sample
    
    def _add_xy_state_information(self, df_sample):
        """Add the x and y state numbers and coordinates to the dataframe."""	
        # Add the x and y state numbers to the dataframe
        df_sample['x_state'] = df_sample['state'] % self.n_x
        df_sample['y_state'] = df_sample['state'] // self.n_x
        df_sample['next_x_state'] = df_sample['next_state'] % self.n_x
        df_sample['next_y_state'] = df_sample['next_state'] // self.n_x 

        # Add the x and y coordinates to the dataframe
        df_sample['x'] = (df_sample['x_state'] + 0.5) / self.n_x * (self.x_lims[1] - self.x_lims[0]) + self.x_lims[0]
        df_sample['y'] = (df_sample['y_state'] + 0.5) / self.n_y * (self.y_lims[1] - self.y_lims[0]) + self.y_lims[0]
        df_sample['next_x'] = (df_sample['next_x_state'] + 0.5) / self.n_x * (self.x_lims[1] - self.x_lims[0]) + self.x_lims[0]
        df_sample['next_y'] = (df_sample['next_y_state'] + 0.5) / self.n_y * (self.y_lims[1] - self.y_lims[0]) + self.y_lims[0]

        return df_sample

    def _sample_Markov_chain(self, n_samples: int, random_state: int = 42, initial_distribution: str = 'begin of chain'):
        """Sample from the xThreat Markov chain."""
        # Precalculate the cumulative transition matrix and initial distribution
        cumulative_transition_matrix = np.cumsum(self.transition_matrix, axis=1)

        if initial_distribution == 'begin of chain':
            if self.initial_distribution is None:
                raise ValueError('The initial distribution is not calculated. This can be because the column "chain_length" was missing in df_events.')
            cumulative_initial_distribution = np.cumsum(self.initial_distribution)
        elif initial_distribution == 'average occurences':
            cumulative_initial_distribution = np.cumsum(self.average_distribution)

        # Sample the Markov chain with a numba wrapped function for speed
        sample = _numba_sample_Markov_chain(
            n_samples,
            self.loss_state,
            self.shot_state,
            cumulative_transition_matrix,
            cumulative_initial_distribution,
            random_state,
        )

        # Store the results in a dataframe and add the goals as samples
        df_sample = self._restructure_and_add_goals_sample(sample, random_state)

        # Add the x and y state numbers and coordinates to the dataframe
        df_sample = self._add_xy_state_information(df_sample)

        return df_sample
    
    def _sample_bootstrap(self, n_samples: int, random_state: int = 42):
        """Sample from the xThreat in an equivalent way of performing bootstrap."""

        # Precalculate the cumulative transition matrix and initial distribution
        cumulative_transition_matrix = np.cumsum(self.transition_matrix, axis=1)

        cumulative_average_distribution = np.cumsum(self.average_distribution)

        # Sample the data points with a numba wrapped function for speed
        sample = _numba_sample_bootstrap(
            n_samples,
            cumulative_transition_matrix,
            cumulative_average_distribution,
            random_state,
        )

        # Store the results in a dataframe and add the goals as samples
        df_sample = self._restructure_and_add_goals_sample(sample, random_state)

        # Add the x and y state numbers and coordinates to the dataframe
        df_sample = self._add_xy_state_information(df_sample)

        return df_sample

    def sample(self, n_samples: int, method: str = 'Markov chain', random_state: int = 42, initial_distribution: str = 'begin of chain'):
        """Sample from the xThreat model with the current method.
        Parameters:
        - n_samples: The number of samples to generate.
        - method: The method to use for sampling. Options are 'Markov chain' and 'bootstrap'.
        - random_state: The random state to use for sampling.
        - initial_distribution (optional): The distribution to use for sampling the initial state in the Markov chain method. Options are 'begin of chain' and 'average occurences'.
        
        Returns:
        - df_sample: The dataframe with the samples.
        """
        if method == 'Markov chain':
            return self._sample_Markov_chain(n_samples, random_state, initial_distribution)
        elif method == 'bootstrap':
            return self._sample_bootstrap(n_samples, random_state)
        else:
            raise ValueError('The method should be either "Markov chain" or "bootstrap".')
        
    def quantify_action_quality(self, df_events: pd.DataFrame):
        """Give the increase in xT by an action.
        Shots are not considered. Loss of possession is valued of 0.

        Parameters:
        - df_events: The dataframe with the events, should include the columns 'x', 'y', 'next_x', 'next_y', 'goal', 'shot', 'possession_kept'.

        Returns
        - df_events: The dataframe with the quality of the events added. This is described in the new columns 'current_xT', 'next_xT', 'action_xT'.
        """
        
        df_events_ = df_events.copy()

        # Filter out events that are out of bounds
        mask_current_state = (df_events['x'] < self.x_lims[1]) & (df_events['y'] < self.y_lims[1]) & (df_events['x'] >= self.x_lims[0]) & (df_events['y'] >= self.y_lims[0])
        mask_next_state = (df_events['possession_kept'] == False) | (df_events['shot'] == True) | ((df_events['next_x'] < self.x_lims[1]) & (df_events['next_y'] < self.y_lims[1]) & (df_events['next_x'] >= self.x_lims[0]) & (df_events['next_y'] >= self.y_lims[0]))
        df_events_ = df_events[(mask_current_state & mask_next_state)].copy()

        # Adds x_state, y_state, next_x_state, next_y_state, state, and next_state
        temp_columns = ['x_state', 'y_state', 'next_x_state', 'next_y_state', 'state', 'next_state']
        df_events_= self._add_state_numbers(df_events_)

        # Add 0 for a loss of possession and np.nan for the shot state
        xT = np.append(self.xT, (0, np.nan))

        ### Match states to xT values
        state_nan = df_events_['state'].isna()
        df_events_.loc[~state_nan, 'current_xT'] = xT[df_events_['state'].dropna().astype(int)]
        next_state_nan = df_events_['next_state'].isna()
        df_events_.loc[~next_state_nan, 'next_xT'] = xT[df_events_['next_state'].dropna().astype(int)]

        # Calculate difference in xT before and after action
        df_events_['action_xT'] = df_events_['next_xT'] - df_events_['current_xT']

        # TO CHECK: what if goal? or shot? or ball possession loss? These state values are not in self.xT

        # Remove added columns
        new_cols = ['current_xT', 'next_xT', 'action_xT']
        df_events.loc[(mask_current_state & mask_next_state), new_cols] = df_events_[new_cols]

        return df_events


@njit
def _numba_sample_next_state_Markov_chain(
        current_state,
        chain_length,
        loss_state,
        shot_state,
        cumulative_transition_matrix, 
        cumulative_initial_distribution, 
        ):
    """Sample the next state in the Markov chain."""
    # If the previous transition was to the loss state, sample a new starting point
    if (current_state == loss_state) or (current_state == shot_state):
        chain_length = 0
        current_state = np.searchsorted(cumulative_initial_distribution, np.random.rand())   
    else:
        chain_length += 1
    
    # Sample the next state
    next_state = np.searchsorted(cumulative_transition_matrix[current_state], np.random.rand())

    return current_state, next_state, chain_length

@njit
def _numba_sample_Markov_chain(
    n_samples,
    loss_state,
    shot_state,
    cumulative_transition_matrix,
    cumulative_initial_distribution,
    random_state,
):
    """Sample from the xThreat Markov chain."""
    # Set the seed
    np.random.seed(random_state)

    # Initialize so that a new starting point is sampled
    current_state = loss_state
    chain_length = 0

    # Initialize the storage of results
    sample = np.zeros((n_samples, 3))

    for i in range(n_samples):
        # Sample the transition
        current_state, next_state, chain_length = _numba_sample_next_state_Markov_chain(
            current_state,
            chain_length,
            loss_state,
            shot_state,
            cumulative_transition_matrix, 
            cumulative_initial_distribution, 
        )

        # Store the results
        sample[i] = current_state, next_state, chain_length

        # Update the current state
        current_state = next_state
    
    return sample

@njit
def _numba_sample_next_state_bootstrap(
        cumulative_transition_matrix, 
        cumulative_average_distribution, 
        ):
    """Sample the next state in the Markov chain."""
    # Sample the starting point of the transition
    current_state = np.searchsorted(cumulative_average_distribution, np.random.rand())   
    
    # Sample the next state
    next_state = np.searchsorted(cumulative_transition_matrix[current_state], np.random.rand())

    return current_state, next_state

@njit
def _numba_sample_bootstrap(
    n_samples,
    cumulative_transition_matrix,
    cumulative_average_distribution,
    random_state,
):
    """Sample from the xThreat Markov chain."""
    # Set the seed
    np.random.seed(random_state)

    # Initialize the storage of results
    sample = np.zeros((n_samples, 2))

    for i in range(n_samples):
        # Sample the transition
        sample[i] = _numba_sample_next_state_bootstrap(
            cumulative_transition_matrix, 
            cumulative_average_distribution, 
        )
    
    return sample

@njit
def _get_state_number(x_i, y_i, n_x):
    return n_x*y_i + x_i

@njit
def _get_xy_states(state, n_x):
    x_i = state % n_x
    y_i = state // n_x
    return x_i, y_i


