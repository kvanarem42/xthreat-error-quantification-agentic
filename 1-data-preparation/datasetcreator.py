import os

import pandas as pd
from tqdm import tqdm

column_dtypes_final = {
    'x': 'float64', 
    'y': 'float64', 
    'possession_kept': 'bool', 
    'possession': 'int64', 
    'possession_length': 'int64', 
    'chain_length': 'int64', 
    'previous_deleted': 'bool', 
    'match_id': 'int64', 
    'id': 'object',
    'period': 'int64', 
    'player_id': 'float64', 
    'type': 'object', 
    'shot_outcome': 'object', 
    'shot_statsbomb_xg': 'float64', 
    'pass_height': 'object', 
    'under_pressure': 'float64', 
    'next_type': 'object', 
    'next_x': 'float64', 
    'next_y': 'float64', 
    'next_under_pressure': 'float64', 
    'next_team_id': 'float64', 
    'next_player_id': 'float64',
}

column_dtypes = {
    'index': 'int64',
    'type': 'object',
    'team': 'object',
    'player': 'object',
    'x': 'float64',
    'y': 'float64',
    'under_pressure': 'bool',
    'counterpress': 'bool',
    'duration': 'float64',
    'id': 'object',
    'match_id': 'int64',
    'minute': 'int64',
    'second': 'int64',
    'out': 'float64',
    'pass_aerial_won': 'float64',
    'pass_angle': 'float64',
    'pass_body_part': 'object',
    'pass_cross': 'object',
    'pass_height': 'object',
    'pass_length': 'float64',
    'pass_technique': 'object',
    'pass_type': 'object',
    'period': 'int64',
    'play_pattern': 'object',
    'player_id': 'float64',
    'position': 'object',
    'possession': 'int64',
    'possession_length': 'int64',
    'possession_team': 'object',
    'possession_team_id': 'int64',
    'shot_outcome': 'object',
    'shot_statsbomb_xg': 'float64',
    'shot_technique': 'object',
    'shot_type': 'object',
    'team_id': 'int64',
    'timestamp': 'object',
    'blocked': 'bool',
    'freekick': 'bool',
    'end_x_data': 'float64',
    'end_y_data': 'float64',
    'actions_since_free_kick': 'float64',
    'actions_since_throw_in': 'float64',
    'actions_since_goal_kick': 'float64',
    'actions_since_corner': 'float64',
    'next_type': 'object',
    'next_x': 'float64',
    'next_y': 'float64',
    'next_under_pressure': 'bool',
    'next_team_id': 'float64',
    'next_player_id': 'float64',
    'possession_kept': 'bool',
    'actions_since_kick_off': 'float64'
}

# columns_to_keep = column_dtypes.keys()

columns_to_keep = ['index',
 'type',
 'team',
 'player',
 'x',
 'y',
 'under_pressure',
 'counterpress',
 'duration',
 'id',
 'match_id',
#  'match_date', # doesn't exist
 'minute',
 'second',
 'out',
 'pass_aerial_won',
 'pass_angle',
 'pass_body_part',
 'pass_cross',
#  'pass_cut_back', # Not in LaLiga 2019-2020
#  'pass_deflected',
 'pass_height',
 'pass_length',
#  'pass_outswinging',
 'pass_technique',
#  'pass_through_ball',
 'pass_type',
 'period',
 'play_pattern',
 'player_id',
 'position',
 'possession',
#  'possession_length', # Is created
 'possession_team',
 'possession_team_id',
 'shot_outcome',
 'shot_statsbomb_xg',
 'shot_technique',
 'shot_type',
 'team_id',
 'timestamp',
 'blocked',
 'freekick',
 'end_x_data',
 'end_y_data',
#  'pass_end_x',
#  'pass_end_y',
#  'carry_end_x',
#  'carry_end_y',
 ]

final_columns = [
    'x',
    'y',
    'possession_kept',
    'possession',
    'possession_length',
    'chain_length',
    'previous_deleted',
    'match_id',
    'id',
    'period',
    'player_id',
    'type',
    'shot_outcome',
    'shot_statsbomb_xg',
    'pass_height',
    'under_pressure',
    'position',
]

redundant_events = [
    'Miscontrol',
    'Ball Recovery',
    'Dribble',
    '50/50',
    ]

remaining_action_types = [
    'Pass', 
    'Carry', 
    'Clearance', 
    'Error', 
    'Shot',
    ]

class DataSetCreator:
    def __init__(self, 
                 next_state_columns: list[str], 
                 storage_directory: str = ".",
                 columns_to_keep: list[str] = columns_to_keep,
                 redundant_events: list[str] = redundant_events,
                 x_lims = [0, 105],
                 y_lims = [0, 68],
                 remaining_action_types = remaining_action_types,
                 final_columns = final_columns,
                 ):
        self.next_state_columns = next_state_columns
        self.storage_directory = storage_directory
        self.columns_to_keep = columns_to_keep
        self.redundant_events = redundant_events
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.remaining_action_types = remaining_action_types
        self.final_columns = final_columns

    def get_match_events(self, file_path):
        df_events = pd.read_parquet(path=file_path).sort_values(['match_id', 'period', 'minute', 'second'])
        return df_events
    
    def select_columns(self, df_events: pd.DataFrame, columns_to_keep: list[str] = None) -> pd.DataFrame:
        if columns_to_keep == None:
            columns_to_keep = self.columns_to_keep
        df_events = df_events[columns_to_keep]
        return df_events
    
    def vectorize_field_locations(self, df_events: pd.DataFrame) -> pd.DataFrame:
        df_events = df_events.copy() # to delete
        df_events[['x', 'y']] = df_events['location'].apply(pd.Series)
        df_events[['pass_end_x', 'pass_end_y']] = df_events['pass_end_location'].apply(pd.Series)
        df_events[['carry_end_x', 'carry_end_y']] = df_events['carry_end_location'].apply(pd.Series)
        df_events.drop(columns=['location', 'pass_end_location', 'carry_end_location', 'shot_freeze_frame'], inplace=True)
        return df_events

    def adjust_end_field_locations(self, df_events: pd.DataFrame):
        df_events = df_events.copy() # to delete
        df_events.loc[~df_events['pass_end_x'].isna(), 'end_x_data'] = df_events['pass_end_x']
        df_events.loc[~df_events['pass_end_y'].isna(), 'end_y_data'] = df_events['pass_end_y']
        df_events.loc[~df_events['carry_end_x'].isna(), 'end_x_data'] = df_events['carry_end_x']
        df_events.loc[~df_events['carry_end_y'].isna(), 'end_y_data'] = df_events['carry_end_y']
        df_events.drop(columns=['pass_end_x', 'pass_end_y', 'carry_end_x', 'carry_end_y'], inplace=True)
        return df_events
    
    def rename_free_kicks(self, 
                          df_events: pd.DataFrame, 
                          count_actions_from_free_kick: int = 5,
                          ) -> pd.DataFrame:
        for i in range(count_actions_from_free_kick):
            df_events.loc[df_events['pass_type'].shift(i) == 'Free Kick', 'actions_since_free_kick'] = i
            df_events.loc[df_events['shot_type'].shift(i) == 'Free Kick', 'actions_since_free_kick'] = i
        df_events.loc[df_events['actions_since_free_kick'] == 0, 'type'] = 'Free Kick'
        return df_events

    def rename_throw_ins(self, 
                          df_events: pd.DataFrame, 
                          count_actions_from_throw_in: int = 5,
                          ) -> pd.DataFrame:
        for i in range(count_actions_from_throw_in):
            df_events.loc[df_events['pass_type'].shift(i) == 'Throw-in', 'actions_since_throw_in'] = i
        df_events.loc[df_events['actions_since_throw_in'] == 0, 'type'] = 'Throw-in'
        return df_events
    
    def rename_goal_kick(self,
                          df_events: pd.DataFrame, 
                          count_actions_from_goal_kick: int = 5,
                          ):
        for i in range(count_actions_from_goal_kick):
            df_events.loc[df_events['pass_type'].shift(i) == 'Goal Kick', 'actions_since_goal_kick'] = i
        df_events.loc[df_events['actions_since_goal_kick'] == 0, 'type'] = 'Goal Kick'
        return df_events

    def rename_corner(self,
                          df_events: pd.DataFrame, 
                          count_actions_from_corner: int = 5,
                          ):
        for i in range(count_actions_from_corner):
            df_events.loc[df_events['pass_type'].shift(i) == 'Corner', 'actions_since_corner'] = i
        df_events.loc[df_events['actions_since_corner'] == 0, 'type'] = 'Corner'
        return df_events
    
    def rename_penalty(self,
                       df_events: pd.DataFrame,
                       count_actions_from_penalty: int = 5,
                       ):
        for i in range(count_actions_from_penalty):
            df_events.loc[df_events['shot_type'].shift(i) == 'Penalty', 'actions_since_penalty'] = i
        df_events.loc[df_events['actions_since_penalty'] == 0, 'type'] = 'Penalty'
        return df_events
    
    def remove_redundant_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        df_events = df_events.drop(df_events[df_events['type'].isin(self.redundant_events)].index)
        return df_events
   
    def add_info_possession_kept(self, df_events: pd.DataFrame):
        df_events['possession_kept'] = df_events['team_id'] == df_events['team_id'].shift(-1)
        return df_events
    
    def process_kick_offs(self, 
                          df_events: pd.DataFrame, 
                          del_actions_from_kickoff: int = 2, 
                          count_actions_from_kickoff: int = 5,
                          ) -> pd.DataFrame:
        for i in range(count_actions_from_kickoff+1):
            df_events.loc[df_events['pass_type'].shift(i) == 'Kick Off', 'actions_since_kick_off'] = i
        df_events = df_events.drop(df_events[df_events['actions_since_kick_off'] <= del_actions_from_kickoff].index)
        return df_events
    
    def process_end_of_half(self, df_events: pd.DataFrame) -> pd.DataFrame:
        df_events = df_events.drop(df_events[df_events['type'].shift(-1) == 'Half End'].index)
        return df_events
    
    def add_statsbomb_possession_length(self, df_events: pd.DataFrame, possession_col: str = 'possession'):
        df_events['possession_length'] = df_events.groupby(possession_col)['player_id'].cumcount()
        return df_events
    
    def add_custom_chain_id(self, df_events: pd.DataFrame):
        chain_start = (df_events['possession_length'] == 0) | \
                (df_events['possession_kept'] == False) | \
                (df_events['type'].shift(1) == 'Half End') | \
                (df_events['type'].shift(1) == 'Free Kick') | \
                (df_events['type'].shift(1) == 'Referee Ball-Drop') | \
                (df_events['type'].shift(1) == 'Offside') | \
                (df_events['type'].shift(1) == 'Goal Kick') | \
                (df_events['type'].shift(1) == 'Throw-In') | \
                (df_events['type'].shift(1) == 'Penalty') | \
                (df_events['type'].shift(1) == 'Corner')
        df_events['chain_start'] = chain_start
        df_events['chain_id'] = df_events['chain_start'].cumsum()
        return df_events

    def keep_open_play_events(self, df_events: pd.DataFrame):
        # Check which actions start from set-pieces or go to set-pieces
        keep_mask = df_events['type'].isin(self.remaining_action_types) & df_events['type'].shift(-1).isin(self.remaining_action_types)
        df_events['previous_deleted'] = (~keep_mask).shift(1, fill_value=False)
        df_events = df_events[keep_mask]
        return df_events
    
    def rescale_xy_columns(self, df_events: pd.DataFrame):
        df_events['x'] = df_events['x'] / 120 * (self.x_lims[1] - self.x_lims[0]) + self.x_lims[0]
        df_events['y'] = df_events['y'] / 80 * (self.y_lims[1] - self.y_lims[0]) + self.y_lims[0]
        # df_events['next_x'] = df_events['next_x'] / 120 * (self.x_lims[1] - self.x_lims[0]) + self.x_lims[0]
        # df_events['next_y'] = df_events['next_y'] / 80 * (self.y_ly_limsim[1] - self.y_lims[0]) + self.y_lims[0]
        return df_events

    def filter_out_of_bounds(self, df_events: pd.DataFrame):
        mask_within_bounds = (df_events['x'] < self.x_lims[1]) & (df_events['y'] < self.y_lims[1]) \
            & (df_events['x'] > self.x_lims[0]) & (df_events['y'] > self.y_lims[0])
        df_events.loc[(~mask_within_bounds).shift(1, fill_value=True), 'previous_deleted'] = True
        df_events = df_events[mask_within_bounds]
        df_events.loc[
            (df_events['previous_deleted'] == True) & (df_events['possession_length'] == 1), 
            'possession_length'] = 0
        return df_events

    def add_custom_chain_length(self, df_events: pd.DataFrame):
        df_events['chain_length'] = df_events.groupby('chain_id').cumcount()
        df_events.drop(columns=['chain_id'], inplace=True)
        return df_events

    # def add_info_next_event(self, df_events: pd.DataFrame):
    #     for col in self.next_state_columns:
    #         df_events[f'next_{col}'] = df_events[col].shift(-1)
    #     return df_events
    
    def add_info_next_event(self, df_events: pd.DataFrame):
        # print(df_events[self.next_state_columns].isna().sum())
        for col in self.next_state_columns:
            df_events[f'next_{col}'] = df_events[col].shift(-1)
        
        no_next_col = df_events[f'next_{self.next_state_columns[0]}'].isna()
        # print(no_next_col.sum())
        

        return df_events[~no_next_col]
    
    def filter_columns(self, df_events: pd.DataFrame):
        cols = final_columns + [f'next_{col}' for col in self.next_state_columns]
        return df_events[cols]
            
    def store_match_events(self, df_events: pd.DataFrame, match_file_name: str) -> None:
        relevant_cols_dtypes = {
            col: column_dtypes[col] for col in column_dtypes.keys() if col in df_events.columns.tolist()
            }
        df_events = df_events.astype(column_dtypes_final)
        os.makedirs(self.storage_directory, exist_ok=True)
        file_path = f'{self.storage_directory}\\{match_file_name}'
        if os.path.exists(file_path):
            df_events.to_parquet(file_path, engine='fastparquet', append=True)
        else:
            df_events.to_parquet(file_path, engine='fastparquet')

    def process_single_match(self, file_path, match_file_name) -> None:
        df_events = self.get_match_events(file_path=file_path)

        df_events = self.vectorize_field_locations(df_events)
        df_events = self.adjust_end_field_locations(df_events)

        df_events = self.select_columns(df_events)

        df_events = self.rename_free_kicks(df_events)
        df_events = self.rename_throw_ins(df_events)
        df_events = self.rename_goal_kick(df_events)
        df_events = self.rename_corner(df_events)
        df_events = self.rename_penalty(df_events) # Added at 20-3-2025 -> it happened in the data

        df_events = self.remove_redundant_events(df_events)
        # df_events = self.add_info_next_event(df_events)
        df_events = self.add_info_possession_kept(df_events)
        df_events = self.process_kick_offs(df_events)
        df_events = self.process_end_of_half(df_events)
        df_events = self.add_statsbomb_possession_length(df_events)

        # # Based on the file depricated 'prepare_data_for_counts_v2.ipynb'
        df_events = self.add_custom_chain_id(df_events)
        df_events = self.keep_open_play_events(df_events) # Should also remove penalties
        df_events = self.rescale_xy_columns(df_events)
        df_events = self.filter_out_of_bounds(df_events)
        df_events = self.add_custom_chain_length(df_events)
        df_events = self.add_info_next_event(df_events) # Moved over here
        df_events = self.filter_columns(df_events)

        self.store_match_events(df_events, match_file_name)



    def get_file_paths(self, storage_path) -> list[str]:
        file_list = []

        # Traverse the directory tree
        for dirpath, dirnames, filenames in os.walk(storage_path):
            for filename in filenames:
                if filename.endswith('.parquet') and 'combined' not in filename:
                    full_path = os.path.join(dirpath, filename)
                    file_list.append(full_path)
        return file_list

    def create_data_set(self, storage_path, match_file_name) -> None:
        file_list = self.get_file_paths(storage_path)
        for file_path in tqdm(file_list, desc="Processing files"):
            self.process_single_match(file_path, match_file_name)