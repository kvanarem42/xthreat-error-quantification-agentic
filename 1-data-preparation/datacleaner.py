import numpy as np
import pandas as pd
from typing import List

class DataCleaner:
    def __init__(self):
        self.redundant_events = [
            'Pressure',
            'Dribbled Past',
            'Substitution',
            'Starting XI',
            'Ball Receipt*',
            'Tactical Shift',
            'Interception',
            'Bad Behaviour',
            'Injury Stoppage',
            'Shield',
            'Foul Committed',
            'Foul Won',
            'Half Start',
        ]

    def del_redundant_events(self, df_events: pd.DataFrame, del_event_list: List = None):
        # if del_event_list == None:
        #     del_event_list = self.redundant_events
        return df_events[~df_events['type'].isin(del_event_list)].copy()
    
    def handle_goalkeeper_event(self, df_events: pd.DataFrame):
        df_events.loc[df_events['goalkeeper_type'] == 'Keeper Sweeper', 'type'] = 'Clearance'
        df_events = self.del_redundant_events(df_events, ['Goal Keeper'])
        return df_events

    def handle_kickoff_event(self, df_events: pd.DataFrame):
        """NOT NECESSARY: pass_type == "Kick Off" gives desired result
        """
        return df_events

    def handle_dispossessed(self, df_events: pd.DataFrame):
        """ NOT IMPLEMENTED YET: actions are deleted for now
        """
        df_events = self.del_redundant_events(df_events, ['Dispossessed', 'Duel'])
        return df_events

    def handle_blocks(self, df_events: pd.DataFrame):
        df_events['blocked'] = df_events['type'].shift(-1) == 'Block'
        df_events = self.del_redundant_events(df_events, ['Block'])
        return df_events

    def handle_freekicks(self, df_events: pd.DataFrame):
        df_events['freekick'] = (df_events['shot_type'] == 'Free Kick') | (df_events['pass_type'] == 'Free Kick')
        return df_events

    def handle_own_goals(self, df_events: pd.DataFrame):
        """NOT IMPLEMENTED YET: actions are deleted for now
        """
        return self.del_redundant_events(df_events, ['Own Goal For','Own Goal Against'])

    def clean(self, df_events: pd.DataFrame):
        # Sort values
        df_events.sort_values('index', inplace=True)

        # Delete unnecessary events
        df_events = self.del_redundant_events(df_events, self.redundant_events)

        # Handling that inserts new events
        df_events = self.handle_goalkeeper_event(df_events)
        df_events = self.handle_dispossessed(df_events)

        # Handling that inserts new columns
        df_events = self.handle_kickoff_event(df_events)
        df_events = self.handle_blocks(df_events)
        df_events = self.handle_freekicks(df_events)

        # Handle own goals
        df_events = self.handle_own_goals(df_events)

        # Handle the indices
        return df_events.copy()

