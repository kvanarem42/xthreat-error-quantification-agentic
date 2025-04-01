import os
import time

import pandas as pd
from tqdm import tqdm

from statsbombpy import sb

from datacleaner import DataCleaner


class DataDownloader:
    def __init__(self, seasons_df: pd.DataFrame, directory: str = "."):
        self.seasons_df = seasons_df
        self.directory = directory
    
    def download_match_events(self, match_id: int) -> pd.DataFrame:
        df_match_events = sb.events(match_id=match_id).sort_values('index')
        return df_match_events
    
    def clean_match_events(self, df_match_events: pd.DataFrame) -> pd.DataFrame:
        cleaner = DataCleaner()
        return cleaner.clean(df_match_events)
    
    def make_file_name(self, match_series: pd.Series) -> str:
        match_id = match_series['match_id']
        match_home_team = match_series['home_team']
        match_home_score = match_series['home_score']
        match_away_team = match_series['away_team']
        match_away_score = match_series['away_score']
        match_date = match_series['match_date']
        return f'{match_id}-{match_date}-{match_home_team}({match_home_score})-{match_away_team}({match_away_score}).parquet'
    
    def store_match_events(self, df_match_events: pd.DataFrame, match_file_name: str, directory: str = None) -> None:
        if directory == None:
            directory = self.directory
        os.makedirs(directory, exist_ok=True)
        df_match_events.to_parquet(f'{directory}\\{match_file_name}')

    
    def download_store_match(self, match_series: pd.Series, directory: str) -> None:
        df_match_events = self.download_match_events(match_series['match_id'])
        df_match_events = self.clean_match_events(df_match_events)
        match_file_name = self.make_file_name(match_series)
        self.store_match_events(df_match_events, match_file_name, directory)

    def make_season_directory_name(self, season_series: pd.Series) -> str:
        competition_id = season_series['competition_id']
        competition_name = season_series['competition_name']
        season_id = season_series['season_id']
        season_name = season_series['season_name'].replace('/', '_')
        print(season_name)
        return f'{self.directory}\\{competition_id}-{competition_name}\\{season_id}-{season_name}'
    
    def download_store_season(self, season_series):
        begin_time = time.time()

        # Get all matches in season
        df_matches = sb.matches(competition_id=season_series['competition_id'], season_id=season_series['season_id']).sort_values(by='match_date', ascending=False)

        # Get directory to store
        directory = self.make_season_directory_name(season_series)

        # Download and store all matches
        for i in tqdm(range(len(df_matches))):
            match_series = df_matches.iloc[i]
            self.download_store_match(match_series, directory)

        print(f"Downloaded, processed, and stored all available events of {season_series['competition_name']} ({season_series['season_name']}) in {(time.time() - begin_time)/60:.2f} minutes.")

    def download_store_seasons(self):
        for i in range(len(self.seasons_df)):
            season_series = self.seasons_df.iloc[i]
            self.download_store_season(season_series)