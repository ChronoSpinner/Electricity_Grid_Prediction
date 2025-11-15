import pandas as pd
import yaml
import os

def load_tidy_country_csvs_from_config(config_path='src/config.yaml'):
    """
    Read 3 separate tidy CSVs per country as per config settings
    Each CSV contains timestamp, load, wind (optional), solar (optional)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    countries = config['countries']
    input_folder = config['dataset']['input_folder']
    
    dfs = {}
    for cc in countries:
        file_path = os.path.join(input_folder, f"{cc}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: file for country '{cc}' not found at {file_path}, skipping")
            continue

        df = pd.read_csv(file_path)

        # safely drop optional column
        df = df.drop(columns=['cet_cest_timestamp'], errors='ignore')

        # rename known columns to standard names (missing keys are ignored)
        df.rename(columns={
            'utc_timestamp': 'timestamp',
            f'{cc}_load_actual_entsoe_transparency': 'load',
            f'{cc}_solar_generation_actual': 'solar',
            f'{cc}_wind_generation_actual': 'wind'
        }, inplace=True)

        # ensure timestamp exists and convert to datetime
        if 'timestamp' not in df.columns:
            raise ValueError(f"Missing column provided to 'parse_dates': 'timestamp' — file {file_path} columns: {list(df.columns)}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # ensure load exists
        if 'load' not in df.columns:
            raise ValueError(f"No 'load' column found in {file_path}. Columns: {list(df.columns)}")

        # drop rows with missing load and sort
        df.dropna(subset=['load'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        dfs[cc] = df
        print(f"Loaded {cc} from {file_path}")
    
    return dfs


dfs = load_tidy_country_csvs_from_config()
print(dfs['DK'].head())
