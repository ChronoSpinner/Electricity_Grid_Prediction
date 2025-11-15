import pandas as pd
import yaml
import os

def load_tidy_country_csvs_from_config(config_path='src\config.yaml'):
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
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = df.drop('cet_cest_timestamp', axis=1)
        df.rename(columns={'utc_timestamp': 'timestamp',f'{cc}_load_actual_entsoe_transparency': 'load',f'{cc}_solar_generation_actual': 'solar',f'{cc}_wind_generation_actual': 'wind'},inplace=True)
        #drop rows with missing load and sort
        df.dropna(subset=['load'], inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        dfs[cc] = df
        print(f"Loaded {cc} from {file_path}")
    
    return dfs


dfs = load_tidy_country_csvs_from_config()
print(dfs['NL'].head())
