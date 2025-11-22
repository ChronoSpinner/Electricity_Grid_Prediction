import pandas as pd
import yaml
import os
import warnings

warnings.filterwarnings('ignore')


def check_and_impute_weather(df, columns=['wind', 'solar']):
    for col in columns:
        if col in df.columns:
            # Interpolate is better for weather than simple ffill/bfill
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            df[col] = df[col].clip(lower=0)
    return df


def load_tidy_country_csvs_from_config(config_path='src/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    countries = config['countries']
    input_folder = config['dataset']['input_folder']
    dfs = {}

    print("\n" + "=" * 80)
    print("LOADING COUNTRY DATA FROM CSV FILES")
    print("=" * 80)

    for cc in countries:
        file_path = os.path.join(input_folder, f"{cc}.csv")

        if not os.path.exists(file_path):
            print(f"\n❌ Warning: file for country '{cc}' not found at {file_path}, skipping")
            continue

        print(f"\n🌍 Loading {cc} from {file_path}...")
        df = pd.read_csv(file_path)

        # Rename columns
        df = df.drop(columns=['cet_cest_timestamp'], errors='ignore')
        df.rename(columns={
            'utc_timestamp': 'timestamp',
            f'{cc}_load_actual_entsoe_transparency': 'load',
            f'{cc}_solar_generation_actual': 'solar',
            f'{cc}_wind_generation_actual': 'wind'
        }, inplace=True)

        if 'timestamp' not in df.columns:
            raise ValueError(f"Missing column 'timestamp' — file {file_path}")

        # Parse Dates
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        # Drop rows where TIMESTAMP is garbage (unlikely, but safe)
        df.dropna(subset=['timestamp'], inplace=True)
        
        # Set Index and Sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # --- CRITICAL FIX: ENFORCE HOURLY FREQUENCY ---
        # This inserts missing rows if any hours are skipped in the CSV
        # It prevents .shift() from misaligning data later.
        df = df.asfreq('H')
        
        # Impute LOAD (Target) - Small gaps only
        # If we have a 1-2 hour gap, linear interpolation is safe.
        # If 'load' was missing, asfreq created a NaN. We fill it now.
        if 'load' in df.columns:
            missing_load = df['load'].isna().sum()
            if missing_load > 0:
                print(f"   ⚠️ Found {missing_load} missing load hours. Interpolating...")
                df['load'] = df['load'].interpolate(method='time', limit=24)
                
                # If gaps are huge (>24h), we might still have NaNs. Drop those edges.
                df.dropna(subset=['load'], inplace=True)

        # Impute Wind/Solar
        df = check_and_impute_weather(df)

        # Reset index for compatibility with the rest of your pipeline
        df.reset_index(inplace=True)

        dfs[cc] = df

        print(f"   ✅ Loaded {len(df)} rows (Hourly aligned)")
        print(f"      Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("\n" + "=" * 80)
    return dfs

if __name__ == '__main__':
    dfs = load_tidy_country_csvs_from_config()