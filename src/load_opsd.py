import pandas as pd
import yaml
import os
import warnings

warnings.filterwarnings('ignore')


def check_and_impute_weather(df, columns=['wind', 'solar']):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
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

        print(f"   📊 Raw shape: {df.shape}")
        print(f"      Columns: {list(df.columns)[:10]}...")

        df = df.drop(columns=['cet_cest_timestamp'], errors='ignore')

        df.rename(columns={
            'utc_timestamp': 'timestamp',
            f'{cc}_load_actual_entsoe_transparency': 'load',
            f'{cc}_solar_generation_actual': 'solar',
            f'{cc}_wind_generation_actual': 'wind'
        }, inplace=True)

        if 'timestamp' not in df.columns:
            raise ValueError(f"Missing column 'timestamp' — file {file_path} columns: {list(df.columns)}")

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        if 'load' not in df.columns:
            raise ValueError(f"No 'load' column found in {file_path}. Columns: {list(df.columns)}")

        before_drop = len(df)
        df.dropna(subset=['load'], inplace=True)
        after_drop = len(df)

        if before_drop > after_drop:
            print(f"   🗑️  Dropped {before_drop - after_drop} rows with NaN in load")

        # Impute missing wind and solar data only, no time features added
        df = check_and_impute_weather(df)

        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        dfs[cc] = df

        print(f"   ✅ Loaded {len(df)} rows")
        print(f"      Columns: {list(df.columns)}")
        print(f"      Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("\n" + "=" * 80)
    print(f"✅ Successfully loaded {len(dfs)} countries: {list(dfs.keys())}")
    print("=" * 80 + "\n")

    return dfs


if __name__ == '__main__':
    dfs = load_tidy_country_csvs_from_config()
    for cc, df in dfs.items():
        print(f"\n{cc} - First 5 rows:")
        print(df.head())
        print(f"\n{cc} - Statistics:")
        print(df.describe())
