"""
exogenous_features.py
---------------------
OPTIONAL: Generate exogenous features for SARIMAX models
- Hour-of-day one-hot encoding
- Day-of-week one-hot encoding
- Wind and solar variables (if available in data)
"""

import pandas as pd
import numpy as np


def add_hour_of_day(df):
    """
    Add hour-of-day as one-hot encoded features (0-23).
    Returns: df with columns hour_0, hour_1, ..., hour_23
    """
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    
    for h in range(24):
        df[f'hour_{h}'] = (df['hour'] == h).astype(int)
    
    df.drop('hour', axis=1, inplace=True)
    return df


def add_day_of_week(df):
    """
    Add day-of-week as one-hot encoded features (0=Monday, 6=Sunday).
    Returns: df with columns dow_0, dow_1, ..., dow_6
    """
    df = df.copy()
    df['dow'] = df['timestamp'].dt.dayofweek
    
    for d in range(7):
        df[f'dow_{d}'] = (df['dow'] == d).astype(int)
    
    df.drop('dow', axis=1, inplace=True)
    return df


def add_calendar_features(df):
    """
    Add combined hour and day-of-week features.
    Returns: df with hour_0...hour_23, dow_0...dow_6
    """
    df = add_hour_of_day(df)
    df = add_day_of_week(df)
    return df


def prepare_exogenous_for_sarimax(df, include_wind=True, include_solar=True, 
                                   include_calendar=True):
    """
    Prepare exogenous variables for SARIMAX.
    - Calendar features (hour, day-of-week)
    - Wind and solar (if available and requested)
    
    Returns: DataFrame of exogenous variables (same index as load)
    """
    exog = pd.DataFrame(index=df.index)
    
    # Calendar features
    if include_calendar:
        df_cal = add_calendar_features(df[['timestamp']].copy())
        for col in df_cal.columns:
            exog[col] = df_cal[col].values
    
    # Wind and solar
    if include_wind and 'wind' in df.columns:
        exog['wind'] = df['wind'].fillna(df['wind'].mean())
    
    if include_solar and 'solar' in df.columns:
        exog['solar'] = df['solar'].fillna(df['solar'].mean())
    
    return exog


if __name__ == '__main__':
    # Example usage
    import yaml
    from load_opsd import load_tidy_country_csvs_from_config
    
    dfs = load_tidy_country_csvs_from_config()
    
    for cc, df in dfs.items():
        exog = prepare_exogenous_for_sarimax(df)
        print(f"{cc}: Exogenous features shape: {exog.shape}")
        print(f"   Columns: {list(exog.columns)[:10]}...")
