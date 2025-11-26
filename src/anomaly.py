"""
anomaly.py - Part 3.1: Anomaly Detection using Z-score + CUSUM
==============================================================

Assignment Requirements:
- 3.1.i: Compute 1-step-ahead residuals on Test: et = yt - ŷt
- 3.1.ii: Rolling z-score with window = 336h (14d), min_periods = 168
- 3.1.iii: Flag anomaly if |zt| ≥ 3.0 → flag_z
- 3.1.iv: Optional CUSUM: k = 0.5, h = 5.0

Output: outputs/<CC>_anomalies.csv
Columns: timestamp, y_true, yhat, z_resid, flag_z, flag_cusum
"""

import os
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple


def load_config(config_path: str = 'src/config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_rolling_zscore(
    residuals: pd.Series, 
    window: int = 336, 
    min_periods: int = 168
) -> pd.Series:
    """
    Compute rolling z-score
    
    Args:
        residuals: Series of residuals (et = yt - yhat)
        window: Rolling window size (default 336h = 14 days)
        min_periods: Minimum observations required (default 168h = 7 days)
    
    Returns:
        z-scores
    """
    rolling_mean = residuals.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = residuals.rolling(window=window, min_periods=min_periods).std()
    
    z_scores = (residuals - rolling_mean) / (rolling_std + 1e-10)
    
    return z_scores


def detect_anomalies_zscore(z_scores: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Flag anomalies based on z-score threshold
    
    Args:
        z_scores: Series of z-scores
        threshold: Absolute z-score threshold (default 3.0)
    
    Returns:
        Binary flags (1 = anomaly, 0 = normal)
    """
    flags = (np.abs(z_scores) >= threshold).astype(int)
    return flags


def compute_cusum(
    z_scores: pd.Series, 
    k: float = 0.5, 
    h: float = 5.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute CUSUM (Cumulative Sum Control Chart)
    
    Args:
        z_scores: Series of z-scores
        k: Allowance (slack) parameter (default 0.5)
        h: Threshold for alarm (default 5.0)
    
    Returns:
        (S_plus, S_minus): Cumulative sums for positive and negative deviations
    """
    S_plus = np.zeros(len(z_scores))
    S_minus = np.zeros(len(z_scores))
    
    for i in range(1, len(z_scores)):
        S_plus[i] = max(0, S_plus[i-1] + z_scores.iloc[i] - k)
        S_minus[i] = max(0, S_minus[i-1] - z_scores.iloc[i] - k)
    
    return pd.Series(S_plus, index=z_scores.index), pd.Series(S_minus, index=z_scores.index)


def detect_anomalies_cusum(
    S_plus: pd.Series, 
    S_minus: pd.Series, 
    h: float = 5.0
) -> pd.Series:
    """
    Flag anomalies based on CUSUM alarm threshold
    
    Args:
        S_plus: Positive cumulative sum
        S_minus: Negative cumulative sum
        h: Alarm threshold (default 5.0)
    
    Returns:
        Binary flags (1 = anomaly, 0 = normal)
    """
    flags = ((S_plus > h) | (S_minus > h)).astype(int)
    return flags


def process_country_anomalies(
    cc: str,
    forecasts_path: str,
    output_folder: str = 'outputs',
    window: int = 336,
    min_periods: int = 168,
    zscore_threshold: float = 3.0,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
    use_cusum: bool = True
) -> pd.DataFrame:
    """
    Process anomaly detection for one country
    
    Args:
        cc: Country code (BE, DK, NL)
        forecasts_path: Path to forecasts CSV (test set)
        output_folder: Output directory
        window: Rolling z-score window (default 336h = 14 days)
        min_periods: Min periods for rolling stats (default 168h = 7 days)
        zscore_threshold: Z-score threshold (default 3.0)
        cusum_k: CUSUM slack parameter (default 0.5)
        cusum_h: CUSUM alarm threshold (default 5.0)
        use_cusum: Whether to compute CUSUM (default True)
    
    Returns:
        DataFrame with anomaly flags
    """
    print(f"\n{'='*80}")
    print(f"Processing anomalies for: {cc}")
    print(f"{'='*80}")
    
    # Load forecasts
    df = pd.read_csv(forecasts_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df):,} forecasts from {forecasts_path}")
    
    # Compute residuals: et = yt - ŷt
    df['residual'] = df['y_true'] - df['yhat']
    
    # Compute rolling z-score
    print(f"Computing rolling z-scores (window={window}h, min_periods={min_periods}h)...")
    df['z_resid'] = compute_rolling_zscore(
        df['residual'], 
        window=window, 
        min_periods=min_periods
    )
    
    # Flag anomalies based on z-score
    print(f"Flagging anomalies (|z| ≥ {zscore_threshold})...")
    df['flag_z'] = detect_anomalies_zscore(df['z_resid'], threshold=zscore_threshold)
    
    n_anomalies_z = df['flag_z'].sum()
    pct_anomalies_z = 100 * n_anomalies_z / len(df)
    print(f"Z-score anomalies: {n_anomalies_z} ({pct_anomalies_z:.2f}%)")
    
    # Optional CUSUM
    if use_cusum:
        print(f"Computing CUSUM (k={cusum_k}, h={cusum_h})...")
        S_plus, S_minus = compute_cusum(df['z_resid'].fillna(0), k=cusum_k, h=cusum_h)
        df['flag_cusum'] = detect_anomalies_cusum(S_plus, S_minus, h=cusum_h)
        
        n_anomalies_cusum = df['flag_cusum'].sum()
        pct_anomalies_cusum = 100 * n_anomalies_cusum / len(df)
        print(f"CUSUM anomalies: {n_anomalies_cusum} ({pct_anomalies_cusum:.2f}%)")
    
    # Select output columns
    output_cols = ['timestamp', 'y_true', 'yhat', 'z_resid', 'flag_z']
    if use_cusum:
        output_cols.append('flag_cusum')
    
    df_out = df[output_cols].copy()
    
    # Save to CSV
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'{cc}_anomalies.csv')
    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved anomalies to: {output_path}")
    
    # Show top anomalies
    if n_anomalies_z > 0:
        print(f"\nTop 10 anomalies by |z_resid|:")
        top_anomalies = df[df['flag_z'] == 1].nlargest(10, 'z_resid', keep='all')[
            ['timestamp', 'y_true', 'yhat', 'residual', 'z_resid']
        ]
        print(top_anomalies.to_string(index=False))
    
    return df_out


def run_anomaly_detection(config_path: str = 'src/config.yaml'):
    """
    Run anomaly detection for all countries
    
    Args:
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("PART 3.1: ANOMALY DETECTION (Z-SCORE + CUSUM)")
    print("="*80)
    
    config = load_config(config_path)
    
    # Get country codes
    country_codes = config.get('countries', ['BE', 'DK', 'NL'])
    
    # Get output folder
    output_folder = config.get('outputs', {}).get('forecasts_folder', 'outputs')
    
    # Anomaly detection parameters
    anomaly_cfg = config.get('anomaly', {})
    window = anomaly_cfg.get('zscore_window', 336)  # 14 days
    min_periods = anomaly_cfg.get('zscore_min_periods', 168)  # 7 days
    zscore_threshold = anomaly_cfg.get('zscore_threshold', 3.0)
    use_cusum = anomaly_cfg.get('use_cusum', True)
    cusum_k = anomaly_cfg.get('cusum_k', 0.5)
    cusum_h = anomaly_cfg.get('cusum_h', 5.0)
    
    # Process each country
    results = {}
    for cc in country_codes:
        test_forecasts_path = os.path.join(output_folder, f'{cc}_sarima_forecasts_test.csv')
        
        if not os.path.exists(test_forecasts_path):
            print(f"\n⚠️  Warning: {test_forecasts_path} not found. Skipping {cc}.")
            continue
        
        df_anomalies = process_country_anomalies(
            cc=cc,
            forecasts_path=test_forecasts_path,
            output_folder=output_folder,
            window=window,
            min_periods=min_periods,
            zscore_threshold=zscore_threshold,
            cusum_k=cusum_k,
            cusum_h=cusum_h,
            use_cusum=use_cusum
        )
        
        results[cc] = df_anomalies
    
    print("\n" + "="*80)
    print("✅ PART 3.1 COMPLETE: Anomaly detection finished for all countries")
    print("="*80)
    
    return results


if __name__ == '__main__':
    run_anomaly_detection()
