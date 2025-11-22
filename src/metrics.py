"""
evaluate_metrics.py - Calculate and Compare All Metrics
========================================================

Generates comprehensive metrics for all countries on Dev and Test sets.

Output:
- outputs/metrics/<CC>_metrics_summary.json (per country)
- outputs/metrics/comparison_table.csv (cross-country comparison)
"""

import os
import pandas as pd
import numpy as np
import yaml
import json
from typing import Dict, Tuple


def load_config(config_path: str = 'src/config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 24) -> float:
    """
    Calculate MASE (Mean Absolute Scaled Error)
    
    MASE = MAE / MAE_naive_seasonal
    where naive forecast = y[t-seasonality]
    """
    # MAE of predictions
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAE of naive seasonal forecast on training set
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    mae_naive = np.mean(naive_errors)
    
    if mae_naive == 0:
        return np.nan
    
    mase = mae / mae_naive
    return mase


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate sMAPE (Symmetric Mean Absolute Percentage Error)
    
    sMAPE = 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    mask = denominator != 0
    
    if not np.any(mask):
        return np.nan
    
    smape = 100 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    return smape


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MSE (Mean Squared Error)"""
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE (Root Mean Squared Error)"""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MAPE (Mean Absolute Percentage Error)
    
    MAPE = 100 * mean(|y_true - y_pred| / |y_true|)
    """
    # Avoid division by zero
    mask = y_true != 0
    
    if not np.any(mask):
        return np.nan
    
    mape = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mape


def calculate_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """
    Calculate prediction interval coverage
    
    Coverage = % of actual values within [lo, hi]
    """
    within_interval = (y_true >= lo) & (y_true <= hi)
    coverage = 100 * np.mean(within_interval)
    return coverage


def evaluate_forecasts(
    forecasts_path: str,
    train_data: pd.DataFrame,
    seasonality: int = 24
) -> Dict:
    """
    Evaluate forecasts and calculate all metrics
    
    Args:
        forecasts_path: Path to forecasts CSV
        train_data: Training data for MASE calculation
        seasonality: Seasonal period (default 24)
    
    Returns:
        Dictionary with all metrics
    """
    # Load forecasts
    df = pd.read_csv(forecasts_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    y_true = df['y_true'].values
    y_pred = df['yhat'].values
    lo = df['lo'].values if 'lo' in df.columns else None
    hi = df['hi'].values if 'hi' in df.columns else None
    
    # Calculate metrics
    metrics = {
        'n_samples': len(y_true),
        'MASE': calculate_mase(y_true, y_pred, train_data['load'].values, seasonality),
        'sMAPE': calculate_smape(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred)
    }
    
    # Add coverage if intervals available
    if lo is not None and hi is not None:
        metrics['Coverage_80%'] = calculate_coverage(y_true, lo, hi)
    else:
        metrics['Coverage_80%'] = np.nan
    
    return metrics


def evaluate_country(
    cc: str,
    output_folder: str,
    data_folder: str,
    config: dict
) -> Dict:
    """
    Evaluate metrics for one country on Dev and Test sets
    
    Args:
        cc: Country code
        output_folder: Output directory
        data_folder: Data directory
        config: Configuration dict
    
    Returns:
        Dictionary with Dev and Test metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating metrics for: {cc}")
    print(f"{'='*80}")
    
    # Load tidy data for training set (for MASE)
    from load_opsd import load_tidy_country_csvs_from_config
    dfs = load_tidy_country_csvs_from_config()
    
    if cc not in dfs:
        print(f"⚠️  Data not found for {cc}")
        return None
    
    df_full = dfs[cc]
    
    # Split into train/val/test (80/10/10)
    n = len(df_full)
    n_train = int(n * 0.80)
    
    df_train = df_full.iloc[:n_train]
    
    # Paths to forecast CSVs
    dev_path = os.path.join(output_folder, f'{cc}_forecasts_dev.csv')
    test_path = os.path.join(output_folder, f'{cc}_forecasts_test.csv')
    
    results = {'country': cc}
    
    # Evaluate Dev
    if os.path.exists(dev_path):
        print(f"Evaluating Dev set...")
        dev_metrics = evaluate_forecasts(dev_path, df_train, seasonality=24)
        results['dev'] = dev_metrics
        
        print(f"  MASE: {dev_metrics['MASE']:.4f}")
        print(f"  sMAPE: {dev_metrics['sMAPE']:.2f}%")
        print(f"  RMSE: {dev_metrics['RMSE']:.2f}")
        print(f"  Coverage: {dev_metrics['Coverage_80%']:.2f}%")
    else:
        print(f"⚠️  Dev forecasts not found: {dev_path}")
        results['dev'] = None
    
    # Evaluate Test
    if os.path.exists(test_path):
        print(f"Evaluating Test set...")
        test_metrics = evaluate_forecasts(test_path, df_train, seasonality=24)
        results['test'] = test_metrics
        
        print(f"  MASE: {test_metrics['MASE']:.4f}")
        print(f"  sMAPE: {test_metrics['sMAPE']:.2f}%")
        print(f"  RMSE: {test_metrics['RMSE']:.2f}")
        print(f"  Coverage: {test_metrics['Coverage_80%']:.2f}%")
    else:
        print(f"⚠️  Test forecasts not found: {test_path}")
        results['test'] = None
    
    return results


def create_comparison_table(all_results: Dict) -> pd.DataFrame:
    """
    Create comparison table across countries
    
    Args:
        all_results: Dictionary of results per country
    
    Returns:
        DataFrame with comparison
    """
    rows = []
    
    for cc, results in all_results.items():
        if results is None:
            continue
        
        # Dev row
        if results['dev'] is not None:
            row = {
                'Country': cc,
                'Split': 'Dev',
                'MASE': results['dev']['MASE'],
                'sMAPE': results['dev']['sMAPE'],
                'MSE': results['dev']['MSE'],
                'RMSE': results['dev']['RMSE'],
                'MAPE': results['dev']['MAPE'],
                'Coverage_80%': results['dev']['Coverage_80%']
            }
            rows.append(row)
        
        # Test row
        if results['test'] is not None:
            row = {
                'Country': cc,
                'Split': 'Test',
                'MASE': results['test']['MASE'],
                'sMAPE': results['test']['sMAPE'],
                'MSE': results['test']['MSE'],
                'RMSE': results['test']['RMSE'],
                'MAPE': results['test']['MAPE'],
                'Coverage_80%': results['test']['Coverage_80%']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def run_evaluation(config_path: str = 'src/config.yaml'):
    """
    Run evaluation for all countries
    
    Args:
        config_path: Path to config file
    """
    print("\n" + "="*80)
    print("PART 5: METRICS EVALUATION")
    print("="*80)
    
    config = load_config(config_path)
    
    country_codes = config.get('countries', ['BE', 'DK', 'NL'])
    output_folder = config.get('outputs', {}).get('forecasts_folder', 'outputs')
    data_folder = config.get('dataset', {}).get('input_folder', 'data')
    
    # Create metrics folder
    metrics_folder = os.path.join(output_folder, 'metrics')
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Evaluate each country
    all_results = {}
    for cc in country_codes:
        results = evaluate_country(cc, output_folder, data_folder, config)
        all_results[cc] = results
        
        # Save individual country metrics
        if results is not None:
            metrics_path = os.path.join(metrics_folder, f'{cc}_metrics_summary.json')
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Saved metrics to: {metrics_path}")
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("Creating comparison table...")
    print(f"{'='*80}")
    
    comparison_df = create_comparison_table(all_results)
    
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # Save comparison table
    comparison_path = os.path.join(metrics_folder, 'comparison_table.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✅ Saved comparison table to: {comparison_path}")
    
    print("\n" + "="*80)
    print("✅ PART 5 COMPLETE: Metrics evaluation finished")
    print("="*80)
    
    return all_results, comparison_df


if __name__ == '__main__':
    run_evaluation()
