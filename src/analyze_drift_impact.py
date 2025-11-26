"""
analyze_drift_impact.py - Calculate Before/After Metrics (USES EXISTING FORECASTS)
===================================================================================

OPTIMIZED VERSION - Uses pre-generated forecasts!

Uses BE_forecasts_test.csv (already generated in Part 2)
Matches the simulation period automatically!
"""

import pandas as pd
import numpy as np
import os
import yaml


def load_config(config_path: str = 'src/config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_mase(residuals: np.ndarray, y_train: np.ndarray, seasonality: int = 24) -> float:
    """Calculate MASE"""
    mae = np.mean(np.abs(residuals))
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    mae_naive = np.mean(naive_errors)
    return mae / mae_naive if mae_naive > 0 else np.nan


def calculate_coverage(actuals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Calculate PI coverage"""
    within = np.sum((actuals >= lo) & (actuals <= hi))
    return 100 * within / len(actuals)


def analyze_drift_event(
    drift_timestamp: pd.Timestamp,
    df_forecasts: pd.DataFrame,
    y_train: np.ndarray,
    window_hours: int = 168
) -> dict:
    """Analyze one drift event"""
    drift_ts = pd.to_datetime(drift_timestamp)
    
    # Before: 7 days before drift
    before_start = drift_ts - pd.Timedelta(hours=window_hours)
    before_end = drift_ts
    
    df_before = df_forecasts[
        (df_forecasts['timestamp'] >= before_start) &
        (df_forecasts['timestamp'] < before_end)
    ].copy()
    
    # After: 7 days after drift
    after_start = drift_ts
    after_end = drift_ts + pd.Timedelta(hours=window_hours)
    
    df_after = df_forecasts[
        (df_forecasts['timestamp'] >= after_start) &
        (df_forecasts['timestamp'] < after_end)
    ].copy()
    
    result = {
        'drift_timestamp': drift_timestamp,
        'n_before': len(df_before),
        'n_after': len(df_after)
    }
    
    # Before metrics
    if len(df_before) >= 24:
        residuals_before = (df_before['y_true'] - df_before['yhat']).values
        result['mase_before_7d'] = calculate_mase(residuals_before, y_train)
        
        if 'lo' in df_before.columns and 'hi' in df_before.columns:
            result['coverage_before_7d'] = calculate_coverage(
                df_before['y_true'].values,
                df_before['lo'].values,
                df_before['hi'].values
            )
        else:
            result['coverage_before_7d'] = np.nan
    else:
        result['mase_before_7d'] = np.nan
        result['coverage_before_7d'] = np.nan
    
    # After metrics
    if len(df_after) >= 24:
        residuals_after = (df_after['y_true'] - df_after['yhat']).values
        result['mase_after_7d'] = calculate_mase(residuals_after, y_train)
        
        if 'lo' in df_after.columns and 'hi' in df_after.columns:
            result['coverage_after_7d'] = calculate_coverage(
                df_after['y_true'].values,
                df_after['lo'].values,
                df_after['hi'].values
            )
        else:
            result['coverage_after_7d'] = np.nan
    else:
        result['mase_after_7d'] = np.nan
        result['coverage_after_7d'] = np.nan
    
    # Calculate improvement
    if not np.isnan(result['mase_before_7d']) and not np.isnan(result['mase_after_7d']):
        result['mase_improvement_pct'] = 100 * (
            result['mase_before_7d'] - result['mase_after_7d']
        ) / result['mase_before_7d']
    else:
        result['mase_improvement_pct'] = np.nan
    
    if not np.isnan(result['coverage_before_7d']) and not np.isnan(result['coverage_after_7d']):
        result['coverage_improvement_pct'] = result['coverage_after_7d'] - result['coverage_before_7d']
    else:
        result['coverage_improvement_pct'] = np.nan
    
    return result


def run_drift_impact_analysis(
    cc: str = 'BE',
    output_folder: str = 'outputs/forecasts',
    config_path: str = 'src/config.yaml'
):
    """Run drift impact analysis using existing test forecasts"""
    print(f"\n{'='*80}")
    print(f"DRIFT IMPACT ANALYSIS: {cc}")
    print(f"{'='*80}")
    
    # Load updates log
    updates_path = os.path.join(output_folder, f'{cc}_online_updates.csv')
    if not os.path.exists(updates_path):
        print(f"❌ Updates log not found: {updates_path}")
        print(f"   Run live_loop.py first!")
        return None
    
    df_updates = pd.read_csv(updates_path)
    df_updates['timestamp'] = pd.to_datetime(df_updates['timestamp'])
    
    # Find drift events
    drift_events = df_updates[df_updates['reason'] == 'drift']
    
    if len(drift_events) == 0:
        print(f"ℹ️  No drift events found.")
        return pd.DataFrame()
    
    print(f"\n✅ Found {len(drift_events)} drift event(s):")
    for idx, (_, event) in enumerate(drift_events.iterrows(), 1):
        print(f"   {idx}. {event['timestamp']}")
    
    # ✅ Load EXISTING test forecasts (generated in Part 2)
    forecasts_path = os.path.join(output_folder, f'{cc}_sarima_forecasts_test.csv')
    if not os.path.exists(forecasts_path):
        print(f"\n❌ Test forecasts not found: {forecasts_path}")
        print(f"   Run forecast.py first!")
        return None
    
    df_forecasts = pd.read_csv(forecasts_path)
    df_forecasts['timestamp'] = pd.to_datetime(df_forecasts['timestamp'])
    
    print(f"\n✅ Loaded test forecasts:")
    print(f"   Total hours: {len(df_forecasts):,}")
    print(f"   Period: {df_forecasts['timestamp'].min()} to {df_forecasts['timestamp'].max()}")
    
    # Load training data for MASE baseline
    print(f"\n✅ Loading training data...")
    from load_opsd import load_tidy_country_csvs_from_config
    dfs = load_tidy_country_csvs_from_config(config_path)
    
    if cc not in dfs:
        print(f"❌ Data not found for {cc}")
        return None
    
    df_full = dfs[cc]
    n = len(df_full)
    n_train = int(n * 0.80)
    y_train = df_full['load'].iloc[:n_train].values
    
    print(f"   Training samples: {len(y_train):,}")
    
    # Analyze each drift event
    print(f"\n{'='*80}")
    print("ANALYZING DRIFT EVENTS...")
    print(f"{'='*80}")
    
    results = []
    for idx, (_, event) in enumerate(drift_events.iterrows(), 1):
        print(f"\n[{idx}/{len(drift_events)}] Drift at {event['timestamp']}")
        
        result = analyze_drift_event(
            event['timestamp'],
            df_forecasts,
            y_train,
            window_hours=168
        )
        
        results.append(result)
        
        if not np.isnan(result['mase_before_7d']):
            print(f"   BEFORE (7d): MASE={result['mase_before_7d']:.4f}, Coverage={result['coverage_before_7d']:.1f}%")
            print(f"   AFTER  (7d): MASE={result['mase_after_7d']:.4f}, Coverage={result['coverage_after_7d']:.1f}%")
            print(f"   CHANGE:      MASE {result['mase_improvement_pct']:+.1f}%, Coverage {result['coverage_improvement_pct']:+.1f}%")
            
            if result['mase_improvement_pct'] > 0:
                print(f"   ✅ Model improved after drift refit!")
            else:
                print(f"   ⚠️  Model performance degraded or unchanged")
        else:
            print(f"   ⚠️  Insufficient data")
            print(f"      Before: {result['n_before']}, After: {result['n_after']}")
    
    # Save results
    df_results = pd.DataFrame(results)
    results_path = os.path.join(output_folder, 'drift_impact_analysis.csv')
    df_results.to_csv(results_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"💾 Saved to: {results_path}")
    
    if len(df_results) > 0:
        print(f"\n📊 SUMMARY TABLE:")
        print(f"\n{df_results.to_string(index=False)}")
    
    return df_results


if __name__ == '__main__':
    import sys
    cc = sys.argv[1] if len(sys.argv) > 1 else 'BE'
    run_drift_impact_analysis(cc=cc)
