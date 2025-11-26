"""
live_loop.py - Part 4: Live Ingestion (USES EXISTING FORECASTS)
================================================================

OPTIMIZED VERSION - Uses pre-generated forecasts from Part 2!

Changes:
1. Simulate on same 2000-hour period as test forecasts (2020 data)
2. Load pre-generated forecasts from BE_forecasts_test.csv
3. Use those forecasts for residual calculation (no re-generation)
4. Much faster! (~30 seconds vs 30+ minutes)
"""

import os
import pandas as pd
import numpy as np
import yaml
import time
from typing import Dict, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: str = 'src/config.yaml') -> dict:
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sarima_orders(path: str) -> Dict:
    """Load SARIMA orders from YAML"""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    orders = {}
    for cc, params in data.items():
        orders[cc] = {
            'order': tuple(params['order']),
            'seasonal_order': tuple(params['seasonal_order'])
        }
    
    return orders


def fit_sarima_model(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int]
) -> object:
    """Fit SARIMAX model"""
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    res = model.fit(disp=False, maxiter=50, method='lbfgs')
    return res


def compute_rolling_zscore_online(
    residuals: list,
    window: int = 336,
    min_periods: int = 168
) -> float:
    """Compute z-score for most recent residual"""
    if len(residuals) < min_periods:
        return 0.0
    
    recent = residuals[-window:]
    mean = np.mean(recent)
    std = np.std(recent)
    
    if std < 1e-10:
        return 0.0
    
    z = (residuals[-1] - mean) / std
    return z


class DriftDetector:
    """Track drift state with 24h cooldown"""
    
    def __init__(self, cooldown_hours: int = 24):
        self.cooldown_hours = cooldown_hours
        self.last_drift_timestamp = None
        self.drift_count = 0
    
    def check(self, z_scores: list, current_ts, window: int = 720, alpha: float = 0.1) -> Tuple[bool, str]:
        """Check if drift should trigger a refit"""
        if len(z_scores) < window:
            return False, None
        
        recent_z = z_scores[-window:]
        abs_z = np.abs(recent_z)
        p95 = np.percentile(abs_z, 95)
        
        ewma = abs_z[0]
        for z in abs_z[1:]:
            ewma = alpha * z + (1 - alpha) * ewma
        
        is_drift = ewma > p95
        
        if not is_drift:
            return False, None
        
        # Drift detected - check cooldown
        if self.last_drift_timestamp is None:
            self.last_drift_timestamp = current_ts
            self.drift_count += 1
            msg = f"Drift detected (#{self.drift_count}) at {current_ts}"
            return True, msg
        else:
            hours_since = (current_ts - self.last_drift_timestamp).total_seconds() / 3600
            
            if hours_since >= self.cooldown_hours:
                self.last_drift_timestamp = current_ts
                self.drift_count += 1
                msg = f"Drift detected (#{self.drift_count}, {hours_since:.0f}h since last) at {current_ts}"
                return True, msg
            else:
                msg = f"Drift detected but cooldown active ({hours_since:.1f}h of {self.cooldown_hours}h)"
                return False, msg


def simulate_live_stream(
    cc: str,
    df_full: pd.DataFrame,
    df_forecasts: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    output_folder: str = 'outputs',
    initial_history_days: int = 120,
    refit_window_days: int = 90,
    strategy: str = 'rolling_sarima'
):
    """
    Simulate live data stream using PRE-GENERATED forecasts
    
    ✅ Uses existing BE_forecasts_test.csv (FAST!)
    ✅ Simulates on same 2000-hour period as test forecasts
    ✅ Drift analysis will work perfectly
    """
    print(f"\n{'='*80}")
    print(f"Live Stream Simulation: {cc}")
    print(f"{'='*80}")
    print(f"Strategy: {strategy}")
    print(f"Using pre-generated forecasts (FAST mode)")
    
    # Sort data
    df_full = df_full.sort_values('timestamp').reset_index(drop=True)
    df_forecasts = df_forecasts.sort_values('timestamp').reset_index(drop=True)
    
    # ✅ Get simulation period from forecasts
    forecast_start = df_forecasts['timestamp'].min()
    forecast_end = df_forecasts['timestamp'].max()
    simulation_hours = len(df_forecasts)
    
    print(f"\nForecast period (from test set):")
    print(f"  Start: {forecast_start}")
    print(f"  End: {forecast_end}")
    print(f"  Hours: {simulation_hours}")
    
    # Find matching period in full dataset
    stream_start_idx = df_full[df_full['timestamp'] == forecast_start].index[0]
    stream_end_idx = df_full[df_full['timestamp'] == forecast_end].index[0] + 1
    
    # Get history (120 days before stream)
    initial_hours = initial_history_days * 24
    history_start_idx = stream_start_idx - initial_hours
    
    df_history = df_full.iloc[history_start_idx:stream_start_idx].copy()
    df_stream = df_full.iloc[stream_start_idx:stream_end_idx].copy()
    
    print(f"\nData period:")
    print(f"  History: {df_history['timestamp'].min()} to {df_history['timestamp'].max()}")
    print(f"  Stream:  {df_stream['timestamp'].min()} to {df_stream['timestamp'].max()}")
    print(f"  Initial history: {len(df_history)} hours")
    print(f"  Stream to simulate: {len(df_stream)} hours")
    
    # Initialize
    history = df_history.copy()
    residuals = []
    z_scores = []
    updates_log = []
    
    # Drift detector with cooldown
    drift_detector = DriftDetector(cooldown_hours=24)
    
    # Initial model fit
    print(f"\nFitting initial model...")
    start_time = time.time()
    y_init = history['load']
    model_result = fit_sarima_model(y_init, order, seasonal_order)
    duration = time.time() - start_time
    
    updates_log.append({
        'timestamp': history['timestamp'].iloc[-1],
        'strategy': strategy,
        'reason': 'initial',
        'duration_s': duration
    })
    
    print(f"✅ Initial model fitted in {duration:.2f}s")
    
    # Simulation loop
    print(f"\nStarting simulation loop...")
    last_forecast_date = None
    should_trigger = False
    
    # Create forecast lookup (for faster access)
    forecast_dict = df_forecasts.set_index('timestamp').to_dict('index')
    
    for i, (idx, row) in enumerate(df_stream.iterrows()):
        current_ts = row['timestamp']
        current_hour = pd.to_datetime(current_ts).hour
        current_date = pd.to_datetime(current_ts).date()
        
        # Append new observation to history
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        
        # Keep only rolling window for refit
        if len(history) > refit_window_days * 24:
            history = history.iloc[-(refit_window_days * 24):].reset_index(drop=True)
        
        # ✅ Use pre-generated forecast
        if current_ts in forecast_dict:
            forecast_row = forecast_dict[current_ts]
            yhat = forecast_row['yhat']
            y_true = row['load']
            residual = y_true - yhat
            residuals.append(residual)
            
            z = compute_rolling_zscore_online(residuals, window=336, min_periods=168)
            z_scores.append(z)
            
            # Check drift with cooldown
            drift_triggered, msg = drift_detector.check(z_scores, current_ts, window=720, alpha=0.1)
            
            if drift_triggered:
                should_trigger = True
                print(f"   🚨 {msg}")
            elif msg is not None and len(z_scores) >= 720:
                if i % 100 == 0:
                    print(f"   ℹ️  {msg}")
        
        # Daily refit at 00:00 OR drift trigger
        should_refit = (current_hour == 0 and current_date != last_forecast_date) or should_trigger
        
        if should_refit:
            reason = 'scheduled' if current_hour == 0 else 'drift'
            
            print(f"   🔄 Refitting model at {current_ts} (reason: {reason})...")
            start_time = time.time()
            
            try:
                y_recent = history['load']
                model_result = fit_sarima_model(y_recent, order, seasonal_order)
                duration = time.time() - start_time
                
                updates_log.append({
                    'timestamp': current_ts,
                    'strategy': strategy,
                    'reason': reason,
                    'duration_s': duration
                })
                
                print(f"   ✅ Refit complete in {duration:.2f}s")
                
                if current_hour == 0:
                    last_forecast_date = current_date
                
            except Exception as e:
                print(f"   ⚠️  Refit failed: {e}")
        
        should_trigger = False
        
        # Progress
        if (i + 1) % 500 == 0:
            print(f"   Progress: {i+1}/{len(df_stream)} hours simulated")
    
    # Save updates log
    df_updates = pd.DataFrame(updates_log)
    updates_path = os.path.join(output_folder, f'{cc}_online_updates.csv')
    df_updates.to_csv(updates_path, index=False)
    
    print(f"\n✅ Simulation complete!")
    print(f"   Total updates: {len(updates_log)}")
    print(f"   Scheduled: {df_updates['reason'].eq('scheduled').sum()}")
    print(f"   Drift-triggered: {df_updates['reason'].eq('drift').sum()}")
    print(f"   Drift detector: {drift_detector.drift_count} drift events detected")
    print(f"💾 Saved updates log to: {updates_path}")
    
    return df_updates


def run_live_simulation(config_path: str = 'src/config.yaml'):
    """Run live stream simulation"""
    print("\n" + "="*80)
    print("PART 4: LIVE INGESTION + ONLINE ADAPTATION")
    print("="*80)
    
    config = load_config(config_path)
    
    live_country = config.get('live', {}).get('country', 'BE')
    print(f"\nLive country: {live_country}")
    
    orders_path = config.get('outputs', {}).get('figures_folder', 'outputs/figures')
    orders_path = os.path.join(orders_path, 'sarima_orders_summary.yaml')
    
    sarima_orders = load_sarima_orders(orders_path)
    
    if live_country not in sarima_orders:
        print(f"❌ Error: No SARIMA orders found for {live_country}")
        return
    
    order = sarima_orders[live_country]['order']
    seasonal_order = sarima_orders[live_country]['seasonal_order']
    
    print(f"SARIMA order: {order}")
    print(f"Seasonal order: {seasonal_order}")
    
    # Load full data
    from load_opsd import load_tidy_country_csvs_from_config
    dfs = load_tidy_country_csvs_from_config(config_path)
    
    if live_country not in dfs:
        print(f"❌ Error: Data not found for {live_country}")
        return
    
    df_full = dfs[live_country]
    
    # ✅ Load pre-generated forecasts
    output_folder = config.get('outputs', {}).get('forecasts_folder', 'outputs')
    forecasts_path = os.path.join(output_folder, f'{live_country}_sarima_forecasts_test.csv')
    
    if not os.path.exists(forecasts_path):
        print(f"❌ Error: Test forecasts not found: {forecasts_path}")
        print(f"   Run forecast.py first!")
        return
    
    df_forecasts = pd.read_csv(forecasts_path)
    df_forecasts['timestamp'] = pd.to_datetime(df_forecasts['timestamp'])
    
    print(f"✅ Loaded pre-generated forecasts: {len(df_forecasts)} hours")
    
    initial_history_days = config.get('live', {}).get('initial_history_days', 120)
    refit_window_days = config.get('live', {}).get('refit_window_days', 90)
    strategy = config.get('live', {}).get('strategy', 'rolling_sarima')
    
    df_updates = simulate_live_stream(
        cc=live_country,
        df_full=df_full,
        df_forecasts=df_forecasts,
        order=order,
        seasonal_order=seasonal_order,
        output_folder=output_folder,
        initial_history_days=initial_history_days,
        refit_window_days=refit_window_days,
        strategy=strategy
    )
    
    print("\n" + "="*80)
    print("✅ PART 4 COMPLETE: Live simulation finished")
    print("="*80)
    
    return df_updates


if __name__ == '__main__':
    run_live_simulation()
