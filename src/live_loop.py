"""
live_loop_FIXED.py - Part 4: Live Ingestion + Online Adaptation (FIXED)
=======================================================================

Fixed: Drift detection now has 24-hour cooldown to prevent excessive refits
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
    seasonal_order: Tuple[int, int, int, int],
    exog: pd.DataFrame = None
) -> object:
    """Fit SARIMAX model"""
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    res = model.fit(disp=False, maxiter=50, method='lbfgs')
    return res


def forecast_next_24h(
    model_result: object,
    exog_future: pd.DataFrame = None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Generate 24-hour ahead forecast"""
    fc = model_result.get_forecast(steps=24, exog=exog_future)
    mean = fc.predicted_mean
    conf_int = fc.conf_int(alpha=0.2)
    lo = conf_int.iloc[:, 0]
    hi = conf_int.iloc[:, 1]
    
    return mean, lo, hi


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


def check_drift_trigger(
    z_scores: list,
    window: int = 720,
    alpha: float = 0.1
) -> bool:
    """Check if drift is detected using EWMA"""
    if len(z_scores) < window:
        return False
    
    recent_z = z_scores[-window:]
    abs_z = np.abs(recent_z)
    
    p95 = np.percentile(abs_z, 95)
    
    ewma = abs_z[0]
    for z in abs_z[1:]:
        ewma = alpha * z + (1 - alpha) * ewma
    
    drift = ewma > p95
    
    return drift


class DriftDetector:
    """Track drift state to prevent excessive consecutive triggers"""
    
    def __init__(self, cooldown_hours: int = 24):
        self.cooldown_hours = cooldown_hours
        self.last_drift_timestamp = None
        self.drift_count = 0
        self.drift_history = []
    
    def check(self, z_scores: list, current_ts, window: int = 720, alpha: float = 0.1) -> Tuple[bool, str]:
        """
        Check if drift should trigger a refit
        
        Returns:
            (should_trigger, message)
        """
        # Not enough data
        if len(z_scores) < window:
            return False, None
        
        # Calculate drift
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
            # First drift detection
            self.last_drift_timestamp = current_ts
            self.drift_count += 1
            self.drift_history.append({
                'timestamp': current_ts,
                'count': self.drift_count,
                'ewma': ewma,
                'p95': p95
            })
            msg = f"Drift detected (#{self.drift_count}) at {current_ts}"
            return True, msg
        else:
            hours_since = (current_ts - self.last_drift_timestamp).total_seconds() / 3600
            
            if hours_since >= self.cooldown_hours:
                # Cooldown expired, allow new drift refit
                self.last_drift_timestamp = current_ts
                self.drift_count += 1
                self.drift_history.append({
                    'timestamp': current_ts,
                    'count': self.drift_count,
                    'ewma': ewma,
                    'p95': p95
                })
                msg = f"Drift detected (#{self.drift_count}, {hours_since:.0f}h since last) at {current_ts}"
                return True, msg
            else:
                # Still in cooldown
                msg = f"Drift detected but cooldown active ({hours_since:.1f}h of {self.cooldown_hours}h)"
                return False, msg


def simulate_live_stream(
    cc: str,
    df_full: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    output_folder: str = 'outputs',
    initial_history_days: int = 120,
    simulation_hours: int = 2000,
    refit_window_days: int = 90,
    strategy: str = 'rolling_sarima'
):
    """
    Simulate live data stream with online adaptation (FIXED VERSION)
    """
    print(f"\n{'='*80}")
    print(f"Live Stream Simulation: {cc}")
    print(f"{'='*80}")
    print(f"Strategy: {strategy}")
    print(f"Initial history: {initial_history_days} days")
    print(f"Simulation hours: {simulation_hours}")
    print(f"Refit window: {refit_window_days} days")
    
    df_full = df_full.sort_values('timestamp').reset_index(drop=True)
    
    # Split into initial history and simulation stream
    initial_hours = initial_history_days * 24
    df_history = df_full.iloc[:initial_hours].copy()
    df_stream = df_full.iloc[initial_hours:initial_hours + simulation_hours].copy()
    
    print(f"\nInitial history: {len(df_history)} hours")
    print(f"Stream to simulate: {len(df_stream)} hours")
    
    # Initialize
    history = df_history.copy()
    residuals = []
    z_scores = []
    forecasts_log = []
    updates_log = []
    
    # ✅ FIXED: Add drift detector with cooldown
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
    
    for i, (idx, row) in enumerate(df_stream.iterrows()):
        current_ts = row['timestamp']
        current_hour = pd.to_datetime(current_ts).hour
        current_date = pd.to_datetime(current_ts).date()
        
        # Append new observation to history
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        
        # Keep only rolling window for refit
        if len(history) > refit_window_days * 24:
            history = history.iloc[-(refit_window_days * 24):].reset_index(drop=True)
        
        # At 00:00 UTC: forecast next 24h
        if current_hour == 0 and current_date != last_forecast_date:
            try:
                mean, lo, hi = forecast_next_24h(model_result)
                
                forecasts_log.append({
                    'timestamp': current_ts,
                    'forecast_mean': mean.values,
                    'forecast_lo': lo.values,
                    'forecast_hi': hi.values
                })
                
                last_forecast_date = current_date
                
            except Exception as e:
                print(f"   ⚠️  Forecast failed at {current_ts}: {e}")
        
        # Compute residual if forecast available
        if len(forecasts_log) > 0:
            last_fc = forecasts_log[-1]
            fc_ts = last_fc['timestamp']
            fc_mean = last_fc['forecast_mean']
            
            hours_since_fc = (pd.to_datetime(current_ts) - pd.to_datetime(fc_ts)).total_seconds() / 3600
            
            if 0 < hours_since_fc <= 24:
                step = int(hours_since_fc) - 1
                if step < len(fc_mean):
                    yhat = fc_mean[step]
                    residual = row['load'] - yhat
                    residuals.append(residual)
                    
                    z = compute_rolling_zscore_online(residuals, window=336, min_periods=168)
                    z_scores.append(z)
                    
                    drift_triggered, msg = drift_detector.check(z_scores, current_ts, window=720, alpha=0.1)
                    
                    if drift_triggered:
                        should_trigger = True  # ✅ Set flag
                        print(f"   🚨 {msg}")
                    elif msg is not None and len(z_scores) >= 720:
                        # Only print cooldown messages occasionally (every 100 hours)
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
                
            except Exception as e:
                print(f"   ⚠️  Refit failed: {e}")
        
        should_trigger = False  # ✅ Reset flag
        
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
    
    orders_path = config.get('outputs', {}).get('figures_folder', 'outputs')
    orders_path = os.path.join(orders_path, 'sarima_orders_summary.yaml')
    
    sarima_orders = load_sarima_orders(orders_path)
    
    if live_country not in sarima_orders:
        print(f"❌ Error: No SARIMA orders found for {live_country}")
        return
    
    order = sarima_orders[live_country]['order']
    seasonal_order = sarima_orders[live_country]['seasonal_order']
    
    print(f"SARIMA order: {order}")
    print(f"Seasonal order: {seasonal_order}")
    
    from load_opsd import load_tidy_country_csvs_from_config
    dfs = load_tidy_country_csvs_from_config(config_path)
    
    if live_country not in dfs:
        print(f"❌ Error: Data not found for {live_country}")
        return
    
    df_full = dfs[live_country]
    
    output_folder = config.get('outputs', {}).get('forecasts_folder', 'outputs')
    initial_history_days = config.get('live', {}).get('initial_history_days', 120)
    simulation_hours = config.get('live', {}).get('simulation_hours', 2000)
    refit_window_days = config.get('live', {}).get('refit_window_days', 90)
    strategy = config.get('live', {}).get('strategy', 'rolling_sarima')
    
    df_updates = simulate_live_stream(
        cc=live_country,
        df_full=df_full,
        order=order,
        seasonal_order=seasonal_order,
        output_folder=output_folder,
        initial_history_days=initial_history_days,
        simulation_hours=simulation_hours,
        refit_window_days=refit_window_days,
        strategy=strategy
    )
    
    print("\n" + "="*80)
    print("✅ PART 4 COMPLETE: Live simulation finished")
    print("="*80)
    
    return df_updates


if __name__ == '__main__':
    run_live_simulation()
