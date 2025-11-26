import os
import warnings
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from .load_opsd import load_tidy_country_csvs_from_config
    from .exogenous_features import (
        prepare_endog_exog,
        build_future_exogenous,
        build_exogenous,
        _CANONICAL_CALENDAR,
    )
except ImportError:
    from load_opsd import load_tidy_country_csvs_from_config
    from exogenous_features import (
        prepare_endog_exog,
        build_future_exogenous,
        build_exogenous,
        _CANONICAL_CALENDAR,
    )

warnings.filterwarnings("ignore")
DEFAULT_CONFIG_PATH = os.path.join('src', 'config.yaml')
DEFAULT_ORDERS_SUMMARY_PATH = os.path.join('outputs', 'figures', 'sarima_orders_summary.yaml')


def _log(msg: str):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")


def _read_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _load_preselected_orders(path: str) -> Dict[str, Dict[str, Tuple]]:
    try:
        with open(path, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except Exception as e_safe:
                _log(f"safe_load failed on {path} ({e_safe}); retrying with full loader.")
                f.seek(0)
                data = yaml.full_load(f)
        data = data or {}
        orders: Dict[str, Dict[str, Tuple]] = {}
        for cc, v in data.items():
            if not isinstance(v, dict):
                continue
            raw_o = v.get('order')
            raw_so = v.get('seasonal_order')
            if raw_o is not None and raw_so is not None:
                try:
                    o = tuple(raw_o)
                    so = tuple(raw_so)
                    if len(o) == 3 and len(so) == 4:
                        orders[cc] = {'order': o, 'seasonal_order': so}
                    else:
                        _log(f"Ignoring malformed order for {cc}: {o}, {so}")
                except Exception as e_item:
                    _log(f"Failed parsing order for {cc}: {e_item}")
        return orders
    except FileNotFoundError:
        _log(f"Orders file not found: {path}")
        return {}
    except Exception as e:
        _log(f"Failed to load preselected orders from {path}: {e}")
        return {}


def fit_sarimax_order(y, X, order, seasonal_order):
    try:
        model = SARIMAX(
            y,
            exog=X,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        # Powell is robust for tricky exogenous convergence
        res = model.fit(disp=False, method='powell')
        return (order, seasonal_order, res.bic, res.aic)
    except Exception:
        return None


def select_sarima_order(
    y: pd.Series,
    X: Optional[pd.DataFrame],
    sarima_cfg: dict,
    max_workers: int = 8
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    p_range = sarima_cfg.get('p_range', [1])
    d_range = sarima_cfg.get('d_range', [1])
    q_range = sarima_cfg.get('q_range', [1])
    P_range = sarima_cfg.get('P_range', [1])
    D_range = sarima_cfg.get('D_range', [1])
    Q_range = sarima_cfg.get('Q_range', [0])
    s = sarima_cfg.get('s', 24)

    orders_to_fit = []
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, s)
                            orders_to_fit.append((order, seasonal_order))

    _log(f"Order search: {len(orders_to_fit)} combinations to evaluate (grid from config)")

    best = None
    best_metrics = (np.inf, np.inf)  # (BIC, AIC)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fit_sarimax_order, y, X, order, seasonal_order): (order, seasonal_order)
            for order, seasonal_order in orders_to_fit
        }
        count = 0
        total = len(futures)
        interval = max(1, total // 5)

        for future in as_completed(futures):
            count += 1
            result = future.result()
            if result is None:
                continue
            order, seasonal_order, bic, aic = result
            if (bic < best_metrics[0]) or (np.isclose(bic, best_metrics[0]) and aic < best_metrics[1]):
                best_metrics = (bic, aic)
                best = (order, seasonal_order)
            if count % interval == 0 or count == total:
                _log(f"Order search progress: {count}/{total} | current best order={best[0]}, seasonal={best[1]} (BIC={best_metrics[0]:.1f})")

    if best is None:
        best = ((1, 1, 1), (1, 1, 0, s))
    return best


def forecast_step(
    step_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    y: pd.Series,
    df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    config: dict,
    include_calendar: bool,
    include_wind: bool,
    include_solar: bool,
    cc: Optional[str],
    ratios: dict,
    calendar_cache,
    warmup_hours: int,
    horizon: int,
    alpha: float,
    stride: int,
    fit_history_days: Optional[int],
    fit_window_hours: Optional[int],
):
    rows = []
    if step_ts < idx[0] + pd.Timedelta(hours=warmup_hours):
        return pd.DataFrame(columns=['timestamp', 'y_true', 'yhat', 'lo', 'hi', 'horizon', 'trainend'])

    # 1. Define Training Window (The specific data we want to fit on)
    if fit_window_hours is not None:
        train_start = max(idx[0], step_ts - pd.Timedelta(hours=fit_window_hours))
    else:
        train_start = idx[0]

    # 2. Define Feature Engineering Window (Must be larger for Lags)
    # We need at least 1 year of data prior to train_start to compute the lag column
    feat_eng_start = max(idx[0], train_start - pd.Timedelta(hours=8800))
    
    # Slice for Feature Engineering
    df_subset = df.loc[feat_eng_start:step_ts].copy()

    X_hist = None
    Xf = None
    if config['forecasting'].get('use_exogenous', True):
        try:
            # Build Exog on the subset (which includes the buffer for lags)
            X_full_subset = build_exogenous(
                df_subset,
                precomputed_calendar=calendar_cache,
                include_calendar=include_calendar,
                include_wind=include_wind,
                include_solar=include_solar,
            )
            
            # Now slice X_hist to match the actual Training Window
            X_hist = X_full_subset.loc[train_start:step_ts]

            # IMPORTANT: Pass the FULL df as reference to find 52-week lag history
            Xf = build_future_exogenous(
                step_ts,
                periods=horizon,
                precomputed_calendar=calendar_cache,
                include_calendar=include_calendar,
                include_wind=include_wind,
                include_solar=include_solar,
                reference_df=df 
            )

        except Exception as e:
            _log(f"[{cc or ''}][forecast_step] exogenous building failed: {e}")
            X_hist = None
            Xf = None

    # Slice Target y to match Training Window
    y_hist = y.loc[train_start:step_ts]

    try:
        model = SARIMAX(y_hist, exog=X_hist, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, method='lbfgs', maxiter=50)

        fc = res.get_forecast(steps=horizon, exog=Xf)
        mean = fc.predicted_mean
        conf_int = fc.conf_int(alpha=alpha)
        lo = conf_int.iloc[:, 0]
        hi = conf_int.iloc[:, 1]

        step_df = pd.DataFrame({
            'timestamp': mean.index,
            'yhat': mean.values,
            'lo': lo.values,
            'hi': hi.values,
        })
        
        # Clip negatives
        step_df['yhat'] = step_df['yhat'].clip(lower=0)
        step_df['lo'] = step_df['lo'].clip(lower=0)
        step_df['hi'] = step_df['hi'].clip(lower=0)

        step_df['horizon'] = np.arange(1, len(step_df) + 1)
        step_df['train_end'] = step_ts
        step_df = step_df.set_index('timestamp').join(y.rename('y_true'))
        step_df = step_df.reset_index()

        step_df = step_df[(step_df['timestamp'] > step_ts) & (step_df['timestamp'] <= end_ts)]
        rows.append(step_df)
    except Exception as e:
        # Only log critical failures
        pass

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.sort_values(['timestamp', 'horizon'], inplace=True)
        return out
    else:
        return pd.DataFrame(columns=['timestamp', 'y_true', 'yhat', 'lo', 'hi', 'horizon', 'train_end'])


def expanding_backtest(
    df: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    config: dict,
    include_calendar: bool,
    include_wind: bool,
    include_solar: bool,
    cc: Optional[str] = None,
    max_workers: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # NO manual lag features (add_lag_features) - handled by SARIMA or yearly exog
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    y = df['load'].astype(float)
    idx = y.index

    calendar_cache = None
    if include_calendar:
        # Compute calendar for full DF once
        calendar_cache = build_exogenous(df, include_calendar=True, include_wind=False, include_solar=False)
        # Only keep DOW to avoid conflict with SARIMA seasonality
        calendar_cache = calendar_cache[[c for c in calendar_cache.columns if 'dow' in c]]

    ratios = config['forecasting']
    train_ratio = float(ratios.get('train_ratio', 0.8))
    val_ratio = float(ratios.get('val_ratio', 0.1))
    test_ratio = float(ratios.get('test_ratio', 0.1))

    n = len(y)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_end_ts = idx[n_train - 1]
    dev_end_ts = idx[n_train + n_val - 1]

    horizon = int(ratios.get('horizon', 24))
    stride = int(ratios.get('stride', 24))
    warmup_days = int(ratios.get('warmup_days', 60))
    warmup_hours = warmup_days * 24
    conf = float(ratios.get('confidence_level', 0.80))
    alpha = 1.0 - conf

    fit_history_days = config.get('forecasting', {}).get('fit_history_days', 90)
    fit_window_hours = None
    if fit_history_days is not None:
        try:
            fit_history_days = int(fit_history_days)
            fit_window_hours = fit_history_days * 24
            _log(f"[{cc or ''}] Using rolling fit window: last {fit_history_days} days of history per step")
        except Exception:
            fit_window_hours = None

    def run_segment_parallel(start_ts: pd.Timestamp, end_ts: pd.Timestamp, label: str) -> pd.DataFrame:
        step_ts = max(start_ts, idx[0] + pd.Timedelta(hours=warmup_hours))
        remaining_hours = (end_ts - step_ts).total_seconds() / 3600.0
        total_steps = int(remaining_hours // stride) + 1
        step_list = [step_ts + pd.Timedelta(hours=stride) * i for i in range(total_steps)]

        results = []
        last_percent_reported = -1
        start_time = time.time()
        avg_sec = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    forecast_step,
                    ts,
                    end_ts,
                    y,
                    df,
                    idx,
                    order,
                    seasonal_order,
                    config,
                    include_calendar,
                    include_wind,
                    include_solar,
                    cc,
                    ratios,
                    calendar_cache,
                    warmup_hours,
                    horizon,
                    alpha,
                    stride,
                    fit_history_days,
                    fit_window_hours,
                ): ts for ts in step_list
            }

            count = 0
            total = len(futures)

            for future in as_completed(futures):
                count += 1
                res = future.result()
                if not res.empty:
                    results.append(res)

                current_percent = int(round((count / total) * 100))
                if current_percent != last_percent_reported or count == total:
                    last_percent_reported = current_percent
                    elapsed = time.time() - start_time
                    avg_sec = elapsed / count
                    eta_sec = max(0.0, (total - count) * avg_sec)
                    _log(f"[{cc or ''}][{label}] {count}/{total} steps ({current_percent}%) | avg {avg_sec:.2f}s/step | ETA {eta_sec/60:.1f}m")

        if results:
            out = pd.concat(results, ignore_index=True)
            out.sort_values(['timestamp', 'horizon'], inplace=True)
            return out
        else:
            return pd.DataFrame(columns=['timestamp', 'y_true', 'yhat', 'lo', 'hi', 'horizon', 'train_end'])

    dev_fc = run_segment_parallel(train_end_ts, dev_end_ts, label='DEV')
    test_fc = run_segment_parallel(dev_end_ts, idx[-1], label='TEST')
    return dev_fc, test_fc


def calculate_metrics(df: pd.DataFrame, seasonality: int = 24) -> Dict[str, float]:
    y_true = df['y_true'].values
    yhat = df['yhat'].values
    lo = df['lo'].values
    hi = df['hi'].values

    mask = ~np.isnan(y_true) & ~np.isnan(yhat)
    y_true = y_true[mask]
    yhat = yhat[mask]
    lo = lo[mask]
    hi = hi[mask]

    n = len(y_true)
    if n == 0:
        return {m: np.nan for m in ['MASE', 'sMAPE', 'MSE', 'RMSE', 'MAPE', 'PI_80_Coverage']}

    y = y_true
    naive_forecast_error = np.mean(np.abs(y[seasonality:] - y[:-seasonality])) if n > seasonality else 1e-10
    mase = np.mean(np.abs(y_true - yhat)) / (naive_forecast_error + 1e-10)

    denom = (np.abs(y_true) + np.abs(yhat)) / 2
    smape = np.mean(np.abs(y_true - yhat) / (denom + 1e-10)) * 100

    mse = np.mean((y_true - yhat) ** 2)
    rmse = np.sqrt(mse)

    mape = np.mean(np.abs((y_true - yhat) / (y_true + 1e-10))) * 100
    pi_covered = np.mean((y_true >= lo) & (y_true <= hi))

    return {
        'MASE': mase,
        'sMAPE': smape,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'PI_80_Coverage': pi_covered
    }


def run(config_path: str = DEFAULT_CONFIG_PATH):
    _log("Reading config and preparing outputs...")
    cfg = _read_config(config_path)
    outputs = cfg.get('outputs', {})
    out_folder = outputs.get('forecasts_folder') or outputs.get('folder', 'outputs')
    os.makedirs(out_folder, exist_ok=True)

    _log("Loading tidy country data...")
    dfs_by_cc = load_tidy_country_csvs_from_config(config_path)
    _log(f"Loaded countries: {list(dfs_by_cc.keys())}")

    fcfg = cfg.get('forecasting', {})
    use_exog = bool(fcfg.get('use_exogenous', True))
    include_calendar = bool(fcfg.get('include_calendar_features', True)) if use_exog else False
    include_wind = bool(fcfg.get('include_wind', False)) if use_exog else False
    include_solar = bool(fcfg.get('include_solar', False)) if use_exog else False
    _log(f"Exogenous: use={use_exog}, calendar={include_calendar}, wind={include_wind}, solar={include_solar}")

    orders_summary_path = cfg.get('sarima_orders_summary_path', DEFAULT_ORDERS_SUMMARY_PATH)
    preselected = _load_preselected_orders(orders_summary_path)
    if not preselected:
        _log(f"Warning: No preselected orders found at {orders_summary_path}.")

    for cc, df in dfs_by_cc.items():
        _log(f"[{cc}] Preparing order selection and backtest...")
        df_idx = df.set_index('timestamp').sort_index()
        
        if cc not in preselected:
             # Fallback or error
             raise KeyError(f"No preselected SARIMA order found for {cc}")
             
        order = preselected[cc]['order']
        seasonal_order = preselected[cc]['seasonal_order']
        _log(f"[{cc}] Using preselected order={order}, seasonal_order={seasonal_order}")

        dev_fc, test_fc = expanding_backtest(
            df,
            order,
            seasonal_order,
            cfg,
            include_calendar,
            include_wind,
            include_solar,
            cc=cc,
            max_workers=8,
        )

        dev_path = os.path.join(out_folder, f"{cc}_sarima_forecasts_dev.csv")
        test_path = os.path.join(out_folder, f"{cc}_sarima_forecasts_test.csv")
        dev_fc.to_csv(dev_path, index=False)
        test_fc.to_csv(test_path, index=False)
        _log(f"[{cc}] Wrote dev forecasts -> {dev_path} ({len(dev_fc)} rows)")
        _log(f"[{cc}] Wrote test forecasts -> {test_path} ({len(test_fc)} rows)")

        dev_metrics = calculate_metrics(dev_fc)
        test_metrics = calculate_metrics(test_fc)

        _log(f"[{cc}] DEV metrics: {dev_metrics}")
        _log(f"[{cc}] TEST metrics: {test_metrics}")


if __name__ == '__main__':
    run()