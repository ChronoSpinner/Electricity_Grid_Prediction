import pandas as pd
import numpy as np
from typing import Optional, List

# Only use Day of Week. Hour is handled by SARIMA Seasonality (s=24)
# This prevents multicollinearity which ruins metrics.
_DOW_COLS = [f"dow_{d}" for d in range(1, 7)] 
_CANONICAL_CALENDAR = _DOW_COLS

def precompute_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    dows = pd.Series(index.dayofweek, index=index)
    dow_df = pd.get_dummies(dows, prefix='dow')
    
    for col in _CANONICAL_CALENDAR:
        if col not in dow_df.columns:
            dow_df[col] = 0
            
    return dow_df[_CANONICAL_CALENDAR].astype(float)

def build_exogenous(
    df: pd.DataFrame,
    precomputed_calendar: Optional[pd.DataFrame] = None,
    include_calendar: bool = True,
    include_wind: bool = False,
    include_solar: bool = False,
    expected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    parts = []

    # 1. Calendar (DOW only)
    if include_calendar:
        if precomputed_calendar is None:
            cal = precompute_calendar_features(df.index)
        else:
            cal = precomputed_calendar.loc[df.index.intersection(precomputed_calendar.index)]
        parts.append(cal)

    # 2. Weather
    if include_wind and 'wind' in df.columns:
        parts.append(df[['wind']].copy())
    if include_solar and 'solar' in df.columns:
        parts.append(df[['solar']].copy())

    # 3. Yearly Lag (52 Weeks) - The "Smart" Lag
    # 52 weeks * 7 days * 24 hours = 8736
    # This aligns MONDAY with MONDAY, which is crucial.
    if 'load' in df.columns:
        yearly_lag = df[['load']].shift(8736).rename(columns={'load': 'load_lag_year'})
        # Fill initial NaN with 0 or backfill if needed, usually 0 for SARIMA exog is safe-ish 
        # if the model has intercept, but better to dropna in training if possible.
        parts.append(yearly_lag.fillna(0))

    if not parts:
        X = pd.DataFrame(index=df.index)
    else:
        X = pd.concat(parts, axis=1)

    # Ensure columns exist
    for col in ['wind', 'solar', 'load_lag_year']:
        if col not in X.columns and ((col=='wind' and include_wind) or (col=='solar' and include_solar)):
            X[col] = 0.0

    if expected_columns is not None:
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected_columns]

    return X.astype(float).fillna(0.0)

def build_future_exogenous(
    last_ts: pd.Timestamp,
    periods: int,
    precomputed_calendar: Optional[pd.DataFrame] = None,
    include_calendar: bool = True,
    include_wind: bool = False,
    include_solar: bool = False,
    reference_df: Optional[pd.DataFrame] = None,
    expected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    
    future_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=periods, freq='H')
    parts = []

    if include_calendar:
        cal_future = precompute_calendar_features(future_index)
        parts.append(cal_future)

    # --- IMPROVED WEATHER LOOKUP ---
    # We verify if we have data for these timestamps in the full df (Backtest Mode)
    # In a real live production scenario, you would plug in an API weather forecast here.
    if reference_df is not None:
        available_idx = future_index.intersection(reference_df.index)
        
        if include_wind:
            w = pd.DataFrame({'wind': 0.0}, index=future_index)
            if not available_idx.empty and 'wind' in reference_df.columns:
                w.loc[available_idx, 'wind'] = reference_df.loc[available_idx, 'wind']
            parts.append(w)

        if include_solar:
            s = pd.DataFrame({'solar': 0.0}, index=future_index)
            if not available_idx.empty and 'solar' in reference_df.columns:
                s.loc[available_idx, 'solar'] = reference_df.loc[available_idx, 'solar']
            parts.append(s)
    else:
        # Fallback to 0 if no reference provided
        if include_wind: parts.append(pd.DataFrame({'wind': 0.0}, index=future_index))
        if include_solar: parts.append(pd.DataFrame({'solar': 0.0}, index=future_index))

    # --- Year Lag Construction for Future ---
    # We look back 52 weeks (8736 hours) into the reference_df
    if reference_df is not None and 'load' in reference_df.columns:
        lookback_idx = future_index - pd.Timedelta(hours=8736)
        # Reindex allows us to pull values; if timestamp doesn't exist (data too short), we get NaN
        hist_vals = reference_df['load'].reindex(lookback_idx).values
        parts.append(pd.DataFrame({'load_lag_year': hist_vals}, index=future_index).fillna(0))
    else:
        parts.append(pd.DataFrame({'load_lag_year': 0}, index=future_index))

    Xf = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=future_index)
    
    if expected_columns:
        for col in expected_columns:
            if col not in Xf.columns: Xf[col] = 0.0
        Xf = Xf[expected_columns]

    return Xf.astype(float)

def prepare_endog_exog(df, **kwargs):
    if 'timestamp' in df.columns: df = df.set_index('timestamp')
    y = df['load'].astype(float)
    X = build_exogenous(df, **kwargs)
    X = X.reindex(y.index)
    return y, X