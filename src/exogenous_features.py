import pandas as pd
import numpy as np
from typing import Optional, List

_HOUR_COLS = [f"hr_{h}" for h in range(1, 24)]  # base hour 0 omitted
_DOW_COLS = [f"dow_{d}" for d in range(1, 7)]   # base dow 0 (Monday) omitted
_CANONICAL_CALENDAR = _HOUR_COLS + _DOW_COLS


def precompute_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hours = pd.Series(index.hour, index=index)
    dows = pd.Series(index.dayofweek, index=index)

    hour_df = pd.get_dummies(hours, prefix='hr')
    dow_df = pd.get_dummies(dows, prefix='dow')

    cal = pd.concat([hour_df, dow_df], axis=1)

    for col in _CANONICAL_CALENDAR:
        if col not in cal.columns:
            cal[col] = 0

    cal = cal[_CANONICAL_CALENDAR]
    cal.index = index
    return cal.astype(float)


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

    if include_calendar:
        if precomputed_calendar is None:
            cal = precompute_calendar_features(df.index)
        else:
            cal = precomputed_calendar.loc[df.index.intersection(precomputed_calendar.index)]
        parts.append(cal)

    if include_wind and 'wind' in df.columns:
        parts.append(df[['wind']].copy())

    if include_solar and 'solar' in df.columns:
        parts.append(df[['solar']].copy())

    if not parts:
        X = pd.DataFrame(index=df.index)
    else:
        X = pd.concat(parts, axis=1)

    if include_wind and 'wind' not in X.columns:
        X['wind'] = 0.0
    if include_solar and 'solar' not in X.columns:
        X['solar'] = 0.0

    if expected_columns is not None:
        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected_columns]

    return X.astype(float).fillna(method='ffill').fillna(0.0)


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
        if precomputed_calendar is None:
            cal_future = precompute_calendar_features(future_index)
        else:
            cal_future = precompute_calendar_features(future_index)
        parts.append(cal_future)

    if include_wind:
        if reference_df is not None and 'wind' in reference_df.columns and not reference_df['wind'].dropna().empty:
            last_wind = reference_df['wind'].dropna().iloc[-1]
        else:
            last_wind = 0.0
        parts.append(pd.DataFrame({'wind': np.repeat(last_wind, periods)}, index=future_index))

    if include_solar:
        if reference_df is not None and 'solar' in reference_df.columns and not reference_df['solar'].dropna().empty:
            last_solar = reference_df['solar'].dropna().iloc[-1]
        else:
            last_solar = 0.0
        parts.append(pd.DataFrame({'solar': np.repeat(last_solar, periods)}, index=future_index))

    if not parts:
        Xf = pd.DataFrame(index=future_index)
    else:
        Xf = pd.concat(parts, axis=1)

    if include_wind and 'wind' not in Xf.columns:
        Xf['wind'] = 0.0
    if include_solar and 'solar' not in Xf.columns:
        Xf['solar'] = 0.0

    if expected_columns is not None:
        for col in expected_columns:
            if col not in Xf.columns:
                Xf[col] = 0.0
        Xf = Xf[expected_columns]

    return Xf.astype(float)


def prepare_endog_exog(
    df: pd.DataFrame,
    precomputed_calendar: Optional[pd.DataFrame] = None,
    include_calendar: bool = True,
    include_wind: bool = False,
    include_solar: bool = False,
    expected_columns: Optional[List[str]] = None,
):
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    y = df['load'].astype(float)
    X = build_exogenous(df, precomputed_calendar, include_calendar, include_wind, include_solar, expected_columns=expected_columns)
    X = X.reindex(y.index)
    return y, X


def derive_expected_columns(include_calendar: bool, include_wind: bool, include_solar: bool) -> List[str]:
    cols = []
    if include_calendar:
        cols.extend(_CANONICAL_CALENDAR)
    if include_wind:
        cols.append('wind')
    if include_solar:
        cols.append('solar')
    return cols
