"""
metrics.py
----------
Helper functions for metric calculations.
- MASE (Mean Absolute Scaled Error) - PRIMARY
- sMAPE (Symmetric Mean Absolute Percentage Error)
- MSE, RMSE, MAPE
- 80% PI coverage check
"""

import numpy as np
import pandas as pd


def calculate_mase(y_true, y_pred, seasonal=24):
    """
    Mean Absolute Scaled Error.
    Uses seasonal naive forecast (shift by 'seasonal' periods) as baseline.
    MASE = MAE / (sum of seasonal naive errors)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Seasonal naive baseline: y_{t-seasonal}
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Seasonal naive error (if enough data)
    if len(y_true) > seasonal:
        seasonal_errors = np.abs(y_true[seasonal:] - y_true[:-seasonal])
        seasonal_mae = np.mean(seasonal_errors)
        mase = mae / (seasonal_mae + 1e-10)
    else:
        mase = mae  # Fallback
    
    return mase


def calculate_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    sMAPE = 2 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + eps))
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    denominator = np.abs(y_true) + np.abs(y_pred)
    numerator = np.abs(y_true - y_pred)
    
    # Avoid division by zero
    ratio = np.zeros_like(numerator)
    mask = denominator != 0
    ratio[mask] = numerator[mask] / denominator[mask]
    
    smape = 2.0 * np.mean(ratio)
    return smape


def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    MAPE = mean(|y_true - y_pred| / |y_true|)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Avoid division by zero
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    else:
        mape = np.inf
    
    return mape


def calculate_mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)


def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_metrics(y_true, y_pred):
    """
    Calculate all required metrics at once.
    Returns: dict with keys: mse, rmse, mape
    """
    return {
        'mse': calculate_mse(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred)
    }


def calculate_pi_coverage(y_true, lo, hi):
    """
    Calculate Prediction Interval (PI) coverage.
    Coverage = proportion of y_true within [lo, hi]
    """
    y_true = np.array(y_true)
    lo = np.array(lo)
    hi = np.array(hi)
    
    coverage = np.mean((y_true >= lo) & (y_true <= hi))
    return coverage
