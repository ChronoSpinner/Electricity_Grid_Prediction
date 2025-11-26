"""
METRICS_BOTH_MODELS.py - Compare SARIMA and GRU for all countries
==================================================================
Calculates metrics for both models (dev and test sets) from CSV files
Output format: timestamp,yhat,lo,hi,horizon,trainend,y_true
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict

DEFAULT_CONFIG_PATH = os.path.join('src', 'config.yaml')


def _log(msg: str):
    """Log with timestamp"""
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")


def _read_config(path: str) -> dict:
    """Read config.yaml"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def calculate_metrics(df: pd.DataFrame, seasonality: int = 24) -> Dict[str, float]:
    """
    Calculate forecasting metrics
    Expects columns: y_true, yhat, lo, hi
    """
    y_true = df.get('y_true')
    yhat = df.get('yhat')
    lo = df.get('lo')
    hi = df.get('hi')
    
    if y_true is None or yhat is None:
        return {m: np.nan for m in ['MASE', 'sMAPE', 'MSE', 'RMSE', 'MAPE', 'PI_80_Coverage']}

    y_true = y_true.values
    yhat = yhat.values
    lo = lo.values if lo is not None else np.full_like(y_true, np.nan, dtype=float)
    hi = hi.values if hi is not None else np.full_like(y_true, np.nan, dtype=float)

    # Remove NaN rows
    mask = ~np.isnan(y_true) & ~np.isnan(yhat)
    y_true = y_true[mask]
    yhat = yhat[mask]
    lo = lo[mask]
    hi = hi[mask]

    n = len(y_true)
    if n == 0:
        return {m: np.nan for m in ['MASE', 'sMAPE', 'MSE', 'RMSE', 'MAPE', 'PI_80_Coverage']}

    # MASE - Mean Absolute Scaled Error
    if n > seasonality:
        scale = np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality]))
        if scale == 0:
            scale = 1e-10
    else:
        scale = 1e-10
    mase = np.mean(np.abs(y_true - yhat)) / scale

    # sMAPE - Symmetric Mean Absolute Percentage Error
    denom = (np.abs(y_true) + np.abs(yhat)) / 2
    smape = np.mean(np.abs(y_true - yhat) / (denom + 1e-10)) * 100

    # MSE/RMSE
    mse = np.mean((y_true - yhat) ** 2)
    rmse = float(np.sqrt(mse))

    # MAPE - Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - yhat) / (y_true + 1e-10))) * 100

    # PI Coverage (80% prediction interval)
    pi_covered = np.mean((y_true >= lo) & (y_true <= hi)) if (not np.all(np.isnan(lo)) and not np.all(np.isnan(hi))) else np.nan

    return {
        'MASE': float(mase),
        'sMAPE': float(smape),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'PI_80_Coverage': float(pi_covered) if not np.isnan(pi_covered) else np.nan,
    }


def run(config_path: str = DEFAULT_CONFIG_PATH):
    """Main function: compare SARIMA and GRU metrics"""
    cfg = _read_config(config_path)
    countries = cfg.get('countries', ['BE', 'DK', 'NL'])
    outputs = cfg.get('outputs', {})
    forecasts_folder = outputs.get('forecasts_folder') or outputs.get('folder', 'outputs')
    metrics_folder = outputs.get('metrics_folder', os.path.join('outputs', 'metrics'))
    os.makedirs(metrics_folder, exist_ok=True)

    seasonality = int(cfg.get('metrics', {}).get('seasonality', 24))

    _log("="*80)
    _log("METRICS COMPARISON - SARIMA vs GRU")
    _log("="*80)

    rows = []
    
    # For each country and split (dev, test)
    for cc in countries:
        for split in ('dev', 'test'):
            _log(f"\n{cc} - {split.upper()} SET:")
            _log("-" * 80)
            
            split_results = {}
            
            # Process SARIMA
            sarima_fname = f"{cc}_forecasts_{split}.csv"
            sarima_fpath = os.path.join(forecasts_folder, sarima_fname)
            
            if os.path.exists(sarima_fpath):
                try:
                    df_sarima = pd.read_csv(sarima_fpath)
                    metrics_sarima = calculate_metrics(df_sarima, seasonality=seasonality)
                    split_results['sarima'] = metrics_sarima
                    
                    pi_cov = metrics_sarima.get('PI_80_Coverage')
                    pi_cov_str = f"{pi_cov:.1f}%" if (pi_cov is not None and not pd.isna(pi_cov)) else "N/A"
                    
                    _log(
                        f"  SARIMA: rows={len(df_sarima):6d} | "
                        f"MASE={metrics_sarima['MASE']:.4f} | "
                        f"sMAPE={metrics_sarima['sMAPE']:6.2f}% | "
                        f"RMSE={metrics_sarima['RMSE']:8.2f} | "
                        f"PI80={pi_cov_str}"
                    )
                    
                    rows.append({
                        'country': cc,
                        'split': split,
                        'model': 'SARIMA',
                        'rows': len(df_sarima),
                        **metrics_sarima
                    })
                except Exception as e:
                    _log(f"  ❌ SARIMA: Failed to compute metrics: {e}")
            else:
                _log(f"  ⚠️  SARIMA: Missing file {sarima_fpath}")
            
            # Process GRU
            gru_fname = f"{cc}_gru_forecasts_{split}.csv"
            gru_fpath = os.path.join(forecasts_folder, gru_fname)
            
            if os.path.exists(gru_fpath):
                try:
                    df_gru = pd.read_csv(gru_fpath)
                    metrics_gru = calculate_metrics(df_gru, seasonality=seasonality)
                    split_results['gru'] = metrics_gru
                    
                    pi_cov = metrics_gru.get('PI_80_Coverage')
                    pi_cov_str = f"{pi_cov:.1f}%" if (pi_cov is not None and not pd.isna(pi_cov)) else "N/A"
                    
                    _log(
                        f"  GRU:    rows={len(df_gru):6d} | "
                        f"MASE={metrics_gru['MASE']:.4f} | "
                        f"sMAPE={metrics_gru['sMAPE']:6.2f}% | "
                        f"RMSE={metrics_gru['RMSE']:8.2f} | "
                        f"PI80={pi_cov_str}"
                    )
                    
                    rows.append({
                        'country': cc,
                        'split': split,
                        'model': 'GRU',
                        'rows': len(df_gru),
                        **metrics_gru
                    })
                except Exception as e:
                    _log(f"  ❌ GRU: Failed to compute metrics: {e}")
            else:
                _log(f"  ⚠️  GRU: Missing file {gru_fpath}")
            
            # Compare and show winner
            if 'sarima' in split_results and 'gru' in split_results:
                mase_sarima = split_results['sarima']['MASE']
                mase_gru = split_results['gru']['MASE']
                
                if mase_sarima < mase_gru:
                    improvement = (mase_gru - mase_sarima) / mase_gru * 100
                    _log(f"  🏆 Best: SARIMA (improvement: {improvement:.1f}%)")
                else:
                    improvement = (mase_sarima - mase_gru) / mase_sarima * 100
                    _log(f"  🏆 Best: GRU (improvement: {improvement:.1f}%)")

    if not rows:
        _log("❌ No metrics computed. Ensure forecast CSVs exist in the configured folder.")
        return

    # Create summary dataframe
    summary = pd.DataFrame(rows)
    
    # Save full summary
    out_csv = os.path.join(metrics_folder, 'forecast_metrics_summary.csv')
    summary.to_csv(out_csv, index=False)
    _log(f"\n✅ Saved metrics summary: {out_csv}")
    
    # Create pivot tables by metric
    _log("\n" + "="*80)
    _log("PIVOT TABLES - TEST SET ONLY")
    _log("="*80)
    
    test_summary = summary[summary['split'] == 'test']
    
    if len(test_summary) > 0:
        # MASE pivot
        mase_pivot = test_summary.pivot_table(values='MASE', index='country', columns='model', aggfunc='first')
        mase_csv = os.path.join(metrics_folder, 'test_mase_pivot.csv')
        mase_pivot.to_csv(mase_csv)
        _log(f"\nMASE (Test Set):")
        _log(mase_pivot.to_string())
        _log(f"✅ Saved: {mase_csv}")
        
        # sMAPE pivot
        smape_pivot = test_summary.pivot_table(values='sMAPE', index='country', columns='model', aggfunc='first')
        smape_csv = os.path.join(metrics_folder, 'test_smape_pivot.csv')
        smape_pivot.to_csv(smape_csv)
        _log(f"\nsMAPE (Test Set):")
        _log(smape_pivot.to_string())
        _log(f"✅ Saved: {smape_csv}")
        
        # RMSE pivot
        rmse_pivot = test_summary.pivot_table(values='RMSE', index='country', columns='model', aggfunc='first')
        rmse_csv = os.path.join(metrics_folder, 'test_rmse_pivot.csv')
        rmse_pivot.to_csv(rmse_csv)
        _log(f"\nRMSE (Test Set):")
        _log(rmse_pivot.to_string())
        _log(f"✅ Saved: {rmse_csv}")
        
        # Ranking table
        test_summary_sorted = test_summary.sort_values(['country', 'MASE'])
        ranking_csv = os.path.join(metrics_folder, 'test_set_ranking.csv')
        test_summary_sorted[['country', 'model', 'MASE', 'sMAPE', 'RMSE', 'MAPE', 'PI_80_Coverage', 'rows']].to_csv(ranking_csv, index=False)
        _log(f"\n" + "="*80)
        _log("TEST SET RANKING (by MASE)")
        _log("="*80)
        _log(test_summary_sorted[['country', 'model', 'MASE', 'sMAPE', 'RMSE', 'MAPE', 'PI_80_Coverage']].to_string(index=False))
        _log(f"\n✅ Saved ranking: {ranking_csv}")
    
    # Generate summary report
    report_path = os.path.join(metrics_folder, 'metrics_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("METRICS COMPARISON REPORT - SARIMA vs GRU\n")
        f.write("="*80 + "\n\n")
        
        f.write("TEST SET PERFORMANCE\n")
        f.write("-"*80 + "\n")
        if len(test_summary) > 0:
            for cc in countries:
                df_cc = test_summary[test_summary['country'] == cc]
                if len(df_cc) > 0:
                    f.write(f"\n{cc}:\n")
                    for _, row in df_cc.iterrows():
                        f.write(f"  {row['model']:8s}: MASE={row['MASE']:.4f} | sMAPE={row['sMAPE']:6.2f}% | RMSE={row['RMSE']:8.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(summary.to_string(index=False))
    
    _log(f"\n✅ Saved report: {report_path}")
    _log("\n" + "="*80)
    _log("✅ METRICS COMPARISON COMPLETE!")
    _log("="*80)


if __name__ == '__main__':
    run()
