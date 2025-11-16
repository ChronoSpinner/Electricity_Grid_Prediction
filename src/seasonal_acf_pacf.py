"""
seasonal_acf_pacf.py
--------------------
Seasonal ACF/PACF Analysis: Find P and Q parameters
- Shows ACF/PACF for seasonal lags: 24, 48, 72, ..., 1152 (24*48)
- Tests BOTH d=1 and D=1 differencing
- Helps determine P and Q for SARIMA(p,d,q)(P,D,Q,24)

Key:
- ACF at seasonal lags helps find Q (moving average order)
- PACF at seasonal lags helps find P (autoregressive order)
- Significant spikes = included in model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from load_opsd import load_tidy_country_csvs_from_config

sns.set_style("whitegrid")


def plot_seasonal_acf_pacf(config_path='src/config.yaml'):
    """Plot seasonal ACF/PACF (lags: 24, 48, 72, ..., 1152) for finding P,Q"""
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    figures_folder = config['outputs']['figures_folder']
    os.makedirs(figures_folder, exist_ok=True)
    
    # Load data
    dfs = load_tidy_country_csvs_from_config(config_path)
    
    if not dfs:
        print("❌ No countries loaded!")
        return
    
    print("\n" + "="*80)
    print("SEASONAL ACF/PACF ANALYSIS FOR P,Q SELECTION")
    print("Lags: 24, 48, 72, ..., 1152 (multiples of 24)")
    print("="*80)
    
    seasonality = 24  # 24 hours per day
    max_lag = 24 * 48  # 1152 lags = 48 days of seasonal lags
    seasonal_lags = list(range(24, max_lag + 1, 24))  # 24, 48, 72, ..., 1152
    
    print(f"\nSeasonal lags to analyze: {len(seasonal_lags)} lags")
    print(f"First 5 lags: {seasonal_lags[:5]}")
    print(f"Last 5 lags: {seasonal_lags[-5:]}")
    
    for cc, df in dfs.items():
        print(f"\n" + "="*80)
        print(f"Country: {cc}")
        print("="*80)
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        y = df_sorted['load'].values
        
        # ====================================================================
        # Apply differencing d=1, D=1
        # ====================================================================
        print(f"\nApplying differencing: d=1, D=1 (s=24)")
        
        # First differencing (d=1)
        y_d1 = np.diff(y, n=1)
        print(f"   After d=1: {len(y_d1):,} points")
        
        # Seasonal differencing (D=1, s=24)
        y_d1_D1 = np.diff(y_d1, n=seasonality)
        print(f"   After d=1 & D=1: {len(y_d1_D1):,} points")
        
        # ====================================================================
        # PLOT 1: ACF at Seasonal Lags (d=1, D=1)
        # ====================================================================
        print(f"\n   📊 Plot 1: ACF at Seasonal Lags (d=1, D=1)")
        
        fig, ax = plt.subplots(figsize=(16, 5))
        
        # Create ACF manually for seasonal lags only
        from statsmodels.tsa.stattools import acf as calc_acf
        
        # Calculate ACF for all lags up to max
        acf_values = calc_acf(y_d1_D1, nlags=max_lag, fft=False)
        
        # Extract only seasonal lags
        seasonal_acf = [acf_values[lag] for lag in seasonal_lags]
        
        # Plot bars for seasonal lags
        ax.bar(seasonal_lags, seasonal_acf, width=20, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add confidence interval line (95%)
        conf_interval = 1.96 / np.sqrt(len(y_d1_D1))
        ax.axhline(y=conf_interval, color='red', linestyle='--', linewidth=2, label=f'95% CI: ±{conf_interval:.3f}')
        ax.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=1)
        
        ax.set_xlabel('Seasonal Lag (multiples of 24 hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('ACF', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Seasonal ACF (d=1, D=1) - Find Q Parameter', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        acf_path = os.path.join(figures_folder, f'{cc}_seasonal_acf_d1D1.png')
        plt.savefig(acf_path, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {acf_path}")
        plt.close()
        
        # ====================================================================
        # PLOT 2: PACF at Seasonal Lags (d=1, D=1)
        # ====================================================================
        print(f"   📊 Plot 2: PACF at Seasonal Lags (d=1, D=1)")
        
        fig, ax = plt.subplots(figsize=(16, 5))
        
        # Calculate PACF
        from statsmodels.tsa.stattools import pacf as calc_pacf
        
        pacf_values = calc_pacf(y_d1_D1, nlags=max_lag, method='ywm')
        
        # Extract only seasonal lags
        seasonal_pacf = [pacf_values[lag] for lag in seasonal_lags]
        
        # Plot bars for seasonal lags
        ax.bar(seasonal_lags, seasonal_pacf, width=20, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add confidence interval line
        ax.axhline(y=conf_interval, color='red', linestyle='--', linewidth=2, label=f'95% CI: ±{conf_interval:.3f}')
        ax.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=1)
        
        ax.set_xlabel('Seasonal Lag (multiples of 24 hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('PACF', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Seasonal PACF (d=1, D=1) - Find P Parameter', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        pacf_path = os.path.join(figures_folder, f'{cc}_seasonal_pacf_d1D1.png')
        plt.savefig(pacf_path, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {pacf_path}")
        plt.close()
        
        # ====================================================================
        # PLOT 3: Combined ACF & PACF (side by side)
        # ====================================================================
        print(f"   📊 Plot 3: Combined ACF & PACF (d=1, D=1)")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # ACF
        ax1.bar(seasonal_lags, seasonal_acf, width=20, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=conf_interval, color='red', linestyle='--', linewidth=2, label='95% CI')
        ax1.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=2)
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.set_ylabel('ACF', fontsize=11, fontweight='bold')
        ax1.set_title(f'{cc}: Seasonal ACF (Find Q) - d=1, D=1', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(fontsize=9)
        
        # PACF
        ax2.bar(seasonal_lags, seasonal_pacf, width=20, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=conf_interval, color='red', linestyle='--', linewidth=2, label='95% CI')
        ax2.axhline(y=-conf_interval, color='red', linestyle='--', linewidth=2)
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.set_xlabel('Seasonal Lag (multiples of 24 hours)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('PACF', fontsize=11, fontweight='bold')
        ax2.set_title(f'{cc}: Seasonal PACF (Find P) - d=1, D=1', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=9)
        
        plt.tight_layout()
        combined_path = os.path.join(figures_folder, f'{cc}_seasonal_acf_pacf_combined.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {combined_path}")
        plt.close()
        
        # ====================================================================
        # ANALYSIS: Identify P and Q
        # ====================================================================
        print(f"\n   📋 Interpretation Guide for {cc}:")
        print(f"      (Find significant spikes outside red CI lines)\n")
        
        # Find significant lags
        acf_significant = [seasonal_lags[i] for i in range(len(seasonal_acf)) if abs(seasonal_acf[i]) > conf_interval]
        pacf_significant = [seasonal_lags[i] for i in range(len(seasonal_pacf)) if abs(seasonal_pacf[i]) > conf_interval]
        
        print(f"      ACF: Significant seasonal lags: {acf_significant[:10]}")
        print(f"           (These suggest MA order Q)")
        print(f"           → Try Q = 1 (first spike at lag {acf_significant[0] if acf_significant else 'none'}) or Q = 0")
        
        print(f"\n      PACF: Significant seasonal lags: {pacf_significant[:10]}")
        print(f"            (These suggest AR order P)")
        print(f"            → Try P = 1 (first spike at lag {pacf_significant[0] if pacf_significant else 'none'}) or P = 0")
        
        # Recommendations
        print(f"\n      📌 Recommended P,Q values:")
        if len(pacf_significant) == 0:
            print(f"         P = 0 (no significant PACF)")
        else:
            print(f"         P = 1 (first spike at lag {pacf_significant[0]})")
        
        if len(acf_significant) == 0:
            print(f"         Q = 0 (no significant ACF)")
        else:
            print(f"         Q = 1 (first spike at lag {acf_significant[0]})")
        
        print(f"\n      Suggested SARIMA orders to test:")
        P_opts = [0, 1] if len(pacf_significant) > 0 else [0]
        Q_opts = [0, 1] if len(acf_significant) > 0 else [0]
        
        for P in P_opts:
            for Q in Q_opts:
                print(f"         → SARIMA(p,1,q)({P},1,{Q},24)")
    
    print("\n" + "="*80)
    print("✅ Seasonal ACF/PACF Analysis Complete!")
    print(f"📁 Plots saved to: {figures_folder}")
    print("   Generated 3 plots per country (9 total)")
    print("="*80)


if __name__ == '__main__':
    plot_seasonal_acf_pacf()
