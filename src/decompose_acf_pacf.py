"""
decompose_acfpacf.py
-------------------
Part 1.3-1.4: Data Exploration and SARIMA Model Order Selection (BE, DK, NL)
- Basic sanity plots (last 14 days)
- STL decomposition
- Stationarity and differencing analysis (ADF tests)
- ACF/PACF plots for each differencing option
- SARIMA grid search with AIC/BIC criteria
- Save best SARIMA orders per country

FIXED FOR STATSMODELS 0.14.0+
"""

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from load_opsd import load_tidy_country_csvs_from_config

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 4)


def plot_sanity_last_14_days(dfs, config_path='src/config.yaml'):
    """
    Part 1.3: Basic sanity plots - Last 14 days
    Confirm hourly cadence and realistic magnitudes.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    figures_folder = config['outputs']['figures_folder']
    os.makedirs(figures_folder, exist_ok=True)
    
    print("\n" + "="*80)
    print("PART 1.3: BASIC SANITY CHECK - Last 14 Days Plot")
    print("="*80)
    
    for cc, df in dfs.items():
        print(f"\n📊 Sanity plot for {cc}...")
        
        # Last 14 days = 14\ * 24 = 336 hours
        last_14d = df.tail(336).copy()
        
        if len(last_14d) < 336:
            print(f"   ⚠️  Only {len(last_14d)} hours available (less than 14 days)")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(last_14d['timestamp'], last_14d['load'], linewidth=1, color='steelblue')
        ax.set_title(f'{cc} - Load (Last 14 Days)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestamp (UTC)', fontsize=12)
        ax.set_ylabel('Load (MW)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(figures_folder, f'{cc}_sanity_last14days.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {fig_path}")
        plt.close()
        
        # Print statistics
        print(f"   📈 Statistics:")
        print(f"      Date range: {last_14d['timestamp'].min()} to {last_14d['timestamp'].max()}")
        print(f"      Load range: {last_14d['load'].min():.1f} - {last_14d['load'].max():.1f} MW")
        print(f"      Mean: {last_14d['load'].mean():.1f} MW, Std: {last_14d['load'].std():.1f} MW")
    
    print("\n✅ Sanity check complete!")


def plot_stl_decomposition(dfs, config_path='src/config.yaml'):
    """
    Part 1.4.i: STL decomposition - daily seasonality period = 24
    Save figure with Trend, Seasonal, and Remainder.
    
    FIXED FOR STATSMODELS 0.14.0+
    In 0.14.0+, use figsize parameter directly in plot() method
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    figures_folder = config['outputs']['figures_folder']
    seasonality = config['decomposition']['seasonality']
    os.makedirs(figures_folder, exist_ok=True)
    
    print("\n" + "="*80)
    print("PART 1.4.i: STL DECOMPOSITION")
    print("="*80)
    
    for cc, df in dfs.items():
        print(f"\n🔍 STL decomposition for {cc} (period={seasonality})...")
        
        stl = STL(df['load'], period=seasonality, robust=True, seasonal=25)
        result = stl.fit()
        
        # ✅ STATSMODELS 0.14.0+: Pass figsize directly to plot()
        #fig = result.plot(figsize=(14, 8))
        fig = result.plot()
        fig.set_size_inches(14, 8)
        #
        fig.suptitle(f"{cc} - STL Decomposition (Seasonal Period={seasonality})", fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        fig_path = os.path.join(figures_folder, f'{cc}_stl.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {fig_path}")
        plt.close()


def adf_stationarity_check(series, title):
    """
    Perform ADF test and print results.
    """
    result = adfuller(series.dropna())
    print(f"\n   ADF Test: {title}")
    print(f"      Statistic: {result[0]:.6f}, p-value: {result[1]:.6g}")
    if result[1] < 0.05:
        print(f"      ✅ Stationary (reject H0)")
    else:
        print(f"      ❌ Non-stationary (fail to reject H0)")
    return result[1] < 0.05


def plot_acf_pacf_pair(series, cc, d_str, lags, figures_folder):
    """
    Plot ACF and PACF side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    plot_acf(series.dropna(), ax=axes[0], lags=lags, title=f'{cc} - ACF ({d_str})')
    plot_pacf(series.dropna(), ax=axes[1], lags=lags, method='ywm', title=f'{cc} - PACF ({d_str})')
    
    fig.tight_layout()
    fig_path = os.path.join(figures_folder, f'{cc}_acf_pacf_{d_str}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"      ✅ ACF/PACF: {fig_path}")
    plt.close()


def analyze_differencing_and_acf(dfs, config_path='src/config.yaml'):
    """
    Part 1.4.ii-iii: Stationarity, differencing, ACF/PACF
    Test: raw, d=1, D=1 (s=24), d=1+D=1
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    figures_folder = config['outputs']['figures_folder']
    seasonality = config['decomposition']['seasonality']
    lags = config['decomposition']['acf_pacf_lags']
    
    print("\n" + "="*80)
    print("PART 1.4.ii-iii: STATIONARITY, DIFFERENCING & ACF/PACF")
    print("="*80)
    
    for cc, df in dfs.items():
        print(f"\n🔬 Stationarity analysis for {cc}:")
        
        # Raw series
        print(f"\n   1. Raw series:")
        adf_stationarity_check(df['load'], 'Raw')
        plot_acf_pacf_pair(df['load'], cc, 'raw', lags, figures_folder)
        
        # First differencing (d=1)
        df_d1 = df['load'].diff(1)
        print(f"\n   2. First differencing (d=1):")
        adf_stationarity_check(df_d1, 'd=1')
        plot_acf_pacf_pair(df_d1, cc, 'd1', lags, figures_folder)
        
        # Seasonal differencing (D=1, s=24)
        df_D1 = df['load'].diff(seasonality)
        print(f"\n   3. Seasonal differencing (D=1, s={seasonality}):")
        adf_stationarity_check(df_D1, f'D=1 (s={seasonality})')
        plot_acf_pacf_pair(df_D1, cc, 'D1', lags, figures_folder)
        
        # Both d=1 and D=1
        df_d1D1 = df_d1.diff(seasonality)
        print(f"\n   4. Both d=1 & D=1:")
        adf_stationarity_check(df_d1D1, 'd=1 & D=1')
        plot_acf_pacf_pair(df_d1D1, cc, 'd1D1', lags, figures_folder)


def sarima_grid_search(dfs, config_path='src/config.yaml'):
    """
    Part 1.4.iv: SARIMA grid search with AIC/BIC
    Select lowest BIC (tie-break with AIC).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    figures_folder = config['outputs']['figures_folder']
    seasonality = config['decomposition']['seasonality']
    
    # Grid ranges
    p_range = config['sarima']['p_range']
    d_range = config['sarima']['d_range']
    q_range = config['sarima']['q_range']
    P_range = config['sarima']['P_range']
    D_range = config['sarima']['D_range']
    Q_range = config['sarima']['Q_range']
    
    print("\n" + "="*80)
    print("PART 1.4.iv: SARIMA GRID SEARCH (AIC/BIC)")
    print("="*80)
    
    results = {}
    
    for cc, df in dfs.items():
        print(f"\n🧮 SARIMA grid search for {cc}...")
        print(f"   Grid: p={p_range}, d={d_range}, q={q_range}")
        print(f"         P={P_range}, D={D_range}, Q={Q_range}, s={seasonality}")
        
        y = df['load'].dropna().tail(2160)  # Last 90 days for fitting for decreasing computation
        param_list = []
        bic_list = []
        aic_list = []
        
        # Exhaustive grid search
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                order = (p, d, q)
                                seasonal_order = (P, D, Q, seasonality)
                                try:
                                    model = SARIMAX(
                                        y, 
                                        order=order, 
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False, 
                                        enforce_invertibility=False
                                    ).fit(disp=False)
                                    
                                    param_list.append((order, seasonal_order))
                                    bic_list.append(model.bic)
                                    aic_list.append(model.aic)
                                except:
                                    continue
        
        # Build results dataframe
        df_grid = pd.DataFrame({
            'order': [x[0] for x in param_list],
            'seasonal_order': [x[1] for x in param_list],
            'bic': bic_list,
            'aic': aic_list
        })
        df_grid = df_grid.sort_values(by='bic').reset_index(drop=True)
        
        # Save full grid to CSV
        grid_path = os.path.join(figures_folder, f'{cc}_sarima_gridsearch.csv')
        df_grid.head(10).to_csv(grid_path, index=False)
        
        # Print top results
        print(f"\n   ✅ Top 5 models (by BIC):")
        for i in range(min(5, len(df_grid))):
            row = df_grid.iloc[i]
            print(f"      {i+1}. Order={row['order']}, Seasonal={row['seasonal_order']}")
            print(f"         BIC={row['bic']:.2f}, AIC={row['aic']:.2f}")
        
        print(f"\n   📊 Grid saved: {grid_path}")
        
        # Store best result
        best_row = df_grid.iloc[0]
        results[cc] = {
            'order': best_row['order'],
            'seasonal_order': best_row['seasonal_order']
        }
        
        print(f"\n   🎯 CHOSEN for {cc}:")
        print(f"      Order: {best_row['order']}, Seasonal: {best_row['seasonal_order']}")
        print(f"      (Lowest BIC = {best_row['bic']:.2f})")
    
    # Save summary
    summary_path = os.path.join(figures_folder, 'sarima_orders_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(results, f)
    
    print(f"\n✅ All countries: Best SARIMA orders saved to: {summary_path}")
    print("="*80)
    
    return results


def main(config_path='src/config.yaml'):
    """
    Run all decomposition and SARIMA order selection steps.
    """
    print("\n" + "="*80)
    print("PART 1: DATA ANALYSIS & DECOMPOSITION")
    print("="*80)
    
    # Load data
    dfs = load_tidy_country_csvs_from_config(config_path)
    
    if not dfs:
        print("❌ No countries loaded! Check config and data files.")
        return
    
    # 1.3: Sanity plots
    plot_sanity_last_14_days(dfs, config_path)
    
    # 1.4.i: STL decomposition
    plot_stl_decomposition(dfs, config_path)
    
    # 1.4.ii-iii: Stationarity and ACF/PACF
    analyze_differencing_and_acf(dfs, config_path)
    
    # 1.4.iv: SARIMA grid search
    sarima_results = sarima_grid_search(dfs, config_path)
    
    print("\n✅ Part 1 (Data Analysis) Complete!")
    print("   Next step: python src/forecast.py")


if __name__ == '__main__':
    main()
