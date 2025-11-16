"""
trend_analysis.py
-----------------
Trend Analysis: Plot load data for each country at monthly intervals (5 years)
Generates 3 graphs per country showing trends over time

Output: 3 graphs per country (BE, DK, NL) = 9 total graphs
- Graph 1: Full 5-year trend with monthly sampling
- Graph 2: Monthly average load over 5 years
- Graph 3: Year-over-year monthly comparison
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_opsd import load_tidy_country_csvs_from_config

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def generate_trend_analysis(config_path='src/config.yaml'):
    """Generate 3 trend analysis graphs per country"""
    
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
    print("TREND ANALYSIS: 5-Year Monthly Trends for Each Country")
    print("="*80)
    
    for cc, df in dfs.items():
        print(f"\n📊 Generating trend analysis for {cc}...")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
        print(f"   Date range: {date_range}")
        
        # ====================================================================
        # GRAPH 1: Full 5-Year Trend with Monthly Sampling
        # ====================================================================
        print(f"\n   📈 Graph 1: Full 5-Year Trend (Monthly Sampling)")
        
        # Sample data: every month (first day of each month at noon)
        monthly_dates = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq='MS')
        monthly_data = []
        
        for date in monthly_dates:
            # Get data for that month
            month_data = df[(df['timestamp'] >= date) & (df['timestamp'] < date + pd.DateOffset(months=1))]
            if len(month_data) > 0:
                monthly_data.append({
                    'date': date,
                    'load': month_data['load'].mean(),  # Monthly average
                    'load_min': month_data['load'].min(),
                    'load_max': month_data['load'].max(),
                    'load_std': month_data['load'].std()
                })
        
        df_monthly = pd.DataFrame(monthly_data)
        
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(df_monthly['date'], df_monthly['load'], linewidth=2.5, color='steelblue', marker='o', markersize=4, label='Monthly Average')
        ax.fill_between(df_monthly['date'], df_monthly['load_min'], df_monthly['load_max'], alpha=0.2, color='steelblue', label='Min-Max Range')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (MW)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Full 5-Year Load Trend (Monthly Average)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        graph1_path = os.path.join(figures_folder, f'{cc}_trend_graph1_5year.png')
        plt.savefig(graph1_path, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {graph1_path}")
        plt.close()
        
        # ====================================================================
        # GRAPH 2: Monthly Average Load Over 5 Years (Bar Chart)
        # ====================================================================
        print(f"   📊 Graph 2: Monthly Average Load (Box Plot by Month)")
        
        # Extract month from timestamp
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['month_name'] = df['timestamp'].dt.strftime('%B')
        
        # Calculate average load for each month (across all years)
        monthly_avg = df.groupby(['month', 'month_name'])['load'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(range(12), monthly_avg['mean'], color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.errorbar(range(12), monthly_avg['mean'], yerr=monthly_avg['std'], fmt='none', ecolor='red', linewidth=2, capsize=5, label='±1 Std Dev')
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Load (MW)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Average Load by Month (5-Year Average)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(12))
        ax.set_xticklabels(monthly_avg['month_name'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        plt.tight_layout()
        
        graph2_path = os.path.join(figures_folder, f'{cc}_trend_graph2_monthly_avg.png')
        plt.savefig(graph2_path, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {graph2_path}")
        plt.close()
        
        # ====================================================================
        # GRAPH 3: Year-over-Year Monthly Comparison
        # ====================================================================
        print(f"   📋 Graph 3: Year-over-Year Monthly Comparison")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Get years
        years = sorted(df['year'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(years)))
        
        for year_idx, year in enumerate(years):
            year_data = df[df['year'] == year]
            monthly_load = year_data.groupby('month')['load'].mean()
            ax.plot(monthly_load.index, monthly_load.values, marker='o', linewidth=2.5, 
                   label=f'{year}', color=colors[year_idx], markersize=6)
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Load (MW)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Year-over-Year Monthly Load Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, ncol=2)
        plt.tight_layout()
        
        graph3_path = os.path.join(figures_folder, f'{cc}_trend_graph3_yoy.png')
        plt.savefig(graph3_path, dpi=150, bbox_inches='tight')
        print(f"      ✅ Saved: {graph3_path}")
        plt.close()
        
        # ====================================================================
        # SUMMARY STATISTICS
        # ====================================================================
        print(f"\n   📊 Summary Statistics for {cc}:")
        print(f"      Total records: {len(df):,}")
        print(f"      Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"      Average load: {df['load'].mean():.1f} MW")
        print(f"      Min load: {df['load'].min():.1f} MW")
        print(f"      Max load: {df['load'].max():.1f} MW")
        print(f"      Std deviation: {df['load'].std():.1f} MW")
        
        print(f"\n   Seasonal patterns:")
        peak_month = monthly_avg.loc[monthly_avg['mean'].idxmax()]
        low_month = monthly_avg.loc[monthly_avg['mean'].idxmin()]
        print(f"      Peak month: {peak_month['month_name']} ({peak_month['mean']:.1f} MW)")
        print(f"      Low month: {low_month['month_name']} ({low_month['mean']:.1f} MW)")
        print(f"      Seasonal difference: {peak_month['mean'] - low_month['mean']:.1f} MW ({((peak_month['mean']/low_month['mean']-1)*100):.1f}%)")
    
    print("\n" + "="*80)
    print("✅ Trend Analysis Complete!")
    print(f"📁 Graphs saved to: {figures_folder}")
    print("   Generated 3 graphs per country (9 total)")
    print("="*80)


if __name__ == '__main__':
    generate_trend_analysis()
