"""
plot_full_data.py
-----------------
Plot full 43,000+ hours of data for each country (5 years)
Generates high-resolution plots for complete time series visualization

Output: 1 plot per country = 3 total plots
- Plot shows all ~43,800 hourly data points
- Color-coded by year for easy visualization
- Includes seasonal patterns and trends
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_opsd import load_tidy_country_csvs_from_config

sns.set_style("whitegrid")


def plot_full_data(config_path='src/config.yaml'):
    """Plot full 43,000+ hours for each country"""
    
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
    print("PLOTTING FULL 43,000+ HOURS DATA FOR EACH COUNTRY")
    print("="*80)
    
    for cc, df in dfs.items():
        print(f"\n📊 Plotting full data for {cc}...")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"   Total records: {len(df):,}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # ====================================================================
        # OPTION 1: Simple Line Plot (All points)
        # ====================================================================
        print(f"\n   📈 Option 1: Simple Line Plot (All {len(df):,} points)")
        
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(df['timestamp'], df['load'], linewidth=0.5, color='steelblue', alpha=0.8)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (MW)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Full 5-Year Load Data ({len(df):,} hourly records)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot1_path = os.path.join(figures_folder, f'{cc}_full_data_simple.png')
        plt.savefig(plot1_path, dpi=100, bbox_inches='tight')  # Lower DPI for large plot
        print(f"      ✅ Saved: {plot1_path}")
        plt.close()
        
        # ====================================================================
        # OPTION 2: Color-coded by Year
        # ====================================================================
        print(f"   🎨 Option 2: Color-coded by Year")
        
        df['year'] = df['timestamp'].dt.year
        years = sorted(df['year'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(years)))
        
        fig, ax = plt.subplots(figsize=(20, 6))
        
        for year_idx, year in enumerate(years):
            year_data = df[df['year'] == year]
            ax.plot(year_data['timestamp'], year_data['load'], linewidth=0.7, 
                   label=f'{year}', color=colors[year_idx], alpha=0.8)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (MW)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Full 5-Year Load Data - Color-coded by Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, ncol=5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot2_path = os.path.join(figures_folder, f'{cc}_full_data_by_year.png')
        plt.savefig(plot2_path, dpi=100, bbox_inches='tight')
        print(f"      ✅ Saved: {plot2_path}")
        plt.close()
        
        # ====================================================================
        # OPTION 3: With Moving Average (Trend)
        # ====================================================================
        print(f"   📉 Option 3: With 168-Hour Moving Average (Trend)")
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        df_sorted['MA_168'] = df_sorted['load'].rolling(window=168, center=True).mean()  # 7-day MA
        
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(df_sorted['timestamp'], df_sorted['load'], linewidth=0.3, color='steelblue', 
               alpha=0.5, label='Hourly Load')
        ax.plot(df_sorted['timestamp'], df_sorted['MA_168'], linewidth=2.5, color='red', 
               label='168-Hour Moving Avg (Trend)', alpha=0.9)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (MW)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cc}: Full 5-Year Load Data with 168-Hour Moving Average', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot3_path = os.path.join(figures_folder, f'{cc}_full_data_with_ma.png')
        plt.savefig(plot3_path, dpi=100, bbox_inches='tight')
        print(f"      ✅ Saved: {plot3_path}")
        plt.close()
        
        # ====================================================================
        # OPTION 4: Density Plot (see concentration)
        # ====================================================================
        print(f"   🔥 Option 4: Density Plot (Load Concentration)")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        
        # Top: Time series
        ax1.plot(df['timestamp'], df['load'], linewidth=0.4, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Load (MW)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{cc}: Full 5-Year Load Data with Load Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Histogram
        ax2.hist(df['load'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Load (MW)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency (hours)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot4_path = os.path.join(figures_folder, f'{cc}_full_data_density.png')
        plt.savefig(plot4_path, dpi=100, bbox_inches='tight')
        print(f"      ✅ Saved: {plot4_path}")
        plt.close()
        
        # ====================================================================
        # STATISTICS
        # ====================================================================
        print(f"\n   📊 Statistics for {cc}:")
        print(f"      Total hours: {len(df):,}")
        print(f"      Years: {len(years)} ({years[0]}-{years[-1]})")
        print(f"      Average load: {df['load'].mean():.1f} MW")
        print(f"      Min load: {df['load'].min():.1f} MW ({df.loc[df['load'].idxmin(), 'timestamp']})")
        print(f"      Max load: {df['load'].max():.1f} MW ({df.loc[df['load'].idxmax(), 'timestamp']})")
        print(f"      Std deviation: {df['load'].std():.1f} MW")
        print(f"      Median: {df['load'].median():.1f} MW")
        print(f"      Q1 (25%): {df['load'].quantile(0.25):.1f} MW")
        print(f"      Q3 (75%): {df['load'].quantile(0.75):.1f} MW")
    
    print("\n" + "="*80)
    print("✅ Full Data Plotting Complete!")
    print(f"📁 Plots saved to: {figures_folder}")
    print("   Generated 4 plots per country (12 total)")
    print("="*80)


if __name__ == '__main__':
    plot_full_data()
