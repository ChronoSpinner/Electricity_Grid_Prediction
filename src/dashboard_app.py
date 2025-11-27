"""
dashboard_app.py - Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import yaml

st.set_page_config(page_title="OPSD PowerDesk Dashboard", page_icon="⚡", layout="wide")

# --- 1. ROBUST FILE LOADER ---
def find_file(filename, default_folder='outputs/forecasts'):
    """Search for a file in common locations"""
    candidates = [
        os.path.join(default_folder, filename),           # outputs/file.csv
        filename,                                         # ./file.csv (root)
        os.path.join('..', default_folder, filename),     # ../outputs/file.csv
        os.path.join('.', default_folder, filename)       # ./outputs/file.csv
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

@st.cache_data
def load_config(config_path='src/config.yaml'):
    """Load configuration"""
    # Try finding config
    if os.path.exists(config_path):
        path = config_path
    elif os.path.exists('config.yaml'):
        path = 'config.yaml'
    else:
        return {}
            
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_forecasts(cc, split='test'):
    """Load forecast data with smart search"""
    # Priority: SARIMA split -> Generic split -> Full file
    filenames = [
        f"{cc}_sarima_forecasts_{split}.csv",
        f"{cc}_forecasts_{split}.csv"
    ]
    
    for fname in filenames:
        path = find_file(fname)
        if path:
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            return df
    return None

@st.cache_data
def load_anomalies(cc):
    """Load anomaly data"""
    path = find_file(f"{cc}_anomalies.csv")
    if path:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        return df
    return None

@st.cache_data
def load_updates(cc):
    """Load online updates log"""
    path = find_file(f"{cc}_online_updates.csv")
    if path:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        return df
    return None

# --- 2. VISUALIZATION FUNCTIONS ---

def plot_live_series(df, current_ts, window_hours=168, future_hours=24):
    """Plot historical data + forecast cone"""
    start_plot = current_ts - timedelta(hours=window_hours)
    end_plot = current_ts + timedelta(hours=future_hours)
    
    mask = (df['timestamp'] >= start_plot) & (df['timestamp'] <= end_plot)
    plot_df = df.loc[mask].copy()
    
    history_df = plot_df[plot_df['timestamp'] <= current_ts]
    future_df = plot_df[plot_df['timestamp'] > current_ts]
    
    fig = go.Figure()
    
    # 1. Actual Load
    fig.add_trace(go.Scatter(
        x=history_df['timestamp'], y=history_df['y_true'],
        mode='lines', name='Actual Load',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # 2. Forecast Cone
    if not future_df.empty:
        future_df = future_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Upper Bound
        fig.add_trace(go.Scatter(
            x=future_df['timestamp'], y=future_df['hi'],
            mode='lines', line=dict(width=0), showlegend=False, name='Upper 80%'
        ))
        
        # Lower Bound
        fig.add_trace(go.Scatter(
            x=future_df['timestamp'], y=future_df['lo'],
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(255, 127, 14, 0.2)', showlegend=False, name='Lower 80%'
        ))
        
        # Mean Forecast
        fig.add_trace(go.Scatter(
            x=future_df['timestamp'], y=future_df['yhat'],
            mode='lines', name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))

    # 3. Current Time Marker (Numeric Timestamp to fix TypeError)
    line_pos = pd.to_datetime(current_ts).timestamp() * 1000
    
    fig.add_vline(
        x=line_pos, 
        line_width=2, line_dash="dot", line_color="gray",
        annotation_text="Now", annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Live Monitoring (Window: -{window_hours}h / +{future_hours}h)",
        xaxis_title="Time", yaxis_title="Load (MW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_anomaly_tape(df_anom, current_ts, window_hours=168):
    """Visual Tape of anomalies"""
    start_plot = current_ts - timedelta(hours=window_hours)
    mask = (df_anom['timestamp'] >= start_plot) & (df_anom['timestamp'] <= current_ts)
    window_anoms = df_anom.loc[mask].copy()
    
    if window_anoms.empty:
        return None
        
    window_anoms['Status'] = 'Normal'
    if 'flag_z' in window_anoms.columns:
        window_anoms.loc[window_anoms['flag_z'] == 1, 'Status'] = 'Anomaly'
    
    fig = px.scatter(
        window_anoms, x='timestamp', y=[1]*len(window_anoms),
        color='Status', color_discrete_map={'Normal': '#eee', 'Anomaly': 'red'},
        symbol='Status', symbol_map={'Normal': 'circle', 'Anomaly': 'x'},
        height=150, title="Anomaly Tape (Last 7 Days)"
    )
    
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(xaxis_title=None, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- 3. MAIN APP ---

def main():
    st.sidebar.title("⚡ OPSD PowerDesk")
    
    # Config & Data Loading
    config = load_config()
    countries = config.get('countries', ['BE', 'DK', 'NL'])
    selected_cc = st.sidebar.selectbox("Select Country", countries, index=0)
    
    df_forecasts = load_forecasts(selected_cc)
    df_anomalies = load_anomalies(selected_cc)
    df_updates = load_updates(selected_cc)
    
    if df_forecasts is None:
        st.error(f"❌ Could not find forecast CSVs for {selected_cc}. Check your 'outputs/' folder.")
        return

    # Simulation Controls
    st.sidebar.header("🕒 Simulation Control")
    min_date = df_forecasts['timestamp'].min()
    max_date = df_forecasts['timestamp'].max()
    
    # Default to 25% through the dataset
    default_date = min_date + (max_date - min_date) / 4
    
    selected_date = st.sidebar.date_input("Date", value=default_date, min_value=min_date.date(), max_value=max_date.date())
    selected_hour = st.sidebar.slider("Hour (UTC)", 0, 23, 12)
    
    try:
        current_ts = pd.Timestamp(datetime.combine(selected_date, datetime.min.time())) + timedelta(hours=selected_hour)
    except:
        st.error("Invalid Date/Time")
        return

    if current_ts < min_date or current_ts > max_date:
        st.warning(f"Time {current_ts} outside data range.")
        return
        
    # --- KPIs ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # 7-Day Window for KPIs
    window_mask = (df_forecasts['timestamp'] > current_ts - timedelta(hours=24*7)) & (df_forecasts['timestamp'] <= current_ts)
    recent_df = df_forecasts.loc[window_mask]
    
    # KPI 1 & 2: Forecast Accuracy
    if not recent_df.empty:
        mae = np.mean(np.abs(recent_df['y_true'] - recent_df['yhat']))
        scale = recent_df['y_true'].mean() * 0.1  # Approx scale
        mase_est = mae / scale if scale > 0 else 0
        kpi1.metric("7-Day MASE", f"{mase_est:.3f}")
        
        if 'lo' in recent_df.columns:
            covered = ((recent_df['y_true'] >= recent_df['lo']) & (recent_df['y_true'] <= recent_df['hi'])).mean()
            kpi2.metric("7-Day Coverage", f"{covered*100:.1f}%")
        else:
            kpi2.metric("7-Day Coverage", "N/A")
    else:
        kpi1.metric("7-Day MASE", "N/A")
        kpi2.metric("7-Day Coverage", "N/A")
        
    # KPI 3: Anomalies Today (Cumulative)
    if df_anomalies is not None:
        today_start = current_ts.replace(hour=0, minute=0, second=0)
        # Count anomalies from start of day UP TO current simulation hour
        today_mask = (df_anomalies['timestamp'] >= today_start) & (df_anomalies['timestamp'] <= current_ts)
        anoms_today = df_anomalies.loc[today_mask, 'flag_z'].sum()
        kpi3.metric("Anomalies (Today)", int(anoms_today), delta_color="inverse")
    else:
        kpi3.metric("Anomalies", "N/A (File Missing)")
        
    # KPI 4: Last Model Refit
    if df_updates is not None:
        # Find updates that happened BEFORE or AT current simulation time
        past_updates = df_updates[df_updates['timestamp'] <= current_ts]
        if not past_updates.empty:
            last_up = past_updates.iloc[-1]
            time_str = last_up['timestamp'].strftime('%Y-%m-%d %H:%M')
            kpi4.metric("Last Model Refit", last_up['reason'].title(), time_str)
        else:
            kpi4.metric("Last Model Refit", "None Yet")
    else:
        kpi4.metric("Last Model Refit", "N/A (File Missing)")
        
    st.markdown("---")
    
    # Charts
    st.subheader("📈 Live Load & Forecast Cone")
    fig_main = plot_live_series(df_forecasts, current_ts)
    st.plotly_chart(fig_main, use_container_width=True)
    
    if df_anomalies is not None:
        fig_tape = plot_anomaly_tape(df_anomalies, current_ts)
        if fig_tape:
            st.plotly_chart(fig_tape, use_container_width=True)
            
    with st.expander("🔎 View Underlying Data"):
        st.dataframe(recent_df.tail(24))

if __name__ == "__main__":
    main()