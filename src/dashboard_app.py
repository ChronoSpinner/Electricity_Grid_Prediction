"""
dashboard_app.py - Interactive Dashboard (Part 6) - SYNTAX FIXED
=================================================================

Streamlit dashboard for monitoring forecasts, anomalies, and online updates.

Run with: streamlit run src/dashboard_app.py

FIX: Line 245 syntax error (extra closing parenthesis) corrected
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import yaml
import json


@st.cache_data
def load_config(config_path='src/config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_data
def load_forecasts(cc, split='test', folder='outputs'):
    """Load forecast data"""
    path = os.path.join(folder, f'{cc}_sarima_forecasts_{split}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data
def load_anomalies(cc, folder='outputs'):
    """Load anomaly data"""
    path = os.path.join(folder, f'{cc}_anomalies.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data
def load_updates(cc, folder='outputs'):
    """Load online updates log"""
    path = os.path.join(folder, f'{cc}_online_updates.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


@st.cache_data
def load_metrics(cc, folder='outputs/metrics'):
    """Load metrics summary"""
    path = os.path.join(folder, f'{cc}_metrics_summary.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def calculate_rolling_mase(df, window=7*24, seasonality=24):
    """Calculate rolling MASE"""
    residuals = df['y_true'] - df['yhat']
    mae = residuals.abs().rolling(window=window, min_periods=24).mean()
    
    # Naive seasonal MAE
    naive_errors = np.abs(df['y_true'].diff(seasonality))
    mae_naive = naive_errors.rolling(window=window, min_periods=seasonality).mean()
    
    mase = mae / mae_naive
    return mase


def calculate_rolling_coverage(df, window=7*24):
    """Calculate rolling 80% PI coverage"""
    within = ((df['y_true'] >= df['lo']) & (df['y_true'] <= df['hi'])).astype(int)
    coverage = 100 * within.rolling(window=window, min_periods=24).mean()
    return coverage


def plot_live_series(df, days=14):
    """Plot live series with actual and forecast"""
    # Last N days
    cutoff = df['timestamp'].max() - timedelta(days=days)
    df_plot = df[df['timestamp'] >= cutoff].copy()
    
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'],
        y=df_plot['y_true'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'],
        y=df_plot['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title=f'Load: Last {days} Days (Actual vs Forecast)',
        xaxis_title='Timestamp',
        yaxis_title='Load (MW)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_forecast_cone(df, hours=24):
    """Plot next 24h forecast with prediction interval"""
    # Last 24 hours
    df_plot = df.tail(hours).copy()
    
    fig = go.Figure()
    
    # Prediction interval (shaded)
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'],
        y=df_plot['hi'],
        mode='lines',
        name='80% PI Upper',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'],
        y=df_plot['lo'],
        mode='lines',
        name='80% PI Lower',
        line=dict(width=0),
        fillcolor='rgba(255, 127, 14, 0.2)',
        fill='tonexty',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Mean forecast
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'],
        y=df_plot['yhat'],
        mode='lines+markers',
        name='Forecast Mean',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=6)
    ))
    
    # Actual (if available)
    if 'y_true' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['y_true'],
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2)
        ))
    
    fig.update_layout(
        title=f'Forecast Cone: Next {hours} Hours',
        xaxis_title='Timestamp',
        yaxis_title='Load (MW)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_anomaly_tape(df_forecasts, df_anomalies, days=14):
    """Plot anomaly highlights on forecast series"""
    # Last N days
    cutoff = df_forecasts['timestamp'].max() - timedelta(days=days)
    df_plot = df_forecasts[df_forecasts['timestamp'] >= cutoff].copy()
    
    # Merge anomalies
    df_plot = df_plot.merge(
        df_anomalies[['timestamp', 'flag_z', 'flag_cusum']],
        on='timestamp',
        how='left'
    )
    
    fig = go.Figure()
    
    # Actual line
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'],
        y=df_plot['y_true'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Anomaly markers (z-score)
    df_anom_z = df_plot[df_plot['flag_z'] == 1]
    if len(df_anom_z) > 0:
        fig.add_trace(go.Scatter(
            x=df_anom_z['timestamp'],
            y=df_anom_z['y_true'],
            mode='markers',
            name='Anomaly (Z-score)',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    # CUSUM markers
    if 'flag_cusum' in df_plot.columns:
        df_anom_cusum = df_plot[df_plot['flag_cusum'] == 1]
        if len(df_anom_cusum) > 0:
            fig.add_trace(go.Scatter(
                x=df_anom_cusum['timestamp'],
                y=df_anom_cusum['y_true'],
                mode='markers',
                name='Anomaly (CUSUM)',
                marker=dict(color='orange', size=8, symbol='diamond')
            ))
    
    fig.update_layout(
        title=f'Anomaly Tape: Last {days} Days',
        xaxis_title='Timestamp',
        yaxis_title='Load (MW)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )  # ✅ FIX: This was line 245 - removed extra )
    
    return fig


def main():
    """Main dashboard app"""
    st.set_page_config(
        page_title="OPSD PowerDesk Dashboard",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ OPSD PowerDesk: Live Monitoring Dashboard")
    
    # Load config
    config = load_config()
    countries = config.get('countries', ['BE', 'DK', 'NL'])
    live_country = config.get('live', {}).get('country', 'BE')
    output_folder = config.get('outputs', {}).get('forecasts_folder', 'outputs')
    
    # Sidebar: Country selector
    st.sidebar.header("Settings")
    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries,
        index=countries.index(live_country) if live_country in countries else 0
    )
    
    st.sidebar.markdown(f"**Live Country:** {live_country}")
    
    # Load data
    df_forecasts = load_forecasts(selected_country, 'test', output_folder)
    df_anomalies = load_anomalies(selected_country, output_folder)
    df_updates = load_updates(selected_country, output_folder)
    metrics = load_metrics(selected_country, 'outputs/metrics')
    
    if df_forecasts is None:
        st.error(f"⚠️ No forecast data found for {selected_country}")
        return
    
    # === KPI TILES ===
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate rolling metrics
    df_forecasts['rolling_mase'] = calculate_rolling_mase(df_forecasts)
    df_forecasts['rolling_coverage'] = calculate_rolling_coverage(df_forecasts)
    
    with col1:
        latest_mase = df_forecasts['rolling_mase'].iloc[-1]
        st.metric(
            "Rolling-7d MASE",
            f"{latest_mase:.3f}" if not np.isnan(latest_mase) else "N/A",
            delta=None
        )
    
    with col2:
        latest_coverage = df_forecasts['rolling_coverage'].iloc[-1]
        st.metric(
            "80% PI Coverage (7d)",
            f"{latest_coverage:.1f}%" if not np.isnan(latest_coverage) else "N/A",
            delta=None
        )
    
    with col3:
        if df_anomalies is not None:
            today = df_forecasts['timestamp'].max().date()
            anomalies_today = df_anomalies[
                df_anomalies['timestamp'].dt.date == today
            ]['flag_z'].sum()
            st.metric("Anomaly Hours Today", int(anomalies_today))
        else:
            st.metric("Anomaly Hours Today", "N/A")
    
    with col4:
        if df_updates is not None and len(df_updates) > 0:
            last_update = df_updates['timestamp'].max()
            st.metric("Last Update", last_update.strftime("%Y-%m-%d %H:%M"))
        else:
            st.metric("Last Update", "N/A")
    
    # === UPDATE STATUS ===
    if df_updates is not None and len(df_updates) > 0:
        st.header("🔄 Online Adaptation Status")
        
        last_update_row = df_updates.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Last Update:** {last_update_row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            reason_emoji = "📅" if last_update_row['reason'] == 'scheduled' else "🚨"
            st.markdown(f"**Reason:** {reason_emoji} {last_update_row['reason'].capitalize()}")
        
        with col3:
            st.markdown(f"**Duration:** {last_update_row['duration_s']:.1f}s")
        
        # Update summary
        st.markdown(f"""
        **Update Summary:**
        - Total updates: {len(df_updates)}
        - Scheduled: {(df_updates['reason'] == 'scheduled').sum()}
        - Drift-triggered: {(df_updates['reason'] == 'drift').sum()}
        """)
    
    # === LIVE SERIES ===
    st.header("📈 Live Series")
    
    days = st.slider("Days to display", 7, 30, 14)
    fig_series = plot_live_series(df_forecasts, days=days)
    st.plotly_chart(fig_series, use_container_width=True)
    
    # === FORECAST CONE ===
    st.header("🎯 Forecast Cone")
    
    fig_cone = plot_forecast_cone(df_forecasts, hours=24)
    st.plotly_chart(fig_cone, use_container_width=True)
    
    # === ANOMALY TAPE ===
    if df_anomalies is not None:
        st.header("🚨 Anomaly Detection")
        
        fig_anomalies = plot_anomaly_tape(df_forecasts, df_anomalies, days=days)
        st.plotly_chart(fig_anomalies, use_container_width=True)
        
        # Anomaly summary
        st.markdown(f"""
        **Anomaly Summary (Test Set):**
        - Total anomalies (Z-score): {df_anomalies['flag_z'].sum()}
        - Anomaly rate: {100 * df_anomalies['flag_z'].mean():.2f}%
        """)
        
        if 'flag_cusum' in df_anomalies.columns:
            st.markdown(f"""
            - CUSUM anomalies: {df_anomalies['flag_cusum'].sum()}
            - CUSUM rate: {100 * df_anomalies['flag_cusum'].mean():.2f}%
            """)
    
    # === METRICS SUMMARY ===
    if metrics is not None:
        st.header("📊 Metrics Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dev Set")
            if metrics['dev'] is not None:
                st.markdown(f"""
                - **MASE:** {metrics['dev']['MASE']:.4f}
                - **sMAPE:** {metrics['dev']['sMAPE']:.2f}%
                - **RMSE:** {metrics['dev']['RMSE']:.2f} MW
                - **Coverage:** {metrics['dev']['Coverage_80%']:.2f}%
                """)
        
        with col2:
            st.subheader("Test Set")
            if metrics['test'] is not None:
                st.markdown(f"""
                - **MASE:** {metrics['test']['MASE']:.4f}
                - **sMAPE:** {metrics['test']['sMAPE']:.2f}%
                - **RMSE:** {metrics['test']['RMSE']:.2f} MW
                - **Coverage:** {metrics['test']['Coverage_80%']:.2f}%
                """)
    
    # === FOOTER ===
    st.markdown("---")
    st.markdown("**OPSD PowerDesk** | Day-Ahead Forecasting & Anomaly Detection System")


if __name__ == '__main__':
    main()
