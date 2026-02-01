#!/usr/bin/env python3
"""
CooledAI Thermal Lag Analytics Dashboard

Professional-grade dashboard displaying thermal lag analysis from facility logs.
Thermal lag: the delay between cooling/load changes and temperature response.

Run: streamlit run dashboard_thermal_lag.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Thermal Lag Analytics | CooledAI",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CooledAI Professional Theme
st.markdown("""
<style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --cooledai-cyan: #00D4AA;
        --cooledai-teal: #00B4D8;
        --cooledai-dark: #0A0E17;
        --cooledai-card: #12182A;
        --cooledai-border: #1E2A3A;
        --cooledai-text: #E8EEF4;
        --cooledai-muted: #6B7A8F;
    }
    
    /* Main container */
    .stApp {
        background: linear-gradient(180deg, #0A0E17 0%, #0D1321 50%, #0A0E17 100%);
    }
    
    /* Header branding */
    .cooledai-header {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(0, 180, 216, 0.05) 100%);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 32px;
        font-family: 'Outfit', sans-serif;
    }
    
    .cooledai-logo {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4AA, #00B4D8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    .cooledai-subtitle {
        color: #6B7A8F;
        font-size: 14px;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #12182A 0%, #0D1321 100%);
        border: 1px solid #1E2A3A;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
        font-family: 'Outfit', sans-serif;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.3);
        box-shadow: 0 4px 20px rgba(0, 212, 170, 0.08);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #00D4AA;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 12px;
        color: #6B7A8F;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #E8EEF4;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1E2A3A;
    }
    
    /* Info callout */
    .info-callout {
        background: rgba(0, 212, 170, 0.06);
        border-left: 4px solid #00D4AA;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        font-size: 14px;
        color: #B8C4D4;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1321 0%, #0A0E17 100%);
        border-right: 1px solid #1E2A3A;
    }
</style>
""", unsafe_allow_html=True)


def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect column roles from CSV structure."""
    cols = {c.lower(): c for c in df.columns}
    result = {}
    
    # Time column
    time_candidates = ['time', 'timestamp', 'time_s', 'times', 'datetime', 'date', 't']
    for tc in time_candidates:
        for col_lower, col in cols.items():
            if tc in col_lower or col_lower == tc:
                result['time'] = col
                break
        if 'time' in result:
            break
    
    if 'time' not in result and len(df.columns) > 0:
        result['time'] = df.columns[0]  # Assume first column is time
    
    # Temperature column
    temp_candidates = ['temp', 'temperature', 't_', 't_c', 'temperature_c']
    for tc in temp_candidates:
        for col_lower, col in cols.items():
            if tc in col_lower:
                result['temperature'] = col
                break
        if 'temperature' in result:
            break
    
    # Load column
    load_candidates = ['load', 'q_load', 'q_it_load', 'power', 'it_load', 'kw']
    for lc in load_candidates:
        for col_lower, col in cols.items():
            if lc in col_lower:
                result['load'] = col
                break
        if 'load' in result:
            break
    
    # Fan/flow column
    fan_candidates = ['fan', 'flow', 'u_flow', 'velocity', 'speed']
    for fc in fan_candidates:
        for col_lower, col in cols.items():
            if fc in col_lower:
                result['fan'] = col
                break
        if 'fan' in result:
            break
    
    return result


def compute_thermal_lag(temperature: np.ndarray, stimulus: np.ndarray, 
                       max_lag: int = 100) -> tuple:
    """
    Compute thermal lag via cross-correlation.
    Returns (optimal_lag_samples, correlation_at_lag, full_correlation).
    """
    # Normalize
    T_norm = (temperature - np.mean(temperature)) / (np.std(temperature) + 1e-8)
    S_norm = (stimulus - np.mean(stimulus)) / (np.std(stimulus) + 1e-8)
    
    # Cross-correlation
    correlation = signal.correlate(T_norm, S_norm, mode='full')
    lags = signal.correlation_lags(len(T_norm), len(S_norm), mode='full')
    
    # Normalize correlation
    n = min(len(T_norm), len(S_norm))
    correlation = correlation / (n * np.std(T_norm) * np.std(S_norm) + 1e-8)
    
    # Find lag in valid range (positive lag = temp follows stimulus)
    valid_mask = (lags >= 0) & (lags <= max_lag)
    valid_lags = lags[valid_mask]
    valid_corr = correlation[valid_mask]
    
    if len(valid_corr) == 0:
        return 0, 0.0, (lags, correlation)
    
    best_idx = np.argmax(np.abs(valid_corr))
    best_lag = int(valid_lags[best_idx])
    best_corr = float(valid_corr[best_idx])
    
    return best_lag, best_corr, (lags, correlation)


def compute_step_response_lag(temperature: np.ndarray, stimulus: np.ndarray,
                              threshold_pct: float = 0.1) -> float:
    """
    Estimate lag from step changes: when stimulus changes significantly,
    how many samples until temperature responds?
    """
    stimulus_diff = np.diff(stimulus, prepend=stimulus[0])
    stimulus_std = np.std(stimulus)
    
    if stimulus_std < 1e-8:
        return 0.0
    
    # Find significant step changes
    step_threshold = threshold_pct * stimulus_std
    step_indices = np.where(np.abs(stimulus_diff) > step_threshold)[0]
    
    if len(step_indices) < 2:
        return 0.0
    
    lags = []
    temp_diff = np.diff(temperature, prepend=temperature[0])
    
    for i in step_indices[:min(20, len(step_indices))]:  # Limit to 20 steps
        if i + 50 >= len(temperature):
            continue
        # Find when temp starts responding (significant change)
        window = temp_diff[i:i+50]
        response_idx = np.where(np.abs(window) > 0.5 * np.std(temp_diff))[0]
        if len(response_idx) > 0:
            lags.append(response_idx[0])
    
    return np.median(lags) if lags else 0.0


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV with flexible parsing."""
    df = pd.read_csv(file_path)
    # Try parsing datetime if first column looks like timestamp
    first_col = df.columns[0]
    if 'date' in first_col.lower() or 'time' in first_col.lower():
        try:
            df[first_col] = pd.to_datetime(df[first_col])
        except Exception:
            pass
    return df


def create_dashboard(df: pd.DataFrame, col_map: dict, time_resolution: float = 1.0):
    """Build the full thermal lag dashboard."""
    
    time_col = col_map.get('time', df.columns[0])
    temp_col = col_map.get('temperature')
    load_col = col_map.get('load')
    fan_col = col_map.get('fan')
    
    # Create time index if needed
    if df[time_col].dtype in [np.float64, np.int64]:
        t = df[time_col].values
    else:
        t = np.arange(len(df)) * time_resolution
    
    temperature = df[temp_col].values.astype(float) if temp_col else None
    
    if temperature is None:
        st.error("Could not detect temperature column. Please specify in sidebar.")
        return
    
    # Header
    st.markdown("""
    <div class="cooledai-header">
        <div class="cooledai-logo">CooledAI Thermal Lag Analytics</div>
        <div class="cooledai-subtitle">Data Center Thermal Inertia Intelligence • Real-time Lag Detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Compute lags
    lag_load = lag_fan = 0
    corr_load = corr_fan = 0.0
    
    if load_col and len(df[load_col].dropna()) > 10:
        load_vals = df[load_col].ffill().values.astype(float)
        lag_load, corr_load, (lags_full, corr_full) = compute_thermal_lag(temperature, load_vals)
        step_lag_load = compute_step_response_lag(temperature, load_vals)
    else:
        load_vals = None
        step_lag_load = 0
    
    if fan_col and len(df[fan_col].dropna()) > 10:
        fan_vals = df[fan_col].ffill().values.astype(float)
        lag_fan, corr_fan, _ = compute_thermal_lag(temperature, fan_vals)
        step_lag_fan = compute_step_response_lag(temperature, fan_vals)
    else:
        fan_vals = None
        step_lag_fan = 0
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{lag_load if load_col else '—'}</div>
            <div class="metric-label">Load→Temp Lag (samples)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{lag_fan if fan_col else '—'}</div>
            <div class="metric-label">Fan→Temp Lag (samples)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_temp = np.mean(temperature)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_temp:.1f}°C</div>
            <div class="metric-label">Mean Temperature</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        temp_std = np.std(temperature)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{temp_std:.2f}</div>
            <div class="metric-label">Temp Variability (°C)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        n_points = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_points:,}</div>
            <div class="metric-label">Data Points</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main charts - 2 columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-header">📈 Temperature & Stimulus Overlay</div>', 
                    unsafe_allow_html=True)
        
        fig_overlay = go.Figure()
        
        # Temperature (primary)
        fig_overlay.add_trace(go.Scatter(
            x=t[:min(2000, len(t))],
            y=temperature[:min(2000, len(temperature))],
            mode='lines',
            name='Temperature',
            line=dict(color='#00D4AA', width=2),
            yaxis='y'
        ))
        
        # Load (secondary axis) if available
        if load_vals is not None:
            load_norm = (load_vals - np.min(load_vals)) / (np.max(load_vals) - np.min(load_vals) + 1e-8)
            load_scaled = np.min(temperature) + load_norm * (np.max(temperature) - np.min(temperature))
            fig_overlay.add_trace(go.Scatter(
                x=t[:min(2000, len(t))],
                y=load_scaled[:min(2000, len(load_scaled))],
                mode='lines',
                name='IT Load (scaled)',
                line=dict(color='#FF6B6B', width=1.5, dash='dot'),
                opacity=0.8
            ))
        
        # Fan (secondary) if available
        if fan_vals is not None and load_vals is None:
            fan_norm = (fan_vals - np.min(fan_vals)) / (np.max(fan_vals) - np.min(fan_vals) + 1e-8)
            fan_scaled = np.min(temperature) + fan_norm * (np.max(temperature) - np.min(temperature))
            fig_overlay.add_trace(go.Scatter(
                x=t[:min(2000, len(t))],
                y=fan_scaled[:min(2000, len(fan_scaled))],
                mode='lines',
                name='Fan Speed (scaled)',
                line=dict(color='#00B4D8', width=1.5, dash='dot'),
                opacity=0.8
            ))
        
        fig_overlay.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(18, 24, 42, 0.8)',
            font=dict(family='Outfit', color='#E8EEF4', size=12),
            margin=dict(l=50, r=30, t=20, b=50),
            height=350,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(0,0,0,0)'
            ),
            xaxis=dict(
                gridcolor='#1E2A3A',
                title='Time',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='#1E2A3A',
                title='Temperature (°C)',
                showgrid=True
            )
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
    
    with col_right:
        st.markdown('<div class="section-header">⏱️ Cross-Correlation (Lag Analysis)</div>', 
                    unsafe_allow_html=True)
        
        if load_vals is not None:
            _, _, (lags_cc, corr_cc) = compute_thermal_lag(temperature, load_vals, max_lag=200)
            valid = (lags_cc >= 0) & (lags_cc <= 200)
            
            fig_cc = go.Figure()
            fig_cc.add_trace(go.Scatter(
                x=lags_cc[valid],
                y=corr_cc[valid],
                mode='lines',
                name='Load ↔ Temperature',
                line=dict(color='#00D4AA', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.2)'
            ))
            fig_cc.add_vline(x=lag_load, line_dash="dash", line_color="#FF6B6B", 
                           annotation_text=f"Peak: {lag_load}s")
            
            fig_cc.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(18, 24, 42, 0.8)',
                font=dict(family='Outfit', color='#E8EEF4', size=12),
                margin=dict(l=50, r=30, t=20, b=50),
                height=350,
                xaxis_title='Lag (samples)',
                yaxis_title='Correlation',
                xaxis=dict(gridcolor='#1E2A3A'),
                yaxis=dict(gridcolor='#1E2A3A')
            )
            st.plotly_chart(fig_cc, use_container_width=True)
        else:
            st.info("Load column not detected. Cross-correlation requires load or fan data.")
    
    # Second row: Lag-shifted comparison & Insights
    st.markdown("---")
    col_shift, col_insight = st.columns([2, 1])
    
    with col_shift:
        st.markdown('<div class="section-header">🔄 Lag-Corrected Alignment</div>', 
                    unsafe_allow_html=True)
        
        if load_vals is not None and lag_load > 0:
            # Shift load backward by lag to align with temperature
            load_shifted = np.roll(load_vals, -lag_load)
            load_shifted[-lag_load:] = np.nan
            
            load_norm = (load_vals - np.min(load_vals)) / (np.max(load_vals) - np.min(load_vals) + 1e-8)
            temp_norm = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature) + 1e-8)
            
            n_show = min(1500, len(t))
            
            fig_align = go.Figure()
            fig_align.add_trace(go.Scatter(
                x=t[:n_show],
                y=temp_norm[:n_show],
                mode='lines',
                name='Temperature (normalized)',
                line=dict(color='#00D4AA', width=2)
            ))
            fig_align.add_trace(go.Scatter(
                x=t[:n_show],
                y=load_norm[:n_show],
                mode='lines',
                name=f'Load (shifted -{lag_load})',
                line=dict(color='#FF6B6B', width=1.5, dash='dot'),
                opacity=0.8
            ))
            
            fig_align.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(18, 24, 42, 0.8)',
                font=dict(family='Outfit', color='#E8EEF4'),
                height=300,
                legend=dict(orientation='h', y=1.02),
                xaxis=dict(gridcolor='#1E2A3A'),
                yaxis=dict(gridcolor='#1E2A3A')
            )
            st.plotly_chart(fig_align, use_container_width=True)
        else:
            st.info("Enable load column mapping to see lag-corrected alignment.")
    
    with col_insight:
        st.markdown('<div class="section-header">💡 Thermal Lag Insights</div>', 
                    unsafe_allow_html=True)
        
        insights = []
        if load_col:
            lag_sec = lag_load * time_resolution
            insights.append(f"**Load→Temp Lag:** {lag_sec:.1f}s")
            insights.append(f"Temperature responds to IT load changes after ~{lag_sec:.0f} seconds.")
        if fan_col:
            lag_sec_f = lag_fan * time_resolution
            insights.append(f"**Fan→Temp Lag:** {lag_sec_f:.1f}s")
            insights.append(f"Cooling adjustments take ~{lag_sec_f:.0f}s to affect temperature.")
        
        if insights:
            st.markdown(
                "<div class='info-callout'>" + "<br><br>".join(insights) + "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div class="info-callout">
                Map <b>Load</b> and/or <b>Fan</b> columns in the sidebar to compute thermal lag.
                Lag indicates how many seconds pass before temperature responds to changes.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Data Preview**")
        st.dataframe(df.head(10), use_container_width=True, height=250)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #6B7A8F; font-size: 12px;'>"
        "CooledAI Thermal Intelligence • Physics-Informed Lag Analytics • © 2026"
        "</p>",
        unsafe_allow_html=True
    )


def main():
    # Sidebar
    st.sidebar.markdown("### ⚙️ Data Source")
    
    default_path = Path("data/raw/Log-013026.csv")
    if not default_path.exists():
        default_path = Path("data/processed/failure_modes_v1_detailed.csv")  # Fallback for demo
    
    file_option = st.sidebar.radio(
        "Load from:",
        ["Project file (Log-013026.csv)", "Upload CSV"]
    )
    
    df = None
    col_map = {}
    
    if file_option == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Choose CSV", type=['csv'])
        if uploaded:
            df = load_data(uploaded)
            col_map = detect_columns(df)
    else:
        # Try project root first
        for path in ["data/raw/Log-013026.csv", "Log-013026.csv", "data/processed/failure_modes_v1_detailed.csv"]:
            if os.path.exists(path):
                df = load_data(path)
                col_map = detect_columns(df)
                st.sidebar.success(f"✓ Loaded {path}")
                break
    
    if df is None:
        st.sidebar.warning("Log-013026.csv not found. Upload a file or place it in the project root.")
        st.markdown("""
        <div class="cooledai-header">
            <div class="cooledai-logo">CooledAI Thermal Lag Analytics</div>
            <div class="cooledai-subtitle">Place Log-013026.csv in this folder or upload via sidebar</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 Use the sidebar to upload your thermal log CSV.")
        return
    
    # Scenario filter (for multi-scenario data like failure_modes)
    scenario_col = None
    for c in df.columns:
        if 'scenario' in c.lower():
            scenario_col = c
            break
    if scenario_col and df[scenario_col].nunique() > 1:
        scenarios = sorted(df[scenario_col].unique())
        selected_scenario = st.sidebar.selectbox(
            "Filter by scenario (for cleaner lag analysis)",
            ["All"] + [str(s) for s in scenarios],
            index=0
        )
        if selected_scenario != "All":
            try:
                val = int(selected_scenario)
            except ValueError:
                val = float(selected_scenario) if '.' in selected_scenario else selected_scenario
            df = df[df[scenario_col] == val].reset_index(drop=True)
    
    # Column mapping overrides
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Column Mapping")
    
    for role, label in [('time', 'Time'), ('temperature', 'Temperature'), ('load', 'Load'), ('fan', 'Fan')]:
        options = ['Auto'] + list(df.columns)
        default = col_map.get(role, 'Auto')
        if default != 'Auto':
            default = default if default in options else 'Auto'
        choice = st.sidebar.selectbox(label, options, index=options.index(default) if default in options else 0)
        if choice != 'Auto':
            col_map[role] = choice
    
    time_res = st.sidebar.number_input("Time resolution (sec/sample)", min_value=0.1, value=1.0, step=0.1)
    
    create_dashboard(df, col_map, time_resolution=time_res)


if __name__ == "__main__":
    main()
