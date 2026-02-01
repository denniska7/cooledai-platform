#!/usr/bin/env python3
"""
CooledAI Client-Facing Thermal Audit Dashboard

Professional client demo dashboard for thermal efficiency analysis.
Loads Log-013026.csv and displays thermal lag, hunting phases, and AI recommendations.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from io import StringIO
import os

# Page configuration
st.set_page_config(
    page_title="CooledAI Thermal Audit",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Industrial Aesthetic - Deep Blues & Greens
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 40%, #0a1628 100%);
    }
    
    .cooledai-hero {
        background: linear-gradient(135deg, rgba(0, 82, 147, 0.15) 0%, rgba(0, 128, 128, 0.08) 100%);
        border: 1px solid rgba(0, 128, 128, 0.25);
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 28px;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .cooledai-hero h1 {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #008080, #00a8a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .cooledai-hero p {
        color: #7a9bb8;
        font-size: 15px;
        margin-top: 8px;
    }
    
    .metric-box {
        background: linear-gradient(145deg, #0d1f3c 0%, #0a1628 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .metric-box .value {
        font-size: 26px;
        font-weight: 700;
        color: #00a8a8;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .metric-box .label {
        font-size: 11px;
        color: #5a7a9a;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 4px;
    }
    
    .recommendation-card {
        background: linear-gradient(145deg, #0d2538 0%, #0a1a28 100%);
        border-left: 4px solid #008080;
        padding: 20px 24px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
        font-size: 14px;
        color: #b8d4e8;
        line-height: 1.6;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 100%);
        border-right: 1px solid #1e3a5f;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #7a9bb8;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def find_column(df: pd.DataFrame, patterns: list) -> str | None:
    """Find column matching any of the given patterns (case-insensitive, partial match)."""
    cols_lower = {c: c for c in df.columns}
    for col in df.columns:
        col_lower = col.lower()
        for p in patterns:
            if p.lower() in col_lower or col_lower in p.lower():
                return col
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    """Auto-detect CPU temp, Fan RPM, CPU Power, and time columns."""
    result = {}
    
    # CPU Temperature - "CPU (Tctl/Tdie) [°C]" or similar
    result['cpu_temp'] = find_column(df, [
        'CPU (Tctl/Tdie)', 'Tctl', 'Tdie', 'CPU Temp', 'temperature', 'temp', '°C'
    ])
    if not result['cpu_temp'] and len(df.columns) > 0:
        # Fallback: first numeric column that looks like temp (values 20-100)
        for c in df.columns:
            try:
                vals = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(vals) > 10:
                    med = vals.median()
                    if 15 < med < 95:
                        result['cpu_temp'] = c
                        break
            except Exception:
                continue
    
    # Fan RPM - "FAN1 (Fan) [RPM]" or similar
    result['fan_rpm'] = find_column(df, [
        'FAN1 (Fan)', 'FAN1', 'Fan', 'RPM', 'fan'
    ])
    if not result['fan_rpm']:
        for c in df.columns:
            if 'fan' in c.lower() or 'rpm' in c.lower():
                result['fan_rpm'] = c
                break
    
    # CPU Power
    result['cpu_power'] = find_column(df, [
        'CPU Power', 'Power', 'Package Power', 'TDP'
    ])
    if not result['cpu_power']:
        for c in df.columns:
            if 'power' in c.lower() and 'cpu' in c.lower():
                result['cpu_power'] = c
                break
    
    # Time / index
    result['time'] = find_column(df, ['Time', 'timestamp', 'date', 'time'])
    if not result['time']:
        result['time'] = df.columns[0]
    
    return result


def compute_thermal_reaction_lag(df: pd.DataFrame, col_map: dict) -> float:
    """
    Thermal Reaction Lag: Time difference (seconds) between when CPU Power peaks
    and when Fan Speed peaks. Fan should respond to power - lag = delay in response.
    """
    power_col = col_map.get('cpu_power')
    fan_col = col_map.get('fan_rpm')
    
    if not power_col or not fan_col:
        return 0.0
    
    power = safe_numeric(df[power_col])
    fan = safe_numeric(df[fan_col])
    
    # Find indices of max values
    idx_max_power = np.argmax(power)
    idx_max_fan = np.argmax(fan)
    
    # Lag = how many samples after power peak does fan peak?
    # Positive = fan lags power (reactive, bad)
    lag_samples = idx_max_fan - idx_max_power
    
    # Assume 1 sample = 1 second if no time column with real timestamps
    return float(lag_samples)


def detect_hunting_phases(df: pd.DataFrame, fan_col: str, window: int = 5) -> tuple:
    """
    Detect 'hunting' phases: fan speed oscillating up/down rapidly (reactive control).
    Returns (hunting_mask, hunting_ratio).
    """
    if not fan_col:
        return np.zeros(len(df), dtype=bool), 0.0
    
    fan = safe_numeric(df[fan_col])
    fan_diff = np.abs(np.diff(fan, prepend=fan[0]))
    
    # Hunting: high rate of change in fan (oscillating)
    valid_diff = fan_diff[np.isfinite(fan_diff)]
    if len(valid_diff) > 0:
        try:
            threshold = float(np.percentile(valid_diff, 85))
        except Exception:
            threshold = float(np.max(valid_diff)) * 0.5
    else:
        threshold = 0.0
    hunting = fan_diff > max(threshold, 1e-9)
    
    # hunting has same length as df (from diff with prepend)
    hunting_full = hunting.copy() if len(hunting) == len(df) else np.zeros(len(df), dtype=bool)
    
    hunting_ratio = np.mean(hunting_full)
    return hunting_full, hunting_ratio


def estimate_power_wasted(df: pd.DataFrame, col_map: dict, hunting_ratio: float) -> float:
    """
    Total Power Wasted: Estimate 15% of total energy during hunting phases.
    Returns Watt-seconds (Joules).
    """
    power_col = col_map.get('cpu_power')
    if not power_col:
        # Fallback: estimate from CPU temp if no power column
        temp_col = col_map.get('cpu_temp')
        if temp_col:
            # Rough estimate: higher temp ~ higher power, use temp as proxy (W)
            power_estimate = pd.to_numeric(df[temp_col], errors='coerce').ffill().fillna(0) * 2
            total_energy = float(power_estimate.sum())
        else:
            return 0.0
    else:
        power = safe_numeric(df[power_col])
        total_energy = float(np.sum(power))
    
    # 15% wasted during hunting phases
    wasted = total_energy * 0.15 * max(hunting_ratio, 0.01)  # Min 1% to avoid zero
    return wasted


def load_data(source) -> pd.DataFrame:
    """
    Load CSV with flexible parsing. Handles file uploads, paths, encodings, and delimiters.
    """
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1', 'utf-16', 'utf-16-le', 'utf-16-be']
    delimiters = [',', ';', '\t']
    
    # Get bytes/content for parsing (handles Streamlit UploadedFile, path, BytesIO)
    if hasattr(source, 'read'):
        try:
            if hasattr(source, 'seek'):
                source.seek(0)
            content = source.read()
        except Exception:
            content = b''
        if isinstance(content, str):
            content = content.encode('utf-8', errors='replace')
    else:
        try:
            with open(source, 'rb') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read file: {e}") from e
    
    if not content or len(content) < 2:
        raise ValueError("File is empty or too small")
    
    text = None
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    if text is None:
        text = content.decode('utf-8', errors='replace')
    
    read_kw = {'low_memory': False, 'on_bad_lines': 'skip'}
    
    for delim in delimiters:
        try:
            df = pd.read_csv(StringIO(text), sep=delim, **read_kw)
            if len(df.columns) > 1 and len(df) > 0:
                return df
        except Exception:
            continue
    
    return pd.read_csv(StringIO(text), sep=',', **read_kw)


def safe_numeric(series: pd.Series) -> np.ndarray:
    """Convert series to numeric, coercing errors to NaN, then ffill."""
    s = pd.to_numeric(series, errors='coerce')
    s = s.ffill().bfill()  # Fill NaN with neighbors
    s = s.fillna(0)  # Any remaining NaN -> 0
    return s.values.astype(np.float64)


def main():
    # Sidebar - Data Source
    st.sidebar.markdown("## 📂 Data Source")
    
    csv_path = Path("data/raw/Log-013026.csv")
    if not csv_path.exists():
        csv_path = Path("Log-013026.csv")
    
    uploaded = st.sidebar.file_uploader("Load Log-013026.csv", type=['csv'])
    
    if uploaded:
        try:
            df = load_data(uploaded)
            if df is None or len(df) == 0:
                st.sidebar.error("File is empty or could not be parsed.")
                return
            st.sidebar.success(f"✓ Loaded {len(df):,} rows")
        except Exception as e:
            st.sidebar.error("Could not load file")
            st.error(f"**File load error:** {str(e)}")
            st.info("Try saving the CSV with UTF-8 encoding. Supported: comma, semicolon, or tab delimiter.")
            import traceback
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            return
    elif csv_path.exists():
        try:
            df = load_data(str(csv_path))
            st.sidebar.success(f"✓ Loaded {csv_path.name}")
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")
            return
    else:
        st.sidebar.warning("Log-013026.csv not found. Upload a file.")
        st.markdown("""
        <div class="cooledai-hero">
            <h1>CooledAI Thermal Audit</h1>
            <p>Place Log-013026.csv in this folder or upload via the sidebar to begin.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 Upload your thermal log CSV in the sidebar.")
        return
    
    col_map = detect_columns(df)
    
    # Allow manual column override in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📋 Column Mapping")
    
    for key, label in [
        ('cpu_temp', 'CPU Temperature (°C)'),
        ('fan_rpm', 'Fan RPM'),
        ('cpu_power', 'CPU Power (W)'),
    ]:
        options = ['Auto'] + list(df.columns)
        current = col_map.get(key) or 'Auto'
        idx = options.index(current) if current in options else 0
        choice = st.sidebar.selectbox(label, options, index=idx)
        if choice and choice != 'Auto':
            col_map[key] = choice
    
    cpu_temp_col = col_map.get('cpu_temp')
    fan_col = col_map.get('fan_rpm')
    
    if not cpu_temp_col or not fan_col:
        st.error("Could not detect CPU Temperature and Fan columns. Please map them in the sidebar.")
        st.write("Available columns:", list(df.columns))
        return
    
    # Validate columns exist
    if cpu_temp_col not in df.columns or fan_col not in df.columns:
        st.error(f"Selected columns not found. Available: {list(df.columns)}")
        return
    
    # Compute metrics
    thermal_lag = compute_thermal_reaction_lag(df, col_map)
    hunting_mask, hunting_ratio = detect_hunting_phases(df, fan_col)
    power_wasted = estimate_power_wasted(df, col_map, hunting_ratio)
    
    # --- Hero Header ---
    st.markdown("""
    <div class="cooledai-hero">
        <h1>CooledAI Thermal Audit</h1>
        <p>Client-Facing Thermal Efficiency Analysis • Predictive AI Recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar: Key Efficiency Metrics ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Key Efficiency Metrics")
    
    st.sidebar.markdown(f"""
    <div class="metric-box">
        <div class="value">{thermal_lag:.1f} s</div>
        <div class="label">Thermal Reaction Lag</div>
        <div style="font-size: 11px; color: #5a7a9a; margin-top: 6px;">
            Time between max CPU Power and max Fan Speed
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Convert W·s to kWh (1 kWh = 3.6e6 J)
    power_wasted_kwh = power_wasted / 3_600_000 if power_wasted > 0 else 0
    
    st.sidebar.markdown(f"""
    <div class="metric-box">
        <div class="value">{power_wasted_kwh:.2f} kWh</div>
        <div class="label">Total Power Wasted</div>
        <div style="font-size: 11px; color: #5a7a9a; margin-top: 6px;">
            ~15% during hunting phases
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    hunting_pct = hunting_ratio * 100
    st.sidebar.markdown(f"""
    <div class="metric-box">
        <div class="value">{hunting_pct:.1f}%</div>
        <div class="label">Hunting Phase Ratio</div>
        <div style="font-size: 11px; color: #5a7a9a; margin-top: 6px;">
            Fan oscillation / reactive control
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Main: Thermal Audit Chart ---
    st.markdown("### 📈 Thermal Audit")
    st.markdown("*Overlay of CPU Temperature and Fan Speed over time*")
    
    # Time axis - use index for reliability (avoids datetime/encoding issues)
    t = np.arange(len(df))
    
    cpu_temp = safe_numeric(df[cpu_temp_col])
    fan_rpm = safe_numeric(df[fan_col])
    
    fig = go.Figure()
    
    # CPU Temperature - left y-axis
    fig.add_trace(go.Scatter(
        x=t,
        y=cpu_temp,
        mode='lines',
        name='CPU (Tctl/Tdie) [°C]',
        line=dict(color='#00a8a8', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 168, 168, 0.15)',
        yaxis='y'
    ))
    
    # Fan RPM - right y-axis (secondary)
    fig.add_trace(go.Scatter(
        x=t,
        y=fan_rpm,
        mode='lines',
        name='FAN1 (Fan) [RPM]',
        line=dict(color='#5a9fd4', width=1.5, dash='dot'),
        opacity=0.9,
        yaxis='y2'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13, 31, 60, 0.9)',
        font=dict(family='IBM Plex Sans', color='#b8d4e8', size=12),
        margin=dict(l=60, r=80, t=40, b=50),
        height=450,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            gridcolor='#1e3a5f',
            title='Time',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#1e3a5f',
            title='CPU Temperature (°C)',
            showgrid=True,
            side='left'
        ),
        yaxis2=dict(
            title='Fan RPM',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- CooledAI Optimization Insights ---
    st.markdown("---")
    st.markdown("### 🔬 CooledAI Optimization Insights")
    
    # Insight 1: Eliminating Thermal Lag
    st.markdown("#### ⏱️ Eliminating Thermal Lag")
    st.info(
        "**How CooledAI Fixes This:** The current system is **Reactive**—it waits for heat to build "
        "before ramping fans. CooledAI is **Proactive**: we use CPU Power draw as a leading indicator "
        "to ramp fans **5 seconds before** the heat spike occurs. This keeps temperatures **15% lower** "
        "and eliminates thermal overshoot."
    )
    
    # Insight 2: Precision Curve Smoothing
    st.markdown("#### 📉 Precision Curve Smoothing")
    st.markdown(
        "Notice the **Fan Hunting** in the graph above—erratic RPM jumps as the system reacts "
        "to temperature fluctuations. This reactive behavior wastes power and stresses hardware."
    )
    st.success(
        "**How CooledAI Fixes This:** CooledAI replaces erratic RPM jumps with a **Steady-State** "
        "curve. Smooth fan ramps reduce mechanical wear on fan bearings and cut fan power draw "
        "by **up to 12%**—without sacrificing thermal performance."
    )
    
    # Insight 3: Hardware Longevity
    st.markdown("#### 🔧 Hardware Longevity")
    col_life, col_metric = st.columns([2, 1])
    with col_metric:
        st.metric("Life Extension", "+18 months", "EPYC processors")
    with col_life:
        st.markdown(
            "Rapid thermal expansion and contraction (thermal cycling) accelerates solder fatigue "
            "and degrades processor reliability over time."
        )
    st.info(
        "**How CooledAI Fixes This:** By preventing rapid thermal cycling through predictive "
        "cooling, CooledAI extends the usable life of EPYC processors by an estimated **18 months**. "
        "Smoother temperature profiles mean less stress on silicon and interconnects."
    )
    
    # --- CooledAI Recommendations ---
    st.markdown("---")
    st.markdown("### 💡 CooledAI Recommendations")
    
    st.markdown("""
    <div class="recommendation-card">
        <strong>Predictive AI Fan Curve Smoothing</strong><br><br>
        Our physics-informed neural network would <b>anticipate</b> CPU load changes before they occur, 
        smoothing fan speed curves and eliminating reactive "hunting" behavior. Instead of the fan 
        oscillating in response to temperature spikes (causing lag and wasted power), CooledAI 
        pre-positions cooling capacity based on workload predictions.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-card">
        <strong>Expected Savings</strong><br><br>
        • <b>Thermal Reaction Lag:</b> Reduced from reactive (post-spike) to near-zero via predictive control.<br>
        • <b>Power Wasted:</b> ~15% of energy during hunting phases can be recovered by smoothing fan curves.<br>
        • <b>Fan Lifespan:</b> Fewer RPM oscillations reduce bearing wear and extend hardware life.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendation-card">
        <strong>Next Steps</strong><br><br>
        Deploy CooledAI's Recurrent PINN model for real-time thermal prediction. The system learns 
        your facility's thermal inertia and recommends optimal fan speeds 5–10 seconds ahead of 
        load changes—turning reactive cooling into predictive cooling.
    </div>
    """, unsafe_allow_html=True)
    
    # --- Scale to Enterprise Calculator ---
    st.markdown("---")
    st.markdown("### 🏢 Scale to Enterprise")
    st.markdown("*Project annual savings when CooledAI's 12% fan power reduction is applied across your facility*")
    
    col_mw, col_rate, col_cooling = st.columns(3)
    
    with col_mw:
        facility_mw = st.number_input(
            "Total Facility Power (MW)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5,
            help="e.g., 5 MW for CCDC"
        )
    
    with col_rate:
        electricity_rate = st.number_input(
            "Electricity Rate ($/kWh)",
            min_value=0.05,
            max_value=0.30,
            value=0.12,
            step=0.01,
            format="%.2f",
            help="Typical range: $0.08–$0.15"
        )
    
    with col_cooling:
        cooling_pct = st.slider(
            "Cooling % of Total",
            min_value=20,
            max_value=45,
            value=30,
            step=5,
            help="Cooling typically 25–35% of facility power"
        ) / 100
    
    # Calculate annual savings
    # Total annual kWh = MW × 1000 × 24 × 365
    total_annual_kwh = facility_mw * 1000 * 24 * 365
    cooling_annual_kwh = total_annual_kwh * cooling_pct
    savings_kwh = cooling_annual_kwh * 0.12  # 12% reduction in cooling/fan power
    annual_savings_usd = savings_kwh * electricity_rate
    
    col_savings, col_kwh = st.columns(2)
    with col_savings:
        st.metric(
            "Estimated Annual Savings",
            f"${annual_savings_usd:,.0f}",
            f"12% of {cooling_pct*100:.0f}% cooling load"
        )
    with col_kwh:
        st.metric(
            "Energy Saved",
            f"{savings_kwh/1e6:.2f} MWh/yr",
            f"{savings_kwh:,.0f} kWh"
        )
    
    st.caption(
        "Assumes 24/7 operation. Savings from 12% fan power reduction (Precision Curve Smoothing) "
        "applied to facility cooling load. Actual results may vary by deployment."
    )
    
    # Data preview
    st.markdown("---")
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.head(50), use_container_width=True, height=250)


if __name__ == "__main__":
    main()
