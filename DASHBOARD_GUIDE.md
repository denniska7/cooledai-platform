# CoolingAI Dashboard Guide

**Complete guide to all three dashboard versions**

---

## Overview

CoolingAI now offers **three dashboard experiences** for different use cases:

| Dashboard | File | Focus | Audience | Port |
|-----------|------|-------|----------|------|
| **Single Rack** | `app.py` | Individual rack optimization | Engineers, Demos | 8501 |
| **10-Rack Fleet** | `app_fleet.py` | Fleet management | Operations Teams | 8502 |
| **1MW Cluster** | `app_1mw.py` | Site efficiency | Executives, CFOs | 8503 |

---

## Prerequisites

### Python Version
- **Recommended**: Python 3.11 or 3.12
- **Not supported**: Python 3.13 (PyTorch compatibility issue)

### Check your Python version:
```bash
python3 --version
# or
python3.11 --version
```

### Required Packages

```bash
# Install dependencies
pip3.11 install streamlit plotly pandas torch stable-baselines3
```

### Trained Models

All dashboards require:
- ✅ `checkpoints/best_recurrent_pinn.pt` (RecurrentPINN physics engine)
- ✅ `optimizer/ppo_cooling_agent.zip` (PPO RL agent)

Both models are already trained and available in your project directory.

---

## Dashboard 1: Single Rack Command Center

**File**: `app.py`
**Focus**: Individual rack optimization with XAI, Shadow Pilot, and Stability Guard

### Features

- 💰 Money Saved ticker (real-time energy savings)
- 🌡️ 3D thermal heatmap (10×10×5 server grid)
- 🤖 AI vs Human control modes
- 🧠 Explainable AI reasoning
- 🛡️ Stability Guard (TTF-based safety override)
- 📊 Shadow Pilot (CSV upload for historical comparison)

### Launch

```bash
cd /Users/denniswork/Desktop/coolingai_simulator
streamlit run app.py
```

**URL**: http://localhost:8501

### Use Cases

- **Engineering Demos**: Show AI decision-making to technical audiences
- **Algorithm Validation**: Compare AI vs manual control
- **Failure Prediction**: Test TTF calculations and safety overrides
- **Shadow Pilot**: Validate against historical data

### Key Metrics

- Temperature: °C (target < 75°C)
- Fan Speed: m/s (range 0.5-3.0)
- Money Saved: $ (vs 2.0 m/s baseline)
- Energy Reduction: % (typically 50-60%)

---

## Dashboard 2: 10-Rack Fleet Command Center

**File**: `app_fleet.py`
**Focus**: Fleet management with thermal coupling and collective monitoring

### Features

- 🗂️ Row View: 10 rack status cards (red/green safety indicators)
- 💰 Cluster Savings ticker (aggregate across all racks)
- 🔍 Focus Mode: Dropdown to select rack for detailed 3D view
- 🌡️ Thermal bleed visualization (5% heat transfer between racks)
- 🛡️ Collective Stability Guard (per-rack TTF monitoring)
- 📈 Cluster performance history

### Launch

```bash
cd /Users/denniswork/Desktop/coolingai_simulator
streamlit run app_fleet.py --server.port 8502
```

**URL**: http://localhost:8502

### Use Cases

- **Operations Centers**: Monitor entire data center row
- **Thermal Coupling Analysis**: Study inter-rack heat transfer
- **Capacity Planning**: Simulate adding/removing racks
- **Failure Impact**: Toggle rack failures to see cluster impact

### Key Metrics

- Cluster Avg Temp: °C (average across 10 racks)
- Max Temp: °C (hottest rack in cluster)
- Total Cluster Savings: $ (sum across all racks)
- Total IT Load: kW (sum of all rack loads)

---

## Dashboard 3: 1MW Site Efficiency Dashboard

**File**: `app_1mw.py`
**Focus**: Enterprise site-level PUE optimization

### Features

- ⚡ **Hero PUE Gauge**: Real-time Power Usage Effectiveness
- 🌱 **Carbon Ticker**: Annual CO₂ offset (tons) + trees equivalent
- 💰 **Cash Ticker**: Annual OpEx reduction ($) from PUE optimization
- 🏗️ **8 Blackwell Pods**: Each @ 125kW = 1,000 kW total IT load
- 🔴 **Pod Failure Simulation**: Toggle pods to see PUE impact
- 📊 **Business Impact Summary**: Energy, carbon, cash metrics

### Launch

```bash
cd /Users/denniswork/Desktop/coolingai_simulator
streamlit run app_1mw.py --server.port 8503
```

**URL**: http://localhost:8503

### Use Cases

- **Executive Dashboards**: Show PUE and ROI to C-suite
- **Investor Demos**: Demonstrate business value ($420k/year savings)
- **ESG Reporting**: Track carbon offset (1,400 tons CO₂/year)
- **Capacity Planning**: Simulate pod failures on site PUE

### Key Metrics

- **PUE**: Ratio (target 1.1, baseline 1.5)
- **Annual Carbon Offset**: Tons CO₂ (environmental impact)
- **Annual OpEx Reduction**: $ (business value)
- **Total IT Load**: kW (operational capacity)
- **Cooling Power**: kW (overhead cost)

---

## Running Multiple Dashboards Simultaneously

You can run all three dashboards at the same time on different ports:

### Terminal 1: Single Rack
```bash
streamlit run app.py --server.port 8501
```

### Terminal 2: 10-Rack Fleet
```bash
streamlit run app_fleet.py --server.port 8502
```

### Terminal 3: 1MW Cluster
```bash
streamlit run app_1mw.py --server.port 8503
```

### Access URLs
- Single Rack: http://localhost:8501
- Fleet: http://localhost:8502
- 1MW Cluster: http://localhost:8503

---

## Quick Comparison

### Which Dashboard Should I Use?

| Question | Answer |
|----------|--------|
| "I want to demo AI decision-making" | **Single Rack** (app.py) |
| "I need to monitor multiple racks" | **10-Rack Fleet** (app_fleet.py) |
| "Show me PUE and business ROI" | **1MW Cluster** (app_1mw.py) |
| "I'm presenting to engineers" | **Single Rack** (app.py) |
| "I'm presenting to operations" | **10-Rack Fleet** (app_fleet.py) |
| "I'm presenting to executives" | **1MW Cluster** (app_1mw.py) |
| "I want to test failure scenarios" | **1MW Cluster** (app_1mw.py) |
| "I want to compare AI vs human control" | **Single Rack** (app.py) |
| "I need carbon offset metrics" | **1MW Cluster** (app_1mw.py) |

---

## Troubleshooting

### Issue: "streamlit: command not found"

**Fix**:
```bash
python3.11 -m streamlit run app.py
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Fix**:
```bash
pip3.11 install streamlit plotly pandas
```

### Issue: "Port already in use"

**Fix**: Use a different port
```bash
streamlit run app.py --server.port 8504
```

Or kill existing Streamlit processes:
```bash
pkill -f streamlit
```

### Issue: "Model not found"

**Check**:
```bash
ls -lh checkpoints/best_recurrent_pinn.pt
ls -lh optimizer/ppo_cooling_agent.zip
```

If missing, train the models first:
```bash
python3.11 train_recurrent_pinn.py  # ~5 minutes
python3.11 optimizer/train_rl_agent.py  # ~10 minutes
```

### Issue: Dashboard loads but shows errors

**Clear Streamlit cache**:
1. Open dashboard in browser
2. Click "☰" menu (top right)
3. Select "Clear cache"
4. Refresh page (Cmd+R or Ctrl+R)

---

## Performance Tips

### For Best Performance

1. **Use Python 3.11**: Faster than 3.12, more compatible than 3.13
2. **Enable GPU**: If available, set `device='cuda'` in dashboard initialization
3. **Reduce history length**: Modify `max_history = 100` to smaller value if slow
4. **Disable auto-run**: Manual step mode is more responsive for detailed analysis

### Memory Usage

- **Single Rack**: ~400 MB
- **10-Rack Fleet**: ~600 MB
- **1MW Cluster**: ~600 MB

### CPU Usage

- **Idle**: 5-10%
- **Auto-run mode**: 20-40%
- **With GPU**: < 10% (offloads to GPU)

---

## Advanced Configuration

### Streamlit Config File

Create `~/.streamlit/config.toml` for persistent settings:

```toml
[browser]
gatherUsageStats = false

[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#00FF00"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1E1E1E"
textColor = "#FFFFFF"
```

### Dashboard-Specific Settings

Edit the dashboard files to customize:

**app.py** (Single Rack):
```python
# Line 72-75
self.baseline_fan_speed = 2.0  # Change baseline
self.baseline_energy_rate = 0.1 * (self.baseline_fan_speed ** 3)
```

**app_fleet.py** (10-Rack Fleet):
```python
# Line 59
num_racks: int = 10  # Change number of racks
```

**app_1mw.py** (1MW Cluster):
```python
# Line 117-118
self.num_pods = 8  # Change to 10 or 16 for larger sites
self.it_load_per_pod = 125.0  # kW per pod
```

---

## Dashboard Feature Matrix

| Feature | Single Rack | 10-Rack Fleet | 1MW Cluster |
|---------|-------------|---------------|-------------|
| **3D Heatmap** | ✅ | ✅ (per rack) | ❌ |
| **PUE Calculation** | ❌ | ❌ | ✅ |
| **Carbon Offset** | ❌ | ❌ | ✅ |
| **Cash Savings** | ✅ | ✅ | ✅ (annual) |
| **XAI Reasoning** | ✅ | ✅ | ❌ |
| **Shadow Pilot** | ✅ | ❌ | ❌ |
| **TTF Monitoring** | ✅ | ✅ | ❌ |
| **Failure Simulation** | ❌ | ❌ | ✅ |
| **Thermal Bleed** | ❌ | ✅ | ✅ |
| **Row View** | ❌ | ✅ | ✅ (pods) |
| **Business Metrics** | ❌ | ❌ | ✅ |
| **Auto-run Mode** | ✅ | ✅ | ✅ |
| **Manual Override** | ✅ | ✅ | ❌ |

---

## Demo Scripts

### 5-Minute Single Rack Demo

1. Launch: `streamlit run app.py`
2. **Show AI Autopilot** (1 min):
   - Enable "Auto-run"
   - Point out money saved ticker
   - Show AI Reasoning section
3. **Manual Override** (1 min):
   - Switch to "Manual Override"
   - Try to beat the AI
   - Show difficulty of optimization
4. **Shadow Pilot** (2 min):
   - Upload CSV with historical data
   - Show comparison metrics
5. **Stability Guard** (1 min):
   - Increase IT Load to 150kW
   - Watch TTF drop
   - Show safety override log

### 5-Minute 1MW Cluster Demo

1. Launch: `streamlit run app_1mw.py --server.port 8503`
2. **Hero PUE Gauge** (1 min):
   - Point to real-time PUE (1.1 vs 1.5 baseline)
   - Show color-coded rating
3. **Carbon & Cash Tickers** (2 min):
   - Annual carbon offset: ~1,400 tons CO₂
   - Annual OpEx reduction: ~$420k
   - ROI calculation: 14 months
4. **Pod Failure Simulation** (1 min):
   - Toggle Pod 3 to failed
   - Watch PUE increase
   - Show impact on business metrics
5. **Historical Charts** (1 min):
   - PUE trend over time
   - Power consumption breakdown
   - Baseline vs optimized

---

## Next Steps

### For Engineers
- Experiment with `app.py` to understand AI decision-making
- Test different IT loads and fan speed strategies
- Upload historical data to Shadow Pilot

### For Operations
- Use `app_fleet.py` to monitor entire data center row
- Simulate rack failures to understand thermal coupling
- Track cluster-wide energy savings

### For Executives
- Present `app_1mw.py` to board for PUE and ROI metrics
- Include in ESG reports (carbon offset data)
- Use for investor demos (business value proposition)

---

## Support

**Documentation**:
- Single Rack: `PHASE_4_4_ENTERPRISE_READY.md`
- 10-Rack Fleet: `app_fleet.py` (docstrings)
- 1MW Cluster: `PHASE_6_1MW_CLUSTER.md`

**Quick Reference**:
- Installation: `RUN_DASHBOARD.md`
- Quick Start: `QUICKSTART_PHASE_4.md`
- Technical Details: `PHASE_4_2_4_3_COMPLETE.md`

---

**Last Updated**: 2026-01-28
**Status**: Production Ready
**Version**: 1.0
