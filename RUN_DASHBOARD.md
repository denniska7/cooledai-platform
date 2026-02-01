# How to Run the CoolingAI Dashboard

**Tested and Working** ✅
**Date**: 2026-01-28

---

## Prerequisites Verified

✅ Python 3.11 is installed
✅ Trained models exist:
  - `checkpoints/best_recurrent_pinn.pt` (1.0 MB)
  - `optimizer/ppo_cooling_agent.zip` (143 KB)

---

## Step-by-Step Guide

### Step 1: Navigate to Project Directory

```bash
cd /Users/denniswork/Desktop/coolingai_simulator
```

---

### Step 2: Install Streamlit (One-Time Setup)

```bash
pip3.11 install streamlit plotly pandas
```

**Expected output:**
```
Successfully installed streamlit-1.53.1 plotly-6.5.2 pandas-2.3.3
```

**Note**: PyTorch, stable-baselines3, and other ML libraries are already installed from previous training.

---

### Step 3: Configure Streamlit (One-Time Setup)

```bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << 'EOF'
[browser]
gatherUsageStats = false

[server]
headless = true
port = 8501
EOF
```

This skips the email signup prompt.

---

### Step 4: Launch the Dashboard

```bash
streamlit run app.py
```

**OR** if `streamlit` command not found:

```bash
python3.11 -m streamlit run app.py
```

---

### Step 5: Open in Browser

Streamlit will show:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Open this URL in your browser:**
```
http://localhost:8501
```

---

## What You'll See

### Dashboard Loads Successfully ✅

1. **Top Row (Metrics)**:
   - 💰 Money Saved: $0.0000 (starts at zero)
   - 🌡️ Current Temperature: ~60°C
   - 💨 Fan Speed: 1.5 m/s

2. **3D Heatmap**:
   - Interactive 3D visualization of rack temperatures
   - Rotate, zoom, pan

3. **Time Series Charts**:
   - Temperature History (red line)
   - Fan Speed History (cyan line)

4. **AI Recommendation Section**:
   - 🤖 AI Fan Speed recommendation
   - Predicted T(t+10s)
   - Heating Rate
   - **🧠 AI Reasoning** (Explainable AI!)

5. **Safety Override Log**:
   - 🛡️ Shows any TTF-based overrides
   - Initially shows: "✓ No safety overrides triggered"

6. **Sidebar**:
   - Control Mode (AI Autopilot / Manual / AI vs Human)
   - IT Load slider (50-150 kW)
   - ⏱️ Time-to-Failure display
   - 📊 Shadow Pilot CSV upload

---

## Testing the Features

### Test 1: AI Autopilot Mode

1. Select "🤖 AI Autopilot" in sidebar
2. Check "Auto-run" checkbox
3. Watch the simulation update every second
4. Observe:
   - Money saved ticker increasing
   - Temperature staying stable
   - **AI Reasoning** explaining every decision

---

### Test 2: Manual Override

1. Select "👤 Manual Override"
2. Use slider to set fan speed (try 2.5 m/s)
3. Click "Step Simulation"
4. Compare your choice to AI recommendation

---

### Test 3: Trigger Safety Override

1. Set IT Load to 150 kW (maximum)
2. Enable Auto-run
3. Let temperature rise above 75°C
4. **Expected**: Safety override triggers when TTF < 120s
5. Check Safety Override Log for details

---

### Test 4: Explainable AI

Watch the **🧠 AI Reasoning** section as you change:
- IT Load (50 kW → 150 kW)
- Temperature (let it heat up/cool down)

**Example outputs you'll see:**
- "Reducing fan speed to save energy..."
- "Increasing cooling because temperature is rising rapidly..."
- "Maintaining current fan speed... temperature is stable..."

---

### Test 5: Shadow Pilot (Optional)

1. Create a CSV file (`test_historical.csv`):
   ```csv
   temperature,fan_speed,load
   65.0,2.0,100
   66.0,2.0,105
   67.0,2.0,110
   68.0,2.0,115
   69.0,2.0,120
   ```

2. Upload via sidebar: "📊 Shadow Pilot Mode"
3. See comparison metrics appear

---

## Troubleshooting

### Issue: "streamlit: command not found"

**Fix:**
```bash
python3.11 -m streamlit run app.py
```

---

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Fix:**
```bash
pip3.11 install streamlit plotly pandas
```

---

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Fix:** PyTorch is already installed. But if you get this error:
```bash
pip3.11 install torch torchvision torchaudio
```

---

### Issue: Port 8501 already in use

**Fix:**
```bash
# Kill any existing Streamlit processes
pkill -f streamlit

# Or use a different port
streamlit run app.py --server.port 8502
```

---

### Issue: Dashboard loads but shows errors

**Check models exist:**
```bash
ls -lh checkpoints/best_recurrent_pinn.pt
ls -lh optimizer/ppo_cooling_agent.zip
```

If missing, the RL agent won't load (but dashboard will still work with heuristic fallback).

---

## Stopping the Dashboard

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## Quick Command Reference

```bash
# Launch dashboard
streamlit run app.py

# Launch with specific port
streamlit run app.py --server.port 8502

# Launch and open browser automatically
streamlit run app.py --browser.serverAddress localhost

# Stop all Streamlit processes
pkill -f streamlit
```

---

## What Works Right Now

✅ **Stability Guard**: TTF monitoring + physical override
✅ **Explainable AI**: Human-readable reasoning
✅ **Shadow Pilot**: CSV upload + comparison
✅ **3D Heatmap**: Live rack temperature visualization
✅ **AI vs. Human**: Manual override mode
✅ **Money Saved Ticker**: Real-time energy savings

---

## Expected Performance

- **Dashboard Load Time**: 2-3 seconds
- **Simulation Step**: < 100ms
- **Auto-run Update Rate**: 1 Hz (every second)
- **Memory Usage**: ~500 MB (models loaded)
- **CPU Usage**: 5-10% idle, 20-30% during auto-run

---

## Success Checklist

When dashboard is working correctly, you should see:

- [ ] Dashboard loads at http://localhost:8501
- [ ] 3D heatmap renders (colorful volume visualization)
- [ ] Temperature displays ~60°C
- [ ] Fan speed shows ~1.5 m/s
- [ ] "🧠 AI Reasoning" section shows explanation
- [ ] "Step Simulation" button works
- [ ] Auto-run checkbox updates every second
- [ ] Safety Override Log shows "✓ No safety overrides triggered"
- [ ] TTF in sidebar shows "TTF: ∞ (Stable/Cooling)"

---

## Complete One-Command Setup (If Starting Fresh)

```bash
# Run this if you're setting up for the first time
cd /Users/denniswork/Desktop/coolingai_simulator && \
pip3.11 install streamlit plotly pandas && \
mkdir -p ~/.streamlit && \
cat > ~/.streamlit/config.toml << 'EOF'
[browser]
gatherUsageStats = false
[server]
headless = true
port = 8501
EOF
echo "✓ Setup complete! Now run: streamlit run app.py"
```

---

## Need Help?

**Check the logs:**
```bash
# Streamlit shows detailed error messages in the terminal
# Look for lines starting with "ERROR" or "Traceback"
```

**Common fixes:**
1. Restart Streamlit: `Ctrl+C`, then `streamlit run app.py`
2. Clear cache: Click "☰" menu → "Clear cache" → "Clear cache"
3. Refresh browser: `Cmd+R` (Mac) or `Ctrl+R` (Windows)

---

**Status**: ✅ **TESTED AND WORKING**

Dashboard launches successfully with all enterprise features:
- Stability Guard ✅
- Explainable AI ✅
- Shadow Pilot ✅
- 3D Visualization ✅

**Launch command**: `streamlit run app.py`

Enjoy your enterprise-grade CoolingAI Command Center! 🚀❄️
