# CooledAI Pilot Node Setup (Predictive Cooling)

Complete setup for the **cooledai-srv** (pilot) node with predictive cooling, dual-GPU workload, telemetry, and fan control.

---

## Hosts

| Node | Role | Host | Notes |
|------|------|------|-------|
| **cooledai-srv** | Pilot (Predictive) | `100.92.29.44` (Tailscale) | Predictive engine, fan control, dual Ollama |
| **cooledai-control** | Control (Baseline) | `192.168.12.101` | Traditional cooling, dual Ollama |

---

## 0. Full repo on the node (recommended)

If `cd ~/coolingai_simulator` fails, clone once on **pilot** (and repeat on **control**):

```bash
cd ~
git clone https://github.com/denniska7/cooledai-platform.git coolingai_simulator
cd ~/coolingai_simulator && git pull origin main
```

Then use `bash scripts/start_ollama_dual_gpu.sh` and `bash scripts/start_st550_paired_workload.sh` from that directory.  
*(Private repo: `git clone git@github.com:denniska7/cooledai-platform.git coolingai_simulator` after adding a deploy key.)*

The sections below describe an older **scp individual scripts to `~`** workflow; a full clone replaces most of that.

---

## 1. Copy Files to Pilot (from your Mac)

From the project root:

```bash
cd /Users/denniswork/Desktop/coolingai_simulator

# All required scripts
scp -o StrictHostKeyChecking=no \
  scripts/st550_telemetry.py scripts/start_telemetry.sh \
  scripts/llama_workload_scheduler.py scripts/predictive_engine.py \
  scripts/cooledai_agent.py scripts/start_ollama_dual_gpu.sh \
  cooledaiadmin@100.92.29.44:~/
```

---

## 2. SSH to Pilot and Run Setup

```bash
ssh -t cooledaiadmin@100.92.29.44
```

Then on the pilot node:

### 2a. Install dependencies (one-time)

```bash
sudo apt-get install -y ipmitool 2>/dev/null || true
sudo python3 -m pip install --break-system-packages nvidia-ml-py3 requests 2>/dev/null || true
```

### 2b. Stop existing processes

```bash
pkill -f llama_workload_scheduler.py 2>/dev/null || true
pkill -f predictive_engine.py 2>/dev/null || true
pkill -f cooledai_agent.py 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true
sleep 3
```

### 2c. Start dual Ollama (one per GPU)

```bash
chmod +x ~/start_ollama_dual_gpu.sh
~/start_ollama_dual_gpu.sh
sleep 5
```

Verify both instances:

```bash
curl -s http://localhost:11434/api/tags | head -1
curl -s http://localhost:11435/api/tags | head -1
```

### 2d. Start telemetry

```bash
sudo bash ~/start_telemetry.sh --node-id ST550-CooledAI-Predictive
```

### 2e. Start workload scheduler (with GPU spread)

```bash
export OLLAMA_SPREAD_URLS="11434,11435"
nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 &
```

### 2f. Start predictive engine (pilot-only)

```bash
nohup python3 ~/predictive_engine.py > /tmp/cooledai_predictive_engine.log 2>&1 &
```

### 2g. Start CooledAI agent (fan control, IPMI)

```bash
sudo nohup python3 ~/cooledai_agent.py \
  --api-url https://proactive-creativity-production.up.railway.app \
  --api-key sk-osfrVz48r7DCsPwXeAYR4nCF7vhkaRYrN2ahX_2EKgo \
  --node-id ST550-CooledAI-Predictive \
  --ipmi-variant lenovo \
  >> /tmp/cooledai_agent.log 2>&1 &
```

---

## 3. Verify

```bash
sleep 3
ps aux | grep -E "ollama|llama_workload|predictive_engine|cooledai_agent|st550_telemetry" | grep -v grep

# Telemetry
tail -5 /var/log/cooledai_telemetry.log 2>/dev/null || tail -5 /tmp/cooledai_telemetry.log

# Predictive engine
tail -5 /tmp/cooledai_predictive_engine.log

# Workload scheduler
tail -5 /tmp/cooledai_llama_workload.log
```

---

## 4. One-Liner (after files are copied)

If you've already copied files and installed deps, you can restart everything with:

```bash
pkill -f llama_workload_scheduler.py 2>/dev/null || true
pkill -f predictive_engine.py 2>/dev/null || true
pkill -f cooledai_agent.py 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true
sleep 3
~/start_ollama_dual_gpu.sh
sleep 5
sudo bash ~/start_telemetry.sh --node-id ST550-CooledAI-Predictive
export OLLAMA_SPREAD_URLS="11434,11435"
nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 &
nohup python3 ~/predictive_engine.py > /tmp/cooledai_predictive_engine.log 2>&1 &
sudo nohup python3 ~/cooledai_agent.py --api-url https://proactive-creativity-production.up.railway.app --api-key sk-osfrVz48r7DCsPwXeAYR4nCF7vhkaRYrN2ahX_2EKgo --node-id ST550-CooledAI-Predictive --ipmi-variant lenovo >> /tmp/cooledai_agent.log 2>&1 &
```

---

## Component Summary

| Component | Purpose |
|-----------|---------|
| **start_ollama_dual_gpu.sh** | Two Ollama instances (ports 11434, 11435), one per GPU |
| **llama_workload_scheduler.py** | Generates inference load, writes to `/tmp/cooledai_inference_load` |
| **predictive_engine.py** | Reads load file, POSTs `inference_load` + `predictive_active` to API |
| **st550_telemetry.py** | GPU/CPU temps, fan RPM → API every 5s |
| **cooledai_agent.py** | Fan control via IPMI, telemetry, config pull |

---

## Control Node: GPU Power Reporting

For the portal to show **both** CooledAI and Control GPU power on the chart, the control node must run an updated `st550_telemetry.py` that includes `power_draw_w` (from pynvml). Copy the latest script:

```bash
scp scripts/st550_telemetry.py cooledaiadmin@192.168.12.101:~/
# Then restart telemetry on control: sudo bash ~/start_telemetry.sh --node-id ST550-Control-Traditional
```

---

## Troubleshooting

- **start_telemetry.sh fails**: If `eno1np0` doesn't exist (e.g. Tailscale-only), the route fix may fail. Telemetry will still run; check `/tmp/cooledai_telemetry.log`.
- **predictive_engine not reporting**: Ensure workload scheduler is running and writing to `/tmp/cooledai_inference_load`. Check `tail /tmp/cooledai_inference_load`.
- **Agent needs sudo**: IPMI/fan control requires root.
