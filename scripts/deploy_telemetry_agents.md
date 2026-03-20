# Deploy ST550 Comparative Demo

Run one telemetry agent per server. Control node (.101) uses traditional cooling;
Pilot node (.100) runs predictive cooling. Both report to cooledai.com portal.

## Data Port vs Wrench — IMPORTANT

- **Data port (eno1np0)** → Router / WiFi → Internet. Use this for SSH and telemetry.
- **Wrench port (enx…)** → Direct link to laptop. When you unplug it, you lose SSH if you were connected via it.

**To survive Wrench unplug:** Run `setup_data_port_only.sh` once on each server. Then:
- **SSH via data port:** `ssh cooledaiadmin@192.168.12.100` from a machine on the same WiFi (192.168.12.x)
- Unplug the Wrench anytime; telemetry and workload keep running.

## Prerequisites

- SSH access to both servers (via data port or Wrench)
- Python 3 with pip on each server

## One-Time Setup: Data Port Routing (Both Servers)

Run while connected (Wrench or data port):

```bash
scp scripts/setup_data_port_only.sh scripts/enforce_st550_route.sh scripts/cooledai-route.service scripts/cooledai-route.timer cooledaiadmin@192.168.12.100:~/scripts/
ssh cooledaiadmin@192.168.12.100
sudo bash ~/scripts/setup_data_port_only.sh
# Repeat for .101
```

## Server .100 (Pilot Node — ST550-CooledAI-Predictive)

```bash
# 1. Copy scripts (from laptop on same WiFi)
scp scripts/st550_telemetry.py scripts/start_telemetry.sh cooledaiadmin@192.168.12.100:~/

# 2. SSH in via DATA PORT (192.168.12.100 — not Wrench)
ssh cooledaiadmin@192.168.12.100

# 3. Install dependencies (one-time, if needed)
sudo apt-get install -y ipmitool 2>/dev/null || true
sudo python3 -m pip install --break-system-packages nvidia-ml-py3 requests

# 4. Start telemetry (sudo + ipmitool → CPU temp + chassis fan RPM in API; not GPU-only)
sudo bash ~/start_telemetry.sh --node-id ST550-CooledAI-Predictive

# 5. Verify
tail -20 /var/log/cooledai_telemetry.log
# Should see: [telemetry] POST ... Status: 200

# 6. (Optional) Run workload simulator + predictive engine
# Workload: python3 ~/workload_sim.py
# Predictive: python3 ~/predictive_engine.py &
```

## Server .101 (Control Node — ST550-Control-Traditional)

```bash
# 1. Copy scripts
scp scripts/st550_telemetry.py scripts/start_telemetry.sh cooledaiadmin@192.168.12.101:~/

# 2. SSH in and run
ssh cooledaiadmin@192.168.12.101

# 3. Install dependencies (one-time)
sudo apt-get install -y ipmitool 2>/dev/null || true
sudo python3 -m pip install --break-system-packages nvidia-ml-py3 requests

# 4. Start telemetry (**sudo** required so `ipmitool sdr` can read chassis fan RPM + CPU temp)
sudo bash ~/start_telemetry.sh --node-id ST550-Control-Traditional

# 5. Verify
tail -20 /var/log/cooledai_telemetry.log
```

## API & Portal

- API: https://proactive-creativity-production.up.railway.app
- Portal: https://cooledai.com (or your frontend URL)
- Both agents use the same API key and report to the same tenant.

## Workload Simulator (Both Servers)

Run on BOTH servers to generate comparable inference load:

```bash
# Copy workload script
scp scripts/workload_sim.py cooledaiadmin@192.168.12.100:~/
scp scripts/workload_sim.py cooledaiadmin@192.168.12.101:~/

# On each server (ensure Ollama + llama3 are running):
python3 ~/workload_sim.py
```

## Predictive Engine (Pilot Node Only)

On .100 only, run the predictive engine alongside telemetry:

```bash
scp scripts/predictive_engine.py cooledaiadmin@192.168.12.100:~/
python3 ~/predictive_engine.py &
```

## Troubleshooting

- **Ping fails**: Script continues with WARN (ICMP may be blocked). First POST will verify connectivity.
- **pip fails**: Use `--break-system-packages` on Ubuntu 24.04.
- **No GPUs on .101**: The script requires NVIDIA GPUs (pynvml). If .101 has different hardware, a different collector may be needed.
