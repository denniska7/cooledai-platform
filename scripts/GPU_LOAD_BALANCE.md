# GPU Load Balance — Fix Uneven Temperatures

## Problem

On the control node (Node 101), one GPU runs at ~63°C while the other stays at ~30°C. This is **not a telemetry bug** — the telemetry correctly reports per-GPU temps. The cause is **Ollama putting all inference on GPU 0** when using the default single-instance setup.

## Solution: Spread Load Across Both GPUs

Run two Ollama instances (one per GPU) and configure the workload scheduler to send requests to each.

### 1. Start dual Ollama (on each node)

```bash
# From project root
./scripts/start_ollama_dual_gpu.sh
```

Or manually:

```bash
pkill -f "ollama serve" 2>/dev/null || true
sleep 2
CUDA_VISIBLE_DEVICES=0 ollama serve &
sleep 3
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve &
```

> **Note**: Ollama does not support `--port`; use `OLLAMA_HOST=127.0.0.1:11435` for the second instance.

### 2. Load the model on both instances

```bash
ollama pull llama3
# Second instance will load on first request, or:
curl -X POST http://localhost:11435/api/pull -d '{"name":"llama3"}'
```

### 3. Run workload scheduler with spread

```bash
OLLAMA_SPREAD_URLS="11434,11435" python3 ~/llama_workload_scheduler.py
```

Or add to your start script:

```bash
export OLLAMA_SPREAD_URLS="11434,11435"
nohup python3 ~/llama_workload_scheduler.py > /tmp/cooledai_llama_workload.log 2>&1 &
```

### 4. Control node: system Ollama

If Ollama runs as a system service (user `ollama`), `pkill` will fail with "Operation not permitted". Stop it first:

```bash
sudo systemctl stop ollama
# or
sudo pkill -f ollama
sleep 3
```

Then run `start_ollama_dual_gpu.sh`. If you prefer to keep the system Ollama on 11434 and only add a second instance on 11435, start only the second:

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve &
```

### 5. Deploy to both nodes

Update your deploy scripts to:
1. Stop any existing Ollama (use `sudo systemctl stop ollama` on Control if needed)
2. Start dual Ollama (`start_ollama_dual_gpu.sh`) before the workload scheduler
3. Set `OLLAMA_SPREAD_URLS=11434,11435` when starting the scheduler

## Telemetry Verification

- **CPU temp**: From `cooledai_agent` → sysfs thermal zones (`/sys/class/thermal/*/temp`) or lm-sensors. Sent as `node_id/cpu` with `temperature_c` = max of all zones. API stores as `cpu_temp_c`.
- **GPU temp**: From nvidia-smi per GPU. Agent sends `node_id/gpu{i}` with `temperature_c`. API computes `avg_gpu_temp_c` for comparison.
- **Fan RPM**: From IPMI (`ipmitool sdr`) or `fan_rpms` dict. API stores as `fan_rpm`.
