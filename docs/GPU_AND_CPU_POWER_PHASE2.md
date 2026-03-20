# GPU power caps & CPU idle coordination (Phase 2)

## GPU dynamic power limit (implemented)

CooledAI can **proactively cap NVIDIA TDP** with `nvidia-smi -pl` so the GPU stays in a **high-efficiency band** instead of **hard thermal throttling** (steep perf drop, poor Joules per useful work).

### How it works

- **Cool** (≤ `COOLEDAI_GPU_PL_TEMP_FULL_C`, default 62°C): restore **default** board limit.
- **Warm** (between full and `COOLEDAI_GPU_PL_TEMP_SOFT_C`, default 72°C): **blend** toward a mid cap.
- **Hot** (approaching `COOLEDAI_GPU_PL_TEMP_HARD_C`, default 82°C): cap toward a **floor** (~55% of default, not below driver min).

Per-GPU curves use **each GPU’s own temperature** and **driver-reported** min/default/max from `nvidia-smi --query-gpu=power.*`.

### Enable on the agent (pilot / any NVIDIA node)

Requires **root** (or capabilities) for `nvidia-smi -pl` on most drivers.

```bash
sudo env COOLEDAI_GPU_POWER_MGMT=1 python3 scripts/cooledai_agent.py \
  --api-url ... --api-key ... --node-id ... --gpu-power-management
```

Environment (optional tuning):

| Variable | Default | Meaning |
|----------|---------|---------|
| `COOLEDAI_GPU_PL_TEMP_FULL_C` | 62 | Below this → full default TDP |
| `COOLEDAI_GPU_PL_TEMP_SOFT_C` | 72 | Start rolling off toward mid cap |
| `COOLEDAI_GPU_PL_TEMP_HARD_C` | 82 | Approach minimum practical cap |
| `COOLEDAI_GPU_PL_MIN_DELTA_W` | 5 | Ignore changes smaller than this (noise) |
| `COOLEDAI_GPU_PL_MIN_INTERVAL_S` | 8 | Min seconds between `-pl` attempts |

**Note:** ST550 + Quadro P2000 run cooler than H100; adjust bands per ASHRAE / vendor guidance. The narrative “42°C” in business copy may refer to a **different sensor** or marketing example — **use junction / GPU edge temps** from NVML, not inlet air.

### Shutdown

On exit, the agent **best-effort resets** each GPU to **default** limit.

### DCGM / fleet policy (next)

Same curve can be driven from **DCGM** or cloud policy later (`target_power_w` in API). Current path is **local nvidia-smi** for zero extra dependencies.

---

## CPU P-state / C-state coordination (scaffold only)

**Not auto-applied by default.** Changing global idle states can **hurt latency-sensitive** workloads and **conflicts** with orchestrators.

### Intended direction

- When **GPU utilization is high** and **CPU is thermally / load idle**, *hint* deeper package C-states.
- **Preemptively** relax before expected CPU bursts (needs job metadata or heuristics).

### Safe first steps

1. **Measure only** — log CPU %idle vs GPU util (already in telemetry).
2. **Opt-in script** (admin): `cpupower frequency-set` / `intel_pstate` — per-site playbook in `scripts/cpu_idle_coordination_stub.sh`.
3. **Kernel / DC-scoped** tuning (cgroups, `idle=poll` avoidance) with customer SRE sign-off.

Potential savings (1.6–2.4% bill) require **validated** idle policies per CPU vendor — track under a separate **Phase 2b** milestone after GPU PL is stable in pilot.

### Stub script

`scripts/cpu_idle_coordination_stub.sh` — documents checks; does **not** change hardware unless uncommented.

---

**Note:** Governor temperature thresholds auto-calibrate via `gpu_power_governor.py`'s own calibration mechanism — independent of the fan-side `ThermalCalibrator` in `core/optimization/thermal_calibrator.py`. See `docs/THERMAL_CALIBRATION.md` for fan threshold auto-calibration details.

## Code

- `core/optimization/gpu_power_governor.py` — curve + `nvidia-smi` helpers
- `core/optimization/thermal_calibrator.py` — fan threshold auto-calibration (separate from GPU PL)
- `scripts/cooledai_agent.py` — `--gpu-power-management` + env gate
- `tests/test_gpu_power_governor.py`
- `tests/test_thermal_calibrator.py`
