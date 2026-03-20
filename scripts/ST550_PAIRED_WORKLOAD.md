# ST550 Pilot vs Control — Identical Workload

## Why averages differ (e.g. 16.7W vs 9.6W)

The scheduler is **deterministic** (same UTC boundaries, same prompts), but GPU power still diverges if:

| Cause | Fix |
|--------|-----|
| Only one GPU doing inference (`OLLAMA_SPREAD_URLS` unset) | Set `OLLAMA_SPREAD_URLS=11434,11435` on **both** nodes |
| Scheduler not running on one host | Run `start_st550_paired_workload.sh` on **both** |
| Different `OLLAMA_MODEL` / model not pulled on one side | Same model name + `ollama pull` on both |
| Clock skew | Use NTP; tiers align to wall-clock boundaries |
| Different `COOLEDAI_WORKLOAD_INTENSITY` | Use the same env on both (default **1.05** = +5% vs older baselines) |

## Recommended startup (both nodes)

1. Dual Ollama: `bash scripts/start_ollama_dual_gpu.sh` (see `GPU_LOAD_BALANCE.md`).
2. Paired scheduler:

```bash
cd /path/to/coolingai_simulator
bash scripts/start_st550_paired_workload.sh
```

3. Confirm logs show `WORKLOAD_INTENSITY=1.05` and `OLLAMA_SPREAD_URLS=2 URLs` (or similar).

## +5% workload

`COOLEDAI_WORKLOAD_INTENSITY` scales `num_predict` for LIGHT / HEAVIER / DIFFICULT tiers (default **1.05**). Set to `1.0` to revert to pre-bump token counts.

## API policy floor (deploy verification)

After deploying the API that returns `policy_soft_floor_rpm`, call `/api/v1/optimize/control` and confirm:

- `policy_soft_floor_rpm` — active-compute / spike-hold target (often ~2500 at rated 7000).
- `policy_floor_forced_after_layers` — `true` if mechanical slew was overridden to meet the floor.
- `policy_capacity_rpm` — capacity used for policy math (rated max, not stuck-low tach).

If these look correct but **fan tach** barely moves, suspect **IPMI / BIOS / PWM limits** on the host, not the optimizer.

See also: `TELEMETRY_LOG_ACCESS.md` (curl example).

## Run both nodes from your Mac (SSH)

See **`REMOTE_RUN_PAIRED_WORKLOAD.md`** and `run_paired_workload_on_both_st550.sh` (requires SSH keys, no passwords in git).
