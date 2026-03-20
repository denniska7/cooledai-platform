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
2. **Pull the model on both ports** (required on every new machine — otherwise `/api/generate` returns **404** and logs show `ok=0/2`):

```bash
bash scripts/pull_ollama_model_both_ports.sh
```

3. Paired scheduler:

```bash
cd /path/to/coolingai_simulator
bash scripts/start_st550_paired_workload.sh
```

4. Confirm logs show `WORKLOAD_INTENSITY=1.05`, `ok=2/2` (not `ok=0/2`), and no GIN `404` on `/api/generate`.

## Troubleshooting

### Mac: `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED` (192.168.12.100)

The server was reinstalled or its SSH host key changed. On your **Mac**:

```bash
ssh-keygen -R 192.168.12.100
ssh-keygen -R 192.168.12.101   # if that one errors too
```

Then SSH again and accept the new fingerprint. **Only do this if** you expect the key change (same machine, OS reinstall, etc.) — not on untrusted networks.

### Ollama `404` on `POST /api/generate` / `[LIGHT] done. ok=0/2`

The model name in `OLLAMA_MODEL` (default `llama3`) is **not** available on that Ollama process. Run `bash scripts/pull_ollama_model_both_ports.sh` on that node (pulls on **11434 and 11435**).

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
