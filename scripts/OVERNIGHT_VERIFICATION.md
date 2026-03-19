# Overnight Run Verification

## ✅ Both Nodes: Same Workload

The `llama_workload_scheduler.py` runs on **both** pilot and control with **identical** workload:

| Feature | How it works |
|---------|--------------|
| **Wall-clock aligned** | Both run at :00, :05, :10 (LIGHT), :00/:30 (HEAVIER), :00 (DIFFICULT) UTC |
| **Same prompts** | `cycle = timestamp // interval` → deterministic prompt index on both nodes |
| **Same intensity** | LIGHT: 80 tokens, HEAVIER: 220 tokens, DIFFICULT: 450 tokens |
| **Dual GPU** | Both use `OLLAMA_SPREAD_URLS=11434,11435` to spread load across 2 GPUs |

---

## ✅ Laptop Shutdown = No Impact

**Your servers are remote machines.** Shutting down your Mac does **not** stop them:

| Machine | Location | Impact of laptop off |
|---------|----------|---------------------|
| **cooledai-srv** (pilot) | Tailscale 100.92.29.44 | None – runs independently |
| **cooledai-control** | 192.168.12.101 | None – runs independently |
| **Your Mac** | Local | Only used for SSH/monitoring |

All processes were started with `nohup ... &`, so they survive SSH disconnect and keep running.

---

## What Could Stop Them

1. **Server power loss** – If cooledai-srv or cooledai-control loses power or reboots
2. **Network outage** – If either server loses internet (telemetry won’t reach the API)
3. **Process crash** – Unlikely; scripts are stable

---

## Quick Status Check (from any device)

```bash
curl -s -H "X-API-Key: sk-osfrVz48r7DCsPwXeAYR4nCF7vhkaRYrN2ahX_2EKgo" \
  https://proactive-creativity-production.up.railway.app/api/v1/nodes/status
```

Look for `"summary": "both_ok"` – both nodes reporting.

---

## Optional: Survive Reboots (systemd)

If you want processes to restart after a server reboot, you can add systemd units. For now, `nohup` is fine for overnight.
