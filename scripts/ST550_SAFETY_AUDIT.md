# ST550 Safety Audit — Network & Watchdog

## Task 1: Network Continuity

### Primary vs Secondary Route
- **eno1np0** (Router) = primary data path → 192.168.12.x, gw 192.168.12.1
- **enx9a29a6faed33** (Wrench) = secondary, direct link only, no internet

### Verification
| Check | Implementation |
|-------|----------------|
| Default route via eno1np0 | `start_telemetry.sh` sets `ip route add default via 192.168.12.1 dev eno1np0` before launch |
| Telemetry bound to eno1np0 | `st550_telemetry.py` uses `_SourceAddressAdapter(source_ip)` — all HTTP traffic binds to eno1np0's IP |
| Wrench disconnect safe | Yes — traffic is source-bound; unplugging Wrench does not affect telemetry |

### Test Script
```bash
bash scripts/test_connection.sh
```
- Pings 8.8.8.8 via `-I eno1np0`
- Logs SUCCESS or FAIL with timestamp
- Exit 0 = pass, 1 = fail
- Optional: `COOLEDAI_DATA_IFACE=eno1np0 COOLEDAI_PING_TARGET=8.8.8.8`

---

## Task 2: Safe Watchdog

### Previous Issue
- `WatchdogSec=300` (5 min) + no `sd_notify` → systemd killed the agent before slow NVML init completed → reboot loop

### New Configuration (`install.sh` → cooledai-agent.service)
| Setting | Value | Purpose |
|---------|-------|---------|
| WatchdogSec | 600 | 10 min grace for NVML/driver init |
| RestartSec | 30 | Prevents rapid-fire restart loops |
| StartLimitIntervalSec | 0 | No rate limit on restarts during recovery |

### Heartbeat Logic (`cooledai_agent.py`)
- `sd_notify(WATCHDOG=1)` is sent **only after** the first successful telemetry POST
- Before that: no heartbeat → systemd does not expect one during init
- After first POST: heartbeat every loop (~10 s) → watchdog stays fed

### Safe to Enable
```bash
sudo systemctl enable cooledai-agent.service
sudo systemctl start cooledai-agent.service
```
- Agent has 10 minutes to complete init and send first telemetry
- If it fails, 30 s delay before restart; no start limit
