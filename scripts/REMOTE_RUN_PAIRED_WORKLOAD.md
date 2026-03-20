# Run paired workload on both ST550s (from your Mac)

## Security first

- **Never commit SSH passwords or API keys to GitHub.** Run commands from your laptop and type passwords only when `ssh` / `sudo` prompt you—or use **SSH keys**.
- If a password was shared in chat or email, **change it on the server** and prefer `ssh-copy-id` for automation.

## One-time: passwordless SSH

From your Mac (repeat for each host):

```bash
ssh-copy-id cooledaiadmin@100.92.29.44          # pilot (Tailscale — example)
ssh-copy-id cooledaiadmin@192.168.12.101        # control (LAN — example)
```

Use your real hostnames/IPs from `COOLEDAI_NODE_SETUP.md`.

## Repo on each server

Both nodes need the same checkout so these paths exist:

```text
~/coolingai_simulator/scripts/start_st550_paired_workload.sh
~/coolingai_simulator/scripts/start_ollama_dual_gpu.sh
~/coolingai_simulator/scripts/llama_workload_scheduler.py
```

If the directory name differs, set `COOLEDAI_REMOTE_DIR` when running the helper scripts.

## Automated (recommended)

From **project root** on your Mac, after `git pull` on both servers:

```bash
# 1) Dual Ollama on both GPUs (kills existing ollama serve)
./scripts/run_dual_ollama_on_both_st550.sh

# 2) Same model on both (run on EACH server over SSH, or see GPU_LOAD_BALANCE.md)
#    ollama pull llama3

# 3) Paired +5% workload on both nodes
./scripts/run_paired_workload_on_both_st550.sh
```

### Overrides (LAN IPs instead of Tailscale, etc.)

```bash
export COOLEDAI_PILOT_SSH='cooledaiadmin@192.168.12.100'
export COOLEDAI_CONTROL_SSH='cooledaiadmin@192.168.12.101'
export COOLEDAI_REMOTE_DIR='coolingai_simulator'
./scripts/run_paired_workload_on_both_st550.sh
```

## Manual (copy-paste on each node over SSH)

SSH into **each** server, then:

```bash
cd ~/coolingai_simulator
bash scripts/start_ollama_dual_gpu.sh
# pull model(s) as needed
bash scripts/start_st550_paired_workload.sh
tail -f /tmp/cooledai_llama_workload.log
```

## Verify

- `scripts/ST550_PAIRED_WORKLOAD.md` — parity checklist  
- `scripts/TELEMETRY_LOG_ACCESS.md` — API / optimize-control checks  

## Why we don’t push passwords to GitHub

Scripts in this repo are **public or shared**. Embedding credentials would expose your infrastructure. Use SSH keys + optional macOS Keychain / `ssh-agent`.
