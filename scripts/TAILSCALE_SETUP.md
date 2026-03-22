# Tailscale Setup for Nodes .100 and .101

Access your Pilot (.100) and Control (.101) nodes from anywhere (e.g. a coffee shop) via Tailscale.

## Do I need to shut down the servers?

**No.** Tailscale installs as a background service and does **not** require a reboot or shutdown. Your models (predictive_engine, llama_workload_scheduler, cooledai_agent) keep running. The install is non-disruptive.

## Prerequisites

1. **Auth key** – Generate at [Tailscale Admin → Settings → Keys](https://login.tailscale.com/admin/settings/keys)
   - Use **Reusable** (one key for both nodes)
   - Consider **Disable key expiry** for servers
   - Copy the key (starts with `tskey-auth-`)

2. **Same network** – Run the install from your Mac while on the same LAN as the nodes (192.168.12.x)

## Automated install (recommended)

```bash
cd /path/to/coolingai_simulator
TAILSCALE_AUTHKEY=tskey-auth-xxxxxxxxxxxx expect scripts/install_tailscale.exp
```

Replace `tskey-auth-xxxxxxxxxxxx` with your actual auth key.

## Manual install

### Node .100 (Pilot)

```bash
ssh cooledaiadmin@192.168.12.100
# Enter password when prompted

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Connect (replace with your auth key)
sudo tailscale up --authkey=tskey-auth-xxxxxxxxxxxx --accept-routes

# Verify
tailscale ip -4
tailscale status
exit
```

### Node .101 (Control)

Same steps, but SSH to `192.168.12.101`.

## After install

1. Open [Tailscale Admin → Machines](https://login.tailscale.com/admin/machines)
2. Find your nodes (e.g. `cooledai-srv`, `cooledai-control`) and note their Tailscale IPs (100.x.x.x)
3. From your Mac (with Tailscale running), SSH via Tailscale:

   ```bash
   ssh cooledaiadmin@100.x.x.x   # use the Tailscale IP from the admin console
   ```

4. Optional: enable **MagicDNS** in Tailscale admin so you can use hostnames instead of IPs.

## Troubleshooting

- **"Permission denied"** – Check the auth key is correct and not expired
- **Can't reach node** – Ensure your Mac has Tailscale running and is on the same tailnet
- **Key expired** – Generate a new key with "Disable key expiry" for servers
