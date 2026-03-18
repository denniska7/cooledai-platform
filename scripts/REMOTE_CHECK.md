# Remote node check (when you're not on the same WiFi)

When you're away from the office and can't SSH to the ST550 nodes, you can still see if **Pilot** (.100) and **Control** (.101) are sending telemetry to the cloud.

## One command (from anywhere)

Use the same API key as your agents (e.g. the one in `st550_telemetry.py` or in your env):

```bash
curl -s -H "X-API-Key: YOUR_API_KEY" \
  https://proactive-creativity-production.up.railway.app/api/v1/nodes/status
```

Replace `YOUR_API_KEY` with your real key (e.g. from `COOLEDAI_API_KEY` or the key in the deploy scripts).

## Example response

**Both nodes sending recently:**

```json
{
  "summary": "both_ok",
  "pilot": {
    "node_id": "ST550-CooledAI-Predictive",
    "last_seen_iso": "2026-03-19T12:34:56.789+00:00",
    "last_seen_s_ago": 8.2,
    "status": "ok"
  },
  "baseline": {
    "node_id": "ST550-Control-Traditional",
    "last_seen_iso": "2026-03-19T12:34:58.123+00:00",
    "last_seen_s_ago": 6.5,
    "status": "ok"
  },
  "checked_at_iso": "2026-03-19T12:35:05.000+00:00"
}
```

**One or both nodes not sending:**

- `summary`: `both_stale`, `pilot_ok_baseline_stale`, `no_data`, etc.
- `status` per node: `ok` (last seen &lt; 2 min), `stale` (≥ 2 min), or `no_data`.

## Optional: check from the portal

If you're logged in at the CooledAI portal, the dashboard already shows live data. If both nodes appear and the charts update, they're logging. The `/api/v1/nodes/status` endpoint is for a quick curl/script check without opening the app.
