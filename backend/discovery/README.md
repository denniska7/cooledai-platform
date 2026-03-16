# CooledAI Discovery Agent

Analyzes **raw, unmapped Modbus/SNMP telemetry** over a **10-minute window** and uses an **LLM** to compare patterns against known equipment profiles (Liebert, Stulz, Schneider). Returns a **Confidence Score** and **Suggested Mapping Schema** so users don't have to manually label points during onboarding.

## Usage

### Programmatic

```python
from backend.discovery import DiscoveryAgent, DiscoveryResult

agent = DiscoveryAgent(window_seconds=600)

# Option A: Add samples over time (e.g. from a collector loop)
agent.add_sample("modbus", "modbus:4096", "4096", 22.5)
agent.add_sample("modbus", "modbus:4097", "4097", 1200.0)
# ... repeat for ~10 minutes ...

# Option B: Ingest a full window at once
from backend.discovery import RawTelemetryWindow, RawTelemetryPoint
window = RawTelemetryWindow(
    protocol="modbus",
    points=[
        RawTelemetryPoint("modbus:4096", "modbus", "4096", series=[(t1, 22.0), (t2, 22.1), ...]),
        # ...
    ],
)
agent.ingest_window(window)

# Run discovery
result = agent.discover()
print(result.confidence_score)       # 0.0 - 1.0
print(result.suggested_vendor)      # e.g. "Liebert", "Stulz", "Schneider"
print(result.suggested_mapping_schema.point_to_attribute)  # point_id -> thermal_input, etc.

# Use mapping with adapters
adapter_config = result.suggested_mapping_schema.to_adapter_format("modbus")
# -> {"register_map": {"thermal_input": (4096, 1.0, ""), ...}}
```

### API

- **POST /discovery/sample** – Add one raw sample (`protocol`, `point_id`, `address`, `value`, optional `timestamp`).
- **POST /discovery/window** – Ingest a full window (body: `protocol`, `points` with `point_id`, `address`, `series`).
- **POST /discovery/run** – Run discovery; returns `confidence_score`, `suggested_vendor`, `suggested_mapping_schema`, `adapter_format_modbus` / `adapter_format_snmp`, `message`.
- **POST /discovery/clear** – Clear the window for a new run.

## LLM Setup (optional)

For **vendor-specific** discovery (Liebert vs Stulz vs Schneider), set one of:

- **OPENAI_API_KEY** – uses `gpt-4o-mini` (or `COOLEDAI_LLM_MODEL`).
- **ANTHROPIC_API_KEY** – uses Claude (e.g. `claude-3-5-haiku`).

Without an API key, the agent uses a **rule-based fallback** (value-range heuristic) and returns a generic mapping and lower confidence.

```bash
pip install openai    # or anthropic
export OPENAI_API_KEY=sk-...
```

## Equipment Profiles

Built-in profiles (see `equipment_profiles.py`):

| Vendor    | Protocol | Description                    |
|-----------|----------|--------------------------------|
| Liebert   | Modbus   | Vertiv CRAC/InRow registers   |
| Stulz     | Modbus   | Stulz CRAC E2/Cybero          |
| Schneider | Modbus   | APC InRow Modbus              |
| Schneider | SNMP     | Cisco/SNMP BMS                |
| APC       | SNMP     | APC PDU/InRow SNMP            |

## Schemas

- **RawTelemetryPoint** – `point_id`, `protocol`, `address`, `series: [(ts, value), ...]`.
- **MappingSchema** – `point_to_attribute: { point_id: "thermal_input" | "power_draw" | ... }`, optional `scale_factors`.
- **DiscoveryResult** – `confidence_score`, `suggested_vendor`, `suggested_mapping_schema`, `message`, `raw_metrics`.

Normalized attributes match `BaseNode`: `thermal_input`, `power_draw`, `cooling_output`, `utilization`, `ambient_inlet_temp`.
