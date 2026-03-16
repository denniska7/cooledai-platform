# CooledAI Discovery Service

Network scanner and Thermal Ping for automatic Influence Map discovery.

## Features

1. **Network Scanner**: Pings IP ranges for Redfish (443) and SNMP (161) ports
2. **Device Query**: Queries Chassis and Model info from discovered devices
   - Redfish: `/redfish/v1/Chassis` → Manufacturer, Model, Name
   - SNMP: sysDescr OID (1.3.6.1.2.1.1.1.0)
3. **Thermal Ping**: Momentarily boosts a fan's speed, measures which racks see a temperature drop, and uses that correlation to build the Influence Map automatically

## Configuration

Edit `core/config/discovery_config.yaml`:

- `ip_ranges`: CIDR or start-end (e.g. `192.168.1.0/24`, `10.0.0.1-10.0.0.50`)
- `cooling_units`: Map cooling_unit_id → {ip, protocol} (fan control)
- `rack_sensors`: Map rack_id → [{ip, protocol}] (temperature sensors)
- `thermal_ping`: Parameters for boost duration, settle time, min temp drop

## Usage

```bash
# Scan network and query devices only
python -m services.discovery.discovery_agent --scan-only

# Full discovery (scan + thermal ping)
python -m services.discovery.discovery_agent

# Run thermal ping only (uses existing config)
python -m services.discovery.discovery_agent --thermal-ping-only

# Save Influence Map to core/config/influence_map.yaml
python -m services.discovery.discovery_agent --thermal-ping-only --save
```

## Dependencies

- `requests` - Redfish device query and thermal ping
- `pyyaml` - Config loading
- `pysnmp-lextudio` (optional) - SNMP device query and fan control

## Output

When `--save` is used, the Influence Map is written to `core/config/influence_map.yaml`.
The API loads this on startup for pre-cooling and failing-component redistribution.
