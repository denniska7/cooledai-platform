# CooledAI Universal Platform - Architecture Guide

**Hardware-agnostic data center thermal optimization**

## Overview

CooledAI is now a modular, universal platform that can:
- **Ingest** data from any source (CSV, JSON, Modbus, SNMP, Redfish)
- **Normalize** via Hardware Abstraction Layer (HAL)
- **Optimize** via central OptimizationBrain
- **Output** commands regardless of specific hardware

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources (Any Protocol)                   │
│  CSV │ JSON │ Modbus │ SNMP │ Redfish (future)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Adapters (Adapter Pattern)                     │
│  ModbusAdapter │ SNMPAdapter │ DataIngestor (CSV/JSON)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Hardware Abstraction Layer (HAL)                     │
│  BaseNode: thermal_input, power_draw, cooling_output, utilization │
│  ServerNode │ CRAC_Unit │ EnvironmentSensor                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│  SyntheticSensor     │   │  OptimizationBrain   │
│  (Fill missing data) │   │  (Efficiency Gap)    │
└──────────────────────┘   └──────────────────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Optimization Output                           │
│  Thermal Lag │ Over-provisioning │ Oscillation │ Recommendations │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Hardware Abstraction Layer (HAL)

**Location:** `hal/base_node.py`

- **BaseNode**: Abstract base with `thermal_input`, `power_draw`, `cooling_output`, `utilization`
- **ServerNode**: Compute nodes (EPYC, Xeon, GPU)
- **CRAC_Unit**: Industrial chillers, CRAC units
- **EnvironmentSensor**: Inlet/outlet sensors, aisle temp

```python
from hal import ServerNode, CRAC_Unit
node = ServerNode(thermal_input=55, power_draw=250, cooling_output=1800, utilization=45)
```

---

## Task 2: Universal Parser (DataIngestor)

**Location:** `ingestion/data_ingestor.py`

Fuzzy column matching maps vendor-specific names to normalized attributes:
- `Tdie`, `Tctl`, `Temp` → `thermal_input`
- `Power`, `Package Power` → `power_draw`
- `Fan`, `RPM`, `FAN1` → `cooling_output`
- `Utilization`, `Load` → `utilization`

Drop any EPYC/HWiNFO CSV - no code changes required.

```python
from ingestion import DataIngestor
ingestor = DataIngestor()
nodes = ingestor.ingest_csv("Log-013026.csv")
```

---

## Task 3: Protocol Adapters

**Location:** `adapters/`

- **ModbusAdapter**: Industrial chillers, CRAC units (pymodbus)
- **SNMPAdapter**: Network gear, PDUs, BMS (pysnmp)

Both implement `read()` → `List[BaseNode]`. Templates include mock data for testing without hardware.

```python
from adapters import ModbusAdapter, SNMPAdapter
modbus = ModbusAdapter(host="192.168.1.100", port=502)
nodes = modbus.read()  # Returns CRAC_Unit instances
```

---

## Task 4: Optimization Engine

**Location:** `optimization/optimization_brain.py`

**OptimizationBrain** analyzes normalized data and computes **Efficiency Gap**:
1. **Thermal Lag**: Reaction time (power spike → cooling response)
2. **Over-provisioning**: Excess cooling when temp is below target
3. **Oscillation**: Fan/pump hunting (rapid up/down cycling)

```python
from optimization import OptimizationBrain
brain = OptimizationBrain()
gap = brain.analyze(nodes)
print(gap.recommendations)
```

---

## Task 5: Synthetic Sensor (Error Handling)

**Location:** `synthetic/synthetic_sensor.py`

When data is incomplete (e.g., missing fan RPM), **SyntheticSensor** estimates:
- `estimate_cooling(power, temp)` → fan RPM
- `estimate_thermal(power, cooling)` → temperature
- Physics-based heuristic: cooling ∝ power + temp_excess

```python
from synthetic import SyntheticSensor
sensor = SyntheticSensor()
rpm = sensor.estimate_cooling(power_draw=150, thermal_input=65)
```

---

## FastAPI Application

**Location:** `api/main.py`

**Run:** `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ingest/csv` | POST | Upload CSV (universal parser) |
| `/ingest/json` | POST | Ingest JSON node data |
| `/optimize` | GET | Get recommendations (from last ingest) |
| `/optimize` | POST | Analyze inline node data |
| `/adapters` | GET | List available adapters |

---

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy

# Run API
cd coolingai_simulator
uvicorn api.main:app --reload --port 8000

# Ingest CSV
curl -X POST -F "file=@Log-013026.csv" http://localhost:8000/ingest/csv

# Get optimization
curl http://localhost:8000/optimize
```

---

## File Structure

```
coolingai_simulator/
├── hal/                    # Hardware Abstraction Layer
│   ├── __init__.py
│   └── base_node.py       # BaseNode, ServerNode, CRAC_Unit, EnvironmentSensor
├── ingestion/              # Universal Parser
│   ├── __init__.py
│   └── data_ingestor.py   # Fuzzy column matching
├── adapters/               # Protocol Adapters
│   ├── __init__.py
│   ├── base_adapter.py    # Abstract BaseAdapter
│   ├── modbus_adapter.py  # Modbus template
│   └── snmp_adapter.py    # SNMP template
├── optimization/           # Optimization Engine
│   ├── __init__.py
│   └── optimization_brain.py  # Efficiency Gap analysis
├── synthetic/              # Synthetic Sensor
│   ├── __init__.py
│   └── synthetic_sensor.py   # Missing data estimation
├── api/                    # FastAPI Application
│   ├── __init__.py
│   └── main.py
└── UNIVERSAL_PLATFORM.md   # This file
```
