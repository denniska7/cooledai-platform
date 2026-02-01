"""
CooledAI Universal API - FastAPI Application

Hardware-agnostic thermal optimization API. Ingests data from any source
(CSV, JSON, Modbus, SNMP) and outputs optimization commands.

Includes System State Machine: OPTIMIZING, GUARD_MODE, MANUAL_OVERRIDE.
GUARD_MODE triggers when sensor data is missing, AI confidence is low,
or physical anomaly detected. In GUARD_MODE, prioritizes thermal stability
over efficiency (100% cooling).

Endpoints:
- POST /ingest/csv - Ingest CSV file (universal parser with fuzzy matching)
- POST /ingest/json - Ingest JSON data
- GET /optimize - Get optimization recommendations (from last ingested data)
- POST /optimize - Analyze provided node data and return Efficiency Gap
- GET /state - Get current system state
- POST /state - Set system state (e.g., MANUAL_OVERRIDE)
- GET /health - Health check
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import io

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import CooledAI components
from core.hal.base_node import BaseNode, ServerNode
from core.ingestion.data_ingestor import DataIngestor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass  # DataIngestor used for type hints
from core.optimization.optimization_brain import OptimizationBrain, EfficiencyGap
from core.synthetic.synthetic_sensor import SyntheticSensor

# Initialize FastAPI app
app = FastAPI(
    title="CooledAI Universal Thermal Optimization API",
    description="Hardware-agnostic platform for data center thermal optimization",
    version="1.0.0",
)

# CORS - allow requests from CooledAI production and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cooledai.com",
        "https://www.cooledai.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- System State Machine ---

class SystemState(str, Enum):
    """System operating states."""
    OPTIMIZING = "OPTIMIZING"       # Normal AI optimization (efficiency + stability)
    GUARD_MODE = "GUARD_MODE"       # Prioritize thermal stability (100% cooling)
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"  # Human control, AI bypassed


class SystemStateMachine:
    """
    System State Machine for thermal control.
    
    OPTIMIZING: AI optimizes for efficiency within safety bounds.
    GUARD_MODE: Triggered by missing data, low confidence, or physical anomaly.
                Stops efficiency optimization, prioritizes absolute thermal stability.
    MANUAL_OVERRIDE: User has taken control; AI recommendations ignored.
    """
    
    # Thresholds for GUARD_MODE triggers
    LOW_CONFIDENCE_THRESHOLD = 60.0   # efficiency_score below this = low confidence
    MISSING_DATA_THRESHOLD = 0.3      # >30% of nodes with missing critical data
    
    def __init__(self):
        self._state = SystemState.OPTIMIZING
        self._guard_mode_reason: str = ""
    
    @property
    def state(self) -> SystemState:
        return self._state
    
    def set_state(self, new_state: SystemState, reason: str = "") -> None:
        """Set state (for MANUAL_OVERRIDE or explicit transitions)."""
        self._state = new_state
        self._guard_mode_reason = reason
    
    def evaluate_guard_mode(
        self,
        nodes: List[BaseNode],
        gap: Optional[EfficiencyGap],
        synthetic_sensor: SyntheticSensor,
        ingestor: Optional["DataIngestor"] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate if GUARD_MODE should be triggered.
        
        Triggers when:
        1. Stale data (Communication Timeout - data older than 5 seconds)
        2. Any sensor data is missing (>threshold of nodes with missing critical data)
        3. AI confidence score is low (efficiency_score < threshold)
        4. Physical anomaly detected (e.g., fan 0 RPM but temp rising)
        
        Returns:
            (should_enter_guard_mode: bool, reason: str)
        """
        if not nodes:
            return True, "No sensor data available"
        
        # 0. Stale Data Protection (Timestamp Validation)
        if ingestor is not None:
            is_stale, stale_reason = ingestor.check_stale_data(nodes)
            if is_stale:
                return True, stale_reason
        
        # 1. Check for missing sensor data (we had to use synthetic estimation)
        missing_count = 0
        for node in nodes:
            # Critical: thermal and cooling must be present for safe optimization
            # If we used synthetic to fill, real sensor data was missing
            has_synthetic = node.raw_data.get("_synthetic_thermal") or node.raw_data.get("_synthetic_cooling")
            has_invalid = (node.thermal_input <= 0) or (node.cooling_output <= 0 and node.power_draw > 0)
            if has_synthetic or has_invalid:
                missing_count += 1
        missing_ratio = missing_count / len(nodes)
        if missing_ratio > self.MISSING_DATA_THRESHOLD:
            return True, (
                f"Missing sensor data: {missing_count}/{len(nodes)} nodes "
                f"({missing_ratio*100:.0f}%) have incomplete thermal/cooling readings"
            )
        
        # 2. Check AI confidence (efficiency score)
        if gap is not None and gap.efficiency_score < self.LOW_CONFIDENCE_THRESHOLD:
            return True, (
                f"Low AI confidence: efficiency_score={gap.efficiency_score:.1f} "
                f"(threshold {self.LOW_CONFIDENCE_THRESHOLD})"
            )
        
        # 3. Check for physical anomaly (Synthetic Sensor detection)
        anomaly_detected, anomaly_reason = synthetic_sensor.detect_physical_anomaly(nodes)
        if anomaly_detected:
            return True, f"Physical anomaly: {anomaly_reason}"
        
        return False, ""
    
    def apply_state_to_gap(self, gap: EfficiencyGap) -> EfficiencyGap:
        """
        Apply current state to optimization result.
        
        In GUARD_MODE: Override to 100% cooling (thermal stability priority).
        In MANUAL_OVERRIDE: Return gap but flag that AI is bypassed.
        """
        if self._state == SystemState.GUARD_MODE:
            # Stop efficiency optimization - prioritize absolute thermal stability
            gap.recommended_cooling_delta = 1.0  # 100% cooling (Fail-Safe High)
            gap.recommendations.insert(
                0,
                f"GUARD_MODE: {self._guard_mode_reason} "
                "Prioritizing thermal stability over efficiency. Cooling set to 100%.",
            )
            gap.raw_metrics["system_state"] = "GUARD_MODE"
            gap.raw_metrics["guard_mode_reason"] = self._guard_mode_reason
        elif self._state == SystemState.MANUAL_OVERRIDE:
            gap.recommendations.insert(
                0,
                "MANUAL_OVERRIDE: Human control active. AI recommendations ignored.",
            )
            gap.raw_metrics["system_state"] = "MANUAL_OVERRIDE"
        
        return gap


# Global state
_last_nodes: List[BaseNode] = []
_ingestor = DataIngestor(node_type=ServerNode)
_brain = OptimizationBrain()
_synthetic_sensor = SyntheticSensor()
_state_machine = SystemStateMachine()


# --- Pydantic Models for API ---

class NodeInput(BaseModel):
    """Input model for node data (JSON ingest)."""
    thermal_input: float = 0.0
    power_draw: float = 0.0
    cooling_output: float = 0.0
    utilization: float = 0.0
    node_id: Optional[str] = None


class OptimizationResponse(BaseModel):
    """Response model for optimization analysis."""
    thermal_lag_seconds: float
    over_provisioning_ratio: float
    oscillation_ratio: float
    efficiency_score: float
    recommended_cooling_delta: float
    recommendations: List[str]
    raw_metrics: Dict[str, Any] = {}


# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "ok"}


@app.get("/simulated-metrics")
async def simulated_metrics():
    """
    Simulated real-time metrics for System Pulse and Portal visualization.
    Returns synthetic thermal/optimization data. Respects simulation mode from Command Center.
    """
    import time
    import math
    t = time.time()
    eff = 87.0 + 5 * math.sin(t * 0.3)
    sim = _get_simulation_metrics(t, eff)
    return {
        "efficiency_score": sim["efficiency_score"],
        "thermal_lag_seconds": 2.1 + 0.5 * math.sin(t * 0.2),
        "state": sim["state"],
        "nodes_active": 24,
        "cooling_delta": 0.12,
        "timestamp": t,
        "cpu_temp_avg": sim["cpu_temp_avg"],
        "delta_t_inlet": sim["delta_t_inlet"],
        "delta_t_outlet": sim["delta_t_outlet"],
        "power_draw_kw": sim["power_draw_kw"],
        "predicted_load_t10": sim["predicted_load_t10"],
        "current_capacity": sim["current_capacity"],
        "carbon_offset_kg": round(1240 + (int(t) % 3600) * 0.3, 0),
        "opex_reclaimed_usd": round(12400 + (int(t) % 3600) * 1.2, 0),
        "simulation_critical": sim["simulation_critical"],
    }


# --- Simulation Control (Command Center) ---
class SimulationMode(str, Enum):
    STEADY = "STEADY"
    GPU_SPIKE = "GPU_SPIKE"
    CHILLER_FAILURE = "CHILLER_FAILURE"

_simulation_mode: SimulationMode = SimulationMode.STEADY
_simulation_start_time: float = 0.0


def _get_simulation_metrics(t: float, eff: float) -> dict:
    """Apply simulation mode to metrics."""
    global _simulation_mode, _simulation_start_time
    elapsed = t - _simulation_start_time if _simulation_start_time else 0

    if _simulation_mode == SimulationMode.STEADY:
        return {
            "cpu_temp_avg": round(38 + (100 - eff) * 0.2 + (t % 10) * 0.1, 1),
            "delta_t_inlet": round(18 + (t % 5) * 0.2, 1),
            "delta_t_outlet": round(24 + (t % 5) * 0.3, 1),
            "power_draw_kw": round(120 + (t % 8) * 2, 1),
            "predicted_load_t10": round(75 + (t % 6) * 1, 1),
            "current_capacity": 100,
            "efficiency_score": min(95, eff + 5),
            "state": "OPTIMIZING",
            "simulation_critical": False,
        }
    elif _simulation_mode == SimulationMode.GPU_SPIKE:
        return {
            "cpu_temp_avg": 82.0,
            "delta_t_inlet": round(22 + elapsed * 0.5, 1),
            "delta_t_outlet": round(32 + elapsed * 0.8, 1),
            "power_draw_kw": round(180 + elapsed * 2, 1),
            "predicted_load_t10": round(95 + min(elapsed * 2, 5), 1),
            "current_capacity": 100,
            "efficiency_score": max(30, 85 - elapsed * 5),
            "state": "GUARD_MODE",
            "simulation_critical": True,
        }
    else:  # CHILLER_FAILURE
        ambient_climb = min(15, elapsed * 1.2)
        return {
            "cpu_temp_avg": round(45 + ambient_climb + (t % 3) * 0.5, 1),
            "delta_t_inlet": round(20 + ambient_climb * 0.8, 1),
            "delta_t_outlet": round(28 + ambient_climb * 1.2, 1),
            "power_draw_kw": 120.0,
            "predicted_load_t10": round(85 + ambient_climb, 1),
            "current_capacity": 100,
            "efficiency_score": max(40, 90 - ambient_climb * 3),
            "state": "GUARD_MODE",
            "simulation_critical": True,
        }


# Simulated AI decision log for portal System Log
_AI_LOG_ENTRIES = [
    "Adjusting Fan Array 04 for predicted GPU spike in Rack B12.",
    "Pre-cooling Rack A07 before scheduled training job.",
    "Reducing chiller setpoint by 0.5°C—efficiency within bounds.",
    "Ramping CRAC 02—inlet temp trending above threshold.",
    "Holding steady—thermal envelope stable for next 15 min.",
    "Increasing airflow to Zone 3—power draw spike detected.",
    "Deferring optimization—manual override active.",
    "Applying guard mode—sensor anomaly in Rack C04.",
]


class SimulationTrigger(BaseModel):
    mode: str  # gpu_spike | chiller_failure | reset


@app.post("/admin/simulation-control/trigger")
async def trigger_simulation(body: SimulationTrigger):
    """Command Center: Trigger GPU spike, chiller failure, or reset."""
    global _simulation_mode, _simulation_start_time
    import time
    t = time.time()
    mode = body.mode.lower()
    if mode == "gpu_spike":
        _simulation_mode = SimulationMode.GPU_SPIKE
        _simulation_start_time = t
        return {"status": "ok", "mode": "GPU_SPIKE", "message": "GPU spike simulated. Temp 45°C → 82°C."}
    elif mode == "chiller_failure":
        _simulation_mode = SimulationMode.CHILLER_FAILURE
        _simulation_start_time = t
        return {"status": "ok", "mode": "CHILLER_FAILURE", "message": "Chiller failure simulated. Ambient climbing."}
    elif mode == "reset":
        _simulation_mode = SimulationMode.STEADY
        _simulation_start_time = 0.0
        return {"status": "ok", "mode": "STEADY", "message": "Environment reset to steady-state."}
    raise HTTPException(status_code=400, detail="Invalid mode. Use gpu_spike, chiller_failure, or reset")


@app.get("/admin/simulation-control/status")
async def simulation_status():
    """Command Center: Get current simulation mode."""
    return {"mode": _simulation_mode.value}


@app.get("/simulated-metrics/log")
async def simulated_ai_log():
    """Returns AI decision log entries. Critical message when simulation chaos active."""
    import time
    t = time.time()
    if _simulation_mode in (SimulationMode.GPU_SPIKE, SimulationMode.CHILLER_FAILURE):
        return {
            "entry": "[CRITICAL] Thermal Inertia detected. Engaging Predictive Fan Modulation.",
            "timestamp": t,
            "critical": True,
        }
    idx = int(t / 3) % len(_AI_LOG_ENTRIES)
    return {"entry": _AI_LOG_ENTRIES[idx], "timestamp": t, "critical": False}


@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Ingest CSV file with universal fuzzy column matching.
    
    Automatically maps columns (Tdie, Tctl, Temp, Fan, RPM, Power) to
    normalized BaseNode attributes. Drop any EPYC/HWiNFO log - no code changes.
    """
    global _last_nodes
    try:
        content = await file.read()
        # DataIngestor accepts file-like objects
        buffer = io.BytesIO(content)
        nodes = _ingestor.ingest_csv(buffer)
        # Fill missing values with synthetic sensor
        _synthetic_sensor.fill_missing_nodes(nodes)
        _last_nodes = nodes
        return {
            "status": "success",
            "nodes_ingested": len(nodes),
            "column_map": _ingestor.get_column_map(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/json")
async def ingest_json(nodes_data: List[NodeInput]):
    """
    Ingest JSON node data directly.
    Expects list of objects with thermal_input, power_draw, cooling_output, utilization.
    """
    global _last_nodes
    nodes = []
    for i, data in enumerate(nodes_data):
        node = ServerNode(
            thermal_input=data.thermal_input,
            power_draw=data.power_draw,
            cooling_output=data.cooling_output,
            utilization=data.utilization,
            node_id=data.node_id or f"node_{i}",
        )
        _synthetic_sensor.fill_missing_node(node)
        nodes.append(node)
    _last_nodes = nodes
    return {"status": "success", "nodes_ingested": len(nodes)}


def _run_optimization_with_state_machine(nodes: List[BaseNode]) -> OptimizationResponse:
    """
    Run optimization and apply System State Machine.
    Evaluates GUARD_MODE triggers; in GUARD_MODE prioritizes thermal stability.
    """
    gap = _brain.analyze(nodes)
    
    # Evaluate GUARD_MODE triggers (unless in MANUAL_OVERRIDE)
    if _state_machine.state != SystemState.MANUAL_OVERRIDE:
        should_guard, reason = _state_machine.evaluate_guard_mode(
            nodes, gap, _synthetic_sensor, ingestor=_ingestor
        )
        if should_guard:
            _state_machine.set_state(SystemState.GUARD_MODE, reason)
        else:
            _state_machine.set_state(SystemState.OPTIMIZING, "")
    
    # Apply state to gap (GUARD_MODE overrides to 100% cooling)
    gap = _state_machine.apply_state_to_gap(gap)
    gap.raw_metrics["system_state"] = _state_machine.state.value
    
    return OptimizationResponse(
        thermal_lag_seconds=gap.thermal_lag_seconds,
        over_provisioning_ratio=gap.over_provisioning_ratio,
        oscillation_ratio=gap.oscillation_ratio,
        efficiency_score=gap.efficiency_score,
        recommended_cooling_delta=gap.recommended_cooling_delta,
        recommendations=gap.recommendations,
        raw_metrics=gap.raw_metrics,
    )


@app.get("/optimize")
async def get_optimization():
    """
    Get optimization recommendations from last ingested data.
    Requires prior call to /ingest/csv or /ingest/json.
    System State Machine applies: GUARD_MODE if data missing, low confidence, or anomaly.
    """
    global _last_nodes
    if not _last_nodes:
        raise HTTPException(
            status_code=400,
            detail="No data ingested. Call /ingest/csv or /ingest/json first.",
        )
    return _run_optimization_with_state_machine(_last_nodes)


@app.post("/optimize")
async def post_optimize(nodes_data: List[NodeInput]):
    """
    Analyze provided node data and return Efficiency Gap.
    Does not require prior ingest - analyze inline.
    System State Machine applies: GUARD_MODE if data missing, low confidence, or anomaly.
    """
    global _last_nodes
    nodes = []
    for i, data in enumerate(nodes_data):
        node = ServerNode(
            thermal_input=data.thermal_input,
            power_draw=data.power_draw,
            cooling_output=data.cooling_output,
            utilization=data.utilization,
            node_id=data.node_id or f"node_{i}",
        )
        _synthetic_sensor.fill_missing_node(node)
        nodes.append(node)
    _last_nodes = nodes
    return _run_optimization_with_state_machine(nodes)


@app.get("/state")
async def get_state():
    """Get current system state (OPTIMIZING, GUARD_MODE, MANUAL_OVERRIDE)."""
    return {
        "state": _state_machine.state.value,
        "guard_mode_reason": _state_machine._guard_mode_reason or None,
    }


class StateInput(BaseModel):
    """Input for POST /state."""
    state: SystemState


@app.post("/state")
async def set_state(state_input: StateInput):
    """
    Set system state. Use MANUAL_OVERRIDE to take human control (AI bypassed).
    Use OPTIMIZING to return to AI control.
    """
    state = state_input.state
    if state == SystemState.MANUAL_OVERRIDE:
        _state_machine.set_state(SystemState.MANUAL_OVERRIDE, "User requested manual override")
    elif state == SystemState.OPTIMIZING:
        _state_machine.set_state(SystemState.OPTIMIZING, "")
    elif state == SystemState.GUARD_MODE:
        _state_machine.set_state(SystemState.GUARD_MODE, "User requested guard mode")
    return {"state": _state_machine.state.value}


class LeadInput(BaseModel):
    """Input for POST /api/v1/leads."""
    fullName: str
    businessEmail: str
    phone: str
    dataCenterScale: str


@app.post("/api/v1/leads")
async def create_lead(lead: LeadInput):
    """
    Lead capture for Blueprint Request form.
    Stores submissions in data/leads.json and logs to console.
    """
    leads_path = project_root / "data" / "leads.json"
    leads_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "fullName": lead.fullName,
        "businessEmail": lead.businessEmail,
        "phone": lead.phone,
        "dataCenterScale": lead.dataCenterScale,
        "submittedAt": datetime.utcnow().isoformat() + "Z",
    }

    leads = []
    if leads_path.exists():
        try:
            with open(leads_path) as f:
                leads = json.load(f)
        except (json.JSONDecodeError, IOError):
            leads = []
    leads.append(entry)
    with open(leads_path, "w") as f:
        json.dump(leads, f, indent=2)

    simulation_ready_url = "https://cooledai.com/portal"
    print(f"[CooledAI Lead] Blueprint Request: {lead.fullName} | {lead.businessEmail} | {lead.dataCenterScale} MW")
    print(f"[CooledAI Lead] Simulation Ready: {simulation_ready_url}")
    return {"status": "success", "message": "Blueprint Request Received."}


@app.get("/adapters")
async def list_adapters():
    """List available protocol adapters (Modbus, SNMP, etc.)."""
    return {
        "adapters": [
            {"name": "ModbusAdapter", "description": "Industrial chillers, CRAC units"},
            {"name": "SNMPAdapter", "description": "Network gear, PDUs, BMS"},
            {"name": "CSVAdapter", "description": "File ingest via DataIngestor"},
        ],
        "note": "Use /ingest/csv for file upload. Modbus/SNMP adapters require separate integration.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
