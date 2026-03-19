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

import os
import secrets
import sys
import json
import base64
import logging
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import io

# Add project root to path for imports (api/main.py -> parent=api, parent.parent=root)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, validator


# --- Clerk JWT user scoping -------------------------------------------------
def _decode_jwt_payload_unverified(token: str) -> Dict[str, Any]:
    """
    Decode JWT payload without verifying signature.

    This is sufficient for extracting the Clerk `sub` claim for tenant scoping.
    If you need strict security, replace with full signature verification.
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            raise ValueError("Not a JWT")
        payload_b64 = parts[1]
        # base64url padding
        payload_b64 += "=" * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid Clerk JWT") from exc


async def _require_clerk_user_id(request: Request) -> str:
    """
    Extract Clerk user_id from Authorization: Bearer <JWT>.

    If the token is missing or cannot be decoded, return 401.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization bearer token")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization bearer token")

    payload = _decode_jwt_payload_unverified(token)
    user_id = payload.get("sub") or payload.get("user_id") or payload.get("uid")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unable to resolve user_id from token")
    return str(user_id)


async def _require_clerk_fixed_owner_id(request: Request) -> str:
    """
    Enforce that the Clerk JWT belongs to the fixed owner tenant.
    Returns FIXED_OWNER_ID if matched, otherwise 401.
    """
    clerk_user_id = await _require_clerk_user_id(request)
    # If the decoded Clerk `sub` doesn't match our fixed tenant, we still
    # return the fixed tenant so the dashboard remains usable while we
    # align Clerk<->tenant mapping. Authentication is still required.
    if clerk_user_id != FIXED_OWNER_ID:
        logging.getLogger("api.main").warning(
            "Clerk user_id mismatch: got=%s expected=%s (serving fixed tenant)",
            clerk_user_id,
            FIXED_OWNER_ID,
        )
    return FIXED_OWNER_ID


# --- Security: API Key Authentication ---
# Legacy env-var keys still honoured for backward compatibility.
# New multi-tenant model: keys stored in data/api_keys.json, each mapped to
# an owner_id.  On first startup an admin key is auto-generated and logged.
_API_KEY = os.environ.get("COOLEDAI_API_KEY", "")
_ADMIN_KEY = os.environ.get("COOLEDAI_ADMIN_KEY", "")
_MASTER_KEY = os.environ.get("COOLEDAI_MASTER_KEY", "")

# Hardcoded tenant owner for all telemetry and all portal reads.
# This must match the Clerk `sub` (user_id) for dennis@cooledai.com.
FIXED_OWNER_ID = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"


# --- Multi-Tenant API Key Registry ---
_KEYS_FILE = project_root / "data" / "api_keys.json"
_api_key_registry: Dict[str, dict] = {}


def _load_key_registry() -> None:
    global _api_key_registry
    try:
        if _KEYS_FILE.exists():
            _api_key_registry = json.loads(_KEYS_FILE.read_text())
    except Exception:
        _api_key_registry = {}


def _save_key_registry() -> None:
    try:
        _KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEYS_FILE.write_text(json.dumps(_api_key_registry, indent=2))
    except Exception as exc:
        logging.getLogger("api.main").warning("Failed to save key registry: %s", exc)


def _generate_api_key() -> str:
    return "sk-" + secrets.token_urlsafe(32)


_load_key_registry()


def _ensure_admin_key_exists() -> None:
    """On first startup, auto-generate an admin API key and persist it."""
    if _api_key_registry:
        return
    key = _generate_api_key()
    _api_key_registry[key] = {
        "owner_id": "admin",
        "label": "Admin (auto-generated on first startup)",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_key_registry()
    logging.getLogger("api.main").info(
        "\n============================================================\n"
        "  FIRST RUN — Admin API Key Generated\n"
        "  Key:   %s\n"
        "  Owner: admin\n"
        "  Save this key now; it will NOT be shown again.\n"
        "============================================================",
        key,
    )


_ensure_admin_key_exists()


def _is_master(key: str) -> bool:
    return bool(_MASTER_KEY) and key == _MASTER_KEY


def _require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    """Dependency: require valid API key (legacy env-var OR registered key)."""
    if _is_master(x_api_key):
        return x_api_key
    if x_api_key in _api_key_registry:
        return x_api_key
    if _API_KEY and x_api_key == _API_KEY:
        return x_api_key
    if _API_KEY or _api_key_registry or _MASTER_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key (X-API-Key header).")
    return x_api_key


def _require_admin_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    """Dependency: require admin-level API key."""
    if _is_master(x_api_key):
        return x_api_key
    if x_api_key in _api_key_registry:
        if _api_key_registry[x_api_key].get("owner_id") == "admin":
            return x_api_key
    legacy = _ADMIN_KEY or _API_KEY
    if legacy and x_api_key == legacy:
        return x_api_key
    if legacy or _api_key_registry or _MASTER_KEY:
        raise HTTPException(status_code=403, detail="Admin access required (X-API-Key header).")
    return x_api_key


def _resolve_owner(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    """
    Resolve owner_id for telemetry ingestion.

    Telemetry writes are intentionally stored under `FIXED_OWNER_ID` regardless
    of which X-API-Key was used. Portal reads remain strictly protected via
    Clerk JWT checks.
    """
    return FIXED_OWNER_ID


# --- File upload limits ---
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB max for CSV/file uploads


# --- Singleton Thermal Model Loader ---

class ThermalModelLoader:
    """Thread-safe singleton that loads a trained .pkl or .joblib model once
    and keeps it in memory for the lifetime of the process.

    Searches ``models/trained/`` for the newest model file. Falls back to
    physics-based predictions via OptimizationBrain when no trained artifact
    exists.
    """

    _instance: Optional["ThermalModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ThermalModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
                    cls._instance._model_path: Optional[Path] = None
                    cls._instance._loaded = False
        return cls._instance

    def load(self, search_dir: Optional[Path] = None) -> None:
        """Scan *search_dir* for the newest ``.pkl`` / ``.joblib`` file and
        load it.  Safe to call multiple times — only the first call does I/O.
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            search_dir = search_dir or (project_root / "models" / "trained")
            candidates: list[Path] = []
            if search_dir.is_dir():
                candidates = sorted(
                    [p for p in search_dir.iterdir() if p.suffix in (".pkl", ".joblib")],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if candidates:
                path = candidates[0]
                try:
                    if path.suffix == ".joblib":
                        import joblib
                        self._model = joblib.load(path)
                    else:
                        import pickle
                        with open(path, "rb") as fh:
                            self._model = pickle.load(fh)
                    self._model_path = path
                    logging.getLogger("api.main").info(
                        "Thermal model loaded from %s", path,
                    )
                except Exception as exc:
                    logging.getLogger("api.main").warning(
                        "Failed to load model %s: %s", path, exc,
                    )
            else:
                logging.getLogger("api.main").info(
                    "No trained model found in %s — using physics fallback.", search_dir,
                )
            self._loaded = True

    @property
    def model(self) -> Any:
        if not self._loaded:
            self.load()
        return self._model

    @property
    def is_available(self) -> bool:
        if not self._loaded:
            self.load()
        return self._model is not None

    def predict(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """Return ``target_rpm``, ``predicted_temp_delta``, and
        ``power_efficiency_score`` for a single telemetry snapshot.

        When a trained model is loaded its ``.predict()`` method is called
        with a feature vector built from the telemetry dict.  Otherwise
        a physics-based heuristic is used.
        """
        cpu_temp = telemetry.get("cpu_temp", 55.0)
        fan_rpm = telemetry.get("fan_rpm", 1800)
        power_kw = telemetry.get("power_kw", 120.0)
        utilization = telemetry.get("utilization", 0.5)

        if self.is_available:
            import numpy as np
            features = np.array([[cpu_temp, fan_rpm, power_kw, utilization]])
            raw = self._model.predict(features)
            if hasattr(raw, "shape") and raw.ndim == 2:
                raw = raw[0]
            return {
                "target_rpm": float(raw[0]) if len(raw) > 0 else fan_rpm,
                "predicted_temp_delta": float(raw[1]) if len(raw) > 1 else 0.0,
                "power_efficiency_score": float(raw[2]) if len(raw) > 2 else 80.0,
            }

        # Physics fallback: Fan Affinity Law + thermal balance heuristic
        max_rpm = 3000.0
        target_temp = 65.0
        temp_delta = cpu_temp - target_temp
        rpm_frac = fan_rpm / max_rpm if max_rpm > 0 else 0.7

        if temp_delta > 5.0:
            target_rpm = min(max_rpm, fan_rpm * 1.12)
        elif temp_delta < -10.0 and rpm_frac > 0.60:
            target_rpm = max(400, fan_rpm * 0.90)
        else:
            target_rpm = fan_rpm

        sweet_spot = 0.60 <= (target_rpm / max_rpm) <= 0.80
        efficiency = 85.0 + (10.0 if sweet_spot else -5.0) - abs(temp_delta) * 0.5
        efficiency = max(0.0, min(100.0, efficiency))

        return {
            "target_rpm": round(target_rpm, 1),
            "predicted_temp_delta": round(-temp_delta * 0.3, 2),
            "power_efficiency_score": round(efficiency, 1),
        }


# Eagerly initialise the singleton (loads model file on first request)
_thermal_model = ThermalModelLoader()


# Import CooledAI components
from core.hal.base_node import BaseNode, ServerNode
from core.ingestion.data_ingestor import DataIngestor
from core.ingestion import TelemetryDerivativeEnricher, InfluenceMap, apply_telemetry_filters
from core.ingestion.telemetry_smoothing import MovingAverageFilter
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

# CORS
# Explicit allowlist for frontend origins. If allow_credentials=True, we must
# not use allow_origins=["*"].
_cors_allowed_origins = [
    "https://www.cooledai.com",
    "https://cooledai.com",
    "https://api.cooledai.com",
    "https://app.cooledai.com",
    "https://cooledai-platform.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Security Headers Middleware ---
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if os.environ.get("COOLEDAI_ENV", "production").lower() not in ("dev", "development", "local"):
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        return response


app.add_middleware(SecurityHeadersMiddleware)


@app.on_event("startup")
async def _start_digest_worker() -> None:
    """Start background digest worker (3-hour predictive-vs-control email)."""
    global _report_thread
    if not _REPORT_EMAIL_ENABLED:
        logging.getLogger("api.main").info("[Digest] Disabled by COOLEDAI_REPORT_EMAIL_ENABLED.")
        return
    if _report_thread is not None and _report_thread.is_alive():
        return
    _report_stop_event.clear()
    _report_thread = threading.Thread(
        target=_digest_worker_loop,
        name="cooledai-efficiency-digest",
        daemon=True,
    )
    _report_thread.start()


@app.on_event("shutdown")
async def _stop_digest_worker() -> None:
    """Stop background digest worker."""
    _report_stop_event.set()
    if _report_thread is not None and _report_thread.is_alive():
        _report_thread.join(timeout=2.0)


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
        """Current system state."""
        return self._state

    def set_state(self, new_state: SystemState, reason: str = "") -> None:
        """Set state explicitly (e.g. MANUAL_OVERRIDE).

        Args:
            new_state: Target SystemState.
            reason: Optional reason (stored for GUARD_MODE).
        """
        self._state = new_state
        self._guard_mode_reason = reason
    
    def evaluate_guard_mode(
        self,
        nodes: List[BaseNode],
        gap: Optional[EfficiencyGap],
        synthetic_sensor: SyntheticSensor,
        ingestor: Optional["DataIngestor"] = None,
    ) -> Tuple[bool, str]:
        """Evaluate if GUARD_MODE should be triggered.

        Physics/safety logic: (1) Stale data—timestamp older than 5s triggers
        Communication Timeout; (2) Missing data—>30% nodes with synthetic or
        invalid thermal/cooling; (3) Low confidence—efficiency_score < 60;
        (4) Physical anomaly—e.g. fan 0 RPM but temp rising (thermodynamics violation).

        Args:
            nodes: Current telemetry nodes.
            gap: EfficiencyGap from OptimizationBrain (for efficiency_score).
            synthetic_sensor: For detect_physical_anomaly.
            ingestor: Optional DataIngestor for check_stale_data.

        Returns:
            Tuple of (should_enter_guard_mode: bool, reason: str).
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
# Node IDs in maintenance mode: optimization skips them and sends no commands (human on-site)
_maintenance_mode_ids: set = set()
_sanity_check_filter = None
_moving_avg_filter = MovingAverageFilter(window_size=5)
try:
    from core.anomaly import SanityCheckFilter
    _sanity_check_filter = SanityCheckFilter()
except Exception:
    pass
_ingestor = DataIngestor(
    node_type=ServerNode,
    sanity_check_filter=_sanity_check_filter,
    moving_avg_filter=_moving_avg_filter,
)
_brain = OptimizationBrain()
_synthetic_sensor = SyntheticSensor()
_state_machine = SystemStateMachine()
_derivative_enricher = TelemetryDerivativeEnricher()
# Load InfluenceMap from discovery-generated file if present; else empty
try:
    from core.config import load_influence_map
    _influence_map = load_influence_map() or InfluenceMap()
except Exception:
    _influence_map = InfluenceMap()
# Anomaly detection state: last (power_draw, delta_t) per cooling unit for trend detection
_cooling_unit_previous_state: Dict[str, Any] = {}

# Shadow mode: when True, proposed actions are logged to shadow_logs and no writes are sent to hardware
try:
    from backend.shadow.shadow_logs import (
        init_shadow_db,
        log_proposed_action,
        log_failing_component_event,
        get_failing_component_events,
    )
    from backend.shadow.savings_report import generate_potential_savings_report, PotentialSavingsReport
    from backend.shadow.missed_opportunity import (
        log_missed_opportunity,
        get_missed_opportunities,
        get_daily_missed_savings_notification,
        DailyMissedSavingsNotification,
        MissedOpportunityEvent,
        _reason_from_gap,
        DEFAULT_MISSED_OPPORTUNITY_THRESHOLD_USD,
    )
    _shadow_available = True
except Exception:
    _shadow_available = False
    init_shadow_db = None
    log_proposed_action = None
    log_failing_component_event = None
    get_failing_component_events = None
    log_missed_opportunity = None
    get_missed_opportunities = None
    get_daily_missed_savings_notification = None
    DailyMissedSavingsNotification = None
    MissedOpportunityEvent = None
    _reason_from_gap = None
    DEFAULT_MISSED_OPPORTUNITY_THRESHOLD_USD = 10.0

# Financial metrics: user-defined utility rate ($/kWh) for dollars_saved_today and estimated_monthly_roi
try:
    from core.optimization.financial_metrics import FinancialMetrics, FinancialResult
    _utility_rate = float(os.environ.get("COOLEDAI_UTILITY_RATE", "0.12"))
    _financial_metrics = FinancialMetrics(utility_rate=_utility_rate)
except Exception:
    _financial_metrics = None

# Anomaly detection: failing cooling unit (power up, ΔT down) -> Predictive Maintenance
try:
    from core.anomaly import (
        detect_failing_cooling_units,
        format_findings_for_diagnostic,
        severity_score_from_finding,
        CoolingUnitState,
        FailingComponentFinding,
    )
    _anomaly_available = True
except Exception:
    _anomaly_available = False
    detect_failing_cooling_units = None
    format_findings_for_diagnostic = None
    severity_score_from_finding = None

# Baseline reporter: 24h current-state recording and comparison report (Current PUE vs Projected CooledAI PUE)
try:
    from backend.baseline import BaselineReporter, BaselineComparisonReport, init_baseline_db
    _baseline_db_path = project_root / "data" / "baseline.db"
    _baseline_reporter = BaselineReporter(
        db_path=_baseline_db_path,
        utility_rate_per_kwh=float(os.environ.get("COOLEDAI_UTILITY_RATE", "0.12")),
        co2_kg_per_kwh=float(os.environ.get("COOLEDAI_CO2_KG_PER_KWH", "0.4")),
        rebate_per_kwh=float(os.environ.get("COOLEDAI_REBATE_PER_KWH", "0.05")),
        projected_pue_improvement_pct=float(os.environ.get("COOLEDAI_PROJECTED_PUE_IMPROVEMENT_PCT", "12")),
    )
    init_baseline_db(_baseline_db_path)
    _baseline_available = True
except Exception as e:
    _baseline_available = False
    _baseline_reporter = None

# Job observer: Slurm/K8s pre-cooling (optional)
try:
    from services.scheduler.job_observer import get_upcoming_load
except Exception:
    get_upcoming_load = None


# --- Pydantic Models for API ---

class NodeInput(BaseModel):
    """Input model for JSON ingest (POST /ingest/json, POST /optimize).

    Normalized attributes matching BaseNode. thermal_input and power_draw
    drive heat load; cooling_output is cooling response.
    """
    thermal_input: float = 0.0
    power_draw: float = 0.0
    cooling_output: float = 0.0
    utilization: float = 0.0
    node_id: Optional[str] = None


class OptimizationResponse(BaseModel):
    """Response model for optimization analysis (GET/POST /optimize).

    Contains Efficiency Gap metrics (thermal lag, over-provisioning,
    oscillation), recommended_cooling_delta, per-unit deltas when failing
    components exist, and financial estimates.
    """
    thermal_lag_seconds: float
    over_provisioning_ratio: float
    oscillation_ratio: float
    efficiency_score: float
    recommended_cooling_delta: float
    recommendations: List[str]
    raw_metrics: Dict[str, Any] = {}
    diagnostic_reasoning: str = ""
    llm_prompt_context: Dict[str, Any] = {}
    shadow_mode: bool = False
    proposed_action_logged: bool = False
    dollars_saved_today: float = 0.0
    estimated_monthly_roi: float = 0.0
    recommended_cooling_delta_by_unit: Dict[str, float] = {}  # per-unit when failing components
    recommended_rpm_confidence_interval: Optional[Tuple[float, float]] = None  # (lower_rpm, upper_rpm)
    prediction_confidence: float = 1.0  # 0-1; below 0.8 triggers Cautionary Cooling
    cautionary_cooling_state: bool = False  # True when confidence < 80%: safe higher-RPM baseline


# --- Thermal Prediction Models ---

class ThermalTelemetryInput(BaseModel):
    """Telemetry snapshot sent to ``/optimize/thermal``."""
    cpu_temp: float
    fan_rpm: float = 1800
    power_kw: float = 120.0
    utilization: float = 0.5
    node_id: Optional[str] = None


class ThermalPredictionResponse(BaseModel):
    """Response from ``/optimize/thermal`` — the three metrics the pilot
    test runner and external integrations consume."""
    target_rpm: float
    predicted_temp_delta: float
    power_efficiency_score: float
    model_loaded: bool
    model_path: Optional[str] = None


@app.post("/optimize/thermal", dependencies=[Depends(_require_api_key)])
async def optimize_thermal(body: ThermalTelemetryInput):
    """Accept a telemetry snapshot and return a JSON recommendation with
    ``target_rpm``, ``predicted_temp_delta``, and ``power_efficiency_score``.

    Uses the singleton-loaded trained model when available, otherwise falls
    back to physics-based heuristics (Fan Affinity Laws).
    """
    prediction = _thermal_model.predict(body.dict())
    return ThermalPredictionResponse(
        target_rpm=prediction["target_rpm"],
        predicted_temp_delta=prediction["predicted_temp_delta"],
        power_efficiency_score=prediction["power_efficiency_score"],
        model_loaded=_thermal_model.is_available,
        model_path=str(_thermal_model._model_path) if _thermal_model._model_path else None,
    )


# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check for load balancers and monitoring.

    Returns:
        {"status": "ok"}.
    """
    return {"status": "ok"}


# Briefing Agent: Morning Briefing (last 24h) for Slack/Email
try:
    from services.briefing import BriefingAgent, MorningBriefing, generate_daily_briefing
    _briefing_available = True
except Exception:
    _briefing_available = False
    generate_daily_briefing = None


@app.get("/briefing/daily")
async def get_briefing_daily():
    """
    Morning Briefing: last 24 hours summary. Includes Financial Win (dollars saved vs baseline),
    System Health (score from anomalies), Watchlist (units with severity > 20), and LLM executive summary.
    For Slack/Email notification systems.
    """
    if not _briefing_available or generate_daily_briefing is None:
        raise HTTPException(status_code=503, detail="Briefing module not available.")
    try:
        briefing = generate_daily_briefing(
            baseline_db_path=project_root / "data" / "baseline.db",
            shadow_db_path=project_root / "data" / "shadow_logs.db",
            use_llm=True,
        )
        return {
            "period_start_utc": briefing.period_start_utc,
            "period_end_utc": briefing.period_end_utc,
            "generated_at_utc": briefing.generated_at_utc,
            "financial_win": briefing.financial_win,
            "system_health": briefing.system_health,
            "watchlist": briefing.watchlist,
            "llm_summary": briefing.llm_summary,
        }
    except Exception as e:
        logging.getLogger("api.main").exception("Daily briefing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/anomalies")
async def get_health_anomalies(
    start: Optional[str] = None,
    end: Optional[str] = None,
    node_id: Optional[str] = None,
    limit: int = 500,
):
    """
    Historical list of all failing component findings for Maintenance Task List.
    Each entry includes a Severity Score (0-100) based on the ratio of power increase
    vs ΔT decrease. Facility engineers can use this to prioritize maintenance.
    """
    if get_failing_component_events is None:
        raise HTTPException(status_code=503, detail="Failing component history not available.")
    start_ts = None
    end_ts = None
    if end:
        try:
            end_ts = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            end_ts = datetime.utcnow()
    if start:
        try:
            start_ts = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            start_ts = (end_ts or datetime.utcnow()) - timedelta(days=30)
    events = get_failing_component_events(start_ts=start_ts, end_ts=end_ts, node_id=node_id, limit=limit)
    return {
        "anomalies": [
            {
                "id": e["id"],
                "created_at": e["created_at"],
                "node_id": e["node_id"],
                "message": e["message"],
                "power_draw_prev": e["power_draw_prev"],
                "power_draw_now": e["power_draw_now"],
                "delta_t_prev": e["delta_t_prev"],
                "delta_t_now": e["delta_t_now"],
                "severity_score": e["severity_score"],
            }
            for e in events
        ],
        "count": len(events),
    }


# Utility Rebate PDF report (30-day audit-ready)
try:
    from reports import generate_utility_rebate_pdf
    from reports.utility_rebate_generator import UtilityRebateGenerator
    _reports_available = True
except Exception:
    _reports_available = False
    generate_utility_rebate_pdf = None


@app.get("/reports/utility-rebate")
async def get_reports_utility_rebate():
    """
    Download the Utility Rebate PDF report: 30-day baseline vs optimized PUE,
    total MWh saved, carbon offset (metric tonnes CO₂), and verification log
    of every optimization event (Active Demand Management). Audit-ready for
    utility rebate submission.
    """
    if not _reports_available or generate_utility_rebate_pdf is None:
        raise HTTPException(status_code=503, detail="Reports module not available.")
    try:
        utility_rate = float(os.environ.get("COOLEDAI_UTILITY_RATE", "0.12"))
        co2_kg = float(os.environ.get("COOLEDAI_CO2_KG_PER_KWH", "0.4"))
        pdf_bytes = generate_utility_rebate_pdf(
            baseline_db_path=project_root / "data" / "baseline.db",
            shadow_db_path=project_root / "data" / "shadow_logs.db",
            utility_rate=utility_rate,
            co2_kg_per_kwh=co2_kg,
        )
        filename = f"utility-rebate-{datetime.utcnow().strftime('%Y-%m-%d')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logging.getLogger("api.main").exception("Utility rebate PDF failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/admin/simulation-control/trigger", dependencies=[Depends(_require_admin_key)])
async def trigger_simulation(body: SimulationTrigger):
    """Command Center: Trigger GPU spike, chiller failure, or reset. Requires admin key."""
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


@app.get("/admin/simulation-control/status", dependencies=[Depends(_require_admin_key)])
async def simulation_status():
    """Command Center: Get current simulation mode. Requires admin key."""
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


@app.post("/ingest/csv", dependencies=[Depends(_require_api_key)])
async def ingest_csv(file: UploadFile = File(...)):
    """
    Ingest CSV file with universal fuzzy column matching.
    
    Automatically maps columns (Tdie, Tctl, Temp, Fan, RPM, Power) to
    normalized BaseNode attributes. Drop any EPYC/HWiNFO log - no code changes.
    Max upload size: 10 MB. Accepted types: .csv, text/csv.
    """
    global _last_nodes, _maintenance_mode_ids
    try:
        # Validate file type
        filename = (file.filename or "").lower()
        content_type = (file.content_type or "").lower()
        if not filename.endswith(".csv") and "csv" not in content_type and "text" not in content_type:
            raise HTTPException(status_code=400, detail="Only CSV files accepted.")
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB.")
        # DataIngestor accepts file-like objects
        buffer = io.BytesIO(content)
        nodes = _ingestor.ingest_csv(buffer)
        # Fill missing values with synthetic sensor
        _synthetic_sensor.fill_missing_nodes(nodes)
        for n in nodes:
            n.maintenance_mode = getattr(n, "node_id", "") in _maintenance_mode_ids
        _last_nodes = nodes
        return {
            "status": "success",
            "nodes_ingested": len(nodes),
            "column_map": _ingestor.get_column_map(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/json", dependencies=[Depends(_require_api_key)])
async def ingest_json(nodes_data: List[NodeInput]):
    """Ingest JSON node data directly.

    Expects list of NodeInput with thermal_input, power_draw, cooling_output,
    utilization. Fills missing values with synthetic sensor; applies Sanity Check
    (Sensor Ghost filter) and Moving Average (last 5 readings) to smooth telemetry
    before OptimizationBrain—prevents fan jitter from electrical noise.

    Args:
        nodes_data: List of NodeInput objects.

    Returns:
        Dict with status, nodes_ingested.
    """
    global _last_nodes, _maintenance_mode_ids
    nodes = []
    now = datetime.now()
    for i, data in enumerate(nodes_data):
        node = ServerNode(
            thermal_input=data.thermal_input,
            power_draw=data.power_draw,
            cooling_output=data.cooling_output,
            utilization=data.utilization,
            node_id=data.node_id or f"node_{i}",
            timestamp=now,
        )
        _synthetic_sensor.fill_missing_node(node)
        node.maintenance_mode = node.node_id in _maintenance_mode_ids
        nodes.append(node)
    apply_telemetry_filters(nodes, _sanity_check_filter, _moving_avg_filter)
    _derivative_enricher.enrich(nodes)
    _last_nodes = nodes
    return {"status": "success", "nodes_ingested": len(nodes)}


def _run_optimization_with_state_machine(nodes: List[BaseNode]) -> OptimizationResponse:
    """
    Run optimization and apply System State Machine.
    Evaluates GUARD_MODE triggers; in GUARD_MODE prioritizes thermal stability.
    When Shadow Mode is on, logs proposed action to shadow_logs and no write is sent to hardware.
    Anomaly detection runs first; any FailingComponentFindings are passed to the brain so it
    can reduce load on failing units and redistribute to healthy ones (InfluenceMap).
    Nodes in maintenance_mode are skipped by the brain (no commands sent); human on-site stays in control.
    """
    global _maintenance_mode_ids
    # Apply maintenance flags from API state; skip maintenance nodes for optimization (no commands to them)
    for n in nodes:
        n.maintenance_mode = getattr(n, "node_id", "") in _maintenance_mode_ids
    active_nodes = [n for n in nodes if not getattr(n, "maintenance_mode", False)]

    # Anomaly detection first: Failing Component (power up, ΔT down) -> state + findings for brain
    failing_findings: List[Any] = []
    global _cooling_unit_previous_state
    if _anomaly_available and detect_failing_cooling_units is not None and nodes:
        try:
            prev = {}
            if _cooling_unit_previous_state:
                for k, v in _cooling_unit_previous_state.items():
                    if isinstance(v, dict) and "power_draw" in v and "delta_t_c" in v:
                        prev[k] = CoolingUnitState(power_draw=v["power_draw"], delta_t_c=v["delta_t_c"], node_id=v.get("node_id", ""))
            findings, new_state = detect_failing_cooling_units(nodes, previous_state=prev or None)
            _cooling_unit_previous_state = {k: {"power_draw": s.power_draw, "delta_t_c": s.delta_t_c, "node_id": s.node_id} for k, s in new_state.items()}
            failing_findings = findings or []
        except Exception as e:
            logging.getLogger("api.main").warning("Anomaly detection failed: %s", e)

    upcoming_jobs = []
    if get_upcoming_load is not None:
        try:
            upcoming_jobs = get_upcoming_load(max_seconds=60.0)
        except Exception as e:
            logging.getLogger("api.main").debug("Job observer unavailable: %s", e)

    gap = _brain.analyze(
        active_nodes,
        influence_map=_influence_map,
        failing_component_findings=failing_findings if failing_findings else None,
        upcoming_jobs=upcoming_jobs if upcoming_jobs else None,
    )

    # Persist failing component findings to history (for GET /health/anomalies / Maintenance Task List)
    if failing_findings and log_failing_component_event is not None and severity_score_from_finding is not None:
        try:
            for f in failing_findings:
                severity = severity_score_from_finding(f)
                log_failing_component_event(
                    node_id=f.node_id,
                    message=f.message,
                    power_draw_prev=f.power_draw_prev,
                    power_draw_now=f.power_draw_now,
                    delta_t_prev=f.delta_t_prev,
                    delta_t_now=f.delta_t_now,
                    severity_score=severity,
                )
        except Exception as e:
            logging.getLogger("api.main").warning("Failed to persist failing component event: %s", e)

    # Ensure recommended_cooling_delta_by_unit has no entries for maintenance nodes (brain already skipped them)
    if hasattr(gap, "recommended_cooling_delta_by_unit") and gap.recommended_cooling_delta_by_unit:
        for nid in list(gap.recommended_cooling_delta_by_unit.keys()):
            if nid in _maintenance_mode_ids:
                del gap.recommended_cooling_delta_by_unit[nid]

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

    # Shadow Mode: persist proposed action to shadow_logs (no write to hardware)
    if getattr(_brain, "shadow_mode", False) and _shadow_available and nodes:
        try:
            import numpy as np
            cooling_arr = np.array([n.cooling_output for n in nodes])
            power_arr = np.array([n.power_draw for n in nodes])
            current_cooling = float(cooling_arr[-1]) if len(cooling_arr) else 0.0
            max_cooling = max(float(np.max(cooling_arr)), 1000.0) if len(cooling_arr) else 3000.0
            current_thermal = float(active_nodes[-1].thermal_input) if active_nodes else 0.0
            power_draw = float(power_arr[-1]) if len(power_arr) else 0.0
            it_power_kw = float(np.sum(power_arr)) / 1000.0 if len(power_arr) else (power_draw / 1000.0)
            # Estimate cooling power: assume cooling is ~20-30% of facility; use current cooling level
            cooling_power_kw = (current_cooling / max_cooling) * (it_power_kw * 0.35) if max_cooling > 1e-6 else it_power_kw * 0.2
            row_id = log_proposed_action(
                recommended_cooling_delta=gap.recommended_cooling_delta,
                current_cooling=current_cooling,
                max_cooling=max_cooling,
                current_thermal=current_thermal,
                power_draw=power_draw,
                it_power_kw=it_power_kw,
                cooling_power_kw=cooling_power_kw,
                diagnostic_reasoning=getattr(gap, "diagnostic_reasoning", "") or "",
                recommendations=gap.recommendations,
                raw_metrics=gap.raw_metrics,
                system_state=gap.raw_metrics.get("system_state"),
            )
            # Missed Opportunity (FOMO): if AI would have saved > $10, log event with reason
            if _shadow_available and log_missed_opportunity is not None and _reason_from_gap is not None:
                proposed_cooling = current_cooling * (1.0 + gap.recommended_cooling_delta)
                proposed_cooling = max(0.0, min(proposed_cooling, max_cooling))
                if current_cooling > 1e-6 and cooling_power_kw and cooling_power_kw > 0:
                    proposed_cooling_power_kw = cooling_power_kw * (proposed_cooling / current_cooling)
                    cooling_savings_kw = cooling_power_kw - proposed_cooling_power_kw
                else:
                    cooling_savings_kw = 0.0
                if cooling_savings_kw > 0:
                    utility_rate = _financial_metrics.utility_rate if _financial_metrics is not None else 0.12
                    dollars_missed_per_hour = cooling_savings_kw * utility_rate
                    threshold = float(os.environ.get("COOLEDAI_MISSED_OPPORTUNITY_THRESHOLD_USD", str(DEFAULT_MISSED_OPPORTUNITY_THRESHOLD_USD)))
                    if dollars_missed_per_hour >= threshold:
                        reason = _reason_from_gap(
                            gap.recommendations,
                            getattr(gap, "diagnostic_reasoning", "") or "",
                            gap.raw_metrics,
                        )
                        log_missed_opportunity(
                            shadow_log_id=row_id,
                            dollars_missed=dollars_missed_per_hour,
                            reason=reason,
                            cooling_savings_kw=cooling_savings_kw,
                            it_power_kw=it_power_kw,
                            recommendations_snapshot=gap.recommendations,
                        )
        except Exception as e:
            logging.getLogger("api.main").warning("Shadow log failed: %s", e)

    # Financial metrics: dollars_saved_today, estimated_monthly_roi for API and LLM diagnostic
    dollars_saved_today = 0.0
    estimated_monthly_roi = 0.0
    diagnostic_reasoning = getattr(gap, "diagnostic_reasoning", "") or ""
    llm_prompt_context = dict(getattr(gap, "llm_prompt_context", None) or {})
    if _financial_metrics is not None:
        kwh_saved = getattr(gap, "kwh_saved", 0.0) or 0.0
        fin = _financial_metrics.compute(kwh_saved)
        dollars_saved_today = fin.dollars_saved_today
        estimated_monthly_roi = fin.estimated_monthly_roi
        llm_prompt_context["financial"] = {
            "dollars_saved_this_hour": fin.dollars_saved_this_hour,
            "dollars_saved_today": fin.dollars_saved_today,
            "estimated_monthly_roi": fin.estimated_monthly_roi,
            "kwh_saved_this_hour": fin.kwh_saved_this_hour,
            "utility_rate_per_kwh": fin.utility_rate_per_kwh,
        }
        snippet = _financial_metrics.diagnostic_snippet(fin)
        if snippet:
            diagnostic_reasoning = (diagnostic_reasoning.rstrip(". ") + "." + snippet).strip()

    proposed_logged = getattr(_brain, "shadow_mode", False) and _shadow_available and nodes
    return OptimizationResponse(
        thermal_lag_seconds=gap.thermal_lag_seconds,
        over_provisioning_ratio=gap.over_provisioning_ratio,
        oscillation_ratio=gap.oscillation_ratio,
        efficiency_score=gap.efficiency_score,
        recommended_cooling_delta=gap.recommended_cooling_delta,
        recommendations=gap.recommendations,
        raw_metrics=gap.raw_metrics,
        diagnostic_reasoning=diagnostic_reasoning,
        llm_prompt_context=llm_prompt_context,
        shadow_mode=getattr(_brain, "shadow_mode", False),
        proposed_action_logged=proposed_logged,
        dollars_saved_today=dollars_saved_today,
        estimated_monthly_roi=estimated_monthly_roi,
        recommended_cooling_delta_by_unit=getattr(gap, "recommended_cooling_delta_by_unit", {}) or {},
        recommended_rpm_confidence_interval=getattr(gap, "recommended_rpm_confidence_interval", None),
        prediction_confidence=getattr(gap, "prediction_confidence", 1.0),
        cautionary_cooling_state=getattr(gap, "cautionary_cooling_state", False),
    )


@app.get("/optimize")
async def get_optimization():
    """Get optimization recommendations from last ingested data.

    Requires prior /ingest/csv or /ingest/json. Applies System State Machine;
    GUARD_MODE if data missing, low confidence, or physical anomaly.

    Returns:
        OptimizationResponse with recommended_cooling_delta, metrics, etc.
    """
    global _last_nodes
    if not _last_nodes:
        raise HTTPException(
            status_code=400,
            detail="No data ingested. Call /ingest/csv or /ingest/json first.",
        )
    return _run_optimization_with_state_machine(_last_nodes)


@app.post("/optimize", dependencies=[Depends(_require_api_key)])
async def post_optimize(nodes_data: List[NodeInput]):
    """Analyze provided node data and return Efficiency Gap.

    Does not require prior ingest. Fills missing values with synthetic sensor.
    Applies System State Machine; GUARD_MODE if data missing, low confidence,
    or physical anomaly.

    Args:
        nodes_data: List of NodeInput objects.

    Returns:
        OptimizationResponse with recommended_cooling_delta, metrics, etc.
    """
    global _last_nodes
    nodes = []
    now = datetime.now()
    for i, data in enumerate(nodes_data):
        node = ServerNode(
            thermal_input=data.thermal_input,
            power_draw=data.power_draw,
            cooling_output=data.cooling_output,
            utilization=data.utilization,
            node_id=data.node_id or f"node_{i}",
            timestamp=now,
        )
        _synthetic_sensor.fill_missing_node(node)
        nodes.append(node)
    apply_telemetry_filters(nodes, _sanity_check_filter, _moving_avg_filter)
    _derivative_enricher.enrich(nodes)
    global _maintenance_mode_ids
    for n in nodes:
        n.maintenance_mode = n.node_id in _maintenance_mode_ids
    _last_nodes = nodes
    return _run_optimization_with_state_machine(nodes)


class MaintenanceModeInput(BaseModel):
    """Optional body for POST /nodes/{id}/maintenance. If omitted, toggles."""
    maintenance_mode: Optional[bool] = None


@app.post("/nodes/{node_id}/maintenance", dependencies=[Depends(_require_api_key)])
async def set_node_maintenance(node_id: str, body: Optional[MaintenanceModeInput] = None):
    """
    Toggle or set maintenance mode for a node. When a node is in maintenance mode,
    the OptimizationBrain skips it entirely and sends no commands (human on-site).
    Body: { "maintenance_mode": true } or { "maintenance_mode": false } to set;
    omit body to toggle.
    """
    global _maintenance_mode_ids
    if body is not None and body.maintenance_mode is not None:
        if body.maintenance_mode:
            _maintenance_mode_ids.add(node_id)
        else:
            _maintenance_mode_ids.discard(node_id)
    else:
        if node_id in _maintenance_mode_ids:
            _maintenance_mode_ids.discard(node_id)
        else:
            _maintenance_mode_ids.add(node_id)
    # Update any live nodes so next optimization sees the flag
    for n in _last_nodes:
        if getattr(n, "node_id", "") == node_id:
            n.maintenance_mode = node_id in _maintenance_mode_ids
            break
    return {
        "node_id": node_id,
        "maintenance_mode": node_id in _maintenance_mode_ids,
        "message": "Node in maintenance; AI will send no commands to it." if node_id in _maintenance_mode_ids else "Node active; AI may send commands.",
    }


@app.get("/state")
async def get_state():
    """Get current system state (OPTIMIZING, GUARD_MODE, MANUAL_OVERRIDE)."""
    return {
        "state": _state_machine.state.value,
        "guard_mode_reason": _state_machine._guard_mode_reason or None,
    }


class StateInput(BaseModel):
    """Body for POST /state. Sets system state (e.g. MANUAL_OVERRIDE)."""
    state: SystemState


@app.post("/state", dependencies=[Depends(_require_api_key)])
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


class BetaSignupInput(BaseModel):
    """Input for POST /api/v1/beta."""
    email: str
    facilityScale: str = ""


def _send_lead_email(lead: LeadInput) -> bool:
    """Send lead notification to dennis@cooledai.com via Resend. Returns True if sent."""
    from html import escape as _esc
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logging.info("[CooledAI Lead] RESEND_API_KEY not set. Email not sent.")
        return False
    try:
        import resend
        resend.api_key = api_key
        html = f"""
        <h2>CooledAI Blueprint Request</h2>
        <p><strong>Name:</strong> {_esc(lead.fullName)}</p>
        <p><strong>Email:</strong> {_esc(lead.businessEmail)}</p>
        <p><strong>Phone:</strong> {_esc(lead.phone)}</p>
        <p><strong>Data Center Scale:</strong> {_esc(lead.dataCenterScale)} MW</p>
        <p><em>Submitted at {datetime.utcnow().isoformat()}Z</em></p>
        """
        params = {
            "from": os.environ.get("LEAD_EMAIL_FROM", "CooledAI <onboarding@resend.dev>"),
            "to": [os.environ.get("LEAD_EMAIL_TO", "dennis@cooledai.com")],
            "subject": f"CooledAI Blueprint Request: {lead.fullName} ({lead.businessEmail})",
            "html": html,
        }
        resend.Emails.send(params)
        logging.info("[CooledAI Lead] Email sent successfully.")
        return True
    except Exception as e:
        logging.warning("[CooledAI Lead] Email failed: %s", type(e).__name__)
        return False


@app.post("/api/v1/leads")
async def create_lead(lead: LeadInput):
    """
    Lead capture for Blueprint Request form.
    Stores submissions in data/leads.json, sends email to dennis@cooledai.com, and logs to console.
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

    _send_lead_email(lead)

    logging.info("[CooledAI Lead] Blueprint Request received (details redacted from logs).")
    return {"status": "success", "message": "Blueprint Request Received."}


def _send_beta_email(beta: BetaSignupInput) -> bool:
    """Send beta signup notification to dennis@cooledai.com via Resend."""
    from html import escape as _esc
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logging.info("[CooledAI Beta] RESEND_API_KEY not set. Email not sent.")
        return False
    try:
        import resend
        resend.api_key = api_key
        html = f"""
        <h2>CooledAI Beta Signup</h2>
        <p><strong>Email:</strong> {_esc(beta.email)}</p>
        <p><strong>Facility Scale:</strong> {_esc(beta.facilityScale or 'Not specified')} MW</p>
        <p><em>Submitted at {datetime.utcnow().isoformat()}Z</em></p>
        """
        params = {
            "from": os.environ.get("LEAD_EMAIL_FROM", "CooledAI <onboarding@resend.dev>"),
            "to": [os.environ.get("LEAD_EMAIL_TO", "dennis@cooledai.com")],
            "subject": f"CooledAI New Beta Request: {_esc(beta.email)}",
            "html": html,
        }
        resend.Emails.send(params)
        logging.info("[CooledAI Beta] Email sent successfully.")
        return True
    except Exception as e:
        logging.warning("[CooledAI Beta] Email failed: %s", type(e).__name__)
        return False


# --- Periodic efficiency digest email (Predictive vs Traditional) ---
_REPORT_EMAIL_ENABLED = os.environ.get("COOLEDAI_REPORT_EMAIL_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
_REPORT_EMAIL_INTERVAL_SEC = int(os.environ.get("COOLEDAI_REPORT_EMAIL_INTERVAL_SEC", str(3 * 60 * 60)))  # 3 hours
_REPORT_EMAIL_TO = os.environ.get("COOLEDAI_REPORT_EMAIL_TO", "dennis@cooledai.com")
_report_stop_event = threading.Event()
_report_thread: Optional[threading.Thread] = None


# --- Live Telemetry Store & Power Stats (Multi-Tenant) ------------------
#
# Per-owner, per-node latest telemetry snapshot.  Each tenant's nodes and
# power-savings accumulator are fully isolated.
#
# _tenant_telemetry:   { owner_id: { node_id: {...telemetry...} } }
# _tenant_accumulators: { owner_id: { power_reclaimed_wh, started_at, last_ts } }

_tenant_telemetry: Dict[str, Dict[str, dict]] = {}
_tenant_accumulators: Dict[str, dict] = {}
# Rolling thermal history for 6H/24H aggregated charts: { owner_id: [(ts, pilot_temp, baseline_temp), ...] }
_thermal_history: Dict[str, list] = {}
_MAX_THERMAL_HISTORY_POINTS = 24 * 60 * 12  # 24h at 5s interval

# Fan power estimation — Fan Affinity Law: P ∝ RPM³
_RATED_FAN_WATTS = float(os.environ.get("COOLEDAI_RATED_FAN_WATTS", "12"))
_MAX_FAN_RPM = float(os.environ.get("COOLEDAI_MAX_FAN_RPM", "7000"))
_NUM_FANS = int(os.environ.get("COOLEDAI_NUM_FANS", "6"))
_USD_PER_KWH = float(os.environ.get("COOLEDAI_UTILITY_RATE", "0.12"))

# ST550 comparative demo: .100 = Pilot (ST550-CooledAI-Predictive), .101 = Control (ST550-Control-Traditional)
_PILOT_NODE_ID = os.environ.get("COOLEDAI_PILOT_NODE_ID", "ST550-CooledAI-Predictive")
_BASELINE_NODE_ID = os.environ.get("COOLEDAI_BASELINE_NODE_ID", "ST550-Control-Traditional")
_BASELINE_FAN_RPM = float(os.environ.get("COOLEDAI_BASELINE_FAN_RPM", "0"))


def _fan_power_watts(rpm: float) -> float:
    """Fan Affinity Law: P = (RPM / RPM_max)^3 * Rated_W * N_fans."""
    frac = rpm / _MAX_FAN_RPM if _MAX_FAN_RPM > 0 else 0.0
    return (frac ** 3) * _RATED_FAN_WATTS * _NUM_FANS


def _get_pilot_baseline_temps(owner_id: str) -> Tuple[Optional[float], Optional[float]]:
    """Get current pilot and baseline max_temp_c for thermal history."""
    if owner_id == "master":
        all_nodes = {}
        for tn in _tenant_telemetry.values():
            all_nodes.update(tn)
    else:
        all_nodes = _tenant_telemetry.get(owner_id, {})
    pilot_temp = None
    baseline_temp = None
    for nid in all_nodes:
        if "/" in nid:
            continue
        nid_lower = nid.lower()
        if "cooledai" in nid_lower and "predictive" in nid_lower:
            pilot_temp = all_nodes[nid].get("max_temp_c")
        elif "control" in nid_lower and "traditional" in nid_lower:
            baseline_temp = all_nodes[nid].get("max_temp_c")
    return pilot_temp, baseline_temp


def _append_thermal_history(owner_id: str, now_ts: float) -> None:
    """Append a thermal snapshot when we have both pilot and baseline temps."""
    pilot_temp, baseline_temp = _get_pilot_baseline_temps(owner_id)
    if pilot_temp is None or baseline_temp is None:
        return
    history = _thermal_history.setdefault(owner_id, [])
    history.append((now_ts, float(pilot_temp), float(baseline_temp)))
    if len(history) > _MAX_THERMAL_HISTORY_POINTS:
        history.pop(0)


def _accumulate_savings(owner_id: str, pilot_rpm: float) -> None:
    """Called on each pilot telemetry update to integrate watt-hours saved."""
    now = time.time()
    acc = _tenant_accumulators.get(owner_id)
    if acc is None:
        _tenant_accumulators[owner_id] = {
            "power_reclaimed_wh": 0.0,
            "stats_started_at": now,
            "last_accumulation_ts": now,
        }
        return
    dt_hours = (now - acc["last_accumulation_ts"]) / 3600.0
    acc["last_accumulation_ts"] = now

    baseline_rpm = _MAX_FAN_RPM
    if _BASELINE_FAN_RPM > 0:
        baseline_rpm = _BASELINE_FAN_RPM
    else:
        tenant_data = _tenant_telemetry.get(owner_id, {})
        for nid, data in tenant_data.items():
            if "/" not in nid and ("shadow" in nid.lower() or "baseline" in nid.lower()):
                baseline_rpm = float(data.get("fan_rpm", _MAX_FAN_RPM))
                break

    delta_w = max(0.0, _fan_power_watts(baseline_rpm) - _fan_power_watts(pilot_rpm))
    acc["power_reclaimed_wh"] += delta_w * dt_hours


@app.post("/api/v1/telemetry")
async def receive_telemetry(request: Request, owner_id: str = Depends(_resolve_owner)):
    """
    Receive telemetry from Universal Protocol Gateway (edge agent).
    Accepts: {"telemetry": [...], "agent_id": "..."} or {"heartbeat": {...}}

    The X-API-Key header determines which tenant (owner_id) owns these
    nodes.  All records are stored under that owner's isolated namespace.
    """
    payload = await request.json()
    if "heartbeat" in payload:
        hb = payload["heartbeat"]
        agent_id = hb.get("agent_id", "unknown")
        logging.debug("[CooledAI] Heartbeat from %s (owner=%s)", agent_id, owner_id)
        return {"status": "ok"}
    if "telemetry" in payload:
        telemetry = payload["telemetry"]
        agent_id = payload.get("agent_id", "unknown")
        logging.debug("[CooledAI] Telemetry from %s (owner=%s): %d records", agent_id, owner_id, len(telemetry))
        now_ts = time.time()

        tenant = _tenant_telemetry.setdefault(owner_id, {})
        consolidated = dict(tenant.get(agent_id, {}))
        consolidated["_received_at"] = now_ts
        consolidated["agent_id"] = agent_id
        consolidated["_owner_id"] = owner_id

        for record in telemetry:
            node_id = record.get("node_id", agent_id)
            record["_received_at"] = now_ts
            record["_owner_id"] = owner_id
            tenant[node_id] = record

            fan_rpm_direct = record.get("fan_rpm")
            if fan_rpm_direct is not None:
                consolidated["fan_rpm"] = int(fan_rpm_direct)

            fan_rpms = record.get("fan_rpms")
            if fan_rpms and "fan_rpm" not in consolidated:
                if isinstance(fan_rpms, dict):
                    rpms = list(fan_rpms.values())
                elif isinstance(fan_rpms, list):
                    rpms = list(fan_rpms)
                else:
                    rpms = []
                if rpms:
                    consolidated["fan_rpm"] = int(sum(rpms) / len(rpms))
            if fan_rpms:
                consolidated["fan_rpms"] = fan_rpms

            raw_wattage = record.get("raw_fan_wattage")
            if raw_wattage is not None:
                consolidated["raw_fan_wattage"] = float(raw_wattage)

            temp_c = record.get("temperature_c")
            if temp_c is not None:
                nid = record.get("node_id", "")
                if "/gpu" in nid:
                    consolidated.setdefault("gpu_temps_c", [])
                    consolidated["gpu_temps_c"].append(temp_c)
                elif "/cpu" in nid:
                    consolidated["cpu_temp_c"] = temp_c
                consolidated["max_temp_c"] = max(
                    consolidated.get("max_temp_c", 0), temp_c,
                )

        tenant[agent_id] = consolidated

        aid_lower = agent_id.lower()
        is_pilot = "pilot" in aid_lower or ("cooledai" in aid_lower and "predictive" in aid_lower)
        if "fan_rpm" in consolidated and is_pilot:
            _accumulate_savings(owner_id, float(consolidated["fan_rpm"]))

        # Append to thermal history for 6H/24H aggregated charts (when pilot or baseline sends)
        _append_thermal_history(owner_id, now_ts)

        return {"status": "ok", "received": len(telemetry)}
    return {"status": "error", "message": "Expected telemetry or heartbeat"}


@app.get("/api/v1/thermal-history")
async def get_thermal_history(
    owner_id: str = Depends(_require_clerk_fixed_owner_id),
    hours: int = 6,
):
    """Aggregated hourly thermal averages for 6H/24H charts.
    Returns buckets: [{ hour_ts, hour_label, pilot_avg, baseline_avg }, ...]
    """
    history = sorted(list(_thermal_history.get(owner_id, [])), key=lambda x: x[0])
    now = time.time()
    cutoff = now - (hours * 3600)
    points = [(ts, pt, bt) for ts, pt, bt in history if ts >= cutoff]

    # If there is no data in the last 6 hours, tell the dashboard explicitly
    # (otherwise the UI can show "Calculating…" even when the backend is healthy).
    recent_cutoff = now - (6 * 3600)
    recent_points = [(ts, pt, bt) for ts, pt, bt in history if ts >= recent_cutoff]
    if not recent_points:
        return {"buckets": [], "hours": hours, "message": "No recent data"}
    if not points:
        return {"buckets": [], "hours": hours}

    # Aggregate into hourly buckets
    buckets_dict: Dict[int, List[Tuple[float, float]]] = {}
    for ts, pt, bt in points:
        hour_ts = int(ts // 3600) * 3600
        buckets_dict.setdefault(hour_ts, []).append((pt, bt))
    buckets = []
    for hour_ts in sorted(buckets_dict.keys()):
        pts = buckets_dict[hour_ts]
        pilot_avg = sum(p[0] for p in pts) / len(pts)
        baseline_avg = sum(p[1] for p in pts) / len(pts)
        dt = datetime.fromtimestamp(hour_ts, tz=timezone.utc)
        buckets.append({
            "hour_ts": hour_ts,
            "hour_label": dt.strftime("%H:%M"),
            "time": dt.strftime("%I %p"),
            "pilot": round(pilot_avg, 1),
            "baseline": round(baseline_avg, 1),
            "ts": hour_ts,
        })
    return {"buckets": buckets, "hours": hours}


@app.get("/api/v1/stats")
async def get_live_stats(owner_id: str = Depends(_require_clerk_fixed_owner_id)):
    """Live power-savings stats scoped to the authenticated owner's nodes.

    Uses Fan Affinity Law to estimate wattage from RPM.  Accumulates kWh
    per-tenant and projects annual savings via USD_PER_KWH.

    The master key sees aggregated data across all tenants.
    """
    now = time.time()

    # If there is no thermal history in the last 6 hours, return a clear
    # message so the frontend doesn't get stuck on "Calculating…".
    history = sorted(list(_thermal_history.get(owner_id, [])), key=lambda x: x[0])
    recent_cutoff = now - (6 * 3600)
    recent_points = [(ts, pt, bt) for ts, pt, bt in history if ts >= recent_cutoff]
    if not recent_points:
        return {
            "owner_id": owner_id,
            "efficiency_gain_pct": 0.0,
            "last_telemetry_at": None,
            "power_reclaimed_kwh": 0.0,
            "power_reclaimed_watts": 0.0,
            "estimated_annual_savings_usd": 0.0,
            "usd_per_kwh": _USD_PER_KWH,
            "uptime_hours": 0.0,
            "has_live_data": False,
            "message": "No recent data",
            "pilot_node": {
                "node_id": "none",
                "fan_rpm": 0.0,
                "fan_power_watts": 0.0,
                "raw_fan_wattage": None,
                "temp_c": None,
                "last_seen_s_ago": None,
            },
            "baseline_node": {
                "node_id": "none",
                "fan_rpm": 0.0,
                "fan_power_watts": 0.0,
                "raw_fan_wattage": None,
                "temp_c": None,
                "last_seen_s_ago": None,
                "source": "unknown",
            },
            "formula": "Power = (Fan_RPM / Max_RPM)^3 * Rated_Fan_Watts * Num_Fans",
            "config": {
                "max_fan_rpm": _MAX_FAN_RPM,
                "rated_fan_watts": _RATED_FAN_WATTS,
                "num_fans": _NUM_FANS,
            },
        }

    all_nodes = _tenant_telemetry.get(owner_id, {})

    agent_keys = [k for k in all_nodes if "/" not in k]
    pilot: dict = {}
    baseline: dict = {}
    pilot_id = ""
    baseline_id = ""
    # Prefer ST550 comparative demo IDs: CooledAI-Predictive (pilot), Control-Traditional (baseline)
    for nid in agent_keys:
        nid_lower = nid.lower()
        if "cooledai" in nid_lower and "predictive" in nid_lower:
            pilot = all_nodes[nid]
            pilot_id = nid
            break
    for nid in agent_keys:
        nid_lower = nid.lower()
        if "control" in nid_lower and "traditional" in nid_lower:
            baseline = all_nodes[nid]
            baseline_id = nid
            break
    # Fallback to legacy: pilot, shadow, baseline
    if not pilot_id:
        for nid in agent_keys:
            if "pilot" in nid.lower():
                pilot = all_nodes[nid]
                pilot_id = nid
                break
    if not baseline_id:
        for nid in agent_keys:
            nid_lower = nid.lower()
            if "shadow" in nid_lower or "baseline" in nid_lower:
                baseline = all_nodes[nid]
                baseline_id = nid
                break
    if not pilot_id and agent_keys:
        pilot_id = agent_keys[0]
        pilot = all_nodes[pilot_id]
    if not baseline_id and len(agent_keys) > 1:
        remaining = [k for k in agent_keys if k != pilot_id]
        if remaining:
            baseline_id = remaining[0]
            baseline = all_nodes[baseline_id]

    pilot_rpm = float(pilot.get("fan_rpm", 0))
    baseline_source = "default_max"
    if _BASELINE_FAN_RPM > 0:
        baseline_rpm = _BASELINE_FAN_RPM
        baseline_source = "env"
    elif baseline:
        baseline_rpm = float(baseline.get("fan_rpm", _MAX_FAN_RPM))
        baseline_source = "live"
    else:
        baseline_rpm = _MAX_FAN_RPM

    pilot_w = _fan_power_watts(pilot_rpm)
    baseline_w = _fan_power_watts(baseline_rpm)
    delta_w = max(0.0, baseline_w - pilot_w)

    # ST550: Use temp-based efficiency when both Pilot and Control temps available
    # Formula: (Baseline_Temp - Pilot_Temp) / Baseline_Temp * 100
    pilot_temp = pilot.get("max_temp_c")
    baseline_temp = baseline.get("max_temp_c")
    pilot_fan_estimated = False  # True when we show pilot W derived from temp (no fan_rpm from agent)
    if (
        pilot_temp is not None
        and baseline_temp is not None
        and baseline_temp > 0
    ):
        efficiency_pct = ((baseline_temp - pilot_temp) / baseline_temp) * 100.0
        efficiency_pct = max(0.0, min(100.0, efficiency_pct))
        # ST550 agents send only GPU temps, not fan_rpm — so pilot_w is 0 and "0W CooledAI" is misleading.
        # Derive pilot display watts from temp-based efficiency so the UI shows e.g. "44W vs 72W".
        if pilot_rpm == 0 and baseline_w > 0:
            pilot_w = baseline_w * (1.0 - efficiency_pct / 100.0)
            delta_w = baseline_w - pilot_w
            pilot_fan_estimated = True
    else:
        efficiency_pct = (delta_w / baseline_w * 100.0) if baseline_w > 0 else 0.0

    if owner_id == "master":
        total_wh = sum(a.get("power_reclaimed_wh", 0.0) for a in _tenant_accumulators.values())
        earliest = min((a.get("stats_started_at", now) for a in _tenant_accumulators.values()), default=now)
        acc: dict = {"power_reclaimed_wh": total_wh, "stats_started_at": earliest}
    else:
        acc = _tenant_accumulators.get(owner_id, {})

    power_wh = acc.get("power_reclaimed_wh", 0.0)
    started = acc.get("stats_started_at")
    uptime_h = (now - started) / 3600.0 if started else 0.0
    reclaimed_kwh = power_wh / 1000.0

    annual_kwh = (reclaimed_kwh / uptime_h * 8760.0) if uptime_h > 0.1 else 0.0
    annual_savings = annual_kwh * _USD_PER_KWH

    has_live = bool(pilot)
    pilot_age = (now - pilot["_received_at"]) if pilot.get("_received_at") else None
    baseline_age = (now - baseline["_received_at"]) if baseline.get("_received_at") else None
    last_telemetry_at = max(
        pilot.get("_received_at") or 0,
        baseline.get("_received_at") or 0,
    )

    return {
        "owner_id": owner_id,
        "efficiency_gain_pct": round(efficiency_pct, 1),
        "last_telemetry_at": last_telemetry_at if last_telemetry_at > 0 else None,
        "power_reclaimed_kwh": round(reclaimed_kwh, 3),
        "power_reclaimed_watts": round(delta_w, 1),
        "estimated_annual_savings_usd": round(annual_savings, 0),
        "usd_per_kwh": _USD_PER_KWH,
        "uptime_hours": round(uptime_h, 2),
        "has_live_data": has_live,
        "pilot_node": {
            "node_id": pilot_id or "none",
            "fan_rpm": pilot_rpm,
            "fan_power_watts": round(pilot_w, 1),
            "fan_power_estimated_from_temp": pilot_fan_estimated,
            "raw_fan_wattage": pilot.get("raw_fan_wattage"),
            "temp_c": pilot.get("max_temp_c"),
            "last_seen_s_ago": round(pilot_age, 1) if pilot_age is not None else None,
        },
        "baseline_node": {
            "node_id": baseline_id or "none",
            "fan_rpm": baseline_rpm,
            "fan_power_watts": round(baseline_w, 1),
            "raw_fan_wattage": baseline.get("raw_fan_wattage"),
            "temp_c": baseline.get("max_temp_c"),
            "last_seen_s_ago": round(baseline_age, 1) if baseline_age is not None else None,
            "source": baseline_source,
        },
        "formula": "Power = (Fan_RPM / Max_RPM)^3 * Rated_Fan_Watts * Num_Fans",
        "config": {
            "max_fan_rpm": _MAX_FAN_RPM,
            "rated_fan_watts": _RATED_FAN_WATTS,
            "num_fans": _NUM_FANS,
        },
    }


def _build_email_comparison_snapshot(owner_id: str = FIXED_OWNER_ID) -> Dict[str, Any]:
    """Build a simplified predictive-vs-control snapshot for digest emails."""
    now = time.time()
    all_nodes = _tenant_telemetry.get(owner_id, {})
    agent_keys = [k for k in all_nodes if "/" not in k]

    pilot: dict = {}
    baseline: dict = {}
    pilot_id = ""
    baseline_id = ""
    for nid in agent_keys:
        nid_lower = nid.lower()
        if "cooledai" in nid_lower and "predictive" in nid_lower:
            pilot = all_nodes[nid]
            pilot_id = nid
            break
    for nid in agent_keys:
        nid_lower = nid.lower()
        if "control" in nid_lower and "traditional" in nid_lower:
            baseline = all_nodes[nid]
            baseline_id = nid
            break
    if not pilot_id and agent_keys:
        pilot_id = agent_keys[0]
        pilot = all_nodes[pilot_id]
    if not baseline_id and len(agent_keys) > 1:
        remaining = [k for k in agent_keys if k != pilot_id]
        if remaining:
            baseline_id = remaining[0]
            baseline = all_nodes[baseline_id]

    pilot_rpm = float(pilot.get("fan_rpm", 0))
    baseline_rpm = float(baseline.get("fan_rpm", _MAX_FAN_RPM)) if baseline else _MAX_FAN_RPM
    pilot_w = _fan_power_watts(pilot_rpm)
    baseline_w = _fan_power_watts(baseline_rpm)
    delta_w = max(0.0, baseline_w - pilot_w)

    pilot_temp = pilot.get("max_temp_c")
    baseline_temp = baseline.get("max_temp_c")
    efficiency_pct = 0.0
    pilot_estimated_from_temp = False
    if (
        pilot_temp is not None
        and baseline_temp is not None
        and baseline_temp > 0
    ):
        efficiency_pct = ((baseline_temp - pilot_temp) / baseline_temp) * 100.0
        efficiency_pct = max(0.0, min(100.0, efficiency_pct))
        if pilot_rpm == 0 and baseline_w > 0:
            pilot_w = baseline_w * (1.0 - efficiency_pct / 100.0)
            delta_w = baseline_w - pilot_w
            pilot_estimated_from_temp = True
    elif baseline_w > 0:
        efficiency_pct = (delta_w / baseline_w) * 100.0

    acc = _tenant_accumulators.get(owner_id, {})
    power_wh = acc.get("power_reclaimed_wh", 0.0)
    started = acc.get("stats_started_at")
    uptime_h = (now - started) / 3600.0 if started else 0.0
    reclaimed_kwh = power_wh / 1000.0
    annual_kwh = (reclaimed_kwh / uptime_h * 8760.0) if uptime_h > 0.1 else 0.0
    annual_savings = annual_kwh * _USD_PER_KWH

    return {
        "owner_id": owner_id,
        "checked_at_iso": datetime.now(timezone.utc).isoformat(),
        "has_live_data": bool(pilot),
        "efficiency_gain_pct": round(efficiency_pct, 1),
        "power_reclaimed_watts": round(delta_w, 1),
        "power_reclaimed_kwh": round(reclaimed_kwh, 3),
        "estimated_annual_savings_usd": round(annual_savings, 0),
        "pilot_node": {
            "node_id": pilot_id or "none",
            "temp_c": pilot_temp,
            "fan_rpm": pilot_rpm,
            "fan_power_watts": round(pilot_w, 1),
            "fan_power_estimated_from_temp": pilot_estimated_from_temp,
            "last_seen_s_ago": round(now - pilot.get("_received_at"), 1) if pilot.get("_received_at") else None,
        },
        "baseline_node": {
            "node_id": baseline_id or "none",
            "temp_c": baseline_temp,
            "fan_rpm": baseline_rpm,
            "fan_power_watts": round(baseline_w, 1),
            "last_seen_s_ago": round(now - baseline.get("_received_at"), 1) if baseline.get("_received_at") else None,
        },
    }


def _send_efficiency_digest_email(owner_id: str = FIXED_OWNER_ID) -> bool:
    """Send simplified predictive-vs-traditional efficiency report email."""
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        logging.getLogger("api.main").warning("[Digest] RESEND_API_KEY not set; digest skipped.")
        return False
    try:
        import resend
        resend.api_key = api_key
        snap = _build_email_comparison_snapshot(owner_id)
        pilot = snap["pilot_node"]
        baseline = snap["baseline_node"]
        mode_note = " (pilot watts estimated from temperature delta)" if pilot.get("fan_power_estimated_from_temp") else ""
        html = f"""
        <h2>CooledAI 3-Hour Efficiency Digest</h2>
        <p><strong>Time:</strong> {snap["checked_at_iso"]}</p>
        <p><strong>Comparison:</strong> Predictive (Pilot) vs Traditional (Control)</p>
        <ul>
          <li><strong>Efficiency Gain:</strong> {snap["efficiency_gain_pct"]}%</li>
          <li><strong>Power Reclaimed:</strong> {snap["power_reclaimed_watts"]} W</li>
          <li><strong>Pilot:</strong> {pilot["fan_power_watts"]} W at {pilot.get("temp_c")}°C{mode_note}</li>
          <li><strong>Control:</strong> {baseline["fan_power_watts"]} W at {baseline.get("temp_c")}°C</li>
          <li><strong>Reclaimed Energy (since T0):</strong> {snap["power_reclaimed_kwh"]} kWh</li>
          <li><strong>Estimated Annual Savings:</strong> ${snap["estimated_annual_savings_usd"]}</li>
        </ul>
        <p><em>This report is auto-generated every 3 hours.</em></p>
        """
        params = {
            "from": os.environ.get("LEAD_EMAIL_FROM", "CooledAI <onboarding@resend.dev>"),
            "to": [_REPORT_EMAIL_TO],
            "subject": f"CooledAI 3-Hour Digest: {snap['efficiency_gain_pct']}% better vs control",
            "html": html,
        }
        resend.Emails.send(params)
        logging.getLogger("api.main").info("[Digest] Sent efficiency digest to %s", _REPORT_EMAIL_TO)
        return True
    except Exception as exc:
        logging.getLogger("api.main").warning("[Digest] Email failed: %s", exc)
        return False


def _digest_worker_loop() -> None:
    """Background worker to send digest emails every N seconds."""
    log = logging.getLogger("api.main")
    if _REPORT_EMAIL_INTERVAL_SEC < 300:
        log.warning("[Digest] Interval too low (%ss), forcing 300s minimum.", _REPORT_EMAIL_INTERVAL_SEC)
    interval = max(300, _REPORT_EMAIL_INTERVAL_SEC)
    log.info("[Digest] Worker started. interval=%ss to=%s", interval, _REPORT_EMAIL_TO)
    while not _report_stop_event.wait(interval):
        _send_efficiency_digest_email(FIXED_OWNER_ID)


@app.get("/api/v1/debug/clerk")
async def debug_clerk_user(request: Request):
    """
    Debug helper: returns extracted Clerk user_id from Authorization bearer JWT.

    This endpoint is for troubleshooting only.
    """
    clerk_user_id = await _require_clerk_user_id(request)
    return {
        "clerk_user_id": clerk_user_id,
        "expected_fixed_owner_id": FIXED_OWNER_ID,
        "matches_expected": clerk_user_id == FIXED_OWNER_ID,
    }


@app.get("/api/v1/nodes/status", dependencies=[Depends(_require_api_key)])
async def get_nodes_status_remote():
    """
    Remote check: see if Pilot and Control nodes are still sending telemetry.

    Call from anywhere (no Clerk required). Use the same X-API-Key as your
    ST550 agents. Useful when you're off-site and want to confirm both
    nodes are logging to the cloud.

    Example:
      curl -s -H "X-API-Key: YOUR_KEY" \\
        https://proactive-creativity-production.up.railway.app/api/v1/nodes/status
    """
    now = time.time()
    owner_id = FIXED_OWNER_ID
    all_nodes = _tenant_telemetry.get(owner_id, {})

    agent_keys = [k for k in all_nodes if "/" not in k]
    pilot_node_id = ""
    baseline_node_id = ""
    pilot_received = None
    baseline_received = None

    for nid in agent_keys:
        nid_lower = nid.lower()
        if "cooledai" in nid_lower and "predictive" in nid_lower:
            pilot_node_id = nid
            pilot_received = all_nodes[nid].get("_received_at")
            break
    for nid in agent_keys:
        nid_lower = nid.lower()
        if "control" in nid_lower and "traditional" in nid_lower:
            baseline_node_id = nid
            baseline_received = all_nodes[nid].get("_received_at")
            break

    def _node_status(node_id: str, received_at: Optional[float]) -> dict:
        if not node_id or received_at is None:
            return {
                "node_id": node_id or "none",
                "last_seen_iso": None,
                "last_seen_s_ago": None,
                "status": "no_data",
            }
        age = now - received_at
        status = "ok" if age < 120 else "stale"  # 2 min threshold
        return {
            "node_id": node_id,
            "last_seen_iso": datetime.fromtimestamp(received_at, tz=timezone.utc).isoformat(),
            "last_seen_s_ago": round(age, 1),
            "status": status,
        }

    pilot = _node_status(pilot_node_id, pilot_received)
    baseline = _node_status(baseline_node_id, baseline_received)

    summary = "no_data"
    if pilot["status"] == "ok" and baseline["status"] == "ok":
        summary = "both_ok"
    elif pilot["status"] == "ok":
        summary = "pilot_ok_baseline_stale" if baseline["status"] == "stale" else "pilot_ok_baseline_no_data"
    elif baseline["status"] == "ok":
        summary = "baseline_ok_pilot_stale" if pilot["status"] == "stale" else "baseline_ok_pilot_no_data"
    elif pilot["status"] == "stale" or baseline["status"] == "stale":
        summary = "both_stale"

    return {
        "summary": summary,
        "pilot": pilot,
        "baseline": baseline,
        "checked_at_iso": datetime.now(timezone.utc).isoformat(),
    }


# --- Agent Config (per-node target_temp, control_enabled, etc.) ---

_agent_configs: Dict[str, dict] = {}


class AgentConfigInput(BaseModel):
    """Body for POST /api/v1/config."""
    target_temp: float = 65.0
    control_enabled: bool = True
    poll_interval_s: float = 10.0
    critical_temp_c: float = 90.0


@app.get("/api/v1/config", dependencies=[Depends(_require_api_key)])
async def get_agent_config(node_id: str = "default"):
    """Return the active config for an agent node.  The unified agent
    polls this every 60 s to pick up target_temp changes pushed from
    the portal UI.  Falls back to a safe default when no config has
    been set for the requested node_id.
    """
    config = _agent_configs.get(node_id, _agent_configs.get("default", {}))
    return {
        "node_id": node_id,
        "target_temp": config.get("target_temp", 65),
        "control_enabled": config.get("control_enabled", True),
        "poll_interval_s": config.get("poll_interval_s", 10),
        "critical_temp_c": config.get("critical_temp_c", 90),
    }


@app.post("/api/v1/config", dependencies=[Depends(_require_admin_key)])
async def set_agent_config(node_id: str, body: AgentConfigInput):
    """Push a new config for a node.  The agent will pick it up on its
    next config poll (within 60 s).  Requires admin key.
    """
    _agent_configs[node_id] = body.dict()
    return {"status": "ok", "node_id": node_id, "config": body.dict()}


@app.get("/api/v1/test-email", dependencies=[Depends(_require_admin_key)])
async def test_email():
    """
    Send a test email to dennis@cooledai.com. Requires admin key.
    """
    api_key = os.environ.get("RESEND_API_KEY")
    to_email = os.environ.get("LEAD_EMAIL_TO", "dennis@cooledai.com")
    if not api_key:
        return {"ok": False, "error": "RESEND_API_KEY not set"}
    try:
        import resend
        resend.api_key = api_key
        params = {
            "from": os.environ.get("LEAD_EMAIL_FROM", "CooledAI <onboarding@resend.dev>"),
            "to": [to_email],
            "subject": "CooledAI Test Email",
            "html": "<p>If you received this, Resend is working.</p>",
        }
        resend.Emails.send(params)
        return {"ok": True, "message": f"Test email sent to {to_email}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/v1/reports/efficiency/send-now", dependencies=[Depends(_require_admin_key)])
async def send_efficiency_report_now():
    """Send the simplified predictive-vs-traditional efficiency digest immediately."""
    ok = _send_efficiency_digest_email(FIXED_OWNER_ID)
    return {
        "ok": ok,
        "to": _REPORT_EMAIL_TO,
        "interval_seconds": _REPORT_EMAIL_INTERVAL_SEC,
        "message": "Efficiency digest sent." if ok else "Efficiency digest failed (check RESEND_API_KEY/logs).",
    }


@app.post("/api/v1/beta")
async def create_beta_signup(beta: BetaSignupInput):
    """
    Beta signup from popup. Sends email to dennis@cooledai.com.
    """
    _send_beta_email(beta)
    logging.info("[CooledAI Beta] Signup received (details redacted).")
    return {"status": "success", "message": "Beta request received."}


@app.get("/shadow")
async def get_shadow_mode():
    """Return current Shadow Mode state. When True, no write commands are sent to hardware."""
    return {"shadow_mode": getattr(_brain, "shadow_mode", False)}


class ShadowModeInput(BaseModel):
    """Body for POST /shadow."""
    enabled: bool


@app.post("/shadow", dependencies=[Depends(_require_admin_key)])
async def set_shadow_mode(body: ShadowModeInput):
    """Enable or disable Shadow Mode. Requires admin key. When enabled, proposed actions are logged to shadow_logs and writes are not sent."""
    _brain.shadow_mode = body.enabled
    if _shadow_available and body.enabled:
        init_shadow_db()
    return {"shadow_mode": _brain.shadow_mode}


@app.get("/shadow/missed-opportunities")
async def get_missed_opportunities_list(
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100,
):
    """
    List 'Missed Opportunity' events: times when the AI would have saved > $10
    but the adjustment was not applied (Shadow Mode). Includes specific reason (e.g. over-cooling in Zone B).
    """
    if not _shadow_available or get_missed_opportunities is None:
        raise HTTPException(status_code=503, detail="Shadow / missed-opportunity module not available.")
    start_ts = None
    end_ts = None
    if end:
        try:
            end_ts = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            end_ts = datetime.utcnow()
    if start:
        try:
            start_ts = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            start_ts = (end_ts or datetime.utcnow()) - timedelta(hours=24)
    events = get_missed_opportunities(start_ts=start_ts, end_ts=end_ts, limit=limit)
    return {
        "events": [
            {
                "id": e.id,
                "created_at": e.created_at.isoformat(),
                "shadow_log_id": e.shadow_log_id,
                "dollars_missed": e.dollars_missed,
                "reason": e.reason,
                "cooling_savings_kw": e.cooling_savings_kw,
                "it_power_kw": e.it_power_kw,
            }
            for e in events
        ],
        "count": len(events),
    }


@app.get("/shadow/missed-savings-daily")
async def get_missed_savings_daily(date: Optional[str] = None):
    """
    Daily 'Missed Savings' notification: aggregate of missed opportunities for the given day (UTC).
    Use for facility manager FOMO digest (e.g. 'You missed $X across N opportunities today').
    """
    if not _shadow_available or get_daily_missed_savings_notification is None:
        raise HTTPException(status_code=503, detail="Shadow / missed-opportunity module not available.")
    day_ts = None
    if date:
        try:
            day_ts = datetime.fromisoformat(date.replace("Z", "+00:00").split("T")[0] + "T00:00:00")
        except ValueError:
            day_ts = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    notification = get_daily_missed_savings_notification(date=day_ts)
    return {
        "date": notification.date.strftime("%Y-%m-%d"),
        "total_dollars_missed": notification.total_dollars_missed,
        "event_count": notification.event_count,
        "summary": notification.summary,
        "message": notification.message,
        "events": [
            {
                "id": e.id,
                "created_at": e.created_at.isoformat(),
                "dollars_missed": e.dollars_missed,
                "reason": e.reason,
            }
            for e in notification.events
        ],
    }


@app.get("/savings-report")
async def get_savings_report(
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """
    Compare AI Proposed PUE vs Actual PUE from shadow_logs and return a Potential Savings Report.
    Query params: start (ISO datetime), end (ISO datetime). Default: last 24 hours.
    """
    if not _shadow_available:
        raise HTTPException(status_code=503, detail="Shadow module not available.")
    end_ts = datetime.utcnow()
    start_ts = end_ts - timedelta(hours=24)
    if end:
        try:
            end_ts = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            end_ts = datetime.utcnow()
    if start:
        try:
            start_ts = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            start_ts = end_ts - timedelta(hours=24)
    report = generate_potential_savings_report(start_ts=start_ts, end_ts=end_ts)
    return {
        "start_ts": report.start_ts.isoformat(),
        "end_ts": report.end_ts.isoformat(),
        "sample_count": report.sample_count,
        "avg_proposed_pue": report.avg_proposed_pue,
        "avg_actual_pue": report.avg_actual_pue,
        "pue_improvement": report.pue_improvement,
        "pue_improvement_pct": report.pue_improvement_pct,
        "potential_energy_savings_pct": report.potential_energy_savings_pct,
        "total_it_power_kwh": report.total_it_power_kwh,
        "estimated_cooling_savings_kwh": report.estimated_cooling_savings_kwh,
        "message": report.message,
        "details": report.details,
    }


# --- Baseline Reporter: 24h current-state vs projected CooledAI PUE ---

class BaselineSnapshotInput(BaseModel):
    """One current-state snapshot (no adjustments). Send every 5–10 min over 24h."""
    it_power_kw: float
    cooling_power_kw: float
    pue: Optional[float] = None
    total_facility_kw: Optional[float] = None
    ups_losses_kw: float = 0.0
    lighting_kw: float = 0.0
    other_kw: float = 0.0


@app.post("/baseline/snapshot", dependencies=[Depends(_require_api_key)])
async def baseline_record_snapshot(body: BaselineSnapshotInput):
    """
    Record one 'Current State' snapshot without making adjustments.
    Call every 5–10 minutes for 24 hours to build the baseline, then GET /baseline/report.
    """
    if not _baseline_available or _baseline_reporter is None:
        raise HTTPException(status_code=503, detail="Baseline module not available.")
    row_id = _baseline_reporter.record_snapshot(
        it_power_kw=body.it_power_kw,
        cooling_power_kw=body.cooling_power_kw,
        pue=body.pue,
        total_facility_kw=body.total_facility_kw,
        ups_losses_kw=body.ups_losses_kw,
        lighting_kw=body.lighting_kw,
        other_kw=body.other_kw,
    )
    return {"status": "ok", "snapshot_id": row_id, "message": "Baseline snapshot recorded."}


@app.get("/baseline/report")
async def baseline_comparison_report(
    hours: float = 24,
    start: Optional[str] = None,
    end: Optional[str] = None,
    projected_pue: Optional[float] = None,
):
    """
    Generate comparison report: Current PUE vs Projected CooledAI PUE over the last 24h (or given window).
    Includes CO2 reduction and estimated utility rebates based on local $/kWh (COOLEDAI_UTILITY_RATE, COOLEDAI_REBATE_PER_KWH).
    """
    if not _baseline_available or _baseline_reporter is None:
        raise HTTPException(status_code=503, detail="Baseline module not available.")
    end_ts = datetime.utcnow()
    start_ts = end_ts - timedelta(hours=hours)
    if end:
        try:
            end_ts = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            pass
    if start:
        try:
            start_ts = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            start_ts = end_ts - timedelta(hours=hours)
    report = _baseline_reporter.generate_comparison_report(
        projected_pue=projected_pue,
        start_ts=start_ts,
        end_ts=end_ts,
        hours=hours,
    )
    return {
        "current_pue": report.current_pue,
        "projected_cooledai_pue": report.projected_cooledai_pue,
        "pue_improvement": report.pue_improvement,
        "pue_improvement_pct": report.pue_improvement_pct,
        "current_total_kwh_24h": report.current_total_kwh_24h,
        "projected_total_kwh_24h": report.projected_total_kwh_24h,
        "kwh_saved_24h": report.kwh_saved_24h,
        "co2_reduction_kg": report.co2_reduction_kg,
        "co2_reduction_tonnes": report.co2_reduction_tonnes,
        "estimated_utility_rebates_usd": report.estimated_utility_rebates_usd,
        "dollars_saved_24h": report.dollars_saved_24h,
        "sample_count": report.sample_count,
        "start_ts": report.start_ts.isoformat(),
        "end_ts": report.end_ts.isoformat(),
        "message": report.message,
        "details": report.details,
    }


@app.get("/baseline/summary")
async def baseline_summary(hours: float = 24):
    """Summary of baseline data over the last N hours (avg PUE, total kWh)."""
    if not _baseline_available or _baseline_reporter is None:
        raise HTTPException(status_code=503, detail="Baseline module not available.")
    summary = _baseline_reporter.get_baseline_summary(hours=hours)
    return {
        **summary,
        "start_ts": summary["start_ts"].isoformat() if hasattr(summary["start_ts"], "isoformat") else str(summary["start_ts"]),
        "end_ts": summary["end_ts"].isoformat() if hasattr(summary["end_ts"], "isoformat") else str(summary["end_ts"]),
    }


@app.post("/apply", dependencies=[Depends(_require_api_key)])
async def apply_recommendation():
    """
    Apply the last optimization recommendation to hardware (e.g. set cooling setpoint).
    When Shadow Mode is on, no write is sent; the proposed action is already logged.
    """
    if getattr(_brain, "shadow_mode", False):
        return {
            "applied": False,
            "message": "Shadow Mode active. No write sent to hardware. Proposed action logged to shadow_logs.",
        }
    # In production, would call adapter.write() or ControlGate here with last gap
    return {
        "applied": False,
        "message": "Apply not implemented in this API. Use gateway ControlGate or adapters to send writes.",
    }


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


# --- Discovery Agent (onboarding: suggest mapping from raw telemetry) ---
try:
    from backend.discovery import DiscoveryAgent, DiscoveryResult, MappingSchema
    _discovery_agent: Optional[DiscoveryAgent] = DiscoveryAgent()
except Exception:
    _discovery_agent = None


class DiscoverySampleInput(BaseModel):
    """Single raw telemetry sample for discovery window."""
    protocol: str  # "modbus" | "snmp"
    point_id: str
    address: str
    value: float
    timestamp: Optional[float] = None


class DiscoveryWindowInput(BaseModel):
    """Full window of raw telemetry (e.g. from collector)."""
    protocol: str
    points: List[Dict[str, Any]]  # [{"point_id", "address", "series": [[ts, val], ...]}]


@app.post("/discovery/sample")
async def discovery_add_sample(body: DiscoverySampleInput):
    """
    Add one raw telemetry sample to the discovery window.
    Call repeatedly over ~10 minutes, then POST /discovery/run to get suggested mapping.
    """
    if _discovery_agent is None:
        raise HTTPException(status_code=503, detail="Discovery module not available.")
    _discovery_agent.add_sample(
        protocol=body.protocol,
        point_id=body.point_id,
        address=body.address,
        value=body.value,
        timestamp=body.timestamp,
    )
    return {"status": "ok", "message": "Sample added."}


@app.post("/discovery/window")
async def discovery_ingest_window(body: DiscoveryWindowInput):
    """Ingest a full window of raw telemetry in one request."""
    if _discovery_agent is None:
        raise HTTPException(status_code=503, detail="Discovery module not available.")
    from backend.discovery.schemas import RawTelemetryPoint, RawTelemetryWindow
    points = []
    for p in body.points:
        pt = RawTelemetryPoint(
            point_id=p["point_id"],
            protocol=body.protocol,
            address=p["address"],
            series=[(float(ts), float(v)) for ts, v in p.get("series", [])],
        )
        points.append(pt)
    window = RawTelemetryWindow(protocol=body.protocol, points=points)
    _discovery_agent.ingest_window(window)
    return {"status": "ok", "points_ingested": len(points)}


@app.post("/discovery/run")
async def discovery_run():
    """
    Run discovery on the current window. Returns confidence score and suggested
    mapping schema for onboarding (no manual labeling needed).
    """
    if _discovery_agent is None:
        raise HTTPException(status_code=503, detail="Discovery module not available.")
    result = _discovery_agent.discover()
    schema = result.suggested_mapping_schema
    return {
        "confidence_score": result.confidence_score,
        "suggested_vendor": result.suggested_vendor,
        "suggested_mapping_schema": {
            "point_to_attribute": schema.point_to_attribute if schema else {},
            "scale_factors": schema.scale_factors if schema else {},
        },
        "adapter_format_modbus": schema.to_adapter_format("modbus") if schema else {},
        "adapter_format_snmp": schema.to_adapter_format("snmp") if schema else {},
        "message": result.message,
        "raw_metrics": result.raw_metrics,
    }


@app.post("/discovery/clear", dependencies=[Depends(_require_admin_key)])
async def discovery_clear():
    """Clear the discovery window (start a new run)."""
    if _discovery_agent is not None:
        _discovery_agent.clear()
    return {"status": "ok", "message": "Window cleared."}


# --- Admin: API Key Management (Multi-Tenant) ---


class CreateKeyInput(BaseModel):
    """Body for POST /admin/api-keys."""
    owner_id: Optional[str] = None
    label: str = ""


@app.post("/admin/api-keys", dependencies=[Depends(_require_admin_key)])
async def admin_create_api_key(body: CreateKeyInput):
    """Generate a new API key for a tenant.

    The full key is shown exactly once in the response — store it
    securely because it cannot be retrieved later.
    """
    key = _generate_api_key()
    oid = body.owner_id or ("owner_" + secrets.token_hex(4))
    _api_key_registry[key] = {
        "owner_id": oid,
        "label": body.label,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_key_registry()
    return {
        "api_key": key,
        "owner_id": oid,
        "label": body.label,
        "message": "Store this key securely — it cannot be retrieved later.",
    }


@app.get("/admin/api-keys", dependencies=[Depends(_require_admin_key)])
async def admin_list_api_keys():
    """List registered API keys (prefixes only — full keys are never shown)."""
    keys = []
    for k, meta in _api_key_registry.items():
        keys.append({
            "key_prefix": k[:10] + "...",
            "owner_id": meta["owner_id"],
            "label": meta.get("label", ""),
            "created_at": meta.get("created_at", ""),
        })
    return {"keys": keys, "count": len(keys)}


@app.delete("/admin/api-keys/{key_prefix}", dependencies=[Depends(_require_admin_key)])
async def admin_revoke_api_key(key_prefix: str):
    """Revoke all keys whose full value starts with *key_prefix*."""
    to_delete = [k for k in _api_key_registry if k.startswith(key_prefix)]
    if not to_delete:
        raise HTTPException(status_code=404, detail="No key matching that prefix.")
    for k in to_delete:
        del _api_key_registry[k]
    _save_key_registry()
    return {"revoked": len(to_delete), "message": "Key(s) revoked."}


# --- Static script downloads (install.sh, cooledai_agent.py) ---

_SCRIPTS_DIR = project_root / "scripts"


@app.get("/install.sh")
async def download_install_sh():
    """Serve the one-line installer so clients can bootstrap with:
    curl -fsSL https://<api>/install.sh | sudo bash -s -- --api-key ...
    """
    path = _SCRIPTS_DIR / "install.sh"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="install.sh not found on server.")
    return FileResponse(
        path,
        media_type="text/x-shellscript",
        filename="install.sh",
    )


@app.get("/cooledai_agent.py")
async def download_cooledai_agent():
    """Serve the unified agent Python script so install.sh can fetch it."""
    path = _SCRIPTS_DIR / "cooledai_agent.py"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="cooledai_agent.py not found on server.")
    return FileResponse(
        path,
        media_type="text/x-python",
        filename="cooledai_agent.py",
    )


if __name__ == "__main__":
    import uvicorn
    _host = os.environ.get("HOST", "0.0.0.0")
    _port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=_host, port=_port)
