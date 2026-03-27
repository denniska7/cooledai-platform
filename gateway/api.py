"""
CooledAI Edge Gateway — FastAPI application.

Exposes the same /api/v1/optimize/control endpoint contract as the cloud API
so agents only need to change COOLEDAI_BACKEND_URL to switch targets.

Runs on port 8080 (configurable via COOLEDAI_GATEWAY_PORT), bound to 0.0.0.0.
"""

import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request

from core.models.agent_input import AgentOptimizeControlInput

logger = logging.getLogger("cooledai.gateway.api")

GATEWAY_PORT = int(os.environ.get("COOLEDAI_GATEWAY_PORT", "8080"))

# ---------------------------------------------------------------------------
# Control node detection + latest snapshots (module-level, thread-safe)
# ---------------------------------------------------------------------------

CONTROL_NODE_PREFIXES = ["control", "traditional", "baseline"]

_latest_snapshots: Dict[str, dict] = {}
_latest_snapshots_lock = threading.Lock()

# Rate-of-change temperature history: node_id -> [(ts, temp_c), ...]
_temp_history: Dict[str, list] = {}
_temp_history_lock = threading.Lock()
_TEMP_HISTORY_MAX = 60  # Keep last 60 readings (~3 min at 3s interval)


def _is_control_node(node_id: str) -> bool:
    """Return True if node_id is a control/baseline node (no optimization)."""
    lower = node_id.lower()
    return any(prefix in lower for prefix in CONTROL_NODE_PREFIXES)


def _update_latest_snapshot(node_id: str, body: AgentOptimizeControlInput) -> None:
    """Record latest telemetry snapshot for a node (thread-safe)."""
    snap = {
        "temp_c": float(body.temp_c),
        "fan_rpm": float(body.fan_rpm),
        "cpu_temp_c": float(body.cpu_temp_c) if body.cpu_temp_c else None,
        "gpu_power_w": float(body.gpu_power_w),
        "peak_power_w": float(body.peak_power_w) if body.peak_power_w else None,
        "ts": time.time(),
    }
    with _latest_snapshots_lock:
        _latest_snapshots[node_id] = snap


def _get_latest_control_snapshot() -> Optional[dict]:
    """Return the most recent control-node snapshot (if fresh, <30s)."""
    with _latest_snapshots_lock:
        for nid, snap in _latest_snapshots.items():
            if _is_control_node(nid) and (time.time() - snap.get("ts", 0)) < 30:
                return snap
    return None


# ---------------------------------------------------------------------------
# Thermal history helpers (module-level, used by route handlers)
# ---------------------------------------------------------------------------

def _compose_thermal_row(
    body: AgentOptimizeControlInput,
    result: dict,
    baseline_snapshot: Optional[dict] = None,
) -> tuple:
    """Build 11-tuple thermal history row.

    If baseline_snapshot is provided (from control node), use its values
    for baseline columns. Otherwise, use pilot telemetry as baseline.

    Row format: (ts, pilot_temp, baseline_temp, pilot_rpm, baseline_rpm,
                  pilot_cpu, baseline_cpu, pilot_gpu_pwr, baseline_gpu_pwr,
                  pilot_peak_pwr, baseline_peak_pwr)
    """
    optimized_rpm = result.get("target_rpm") or (
        result.get("target_duty", 0) / 100.0 * body.max_fan_rpm
    )
    base = baseline_snapshot or {}
    return (
        time.time(),
        float(body.temp_c),
        float(base.get("temp_c", body.temp_c)),
        float(optimized_rpm),
        float(base.get("fan_rpm", body.fan_rpm)),
        float(body.cpu_temp_c) if body.cpu_temp_c else None,
        float(base.get("cpu_temp_c", body.cpu_temp_c)) if body.cpu_temp_c or base.get("cpu_temp_c") else None,
        float(body.gpu_power_w),
        float(base.get("gpu_power_w", body.gpu_power_w)),
        float(body.peak_power_w) if body.peak_power_w else None,
        float(base.get("peak_power_w", body.peak_power_w)) if body.peak_power_w or base.get("peak_power_w") else None,
    )

# ---------------------------------------------------------------------------
# Simple token-bucket rate limiter (in-memory, per API key)
# ---------------------------------------------------------------------------

class _TokenBucket:
    __slots__ = ("rate", "burst", "_tokens", "_last_refill")

    def __init__(self, rate: float, burst: float):
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last_refill = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class _RateLimiter:
    def __init__(self):
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, rate: float = 100.0) -> bool:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or bucket.rate != rate:
                bucket = _TokenBucket(rate=rate, burst=rate * 2)
                self._buckets[key] = bucket
            return bucket.allow()


# ---------------------------------------------------------------------------
# API Key validation (local copy, synced from cloud via PolicySyncer)
# ---------------------------------------------------------------------------

class _KeyRegistry:
    """Loads and validates API keys from a local copy of data/api_keys.json."""

    def __init__(self, keys_file: Optional[str] = None):
        self._keys_file = Path(keys_file) if keys_file else None
        self._registry: Dict[str, dict] = {}
        self._key_to_owner: Dict[str, str] = {}
        self._key_facilities: Dict[str, Dict[str, dict]] = {}
        self._legacy_key = os.environ.get("COOLEDAI_API_KEY", "")
        self._fixed_owner = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        if self._keys_file:
            self.reload()

    def reload(self, keys_file: Optional[str] = None) -> None:
        fpath = Path(keys_file) if keys_file else self._keys_file
        if not fpath or not fpath.exists():
            return
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
            if "keys" in raw:
                self._registry = raw["keys"]
            else:
                self._registry = raw
            self._rebuild_indexes()
            logger.info("Loaded %d API keys from %s", len(self._registry), fpath)
        except Exception as e:
            logger.warning("Failed to load key registry from %s: %s", fpath, e)

    def _rebuild_indexes(self) -> None:
        self._key_to_owner = {}
        self._key_facilities = {}
        for key_str, entry in self._registry.items():
            owner = entry.get("owner_id", self._fixed_owner)
            self._key_to_owner[key_str] = owner
            self._key_facilities[key_str] = entry.get("facilities", {})

    def validate_key(self, key: str) -> bool:
        if key in self._registry:
            return True
        if self._legacy_key and key == self._legacy_key:
            return True
        return False

    def resolve_owner(self, key: str) -> str:
        if key in self._key_to_owner:
            return self._key_to_owner[key]
        return self._fixed_owner

    def validate_node_access(self, key: str, node_id: str) -> str:
        """Returns facility_id or raises HTTPException(403)."""
        facilities = self._key_facilities.get(key, {})
        if not facilities:
            return "facility-default"
        for fid, fconf in facilities.items():
            nodes = fconf.get("nodes", [])
            if "*" in nodes or node_id in nodes:
                return fid
        if key not in self._registry:
            return "facility-default"
        raise HTTPException(
            status_code=403,
            detail=f"Node '{node_id}' is not authorized for this API key's facilities.",
        )

    def get_rate_limit(self, key: str) -> float:
        """Return rate_limit_rps for the key's first facility, default 100."""
        facilities = self._key_facilities.get(key, {})
        for _fid, fconf in facilities.items():
            return float(fconf.get("rate_limit_rps", 100))
        return 100.0


# ---------------------------------------------------------------------------
# Factory: create_app()
# ---------------------------------------------------------------------------

def create_app(
    optimization_service: Any = None,
    cloud_forwarder: Any = None,
    policy_syncer: Any = None,
    keys_file: Optional[str] = None,
) -> FastAPI:
    """Create the gateway FastAPI application.

    Args:
        optimization_service: LocalOptimizationService instance
        cloud_forwarder: CloudForwarder instance (optional, for telemetry batching)
        policy_syncer: PolicySyncer instance (optional, for mode sync)
        keys_file: Path to api_keys.json (defaults to data/api_keys.json)
    """
    app = FastAPI(
        title="CooledAI Edge Gateway",
        description="Local optimization endpoint for CooledAI agents",
        version="1.0.0",
    )

    _start_time = time.time()

    # Key registry
    if keys_file is None:
        project_root = Path(__file__).resolve().parent.parent
        keys_file = str(project_root / "data" / "api_keys.json")
    key_registry = _KeyRegistry(keys_file=keys_file)
    rate_limiter = _RateLimiter()

    # Dependency: extract API key
    def _extract_key(request: Request) -> str:
        key = request.headers.get("x-api-key", "")
        if key:
            return key
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ""

    # Dependency: require valid API key
    def _require_key(request: Request) -> str:
        key = _extract_key(request)
        if not key_registry.validate_key(key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")
        # Rate limiting
        rate = key_registry.get_rate_limit(key)
        if not rate_limiter.check(key, rate):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({rate} req/s). Retry later.",
            )
        return key

    # ------------------------------------------------------------------
    # POST /api/v1/optimize/control — Agent optimization (same contract as cloud)
    # ------------------------------------------------------------------

    @app.post("/api/v1/optimize/control")
    async def gateway_optimize_control(request: Request, body: AgentOptimizeControlInput):
        key = _require_key(request)
        owner_id = key_registry.resolve_owner(key)
        key_registry.validate_node_access(key, body.node_id)

        if optimization_service is None:
            raise HTTPException(status_code=503, detail="Optimization service not initialized.")

        # Record latest snapshot for every node (pilot or control)
        _update_latest_snapshot(body.node_id, body)

        # Track temperature history for rate-of-change safety
        with _temp_history_lock:
            hist = _temp_history.setdefault(body.node_id, [])
            hist.append((time.time(), float(body.temp_c)))
            if len(hist) > _TEMP_HISTORY_MAX:
                _temp_history[body.node_id] = hist[-_TEMP_HISTORY_MAX:]

        if _is_control_node(body.node_id):
            # Control node: skip optimization, passthrough only
            result = {"target_duty": None, "source": "control_passthrough"}
        else:
            # Determine current mode
            current_mode = "shadow"
            if policy_syncer is not None:
                current_mode = getattr(policy_syncer, "current_mode", "shadow")

            if current_mode == "shadow":
                # Phase 6: Shadow = pure passthrough. Brain runs simulation-only
                # for projected savings, but does NOT produce a target_duty for BMC.
                result = optimization_service.optimize(owner_id, body)
                result["mode"] = "shadow"
                result["applied"] = False
                # The target_rpm in result is what the brain WOULD recommend
                # (used for projected savings calculations in the cloud)
            elif current_mode == "supervised":
                # Phase 6: Supervised = brain controls with BMC baseline clamping.
                # Max 20% reduction from what BMC currently runs.
                result = optimization_service.optimize(owner_id, body)
                result["mode"] = "supervised"
                result["applied"] = True
                if result.get("target_duty") is not None:
                    # Clamp: max 20% reduction from BMC baseline (body.fan_rpm)
                    bmc_baseline_duty = (body.fan_rpm / body.max_fan_rpm) * 100.0
                    floor_duty = int(round(bmc_baseline_duty * 0.80))
                    if result["target_duty"] < floor_duty:
                        result["target_duty"] = floor_duty
                        result["target_rpm"] = floor_duty / 100.0 * body.max_fan_rpm
                        result["source"] = result.get("source", "") + "_supervised_clamped"
            else:
                # Active mode: brain has full authority within physics limits
                result = optimization_service.optimize(owner_id, body)
                result["mode"] = "active"
                result["applied"] = True

        # Rate-of-change safety check (supervised/active only)
        if result.get("applied") and result.get("mode") in ("supervised", "active"):
            from core.optimization.cooling_safety_policy import (
                check_rate_of_change, THERMAL_CEILING_C,
            )
            with _temp_history_lock:
                node_temps = list(_temp_history.get(body.node_id, []))
            roc_status = check_rate_of_change(node_temps)
            result["rate_of_change_status"] = roc_status

            if roc_status == "emergency":
                # Revert to BMC baseline RPM immediately
                bmc_duty = int(round((body.fan_rpm / body.max_fan_rpm) * 100))
                result["target_duty"] = bmc_duty
                result["target_rpm"] = body.fan_rpm
                result["source"] = "rate_of_change_emergency"
                logger.warning("[ROC_SAFETY] Emergency revert for %s: temp rising fast", body.node_id)
            elif roc_status == "caution":
                # Freeze: don't reduce below current RPM
                current_duty = int(round((body.fan_rpm / body.max_fan_rpm) * 100))
                if result.get("target_duty") is not None and result["target_duty"] < current_duty:
                    result["target_duty"] = current_duty
                    result["target_rpm"] = body.fan_rpm
                    result["source"] = result.get("source", "") + "_roc_frozen"

            # Absolute temperature ceiling
            if body.temp_c >= THERMAL_CEILING_C:
                result["target_duty"] = 100
                result["target_rpm"] = body.max_fan_rpm
                result["source"] = "thermal_ceiling_emergency"
                result["applied"] = True
                logger.critical("[THERMAL_CEILING] %s: %.1f°C >= %.1f°C — MAX RPM",
                                body.node_id, body.temp_c, THERMAL_CEILING_C)

        # Write telemetry to thermal history so the brain accumulates time-series data
        try:
            store = optimization_service.control_service._store
            baseline_snap = _get_latest_control_snapshot()
            row = _compose_thermal_row(body, result, baseline_snapshot=baseline_snap)
            store.append_thermal_history(owner_id, row)
        except Exception:
            pass  # Don't fail agent request if history write fails

        # Non-blocking: enqueue for cloud forwarding
        if cloud_forwarder is not None:
            try:
                cloud_forwarder.enqueue(body.node_id, body.dict(), result)
            except Exception:
                pass  # Don't fail agent request if forwarding fails

        return result

    # ------------------------------------------------------------------
    # GET /health — Gateway health check
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        brain_loaded = optimization_service is not None
        redis_connected = False
        if optimization_service is not None and hasattr(optimization_service, "_redis_available"):
            redis_connected = optimization_service._redis_available
        cloud_connected = True
        cloud_disconnected = False
        if cloud_forwarder is not None:
            cloud_connected = getattr(cloud_forwarder, "cloud_connected", True)
            cloud_disconnected = getattr(cloud_forwarder, "cloud_disconnected", False)
        nodes_active = 0
        if optimization_service is not None:
            try:
                profiles = optimization_service.get_all_profiles()
                nodes_active = len(profiles)
            except Exception:
                pass
        status = "healthy"
        if not brain_loaded or cloud_disconnected:
            status = "degraded"
        return {
            "status": status,
            "brain_loaded": brain_loaded,
            "redis_connected": redis_connected,
            "cloud_connected": cloud_connected,
            "cloud_disconnected": cloud_disconnected,
            "nodes_active": nodes_active,
            "uptime_seconds": round(time.time() - _start_time, 0),
        }

    # ------------------------------------------------------------------
    # GET /api/v1/debug/calibrators — Per-node calibration state
    # ------------------------------------------------------------------

    @app.get("/api/v1/debug/calibrators")
    async def debug_calibrators(request: Request):
        key = _require_key(request)
        owner_id = key_registry.resolve_owner(key)
        if optimization_service is None:
            return {"calibrators": {}, "owner_id": owner_id}
        profiles = optimization_service.get_all_profiles()
        result = {}
        for nid, cp in profiles.items():
            try:
                key_registry.validate_node_access(key, nid)
            except HTTPException:
                continue
            result[nid] = cp.to_dict() if hasattr(cp, "to_dict") else {"temp_mean_c": getattr(cp, "temp_mean_c", None)}
        return {"calibrators": result, "owner_id": owner_id}

    # ------------------------------------------------------------------
    # GET /api/v1/debug/calibrators/{node_id} — Single node detail
    # ------------------------------------------------------------------

    @app.get("/api/v1/debug/calibrators/{node_id}")
    async def debug_calibrator_detail(request: Request, node_id: str):
        key = _require_key(request)
        owner_id = key_registry.resolve_owner(key)
        key_registry.validate_node_access(key, node_id)
        if optimization_service is None:
            raise HTTPException(status_code=404, detail="Optimization service not initialized.")
        detail = optimization_service.get_node_calibration(node_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"No calibration profile for node '{node_id}'.")
        return {"node_id": node_id, "owner_id": owner_id, "calibration_profile": detail}

    # ------------------------------------------------------------------
    # GET /api/v1/debug/safety/{node_id} — Per-node safety state
    # ------------------------------------------------------------------

    @app.get("/api/v1/debug/safety/{node_id}")
    async def debug_safety_state(request: Request, node_id: str):
        key = _require_key(request)
        owner_id = key_registry.resolve_owner(key)
        key_registry.validate_node_access(key, node_id)
        if optimization_service is None:
            raise HTTPException(status_code=404, detail="Optimization service not initialized.")
        return optimization_service.get_safety_state(owner_id, node_id)

    # ------------------------------------------------------------------
    # POST /api/v1/telemetry — Telemetry ingestion from collectors
    # ------------------------------------------------------------------

    @app.post("/api/v1/telemetry")
    async def ingest_telemetry(request: Request, body: dict):
        key = _require_key(request)
        # Telemetry from collectors is forwarded to cloud via CloudForwarder
        if cloud_forwarder is not None:
            try:
                cloud_forwarder.enqueue("collector", body, {})
            except Exception:
                pass
        return {"status": "accepted"}

    # ------------------------------------------------------------------
    # GET /api/v1/debug/forwarder — CloudForwarder stats
    # ------------------------------------------------------------------

    @app.get("/api/v1/debug/forwarder")
    async def debug_forwarder(request: Request):
        _require_key(request)
        if cloud_forwarder is None:
            return {"error": "CloudForwarder not initialized"}
        return cloud_forwarder.stats()

    # ------------------------------------------------------------------
    # GET /api/v1/debug/brain-state — Optimization brain state
    # ------------------------------------------------------------------

    @app.get("/api/v1/debug/brain-state")
    async def debug_brain_state(request: Request):
        _require_key(request)
        if optimization_service is None:
            return {"error": "Optimization service not initialized"}
        store = optimization_service.control_service._store
        # Count history points across all owners
        all_history = {}
        if hasattr(store, "get_all_thermal_history"):
            all_history = store.get_all_thermal_history()
        total_points = sum(len(v) for v in all_history.values())
        # Calibration status per node
        profiles = optimization_service.get_all_profiles()
        cal_status = {}
        for nid, cp in profiles.items():
            if hasattr(cp, "sample_count") and cp.sample_count > 0:
                cal_status[nid] = "CALIBRATED"
            else:
                cal_status[nid] = "CALIBRATING"
        return {
            "nodes_in_history": total_points,
            "calibration_status": cal_status,
        }

    # ------------------------------------------------------------------
    # GET /api/v1/debug/thermal-history-stats
    # ------------------------------------------------------------------

    @app.get("/api/v1/debug/thermal-history-stats")
    async def debug_thermal_history_stats(request: Request):
        key = _require_key(request)
        owner_id = key_registry.resolve_owner(key)
        if optimization_service is None:
            return {"point_count": 0}
        store = optimization_service.control_service._store
        history = store.get_thermal_history(owner_id, max_points=999999)
        if not history:
            return {"point_count": 0, "oldest_ts": None, "newest_ts": None, "nodes_tracked": []}
        from datetime import datetime, timezone
        oldest = min(r[0] for r in history)
        newest = max(r[0] for r in history)
        # Track unique nodes from calibration profiles
        profiles = optimization_service.get_all_profiles()
        return {
            "point_count": len(history),
            "oldest_ts": datetime.fromtimestamp(oldest, tz=timezone.utc).isoformat(),
            "newest_ts": datetime.fromtimestamp(newest, tz=timezone.utc).isoformat(),
            "nodes_tracked": list(profiles.keys()),
        }

    # ------------------------------------------------------------------
    # GET /api/v1/debug/savings-validation — Fan power savings
    # ------------------------------------------------------------------

    MAX_FAN_RPM = 7000.0
    RATED_FAN_WATTS = 12.0
    NUM_FANS = 6

    @app.get("/api/v1/debug/savings-validation")
    async def debug_savings_validation(request: Request):
        key = _require_key(request)
        owner_id = key_registry.resolve_owner(key)
        if optimization_service is None:
            return {"error": "Optimization service not initialized"}
        store = optimization_service.control_service._store
        history = store.get_thermal_history(owner_id, max_points=720)  # ~1 hour at 5s
        if not history:
            return {
                "optimization_source": "unknown",
                "pilot_avg_duty": 0, "baseline_avg_duty": 0,
                "savings_watts": 0.0, "savings_percent": 0.0,
                "data_points_last_hour": 0,
            }
        # Extract pilot_rpm (col 3) and baseline_rpm (col 4)
        pilot_rpms = [float(r[3]) for r in history if r[3] is not None]
        baseline_rpms = [float(r[4]) for r in history if r[4] is not None]
        if not pilot_rpms or not baseline_rpms:
            return {
                "optimization_source": "unknown",
                "pilot_avg_duty": 0, "baseline_avg_duty": 0,
                "savings_watts": 0.0, "savings_percent": 0.0,
                "data_points_last_hour": len(history),
            }
        avg_pilot_rpm = sum(pilot_rpms) / len(pilot_rpms)
        avg_baseline_rpm = sum(baseline_rpms) / len(baseline_rpms)
        pilot_duty = int(round((avg_pilot_rpm / MAX_FAN_RPM) * 100))
        baseline_duty = int(round((avg_baseline_rpm / MAX_FAN_RPM) * 100))
        # Fan Affinity Law: power = (rpm/max)^3 * rated * num_fans
        pilot_watts = (avg_pilot_rpm / MAX_FAN_RPM) ** 3 * RATED_FAN_WATTS * NUM_FANS
        baseline_watts = (avg_baseline_rpm / MAX_FAN_RPM) ** 3 * RATED_FAN_WATTS * NUM_FANS
        savings_w = baseline_watts - pilot_watts
        savings_pct = (savings_w / baseline_watts * 100) if baseline_watts > 0 else 0.0
        return {
            "optimization_source": "optimization_brain" if len(history) > 1 else "fallback_simple",
            "pilot_avg_duty": pilot_duty,
            "baseline_avg_duty": baseline_duty,
            "savings_watts": round(savings_w, 2),
            "savings_percent": round(savings_pct, 1),
            "data_points_last_hour": len(history),
        }

    # ------------------------------------------------------------------
    # POST /api/v1/calibration/start — Trigger calibration sweep (stub)
    # ------------------------------------------------------------------

    @app.post("/api/v1/calibration/start")
    async def start_calibration_sweep(request: Request):
        """Trigger a controlled fan duty sweep for thermal model calibration.

        Cycles through 3-4 fan duty setpoints (25%, 40%, 55%, 70%), holds each
        for 60 seconds, and records the thermal response. This gives the brain
        diverse (temp, duty) data points to build confidence.

        NOTE: This is a stub endpoint. The actual sweep logic will be
        implemented when the agent supports duty override commands.
        Manual trigger only — never auto-triggered.
        """
        _require_key(request)
        # TODO: Implement actual sweep logic when agent supports duty override
        return {
            "status": "not_implemented",
            "message": "Calibration sweep endpoint is a stub. Actual sweep logic pending agent duty override support.",
            "planned_setpoints": [25, 40, 55, 70],
            "hold_duration_s": 60,
        }

    return app
