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
    keys_file: Optional[str] = None,
) -> FastAPI:
    """Create the gateway FastAPI application.

    Args:
        optimization_service: LocalOptimizationService instance
        cloud_forwarder: CloudForwarder instance (optional, for telemetry batching)
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

        result = optimization_service.optimize(owner_id, body)

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
        if cloud_forwarder is not None:
            cloud_connected = getattr(cloud_forwarder, "cloud_connected", True)
        nodes_active = 0
        if optimization_service is not None:
            try:
                profiles = optimization_service.get_all_profiles()
                nodes_active = len(profiles)
            except Exception:
                pass
        return {
            "status": "healthy" if brain_loaded else "degraded",
            "brain_loaded": brain_loaded,
            "redis_connected": redis_connected,
            "cloud_connected": cloud_connected,
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

    return app
