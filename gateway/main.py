"""
CooledAI Universal Protocol Gateway - Main Entry Point

Phase 1: The gateway now runs a local OptimizationBrain and exposes
/api/v1/optimize/control on the LAN so agents get <5ms optimization
instead of 200-500ms cloud round-trips.

Threading model:
  Thread 1: Collector loop       (existing, 10s interval)
  Thread 2: Heartbeat loop       (existing, 60s interval)
  Thread 3: Uvicorn/FastAPI      (NEW — serves /optimize/control on :8080)
  Thread 4: asyncio event loop   (NEW — CloudForwarder + PolicySyncer)
"""

import asyncio
import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from gateway.collectors.bacnet_manager import BACnetManager
from gateway.collectors.snmp_manager import SNMPManager
from gateway.collectors.redfish_manager import RedfishManager
from gateway.collectors.redfish_bridge_collector import RedfishBridgeCollector
from gateway.collectors.base_collector import TelemetryObject
from gateway.control_gate import ControlGate, CONTROL_MODE
from gateway.telemetry_buffer import TelemetryBuffer
from gateway.normalizer import Normalizer
from gateway.heartbeat import get_agent_health, HEARTBEAT_INTERVAL_SEC
from gateway.log_scrubber import scrub_log_message
from gateway.cloud_heartbeat import mark_cloud_response_success
from gateway.safety_watchdog import SafetyWatchdog, build_revert_callback

try:
    import requests
except ImportError:
    requests = None

_logger = logging.getLogger("cooledai.gateway")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)


BACKEND_URL = os.environ.get("COOLEDAI_BACKEND_URL", "https://api.cooledai.com")
BACKEND_API_KEY = os.environ.get("COOLEDAI_API_KEY", "")
COLLECT_INTERVAL_SEC = 10
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL_SEC", str(HEARTBEAT_INTERVAL_SEC)))
# SafetyWatchdog: trigger REVERT_TO_DEFAULT if no cloud response for this many seconds
CLOUD_LOSS_TIMEOUT_SEC = int(os.environ.get("COOLEDAI_CLOUD_LOSS_TIMEOUT_SEC", "30"))
# Gateway API port
GATEWAY_PORT = int(os.environ.get("COOLEDAI_GATEWAY_PORT", "8080"))


class Gateway:
    """
    Universal Protocol Gateway - orchestrates collectors, buffer, backend,
    and the local optimization service (Phase 1).
    """

    def __init__(self):
        self.control_gate = ControlGate(log_dir=".")
        self.buffer = TelemetryBuffer(db_path="telemetry_buffer.db")
        self.collectors: List = []
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._collect_thread: Optional[threading.Thread] = None
        self._safety_watchdog: Optional[SafetyWatchdog] = None

        # Phase 1: Local optimization service
        self._optimization_service = None
        self._cloud_forwarder = None
        self._policy_syncer = None
        self._api_thread: Optional[threading.Thread] = None
        self._async_thread: Optional[threading.Thread] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._persistence_thread: Optional[threading.Thread] = None

        self._init_optimization_service()

    def _init_optimization_service(self) -> None:
        """Initialize the local optimization service.

        Tries RedisStateStore first (persistence across restarts).
        Falls back to InMemoryStateStore if Redis is unavailable.
        """
        try:
            from gateway.optimization_service import LocalOptimizationService
            redis_url = os.environ.get("COOLEDAI_REDIS_URL", "redis://localhost:6379/0")

            # Try Redis first
            store = None
            try:
                from gateway.state_store import RedisStateStore
                redis_store = RedisStateStore(redis_url=redis_url)
                if redis_store.available:
                    redis_store.warm_start()
                    store = redis_store
                    _logger.info("Gateway: Using RedisStateStore (%s)", redis_url)
            except Exception as e:
                _logger.debug("Gateway: Redis unavailable: %s", e)

            # Fallback to in-memory with file persistence
            if store is None:
                from core.optimization.in_memory_store import InMemoryStateStore
                store = InMemoryStateStore(
                    thermal_history_file="data/gateway_thermal_history.json",
                )
                store.load_thermal_history_from_file()
                store.load_calibration_from_file("data/gateway_calibration.json")
                _logger.info("Gateway: Using InMemoryStateStore with file persistence")

            self._optimization_service = LocalOptimizationService(store)
            self._optimization_service._redis_available = getattr(store, "available", False)
            _logger.info("Gateway: Local OptimizationBrain initialized")
        except Exception as e:
            _logger.error("Gateway: Failed to initialize optimization service: %s", e)
            _logger.warning("Gateway: Running in collector-only mode (no local optimization)")

    def _init_cloud_forwarder(self) -> None:
        """Initialize the async cloud forwarder."""
        try:
            from gateway.cloud_forwarder import CloudForwarder
            self._cloud_forwarder = CloudForwarder(
                cloud_url=BACKEND_URL,
                api_key=BACKEND_API_KEY,
                batch_interval=5.0,
            )
            _logger.info("Gateway: CloudForwarder initialized (batch every 5s to %s)", BACKEND_URL)
        except Exception as e:
            _logger.warning("Gateway: CloudForwarder unavailable: %s", e)

    def _init_policy_syncer(self) -> None:
        """Initialize the policy syncer."""
        try:
            from gateway.policy_syncer import PolicySyncer
            self._policy_syncer = PolicySyncer(
                cloud_url=BACKEND_URL,
                api_key=BACKEND_API_KEY,
                interval=60.0,
            )
            _logger.info("Gateway: PolicySyncer initialized (sync every 60s)")
        except Exception as e:
            _logger.warning("Gateway: PolicySyncer unavailable: %s", e)

    def add_collector(self, collector) -> None:
        """Register a protocol collector."""
        collector.connect()
        self.collectors.append(collector)

    def _collect_all(self) -> List[TelemetryObject]:
        """Collect from all registered collectors."""
        objs = []
        for c in self.collectors:
            try:
                objs.extend(c.collect())
            except Exception as e:
                _logger.warning("Gateway: Collector %s failed: %s", c.protocol, e)
        return Normalizer.normalize_batch(objs)

    def _send_to_backend(self, payload: dict) -> bool:
        """Send telemetry or heartbeat to backend. Returns True if success. Updates cloud heartbeat for SafetyWatchdog."""
        if not requests:
            return False
        try:
            headers = {"Content-Type": "application/json"}
            if BACKEND_API_KEY:
                headers["X-API-Key"] = BACKEND_API_KEY
            r = requests.post(
                f"{BACKEND_URL}/api/v1/telemetry",
                json=payload,
                headers=headers,
                timeout=10,
            )
            ok = r.status_code in (200, 201, 204)
            if ok:
                mark_cloud_response_success()
            return ok
        except Exception as e:
            _logger.debug("Gateway: Backend unreachable: %s", e)
            return False

    def _collect_loop(self) -> None:
        """Main collect loop: collect, normalize, send or buffer."""
        while self._running:
            try:
                objs = self._collect_all()
                if not objs:
                    time.sleep(COLLECT_INTERVAL_SEC)
                    continue

                payload = {
                    "telemetry": [o.to_dict() for o in objs],
                    "agent_id": os.environ.get("COOLEDAI_AGENT_ID", "default"),
                }

                if self._send_to_backend(payload):
                    # Burst buffer if we had stored data
                    buffered = self.buffer.pop_all()
                    if buffered:
                        for b in buffered:
                            if self._send_to_backend({"telemetry": [b["payload"]]}):
                                self.buffer.delete_ids([b["id"]])
                else:
                    self.buffer.push(objs)
            except Exception as e:
                _logger.error("Gateway: Collect loop error: %s", e)
            time.sleep(COLLECT_INTERVAL_SEC)

    def _heartbeat_loop(self) -> None:
        """Send heartbeat every 60 seconds."""
        while self._running:
            try:
                health = get_agent_health()
                self._send_to_backend({"heartbeat": health})
            except Exception as e:
                _logger.debug("Gateway: Heartbeat failed: %s", e)
            time.sleep(HEARTBEAT_INTERVAL)

    def _persistence_loop(self) -> None:
        """Thread 5: Flush state to disk every 60 seconds (InMemoryStateStore only)."""
        while self._running:
            time.sleep(60)
            self._flush_state_to_disk()

    def _flush_state_to_disk(self) -> None:
        """Flush thermal history + calibration profiles to disk."""
        if self._optimization_service is None:
            return
        # Skip if Redis is handling persistence
        if getattr(self._optimization_service, "_redis_available", False):
            return
        try:
            store = self._optimization_service.control_service._store
            if hasattr(store, "flush_thermal_history_to_file"):
                store.flush_thermal_history_to_file()
            if hasattr(store, "flush_calibration_to_file"):
                store.flush_calibration_to_file("data/gateway_calibration.json")
        except Exception as e:
            _logger.debug("Gateway: persistence flush error: %s", e)

    def _start_api_server(self) -> None:
        """Thread 3: Start uvicorn serving gateway/api.py."""
        try:
            import uvicorn
            from gateway.api import create_app

            app = create_app(
                optimization_service=self._optimization_service,
                cloud_forwarder=self._cloud_forwarder,
            )
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=GATEWAY_PORT,
                log_level="info",
            )
            server = uvicorn.Server(config)
            self._api_thread = threading.Thread(
                target=server.run, daemon=True, name="uvicorn"
            )
            self._api_thread.start()
            _logger.info("Gateway: API server started on 0.0.0.0:%d", GATEWAY_PORT)
        except Exception as e:
            _logger.error("Gateway: Failed to start API server: %s", e)

    def _start_async_loop(self) -> None:
        """Thread 4: Dedicated asyncio loop for CloudForwarder + PolicySyncer."""
        if not self._cloud_forwarder and not self._policy_syncer:
            _logger.info("Gateway: No async services to start")
            return

        self._async_loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(self._async_loop)
            if self._cloud_forwarder:
                self._async_loop.create_task(self._cloud_forwarder.run_forever())
            if self._policy_syncer:
                self._async_loop.create_task(self._policy_syncer.sync_loop())
            self._async_loop.run_forever()

        self._async_thread = threading.Thread(
            target=_run, daemon=True, name="async-services"
        )
        self._async_thread.start()
        _logger.info("Gateway: Async services started (CloudForwarder + PolicySyncer)")

    def start(self) -> None:
        """Start the gateway with all services."""
        self._running = True

        # Thread 1: Collector loop (existing)
        self._collect_thread = threading.Thread(
            target=self._collect_loop, daemon=True, name="collector"
        )
        # Thread 2: Heartbeat loop (existing)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._collect_thread.start()
        self._heartbeat_thread.start()

        # Initialize forwarder + syncer BEFORE the API server so the
        # FastAPI app receives the real CloudForwarder reference (not None).
        self._init_cloud_forwarder()
        self._init_policy_syncer()

        # Thread 3: API server (Phase 1)
        self._start_api_server()

        # Thread 4: Async event loop for CloudForwarder + PolicySyncer
        self._start_async_loop()

        # Thread 5: Persistence (Phase 2 — file-based, 60s interval)
        if not getattr(self._optimization_service, "_redis_available", False):
            self._persistence_thread = threading.Thread(
                target=self._persistence_loop, daemon=True, name="persistence"
            )
            self._persistence_thread.start()
            _logger.info("Gateway: Persistence thread started (60s flush interval)")

        _logger.info(
            "Gateway: Started. CONTROL_MODE=%s. Backend=%s. API port=%d",
            self.control_gate.mode.value,
            BACKEND_URL,
            GATEWAY_PORT,
        )

    def stop(self) -> None:
        """Stop the gateway and all services. Flushes state before exit."""
        self._flush_state_to_disk()
        self._running = False
        if self._async_loop:
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._safety_watchdog:
            self._safety_watchdog.stop()
            self._safety_watchdog = None
        if self._collect_thread:
            self._collect_thread.join(timeout=5)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._async_thread:
            self._async_thread.join(timeout=5)


def main() -> None:
    """Run gateway with default collectors (template mode)."""
    gw = Gateway()
    gw.add_collector(BACnetManager())
    gw.add_collector(SNMPManager())
    gw.add_collector(RedfishBridgeCollector(RedfishManager()))
    gw.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        gw.stop()
        _logger.info("Gateway: Stopped")


if __name__ == "__main__":
    main()
