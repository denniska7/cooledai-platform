"""
CooledAI Universal Protocol Gateway - Main Entry Point

Runs the edge agent: collects telemetry, normalizes, sends to backend.
Store-and-forward on connection failure. Heartbeat every 60s.
"""

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
from gateway.collectors.base_collector import TelemetryObject
from gateway.control_gate import ControlGate, CONTROL_MODE
from gateway.telemetry_buffer import TelemetryBuffer
from gateway.normalizer import Normalizer
from gateway.heartbeat import get_agent_health, HEARTBEAT_INTERVAL_SEC
from gateway.log_scrubber import scrub_log_message

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
COLLECT_INTERVAL_SEC = 10
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL_SEC", str(HEARTBEAT_INTERVAL_SEC)))


class Gateway:
    """
    Universal Protocol Gateway - orchestrates collectors, buffer, and backend.
    """

    def __init__(self):
        self.control_gate = ControlGate(log_dir=".")
        self.buffer = TelemetryBuffer(db_path="telemetry_buffer.db")
        self.collectors: List = []
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._collect_thread: Optional[threading.Thread] = None

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
        """Send telemetry or heartbeat to backend. Returns True if success."""
        if not requests:
            return False
        try:
            r = requests.post(
                f"{BACKEND_URL}/api/v1/telemetry",
                json=payload,
                timeout=10,
            )
            return r.status_code in (200, 201, 204)
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

    def start(self) -> None:
        """Start the gateway."""
        self._running = True
        self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._collect_thread.start()
        self._heartbeat_thread.start()
        _logger.info(
            "Gateway: Started. CONTROL_MODE=%s. Backend=%s",
            self.control_gate.mode.value,
            BACKEND_URL,
        )

    def stop(self) -> None:
        """Stop the gateway."""
        self._running = False
        if self._collect_thread:
            self._collect_thread.join(timeout=5)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)


def main() -> None:
    """Run gateway with default collectors (template mode)."""
    gw = Gateway()
    gw.add_collector(BACnetManager())
    gw.add_collector(SNMPManager())
    gw.add_collector(RedfishManager())
    gw.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        gw.stop()
        _logger.info("Gateway: Stopped")


if __name__ == "__main__":
    main()
