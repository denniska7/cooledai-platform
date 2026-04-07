"""
Async batched cloud forwarder for the CooledAI edge gateway.

Buffers telemetry + optimization results from all nodes and sends to the
cloud API every 5-10 seconds in a single compressed batch. This reduces
cloud load from ~500 req/s (167 agents × 3Hz) to <1 req/s.

Threading model:
- enqueue() is called from the uvicorn thread (Thread 3) — uses threading.Lock
- run_forever() runs on Thread 4's asyncio loop — drains the buffer periodically
"""

import asyncio
import json
import logging
import ssl
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = None  # Fall back to system certs

logger = logging.getLogger("cooledai.gateway.forwarder")


class CloudForwarder:
    """Async, non-blocking cloud forwarder with batched telemetry."""

    def __init__(
        self,
        cloud_url: str,
        api_key: str,
        batch_interval: float = 5.0,
        max_buffer_size: int = 10000,
        gateway_id: str = "gateway-default",
        control_gate: Optional[Any] = None,
    ):
        self.cloud_url = cloud_url.rstrip("/")
        self.api_key = api_key
        self.batch_interval = batch_interval
        self.max_buffer_size = max_buffer_size
        self.gateway_id = gateway_id
        self._control_gate = control_gate

        # Buffer shared between uvicorn thread and asyncio thread
        self._buffer: List[dict] = []
        self._buffer_lock = threading.Lock()  # NOT asyncio.Lock — cross-thread

        self._connected = True
        self._last_success = time.time()
        self._total_batches_sent = 0
        self._total_entries_sent = 0
        self._total_failures = 0
        self._consecutive_failures = 0
        self._cloud_disconnected = False
        self._DISCONNECT_THRESHOLD = 10  # consecutive failures before disconnected mode

    def enqueue(self, node_id: str, telemetry: dict, optimization_result: dict) -> None:
        """Thread-safe enqueue. Called from uvicorn thread (Thread 3)."""
        entry = {
            "node_id": node_id,
            "telemetry": telemetry,
            "optimization": optimization_result,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with self._buffer_lock:
            if len(self._buffer) < self.max_buffer_size:
                self._buffer.append(entry)
            else:
                # Drop oldest to prevent unbounded memory growth
                self._buffer.pop(0)
                self._buffer.append(entry)

    async def run_forever(self) -> None:
        """Background task on Thread 4's asyncio loop. Flushes buffer every N seconds."""
        logger.info("CloudForwarder: started (batch every %.1fs to %s)", self.batch_interval, self.cloud_url)
        while True:
            await asyncio.sleep(self.batch_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Send buffered telemetry to cloud in a single POST."""
        with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer.copy()
            self._buffer.clear()

        payload = {
            "batch": batch,
            "gateway_id": self.gateway_id,
            "batch_size": len(batch),
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Use aiohttp for non-blocking HTTP
            import aiohttp
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX) if _SSL_CTX else None
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{self.cloud_url}/api/v1/gateway/telemetry-batch",
                    json=payload,
                    headers={
                        "X-API-Key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        self._connected = True
                        self._last_success = time.time()
                        self._total_batches_sent += 1
                        self._total_entries_sent += len(batch)
                        logger.info("CloudForwarder: flushed %d entries (total: %d)", len(batch), self._total_entries_sent)
                        if self._cloud_disconnected:
                            logger.info("CloudForwarder: Cloud connection restored")
                        self._consecutive_failures = 0
                        self._cloud_disconnected = False
                        # Process mode directives piggybacked on the response
                        try:
                            resp_body = await resp.json()
                            self._process_mode_directive(resp_body)
                        except Exception:
                            pass  # Non-critical — don't break telemetry flow
                    else:
                        logger.warning(
                            "CloudForwarder: batch rejected (status=%d, entries=%d)",
                            resp.status, len(batch),
                        )
                        self._re_buffer(batch)
        except ImportError:
            # aiohttp not installed — use synchronous fallback
            logger.warning("CloudForwarder: aiohttp not available, using sync fallback")
            self._flush_sync(payload, batch)
        except Exception as e:
            logger.warning("CloudForwarder: flush failed: %s", e)
            self._connected = False
            self._total_failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._DISCONNECT_THRESHOLD and not self._cloud_disconnected:
                self._cloud_disconnected = True
                logger.warning(
                    "CloudForwarder: Operating in cloud-disconnected mode (%d consecutive failures)",
                    self._consecutive_failures,
                )
            self._re_buffer(batch)

    def _flush_sync(self, payload: dict, batch: list) -> None:
        """Synchronous fallback when aiohttp is unavailable."""
        try:
            import requests as _req
            resp = _req.post(
                f"{self.cloud_url}/api/v1/gateway/telemetry-batch",
                json=payload,
                headers={"X-API-Key": self.api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                self._connected = True
                self._last_success = time.time()
                self._total_batches_sent += 1
                self._total_entries_sent += len(batch)
                if self._cloud_disconnected:
                    logger.info("CloudForwarder: Cloud connection restored")
                self._consecutive_failures = 0
                self._cloud_disconnected = False
                # Process mode directives piggybacked on the response
                try:
                    self._process_mode_directive(resp.json())
                except Exception:
                    pass
            else:
                self._consecutive_failures += 1
                self._re_buffer(batch)
        except Exception:
            self._connected = False
            self._total_failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._DISCONNECT_THRESHOLD and not self._cloud_disconnected:
                self._cloud_disconnected = True
                logger.warning(
                    "CloudForwarder: Operating in cloud-disconnected mode (%d consecutive failures)",
                    self._consecutive_failures,
                )
            self._re_buffer(batch)

    def set_control_gate(self, gate: Any) -> None:
        """Late-bind control gate (avoids circular imports)."""
        self._control_gate = gate

    def _process_mode_directive(self, resp_body: dict) -> None:
        """Apply a mode directive from the cloud batch response.

        The cloud API piggybacks a ``desired_mode`` field on the batch
        response when the portal user requests a mode change.  We map
        the gateway mode string (SHADOW / SUPERVISED / PRODUCTION) to
        the ControlGate enum and call switch_mode().
        """
        desired = resp_body.get("desired_mode")
        if not desired or not self._control_gate:
            return

        # Import here to avoid circular dependency at module level
        from .control_gate import CONTROL_MODE

        mode_map = {
            "SHADOW": CONTROL_MODE.SHADOW,
            "SUPERVISED": CONTROL_MODE.SUPERVISED,
            "PRODUCTION": CONTROL_MODE.PRODUCTION,
        }
        target_mode = mode_map.get(desired.upper())
        if target_mode is None:
            logger.warning("CloudForwarder: unknown desired_mode '%s' from cloud", desired)
            return

        current = self._control_gate.mode
        if current == target_mode:
            logger.info("CloudForwarder: already in %s, ignoring directive", desired)
            return

        requested_by = resp_body.get("desired_mode_requested_by", "portal")
        ok = self._control_gate.switch_mode(target_mode, reason=f"cloud_directive:{requested_by}")
        if ok:
            logger.info(
                "CloudForwarder: Mode switched %s → %s (requested by %s)",
                current.value, target_mode.value, requested_by,
            )
        else:
            logger.warning(
                "CloudForwarder: Mode switch %s → %s REJECTED by ControlGate",
                current.value, target_mode.value,
            )

    def _re_buffer(self, batch: list) -> None:
        """On failure, put entries back into buffer (up to max size)."""
        with self._buffer_lock:
            space = self.max_buffer_size - len(self._buffer)
            if space > 0:
                self._buffer = batch[:space] + self._buffer

    @property
    def cloud_connected(self) -> bool:
        return self._connected

    @property
    def seconds_since_last_sync(self) -> float:
        return time.time() - self._last_success

    @property
    def cloud_disconnected(self) -> bool:
        return self._cloud_disconnected

    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "cloud_disconnected": self._cloud_disconnected,
            "consecutive_failures": self._consecutive_failures,
            "seconds_since_last_sync": round(self.seconds_since_last_sync, 1),
            "total_batches_sent": self._total_batches_sent,
            "total_entries_sent": self._total_entries_sent,
            "total_failures": self._total_failures,
            "buffer_size": len(self._buffer),
        }
