"""
CooledAI SafetyWatchdog - Edge Gateway Cloud Connection Monitor

Runs in a dedicated thread, independent of the AI optimization loop.
Monitors the heartbeat/connection to the cloud API; if the connection is lost
for more than 30 seconds, triggers REVERT_TO_DEFAULT on all connected cooling units.
"""

import logging
import threading
import time
from typing import List, Callable, Optional

from gateway.cloud_heartbeat import seconds_since_last_cloud_ok, get_last_cloud_ok_at
from gateway.collectors.base_collector import BaseCollector

_logger = logging.getLogger("cooledai.safety_watchdog")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)

# Default: trigger REVERT if no cloud response for this many seconds
DEFAULT_CLOUD_LOSS_TIMEOUT_SEC = 30
# How often the watchdog checks (seconds)
WATCHDOG_CHECK_INTERVAL_SEC = 5

class SafetyWatchdog:
    """
    Monitors cloud API heartbeat. If connection is lost for longer than
    cloud_loss_timeout_sec, triggers REVERT_TO_DEFAULT on all cooling units.
    Runs in its own thread; does not depend on the AI optimization loop.
    """

    def __init__(
        self,
        revert_callback: Callable[[], None],
        cloud_loss_timeout_sec: float = DEFAULT_CLOUD_LOSS_TIMEOUT_SEC,
        check_interval_sec: float = WATCHDOG_CHECK_INTERVAL_SEC,
    ):
        """
        Args:
            revert_callback: Called when cloud loss is detected. Must send
                REVERT_TO_DEFAULT to all cooling units (e.g. via ControlGate).
            cloud_loss_timeout_sec: Trigger revert after this many seconds
                without a successful cloud response.
            check_interval_sec: How often to check (seconds).
        """
        self.revert_callback = revert_callback
        self.cloud_loss_timeout_sec = cloud_loss_timeout_sec
        self.check_interval_sec = check_interval_sec
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._triggered = False

    def start(self) -> None:
        """Start the watchdog thread (independent of AI loop)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        _logger.info(
            "SafetyWatchdog: Started. Cloud loss timeout=%.0fs, check every %.0fs.",
            self.cloud_loss_timeout_sec,
            self.check_interval_sec,
        )

    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.check_interval_sec * 2)
            self._thread = None

    def _watch_loop(self) -> None:
        while self._running:
            try:
                since = seconds_since_last_cloud_ok()
                if since >= self.cloud_loss_timeout_sec:
                    if not self._triggered:
                        self._triggered = True
                        _logger.warning(
                            "SafetyWatchdog: Cloud connection lost for %.0fs (threshold %.0fs). "
                            "Triggering REVERT_TO_DEFAULT on all cooling units.",
                            since,
                            self.cloud_loss_timeout_sec,
                        )
                        try:
                            self.revert_callback()
                        except Exception as e:
                            _logger.exception("SafetyWatchdog: Revert callback failed: %s", e)
                else:
                    self._triggered = False
            except Exception as e:
                _logger.exception("SafetyWatchdog: Check failed: %s", e)
            time.sleep(self.check_interval_sec)


def build_revert_callback(control_gate, collectors: List[BaseCollector]) -> Callable[[], None]:
    """
    Build a revert callback that sends REVERT_TO_DEFAULT to all collectors
    via the ControlGate (so safety bounds and SHADOW/PRODUCTION still apply).
    """

    def _revert() -> None:
        control_gate.revert_all_to_default(collectors)

    return _revert
