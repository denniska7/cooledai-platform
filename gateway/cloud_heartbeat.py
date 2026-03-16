"""
CooledAI Cloud Heartbeat - Track last successful cloud API response.

Used by SafetyWatchdog to detect connection loss. The gateway updates
last_successful_response_at whenever a request to the cloud succeeds.
Independent of the AI optimization loop.
"""

import time
import threading
import logging

_logger = logging.getLogger("cooledai.cloud_heartbeat")

# Monotonic time of last successful response from cloud API (heartbeat or telemetry)
_last_successful_response_at: float = 0.0
_lock = threading.Lock()


def mark_cloud_response_success() -> None:
    """Call when the edge receives a successful response from the cloud API."""
    global _last_successful_response_at
    with _lock:
        _last_successful_response_at = time.monotonic()


def get_last_cloud_ok_at() -> float:
    """Return monotonic time of last successful cloud response. 0 if never."""
    with _lock:
        return _last_successful_response_at


def seconds_since_last_cloud_ok() -> float:
    """Seconds since last successful cloud response. Large value if never."""
    t = get_last_cloud_ok_at()
    if t <= 0:
        return 999999.0
    return time.monotonic() - t
