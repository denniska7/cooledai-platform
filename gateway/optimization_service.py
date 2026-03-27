"""
Local optimization service for the CooledAI edge gateway.

Wraps ControlService with a per-node result cache and ThreadPoolExecutor.
This is the gateway equivalent of the cloud API's _optimize_control_sync +
_executor + _result_cache pattern.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

from core.models.agent_input import AgentOptimizeControlInput
from core.optimization.control_service import ControlService
from core.optimization.in_memory_store import StateBackend
from core.optimization.thermal_calibrator import CalibrationProfile

logger = logging.getLogger("cooledai.gateway.optimization")


class LocalOptimizationService:
    """Gateway-local optimization service.

    Provides the same optimization logic as the cloud API but runs on-premises
    with <5ms latency. No shadow logging, no financial metrics, no Clerk JWT.
    """

    def __init__(self, state_store: StateBackend):
        self._service = ControlService(store=state_store)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gw-opt")
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()
        self._CACHE_TTL = 1.0  # seconds
        self._redis_available = False  # Updated externally when Redis is wired in

    def optimize(self, owner_id: str, body: AgentOptimizeControlInput) -> dict:
        """Run optimization for a single agent request.

        Returns the same response format as the cloud API:
        {target_duty, recommended_cooling_delta, source, ...}
        """
        node_id = body.node_id
        now = time.time()

        # Check cache (1s TTL prevents duplicate computation)
        with self._cache_lock:
            cached = self._cache.get(node_id)
            # Lazy cleanup
            stale = [k for k, (ts, _) in self._cache.items() if now - ts > 10.0]
            for k in stale:
                del self._cache[k]
        if cached and (now - cached[0]) < self._CACHE_TTL:
            return cached[1]

        # Run optimization synchronously (gateway handles fewer concurrent requests)
        try:
            result = self._service.optimize_control(
                owner_id=owner_id,
                node_id=body.node_id,
                temp_c=body.temp_c,
                fan_rpm=body.fan_rpm,
                gpu_power_w=body.gpu_power_w,
                cpu_temp_c=body.cpu_temp_c,
                max_fan_rpm=body.max_fan_rpm,
                last_commanded_duty=body.last_commanded_duty,
                peak_power_w=body.peak_power_w,
                calibration_profile_dict=body.calibration_profile,
            )
        except Exception as e:
            logger.error("Optimization failed for node %s: %s", node_id, e)
            # Fallback: simple linear curve
            result = ControlService._fallback_curve(body.temp_c, source="fallback_gateway_error")

        # Cache result
        with self._cache_lock:
            self._cache[node_id] = (time.time(), result)

        return result

    def get_all_profiles(self) -> Dict[str, CalibrationProfile]:
        """Return all calibration profiles (for debug endpoints)."""
        return self._service.get_calibration_profiles()

    def get_node_calibration(self, node_id: str) -> Optional[dict]:
        """Return detailed calibration for a single node."""
        return self._service.get_calibration_detail(node_id)

    def get_safety_state(self, owner_id: str, node_id: str) -> dict:
        """Return per-node safety state."""
        return self._service.get_safety_state(owner_id, node_id)

    @property
    def control_service(self) -> ControlService:
        """Expose the underlying ControlService for advanced use."""
        return self._service
