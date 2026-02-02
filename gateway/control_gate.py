"""
CooledAI Control Gate - Shadow vs Production Mode

CONTROL_MODE: SHADOW (intercept) or PRODUCTION (safety bounds then send)
"""

import os
import logging
from pathlib import Path
from typing import Any
from datetime import datetime
from enum import Enum

from .collectors.base_collector import BaseCollector

_logger = logging.getLogger("cooledai.control_gate")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)


class CONTROL_MODE(str, Enum):
    SHADOW = "SHADOW"
    PRODUCTION = "PRODUCTION"


MIN_FAN_RPM_PERCENT = 30.0
MAX_FAN_RPM_PERCENT = 100.0
MIN_TEMP_SETPOINT_C = 18.0
MAX_TEMP_SETPOINT_C = 28.0


class ControlGate:
    """Intercepts writes. SHADOW: log only. PRODUCTION: safety bounds then forward."""

    def __init__(self, log_dir: str = "."):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._shadow_log = self.log_dir / "shadow_actions.log"
        self._mode = self._read_mode()

    def _read_mode(self) -> CONTROL_MODE:
        mode_str = os.environ.get("CONTROL_MODE", "SHADOW").upper()
        if mode_str == "PRODUCTION":
            return CONTROL_MODE.PRODUCTION
        return CONTROL_MODE.SHADOW

    @property
    def mode(self) -> CONTROL_MODE:
        return self._mode

    def write(
        self,
        collector: BaseCollector,
        target: str,
        value: Any,
        unit: str = "",
    ) -> bool:
        if self._mode == CONTROL_MODE.SHADOW:
            self._log_shadow(collector, target, value, unit)
            return True

        safe_value, ok = self._apply_safety_bounds(target, value, unit)
        if not ok:
            _logger.warning("ControlGate: Rejected unsafe value target=%s value=%s", target, value)
            return False

        return collector.write(target, safe_value, unit)

    def _log_shadow(self, collector: BaseCollector, target: str, value: Any, unit: str) -> None:
        line = (
            f"{datetime.now().isoformat()} | protocol={collector.protocol} "
            f"source={collector.source} | target={target} value={value} unit={unit} | INTERCEPTED\n"
        )
        try:
            with open(self._shadow_log, "a") as f:
                f.write(line)
        except Exception as e:
            _logger.error("ControlGate: Failed to write shadow log: %s", e)

    def _apply_safety_bounds(self, target: str, value: Any, unit: str) -> tuple:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return value, True

        if "fan" in target.lower() or "rpm" in unit.lower() or "%" in unit:
            if val < MIN_FAN_RPM_PERCENT:
                return MIN_FAN_RPM_PERCENT, True
            if val > MAX_FAN_RPM_PERCENT:
                return MAX_FAN_RPM_PERCENT, True

        if "temp" in target.lower() or "celsius" in unit.lower():
            if val < MIN_TEMP_SETPOINT_C:
                return MIN_TEMP_SETPOINT_C, True
            if val > MAX_TEMP_SETPOINT_C:
                return MAX_TEMP_SETPOINT_C, True

        return value, True
