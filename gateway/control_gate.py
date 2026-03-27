"""
CooledAI Control Gate - Shadow / Supervised / Production Mode

CONTROL_MODE:
  SHADOW      — log only, no hardware writes
  SUPERVISED  — safety-bounded writes, max 20% reduction, rate-limited
  PRODUCTION  — safety-bounded writes, full brain authority
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    SUPERVISED = "SUPERVISED"
    PRODUCTION = "PRODUCTION"


MIN_FAN_RPM_PERCENT = 30.0
MAX_FAN_RPM_PERCENT = 100.0
MIN_TEMP_SETPOINT_C = 18.0
MAX_TEMP_SETPOINT_C = 28.0

# Supervised mode limits
SUPERVISED_MAX_REDUCTION_PCT = 20.0    # Max 20% reduction from BMC baseline
SUPERVISED_MAX_CHANGE_PER_CYCLE = 5.0  # Max 5% RPM change per 3s cycle
THERMAL_CEILING_C = 85.0              # Auto-revert to SHADOW if GPU exceeds this


class ControlGate:
    """Intercepts writes. SHADOW: log only. SUPERVISED: bounded. PRODUCTION: full."""

    def __init__(self, log_dir: str = "."):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._shadow_log = self.log_dir / "shadow_actions.log"
        self._mode = self._read_mode()
        self._mode_lock = threading.Lock()
        self._mode_history: List[Dict[str, Any]] = [{
            "mode": self._mode.value,
            "ts": time.time(),
            "reason": "startup",
        }]
        # Track last applied values for rate-limiting in supervised mode
        self._last_applied: Dict[str, float] = {}  # target → last value (%)

    def _read_mode(self) -> CONTROL_MODE:
        mode_str = os.environ.get("CONTROL_MODE", "SHADOW").upper()
        if mode_str == "PRODUCTION":
            return CONTROL_MODE.PRODUCTION
        if mode_str == "SUPERVISED":
            return CONTROL_MODE.SUPERVISED
        return CONTROL_MODE.SHADOW

    @property
    def mode(self) -> CONTROL_MODE:
        with self._mode_lock:
            return self._mode

    def switch_mode(self, new_mode: CONTROL_MODE, reason: str = "api") -> bool:
        """Switch control mode. Returns True if switch was accepted."""
        with self._mode_lock:
            old = self._mode
            # Validate transitions
            if old == CONTROL_MODE.SHADOW and new_mode == CONTROL_MODE.PRODUCTION:
                _logger.warning(
                    "ControlGate: SHADOW→PRODUCTION not allowed — must go through SUPERVISED"
                )
                return False
            self._mode = new_mode
            self._mode_history.append({
                "mode": new_mode.value,
                "ts": time.time(),
                "reason": reason,
                "previous": old.value,
            })
            _logger.warning(
                "ControlGate: Mode switch %s → %s (reason: %s)", old.value, new_mode.value, reason
            )
            # Also update env so gateway/api.py reads the latest
            os.environ["CONTROL_MODE"] = new_mode.value
            return True

    @property
    def mode_since(self) -> float:
        """Timestamp of last mode change."""
        with self._mode_lock:
            if self._mode_history:
                return self._mode_history[-1]["ts"]
            return 0.0

    def write(
        self,
        collector: BaseCollector,
        target: str,
        value: Any,
        unit: str = "",
    ) -> bool:
        current_mode = self.mode
        if current_mode == CONTROL_MODE.SHADOW:
            self._log_shadow(collector, target, value, unit)
            return True

        safe_value, ok = self._apply_safety_bounds(target, value, unit)
        if not ok:
            _logger.warning("ControlGate: Rejected unsafe value target=%s value=%s", target, value)
            return False

        if current_mode == CONTROL_MODE.SUPERVISED:
            safe_value = self._apply_supervised_limits(target, safe_value, unit)

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

    def _apply_supervised_limits(self, target: str, value: Any, unit: str) -> Any:
        """Apply supervised mode constraints: max 20% reduction, max 5% change/cycle."""
        try:
            val = float(value)
        except (TypeError, ValueError):
            return value

        is_fan = "fan" in target.lower() or "rpm" in unit.lower() or "%" in unit
        if not is_fan:
            return value

        last = self._last_applied.get(target)
        if last is not None:
            # Rate limit: max 5% change per cycle
            max_change = SUPERVISED_MAX_CHANGE_PER_CYCLE
            if val < last - max_change:
                clamped = last - max_change
                _logger.info(
                    "[SUPERVISED_RATE] target=%s requested=%.1f%% last=%.1f%% "
                    "clamped=%.1f%% (max %.1f%%/cycle)",
                    target, val, last, clamped, max_change,
                )
                val = clamped
            elif val > last + max_change:
                clamped = last + max_change
                _logger.info(
                    "[SUPERVISED_RATE] target=%s requested=%.1f%% last=%.1f%% "
                    "clamped=%.1f%% (max %.1f%%/cycle)",
                    target, val, last, clamped, max_change,
                )
                val = clamped

        # Floor: never reduce below (100% - max_reduction) = 80% of whatever the
        # BMC was running (approximated by the first seen value for this target)
        if target not in self._last_applied:
            # First write — record as baseline
            self._last_applied[target] = val
            return val

        baseline = self._last_applied.get(f"_baseline_{target}")
        if baseline is None:
            baseline = self._last_applied.get(target, val)
            self._last_applied[f"_baseline_{target}"] = baseline

        floor = baseline * (1.0 - SUPERVISED_MAX_REDUCTION_PCT / 100.0)
        if val < floor:
            _logger.info(
                "[SUPERVISED_FLOOR] target=%s requested=%.1f%% baseline=%.1f%% "
                "floor=%.1f%% (max %.0f%% reduction)",
                target, val, baseline, floor, SUPERVISED_MAX_REDUCTION_PCT,
            )
            val = floor

        self._last_applied[target] = val
        return val

    def check_thermal_ceiling(self, gpu_temp_c: float) -> bool:
        """Check if GPU temp exceeds thermal ceiling. Returns True if auto-reverted."""
        if gpu_temp_c > THERMAL_CEILING_C and self.mode in (
            CONTROL_MODE.SUPERVISED, CONTROL_MODE.PRODUCTION
        ):
            _logger.warning(
                "[THERMAL_CEILING] GPU=%.1f°C > %.0f°C — auto-reverting to SHADOW",
                gpu_temp_c, THERMAL_CEILING_C,
            )
            self.switch_mode(CONTROL_MODE.SHADOW, reason=f"thermal_ceiling_{gpu_temp_c:.1f}C")
            return True
        return False

    def revert_all_to_default(self, collectors: List[BaseCollector]) -> None:
        """
        Send REVERT_TO_DEFAULT (100% cooling) to all cooling units.
        Used by SafetyWatchdog when cloud connection is lost.
        """
        REVERT_TARGET = "revert_to_default"
        REVERT_VALUE_PERCENT = 100.0
        for c in collectors:
            try:
                self.write(c, REVERT_TARGET, REVERT_VALUE_PERCENT, "%")
            except Exception as e:
                _logger.warning("ControlGate: revert_to_default failed for %s: %s", getattr(c, "source", ""), e)

    def status(self) -> dict:
        """Return gate status for debug/API endpoints."""
        with self._mode_lock:
            return {
                "mode": self._mode.value,
                "mode_since": self._mode_history[-1]["ts"] if self._mode_history else None,
                "mode_history": self._mode_history[-10:],  # Last 10 transitions
                "last_applied": {
                    k: v for k, v in self._last_applied.items()
                    if not k.startswith("_baseline_")
                },
            }
