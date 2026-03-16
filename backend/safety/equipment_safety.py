"""
CooledAI EquipmentSafety - Middleware Between AI Suggestions and Hardware

Enforces equipment protection before commands reach hardware:
- Minimum Runtime: Prevent compressors from short-cycling (must run at least X minutes).
- Delta Constraints: Limit max change in fan RPM per 5-minute window (mechanical stress).
- Hysteresis: Buffer zone to prevent AI from jittering setpoints around a target.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Default: 5 minutes min runtime, 400 RPM max change per 5 min, 30 RPM hysteresis band
MIN_RUNTIME_MINUTES_DEFAULT = 5.0
MAX_RPM_DELTA_PER_5MIN_DEFAULT = 400.0
HYSTERESIS_RPM_DEFAULT = 30.0
WINDOW_SECONDS = 300  # 5 minutes
_COOLING_OFF_THRESHOLD = 0.05  # Unit "off" below 5% of max


@dataclass
class EquipmentSafetyResult:
    """Result of applying equipment safety: safe delta and which constraints were applied."""
    safe_cooling_delta: float
    applied_constraints: List[str] = field(default_factory=list)
    minimum_runtime_hold: bool = False
    delta_constraint_limited: bool = False
    hysteresis_hold: bool = False


class EquipmentSafety:
    """
    Middleware between AI suggestions and hardware. Enforces minimum runtime,
    RPM delta limits per 5-minute window, and hysteresis to prevent setpoint jitter.
    """

    def __init__(
        self,
        min_runtime_minutes: float = MIN_RUNTIME_MINUTES_DEFAULT,
        max_rpm_delta_per_5min: float = MAX_RPM_DELTA_PER_5MIN_DEFAULT,
        hysteresis_rpm: float = HYSTERESIS_RPM_DEFAULT,
        window_seconds: float = WINDOW_SECONDS,
    ):
        self.min_runtime_seconds = min_runtime_minutes * 60.0
        self.max_rpm_delta_per_5min = max_rpm_delta_per_5min
        self.hysteresis_rpm = hysteresis_rpm
        self.window_seconds = window_seconds

        # Per-unit state
        self._unit_on_since: Dict[str, float] = {}   # unit_id -> monotonic time when turned on
        self._unit_off_since: Dict[str, float] = {}  # unit_id -> monotonic time when turned off
        self._unit_is_on: Dict[str, bool] = {}
        self._rpm_history: Dict[str, deque] = {}    # unit_id -> deque of (monotonic_ts, rpm)
        self._last_applied_rpm: Dict[str, float] = {}

    def _is_on(self, cooling: float, max_cooling: float) -> bool:
        if max_cooling < 1e-6:
            return False
        return cooling / max_cooling > _COOLING_OFF_THRESHOLD

    def _update_runtime_state(self, unit_id: str, current_rpm: float, max_cooling: float) -> None:
        now = time.monotonic()
        is_on = self._is_on(current_rpm, max_cooling)

        if unit_id not in self._unit_is_on:
            self._unit_is_on[unit_id] = is_on
            # Conservative: assume unit just entered current state (so min_runtime/min_off apply)
            self._unit_on_since[unit_id] = now
            self._unit_off_since[unit_id] = now
            return

        prev_on = self._unit_is_on[unit_id]
        if is_on and not prev_on:
            self._unit_on_since[unit_id] = now
        elif not is_on and prev_on:
            self._unit_off_since[unit_id] = now
        self._unit_is_on[unit_id] = is_on

    def _push_rpm_history(self, unit_id: str, rpm: float) -> None:
        now = time.monotonic()
        if unit_id not in self._rpm_history:
            self._rpm_history[unit_id] = deque(maxlen=500)
        q = self._rpm_history[unit_id]
        q.append((now, rpm))
        while q and (now - q[0][0]) > self.window_seconds:
            q.popleft()

    def _rpm_at_window_start(self, unit_id: str) -> Optional[float]:
        """RPM at the start of the 5-minute window (or oldest sample)."""
        q = self._rpm_history.get(unit_id)
        if not q:
            return None
        now = time.monotonic()
        cutoff = now - self.window_seconds
        for ts, rpm in q:
            if ts >= cutoff:
                return rpm
        return q[0][1] if q else None

    def _clamp_rpm_delta(self, unit_id: str, proposed_rpm: float) -> tuple:
        """
        Enforce max RPM change in 5-minute window. History must be updated
        with current (applied) RPM at start of each cycle.
        Returns (clamped_rpm, was_limited).
        """
        ref_rpm = self._rpm_at_window_start(unit_id)
        if ref_rpm is None:
            return proposed_rpm, False
        delta = proposed_rpm - ref_rpm
        if abs(delta) <= self.max_rpm_delta_per_5min:
            return proposed_rpm, False
        clamped = ref_rpm + (self.max_rpm_delta_per_5min if delta > 0 else -self.max_rpm_delta_per_5min)
        return clamped, True

    def _apply_hysteresis(self, unit_id: str, proposed_rpm: float) -> tuple:
        """
        If proposed RPM is within ±hysteresis_rpm of last applied, hold last applied.
        Returns (rpm_to_use, was_held_by_hysteresis).
        """
        last = self._last_applied_rpm.get(unit_id)
        if last is None:
            return proposed_rpm, False
        if abs(proposed_rpm - last) <= self.hysteresis_rpm:
            return last, True
        return proposed_rpm, False

    def apply(
        self,
        recommended_cooling_delta: float,
        current_cooling: float,
        max_cooling: float,
        unit_id: str = "default",
        emergency_bypass: bool = False,
    ) -> EquipmentSafetyResult:
        """
        Apply equipment safety: minimum runtime, delta constraint, hysteresis.
        Returns safe cooling delta and which constraints were applied.

        Args:
            recommended_cooling_delta: AI suggestion (-1 to 1).
            current_cooling: Current fan/cooling RPM (or flow).
            max_cooling: Max cooling capacity (RPM or flow).
            unit_id: Equipment unit identifier (for per-unit state).
            emergency_bypass: If True, skip all constraints (e.g. emergency mode).

        Returns:
            EquipmentSafetyResult with safe_cooling_delta and applied_constraints list.
        """
        if max_cooling < 1e-6:
            max_cooling = 3000.0
        proposed_rpm = current_cooling * (1.0 + recommended_cooling_delta)
        proposed_rpm = max(0.0, min(proposed_rpm, max_cooling))
        applied: List[str] = []
        min_runtime_hold = False
        delta_limited = False
        hysteresis_hold = False

        if emergency_bypass:
            self._last_applied_rpm[unit_id] = proposed_rpm
            self._push_rpm_history(unit_id, proposed_rpm)
            return EquipmentSafetyResult(
                safe_cooling_delta=recommended_cooling_delta,
                applied_constraints=[],
            )

        # Record current (applied) RPM for 5-minute delta window
        self._push_rpm_history(unit_id, current_cooling)
        self._update_runtime_state(unit_id, current_cooling, max_cooling)
        now = time.monotonic()
        is_on = self._unit_is_on.get(unit_id, False)
        proposed_on = self._is_on(proposed_rpm, max_cooling)

        # 1. Minimum runtime: block turn-off if compressor has run less than min_runtime
        if is_on and not proposed_on:
            elapsed = now - self._unit_on_since.get(unit_id, now)
            if elapsed < self.min_runtime_seconds:
                min_runtime_hold = True
                applied.append(
                    f"Minimum runtime: unit has run {elapsed:.0f}s < {self.min_runtime_seconds:.0f}s. "
                    "Holding current state (no turn-off)."
                )
                proposed_rpm = current_cooling
        # Block turn-on if unit has been off less than min_off (optional; same as min_runtime for symmetry)
        elif not is_on and proposed_on:
            elapsed = now - self._unit_off_since.get(unit_id, now)
            if elapsed < self.min_runtime_seconds:
                min_runtime_hold = True
                applied.append(
                    f"Minimum off time: unit off for {elapsed:.0f}s < {self.min_runtime_seconds:.0f}s. "
                    "Holding current state (no turn-on)."
                )
                proposed_rpm = current_cooling

        # 2. Delta constraint: limit max RPM change in 5-minute window
        if not min_runtime_hold:
            proposed_rpm, delta_limited = self._clamp_rpm_delta(unit_id, proposed_rpm)
            if delta_limited:
                applied.append(
                    f"Delta constraint: RPM change limited to ±{self.max_rpm_delta_per_5min:.0f} per "
                    f"{self.window_seconds:.0f}s window to reduce mechanical stress."
                )

        # 3. Hysteresis: don't jitter around target
        if not min_runtime_hold:
            proposed_rpm, hysteresis_hold = self._apply_hysteresis(unit_id, proposed_rpm)
            if hysteresis_hold:
                applied.append(
                    f"Hysteresis: setpoint within ±{self.hysteresis_rpm:.0f} RPM of current. "
                    "No change to prevent jitter."
                )

        self._last_applied_rpm[unit_id] = proposed_rpm

        safe_delta = (proposed_rpm / current_cooling) - 1.0 if current_cooling > 1e-6 else 0.0
        safe_delta = max(-1.0, min(1.0, safe_delta))

        return EquipmentSafetyResult(
            safe_cooling_delta=safe_delta,
            applied_constraints=applied,
            minimum_runtime_hold=min_runtime_hold,
            delta_constraint_limited=delta_limited,
            hysteresis_hold=hysteresis_hold,
        )

    def apply_to_gap(
        self,
        gap: Any,
        current_cooling: float,
        max_cooling: float,
        unit_id: str = "default",
    ) -> Any:
        """
        Apply equipment safety to an EfficiencyGap. Mutates gap.recommended_cooling_delta
        and appends to gap.recommendations. Call after guardrails and mechanical longevity.

        Args:
            gap: Object with recommended_cooling_delta, emergency_mode, recommendations.
            current_cooling: Current cooling RPM.
            max_cooling: Max cooling capacity.
            unit_id: Unit identifier.

        Returns:
            gap (mutated).
        """
        emergency = getattr(gap, "emergency_mode", False)
        delta = getattr(gap, "recommended_cooling_delta", 0.0)
        result = self.apply(
            recommended_cooling_delta=delta,
            current_cooling=current_cooling,
            max_cooling=max_cooling,
            unit_id=unit_id,
            emergency_bypass=emergency,
        )
        gap.recommended_cooling_delta = result.safe_cooling_delta
        for msg in result.applied_constraints:
            gap.recommendations.insert(0, f"⚙️ {msg}")
        if result.minimum_runtime_hold:
            gap.anti_short_cycle_hold = True
        if result.delta_constraint_limited:
            gap.slew_rate_limited = True
        return gap

    def reset_unit(self, unit_id: str) -> None:
        """Clear state for a unit (e.g. after maintenance or config change)."""
        self._unit_on_since.pop(unit_id, None)
        self._unit_off_since.pop(unit_id, None)
        self._unit_is_on.pop(unit_id, None)
        self._rpm_history.pop(unit_id, None)
        self._last_applied_rpm.pop(unit_id, None)
