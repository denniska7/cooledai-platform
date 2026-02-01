"""
CooledAI Safety State Machine - SystemState & SafetyOrchestrator

Manages three system states:
- STABLE_OPTIMIZING: Full AI control, normal operation
- GUARD_MODE: AI "handcuffed" - fan speed changes limited to +/- 5% per cycle
- FAIL_SAFE: AI bypassed entirely, hardware in Safe State

GUARD_MODE triggers when data reliability is questionable.
FAIL_SAFE triggers on Watchdog timeout or repeated guardrail violations.
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.hal.base_node import BaseNode

from .watchdog import Watchdog, SafeStateConfig
from .thermal_forensics import analyze_thermal_forensics, ForensicsResult

_logger = logging.getLogger("cooledai.safety")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)


class SystemState(Enum):
    """
    Four-level system state for CooledAI safety.
    
    STABLE_OPTIMIZING: Full AI control, data is reliable
    GUARD_MODE: Data reliability questionable - limit AI to +/- 5% fan change per cycle
    FAIL_SAFE: AI bypassed - Watchdog triggered or guardrails violated >3x in 10 min
    MAINTENANCE: Human-in-the-loop - AI and Guardrail commands ignored, fans locked
                 or control handed to hardware native IPMI/BIOS
    """
    STABLE_OPTIMIZING = "stable_optimizing"
    GUARD_MODE = "guard_mode"
    FAIL_SAFE = "fail_safe"
    MAINTENANCE = "maintenance"


@dataclass
class GuardModeConfig:
    """Configuration for GUARD_MODE triggers and behavior."""
    # Trigger: SyntheticSensor estimating more than this fraction of data (0-1)
    synthetic_data_threshold: float = 0.20  # 20%
    
    # Trigger: AI confidence score below this (0-1)
    confidence_threshold: float = 0.80  # 0.8
    
    # Trigger: Physically impossible temp change (°C per second)
    # e.g., CPU temp dropping 20°C in 1 second is impossible
    max_temp_rate_c_per_s: float = 20.0
    
    # GUARD_MODE behavior: max fan speed change per cycle (-1 to 1)
    # +/- 5% = 0.05
    max_fan_delta_per_cycle: float = 0.05


@dataclass
class FailSafeConfig:
    """Configuration for FAIL_SAFE triggers."""
    # Trigger: Guardrails triggered more than this many times in window
    guardrail_trigger_limit: int = 3
    
    # Time window for guardrail count (minutes)
    guardrail_window_minutes: float = 10.0


def _count_synthetic_fraction(nodes: List["BaseNode"]) -> float:
    """
    Compute fraction of nodes with at least one synthetic (estimated) value.
    
    Nodes with raw_data["_synthetic_cooling"] or raw_data["_synthetic_thermal"] = True
    are counted as synthetic.
    
    Returns:
        Fraction 0-1 (0 = no synthetic, 1 = all synthetic)
    """
    if not nodes:
        return 0.0
    synthetic_count = 0
    for n in nodes:
        rd = getattr(n, "raw_data", None) or {}
        if rd.get("_synthetic_cooling") or rd.get("_synthetic_thermal"):
            synthetic_count += 1
    return synthetic_count / len(nodes)


def _detect_impossible_temp_rate(
    nodes: List["BaseNode"],
    max_rate_c_per_s: float = 20.0,
) -> Tuple[bool, str]:
    """
    Detect physically impossible temperature rate-of-change.
    
    Example: CPU temp dropping 20°C in 1 second violates thermal mass.
    
    Requires nodes with timestamps. Uses consecutive pairs to compute dT/dt.
    
    Returns:
        (detected: bool, reason: str)
    """
    if not nodes or len(nodes) < 2:
        return False, ""
    
    for i in range(1, len(nodes)):
        prev = nodes[i - 1]
        curr = nodes[i]
        t_prev = getattr(prev, "timestamp", None)
        t_curr = getattr(curr, "timestamp", None)
        
        if t_prev is None or t_curr is None:
            continue
        
        try:
            dt_sec = (t_curr - t_prev).total_seconds()
        except Exception:
            continue
        
        if dt_sec <= 0:
            continue
        
        dT = curr.thermal_input - prev.thermal_input
        rate = abs(dT) / dt_sec
        
        if rate > max_rate_c_per_s:
            return True, (
                f"Physically impossible: temperature changed {dT:.1f}°C "
                f"in {dt_sec:.2f}s ({rate:.1f}°C/s > max {max_rate_c_per_s}°C/s)"
            )
    
    return False, ""


class SafetyOrchestrator:
    """
    Manages system state (STABLE_OPTIMIZING, GUARD_MODE, FAIL_SAFE) and
    applies safety constraints (e.g., fan delta clamping in GUARD_MODE).
    
    Integrates with:
    - SyntheticSensor: via node raw_data to detect >20% synthetic
    - AI Confidence: via evaluate(confidence_score=...)
    - Physical anomaly: via node time-series for impossible dT/dt
    - Watchdog: via optional registration for heartbeat-missed -> FAIL_SAFE
    - Guardrails: via record_guardrail_trigger() for >3 in 10 min -> FAIL_SAFE
    """
    
    def __init__(
        self,
        guard_config: Optional[GuardModeConfig] = None,
        fail_safe_config: Optional[FailSafeConfig] = None,
        watchdog: Optional[Watchdog] = None,
    ):
        """
        Initialize SafetyOrchestrator.
        
        Args:
            guard_config: GUARD_MODE thresholds (defaults if None)
            fail_safe_config: FAIL_SAFE thresholds (defaults if None)
            watchdog: Optional Watchdog - if provided, registers callback
                      so heartbeat miss triggers FAIL_SAFE
        """
        self.guard_config = guard_config or GuardModeConfig()
        self.fail_safe_config = fail_safe_config or FailSafeConfig()
        self._watchdog = watchdog
        
        self._state = SystemState.STABLE_OPTIMIZING
        self._maintenance_mode: bool = False
        self._guardrail_timestamps: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # Register with Watchdog so heartbeat miss triggers FAIL_SAFE
        if watchdog is not None:
            watchdog.register_safe_state_callback(self._on_watchdog_triggered)
            _logger.info("SafetyOrchestrator: registered with Watchdog for heartbeat-miss -> FAIL_SAFE")
    
    def _on_watchdog_triggered(self, config: SafeStateConfig) -> None:
        """Callback when Watchdog triggers Safe State (heartbeat missed)."""
        self.transition_to(SystemState.FAIL_SAFE, reason="Watchdog missed heartbeat")
    
    def toggle_maintenance_mode(self, active: bool) -> None:
        """
        Human-in-the-loop: Enable/disable maintenance mode.
        
        When active=True: Ignores all AI and Guardrail commands. Locks fans at
        current state or hands control to hardware native IPMI/BIOS.
        
        When active=False: Returns to STABLE_OPTIMIZING (AI control restored).
        """
        with self._lock:
            self._maintenance_mode = active
            if active:
                self._state = SystemState.MAINTENANCE
            else:
                self._state = SystemState.STABLE_OPTIMIZING
        _logger.info(
            "SafetyOrchestrator: Maintenance mode %s. AI and Guardrail commands %s.",
            "ENABLED" if active else "DISABLED",
            "ignored (fans locked / IPMI control)" if active else "restored",
        )
    
    def transition_to(self, new_state: SystemState, reason: str = "") -> None:
        """Transition to a new state and log."""
        with self._lock:
            old = self._state
            self._state = new_state
        _logger.warning(
            "SafetyOrchestrator: %s -> %s. Reason: %s",
            old.value,
            new_state.value,
            reason or "manual",
        )
    
    def record_guardrail_trigger(self) -> None:
        """
        Call when OptimizationBrain guardrails fire.
        
        Used to trigger FAIL_SAFE if >3 triggers in 10 minutes.
        """
        import time
        ts = time.time()
        with self._lock:
            self._guardrail_timestamps.append(ts)
            # Prune old entries outside window
            window_sec = self.fail_safe_config.guardrail_window_minutes * 60
            while self._guardrail_timestamps and ts - self._guardrail_timestamps[0] > window_sec:
                self._guardrail_timestamps.popleft()
            
            count = len(self._guardrail_timestamps)
            if count > self.fail_safe_config.guardrail_trigger_limit:
                self._state = SystemState.FAIL_SAFE
                _logger.critical(
                    "SafetyOrchestrator: FAIL_SAFE - Guardrails triggered %d times in %.1f min (limit %d)",
                    count,
                    self.fail_safe_config.guardrail_window_minutes,
                    self.fail_safe_config.guardrail_trigger_limit,
                )
    
    def _count_recent_guardrail_triggers(self) -> int:
        """Count guardrail triggers in the configured time window."""
        import time
        now = time.time()
        window_sec = self.fail_safe_config.guardrail_window_minutes * 60
        with self._lock:
            timestamps = list(self._guardrail_timestamps)
        return sum(1 for t in timestamps if now - t <= window_sec)
    
    def evaluate(
        self,
        nodes: Optional[List["BaseNode"]] = None,
        confidence_score: Optional[float] = None,
    ) -> SystemState:
        """
        Evaluate current conditions and transition to GUARD_MODE or FAIL_SAFE if needed.
        
        GUARD_MODE triggers:
        1. SyntheticSensor estimating >20% of data (nodes with _synthetic_* in raw_data)
        2. AI confidence score < 0.8
        3. Physically impossible sensor value (e.g., temp drop 20°C in 1 second)
        
        FAIL_SAFE triggers (handled elsewhere):
        - Watchdog heartbeat missed (via callback)
        - Guardrails triggered >3 times in 10 min (via record_guardrail_trigger)
        
        Args:
            nodes: List of BaseNodes (may be time-series). Used to check
                   synthetic fraction and impossible temp rate.
            confidence_score: AI confidence 0-1. If < 0.8, trigger GUARD_MODE.
        
        Returns:
            Current SystemState after evaluation
        """
        with self._lock:
            current = self._state
            maintenance = self._maintenance_mode
        
        # MAINTENANCE: Human control - no state transitions from evaluation
        if maintenance or current == SystemState.MAINTENANCE:
            return SystemState.MAINTENANCE
        
        # FAIL_SAFE is sticky - only explicit recovery can leave it
        if current == SystemState.FAIL_SAFE:
            return current
        
        # Check Watchdog (if we have it and it already triggered)
        if self._watchdog is not None and self._watchdog.safe_state_triggered:
            self.transition_to(SystemState.FAIL_SAFE, reason="Watchdog Safe State was triggered")
            return SystemState.FAIL_SAFE
        
        # Check guardrail count
        if self._count_recent_guardrail_triggers() > self.fail_safe_config.guardrail_trigger_limit:
            self.transition_to(
                SystemState.FAIL_SAFE,
                reason=f"Guardrails triggered >{self.fail_safe_config.guardrail_trigger_limit} times in {self.fail_safe_config.guardrail_window_minutes} min",
            )
            return SystemState.FAIL_SAFE
        
        # Evaluate GUARD_MODE triggers
        guard_reasons: List[str] = []
        
        # 1. Synthetic data > 20%
        if nodes:
            synth_frac = _count_synthetic_fraction(nodes)
            if synth_frac > self.guard_config.synthetic_data_threshold:
                guard_reasons.append(
                    f"SyntheticSensor estimating {synth_frac*100:.1f}% of data (>{self.guard_config.synthetic_data_threshold*100:.0f}%)"
                )
        
        # 2. Confidence < 0.8
        if confidence_score is not None and confidence_score < self.guard_config.confidence_threshold:
            guard_reasons.append(
                f"AI confidence {confidence_score:.2f} < {self.guard_config.confidence_threshold}"
            )
        
        # 3. Physically impossible temp rate
        if nodes:
            impossible, reason = _detect_impossible_temp_rate(
                nodes,
                max_rate_c_per_s=self.guard_config.max_temp_rate_c_per_s,
            )
            if impossible:
                guard_reasons.append(reason)

        # 4. Thermal runaway (dT/dt > 2°C/min, power stable) - fan failure / intake blockage
        if nodes:
            forensics = analyze_thermal_forensics(nodes)
            if forensics.trigger_guard_mode:
                guard_reasons.append(forensics.runaway_reason)
        
        if guard_reasons:
            self.transition_to(
                SystemState.GUARD_MODE,
                reason="; ".join(guard_reasons),
            )
            return SystemState.GUARD_MODE
        
        # No GUARD_MODE triggers - can return to STABLE_OPTIMIZING
        if current == SystemState.GUARD_MODE:
            self.transition_to(
                SystemState.STABLE_OPTIMIZING,
                reason="Data reliability restored",
            )
        
        return SystemState.STABLE_OPTIMIZING
    
    @property
    def state(self) -> SystemState:
        """Current system state."""
        with self._lock:
            return self._state
    
    def clamp_fan_delta(self, recommended_delta: float) -> float:
        """
        Apply GUARD_MODE "handcuff": limit fan speed change to +/- 5% per cycle.
        
        In MAINTENANCE: Returns 0 (no change) - ignore AI, lock fans at current state.
        In STABLE_OPTIMIZING or FAIL_SAFE, returns the input unchanged
        (FAIL_SAFE: AI is bypassed anyway; STABLE_OPTIMIZING: no limit).
        
        In GUARD_MODE, clamps to [-max_fan_delta, +max_fan_delta].
        
        Args:
            recommended_delta: AI's recommended cooling delta (-1 to 1,
                              e.g. -0.1 = reduce 10%, 0.1 = increase 10%)
        
        Returns:
            Clamped delta (or original if not in GUARD_MODE)
        """
        with self._lock:
            s = self._state
            maintenance = self._maintenance_mode
        
        if maintenance or s == SystemState.MAINTENANCE:
            return 0.0  # No change - lock fans, hand control to IPMI/BIOS
        
        if s != SystemState.GUARD_MODE:
            return recommended_delta
        
        limit = self.guard_config.max_fan_delta_per_cycle
        return max(-limit, min(limit, recommended_delta))
    
    def get_state_summary(self) -> dict:
        """Return current state and metrics for monitoring."""
        with self._lock:
            s = self._state
            maintenance = self._maintenance_mode
            guard_count = len(self._guardrail_timestamps)
        return {
            "state": s.value,
            "maintenance_mode": maintenance,
            "guardrail_triggers_in_window": guard_count,
            "guardrail_limit": self.fail_safe_config.guardrail_trigger_limit,
            "guardrail_window_minutes": self.fail_safe_config.guardrail_window_minutes,
            "fan_delta_limit": (
                0.0 if (maintenance or s == SystemState.MAINTENANCE)
                else (self.guard_config.max_fan_delta_per_cycle if s == SystemState.GUARD_MODE else None)
            ),
        }
