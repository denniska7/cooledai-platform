"""
Shared optimization control service for CooledAI.

Encapsulates the core optimization pipeline (OptimizationBrain, state machine,
anomaly detection, telemetry enrichment) behind a clean class interface.

Both the cloud API (api/main.py) and the edge gateway (gateway/optimization_service.py)
instantiate this class with different StateBackend implementations:
- Cloud: InMemoryStateStore (with JSON persistence)
- Gateway: RedisStateStore (with InMemoryStateStore fallback)

The cloud API passes its existing module-level singletons to ControlService
so there's no duplicate state. The gateway creates fresh instances.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.hal.base_node import BaseNode, ServerNode
from core.optimization.optimization_brain import OptimizationBrain, EfficiencyGap, InfluenceMap
from core.optimization.thermal_calibrator import CalibrationProfile
from core.optimization.spike_hold_state import record_spike_if_hot
from core.optimization.in_memory_store import StateBackend

logger = logging.getLogger("cooledai.control_service")


def _create_defaults():
    """Create default instances for gateway use (fresh, no circular imports)."""
    defaults = {}
    defaults["brain"] = OptimizationBrain()

    try:
        from core.ingestion import TelemetryDerivativeEnricher
        defaults["derivative_enricher"] = TelemetryDerivativeEnricher()
    except ImportError:
        defaults["derivative_enricher"] = None

    try:
        from core.synthetic.synthetic_sensor import SyntheticSensor
        defaults["synthetic_sensor"] = SyntheticSensor()
    except ImportError:
        defaults["synthetic_sensor"] = None

    try:
        from core.anomaly import SanityCheckFilter
        defaults["sanity_check_filter"] = SanityCheckFilter()
    except Exception:
        defaults["sanity_check_filter"] = None

    try:
        from core.ingestion.telemetry_smoothing import EwmaTelemetryFilter
        defaults["moving_avg_filter"] = EwmaTelemetryFilter(alpha=0.33)
    except ImportError:
        defaults["moving_avg_filter"] = None

    try:
        from core.ingestion.data_ingestor import DataIngestor
        defaults["ingestor"] = DataIngestor(
            node_type=ServerNode,
            sanity_check_filter=defaults["sanity_check_filter"],
            moving_avg_filter=defaults["moving_avg_filter"],
        )
    except ImportError:
        defaults["ingestor"] = None

    try:
        from core.config import load_influence_map
        defaults["influence_map"] = load_influence_map() or InfluenceMap()
    except Exception:
        defaults["influence_map"] = InfluenceMap()

    return defaults


class ControlService:
    """Shared optimization pipeline used by both cloud API and edge gateway.

    Holds the OptimizationBrain, SystemStateMachine, anomaly detection,
    telemetry enrichment, and per-node state management via a pluggable StateBackend.

    Thread safety:
    - self._internal_lock protects _cooling_unit_previous_state and _maintenance_mode_ids
    - StateBackend implementations handle their own locking
    - OptimizationBrain uses per-node-keyed internal buffers (safe when same node_id
      isn't processed concurrently, guaranteed by 1s result cache dedup)
    """

    def __init__(
        self,
        store: StateBackend,
        # Pre-created objects (cloud API passes its existing singletons).
        # If None, fresh instances are created (gateway path).
        brain: Optional[OptimizationBrain] = None,
        state_machine: Any = None,
        system_state_enum: Any = None,
        derivative_enricher: Any = None,
        synthetic_sensor: Any = None,
        sanity_check_filter: Any = None,
        moving_avg_filter: Any = None,
        ingestor: Any = None,
        influence_map: Any = None,
        maintenance_mode_ids: Optional[set] = None,
        cooling_unit_previous_state: Optional[Dict[str, Any]] = None,
    ):
        self._store = store

        # If pre-created objects are provided, use them. Otherwise create defaults.
        if brain is not None:
            self.brain = brain
            self.derivative_enricher = derivative_enricher
            self.synthetic_sensor = synthetic_sensor
            self.sanity_check_filter = sanity_check_filter
            self.moving_avg_filter = moving_avg_filter
            self.ingestor = ingestor
            self.influence_map = influence_map if influence_map is not None else InfluenceMap()
        else:
            # Gateway path: create fresh instances
            defaults = _create_defaults()
            self.brain = defaults["brain"]
            self.derivative_enricher = defaults["derivative_enricher"]
            self.synthetic_sensor = defaults["synthetic_sensor"]
            self.sanity_check_filter = defaults["sanity_check_filter"]
            self.moving_avg_filter = defaults["moving_avg_filter"]
            self.ingestor = defaults["ingestor"]
            self.influence_map = defaults["influence_map"]

        # State machine (api/main.py owns this; gateway may not have it)
        self.state_machine = state_machine
        self._SystemState = system_state_enum

        # Anomaly detection (optional)
        try:
            from core.anomaly import (
                detect_failing_cooling_units,
                severity_score_from_finding,
                CoolingUnitState,
            )
            self._detect_failing = detect_failing_cooling_units
            self._severity_score = severity_score_from_finding
            self._CoolingUnitState = CoolingUnitState
            self._anomaly_available = True
        except Exception:
            self._detect_failing = None
            self._severity_score = None
            self._CoolingUnitState = None
            self._anomaly_available = False

        # Failing component event logging (optional)
        try:
            from backend.shadow.shadow_logs import log_failing_component_event
            self._log_failing_event = log_failing_component_event
        except Exception:
            self._log_failing_event = None

        # Job observer (currently disabled)
        self._get_upcoming_load = None

        # Internal mutable state — protected by _internal_lock
        self._internal_lock = threading.Lock()
        # Accept shared mutable containers from api/main.py (same object reference)
        self._cooling_unit_previous_state: Dict[str, Any] = (
            cooling_unit_previous_state if cooling_unit_previous_state is not None else {}
        )
        self._maintenance_mode_ids: set = (
            maintenance_mode_ids if maintenance_mode_ids is not None else set()
        )

    # ------------------------------------------------------------------
    # Core optimization: brain.analyze + anomaly detection + state machine
    # Returns raw EfficiencyGap (no shadow/financial enrichment)
    # ------------------------------------------------------------------

    def run_with_state_machine(
        self,
        nodes: List[BaseNode],
        policy_context: Optional[dict] = None,
        thermal_session_key: Optional[str] = None,
        current_cpu_temp_c: Optional[float] = None,
        calibration_profile: Optional[Any] = None,
        node_id: Optional[str] = None,
    ) -> EfficiencyGap:
        """Run OptimizationBrain.analyze() with anomaly detection and state machine.

        Returns the raw EfficiencyGap after state machine application.
        Cloud API wraps this with shadow logging + financial metrics.
        Gateway uses the gap directly.
        """
        # Apply maintenance flags
        with self._internal_lock:
            maint_ids = set(self._maintenance_mode_ids)
        for n in nodes:
            n.maintenance_mode = getattr(n, "node_id", "") in maint_ids
        active_nodes = [n for n in nodes if not getattr(n, "maintenance_mode", False)]

        # Anomaly detection: failing component (power up, delta_t down)
        failing_findings: List[Any] = []
        with self._internal_lock:
            if self._anomaly_available and self._detect_failing is not None and nodes:
                try:
                    prev = {}
                    if self._cooling_unit_previous_state:
                        for k, v in self._cooling_unit_previous_state.items():
                            if isinstance(v, dict) and "power_draw" in v and "delta_t_c" in v:
                                prev[k] = self._CoolingUnitState(
                                    power_draw=v["power_draw"],
                                    delta_t_c=v["delta_t_c"],
                                    node_id=v.get("node_id", ""),
                                )
                    findings, new_state = self._detect_failing(nodes, previous_state=prev or None)
                    # Update in-place to preserve shared reference with api/main.py
                    self._cooling_unit_previous_state.clear()
                    self._cooling_unit_previous_state.update({
                        k: {"power_draw": s.power_draw, "delta_t_c": s.delta_t_c, "node_id": s.node_id}
                        for k, s in new_state.items()
                    })
                    failing_findings = findings or []
                except Exception as e:
                    logger.warning("Anomaly detection failed: %s", e)

        # Upcoming jobs
        upcoming_jobs = []
        if self._get_upcoming_load is not None:
            try:
                upcoming_jobs = self._get_upcoming_load(max_seconds=60.0)
            except Exception as e:
                logger.debug("Job observer unavailable: %s", e)

        # Run brain
        gap = self.brain.analyze(
            active_nodes,
            influence_map=self.influence_map,
            failing_component_findings=failing_findings if failing_findings else None,
            upcoming_jobs=upcoming_jobs if upcoming_jobs else None,
            policy_context=policy_context,
            thermal_session_key=thermal_session_key,
            current_cpu_temp_c=current_cpu_temp_c,
            calibration_profile=calibration_profile,
            node_id=node_id,
        )

        # Persist failing component findings
        if failing_findings and self._log_failing_event is not None and self._severity_score is not None:
            try:
                for f in failing_findings:
                    severity = self._severity_score(f)
                    self._log_failing_event(
                        node_id=f.node_id,
                        message=f.message,
                        power_draw_prev=f.power_draw_prev,
                        power_draw_now=f.power_draw_now,
                        delta_t_prev=f.delta_t_prev,
                        delta_t_now=f.delta_t_now,
                        severity_score=severity,
                    )
            except Exception as e:
                logger.warning("Failed to persist failing component event: %s", e)

        # Clean maintenance nodes from per-unit deltas
        if hasattr(gap, "recommended_cooling_delta_by_unit") and gap.recommended_cooling_delta_by_unit:
            with self._internal_lock:
                for nid in list(gap.recommended_cooling_delta_by_unit.keys()):
                    if nid in self._maintenance_mode_ids:
                        del gap.recommended_cooling_delta_by_unit[nid]

        # State machine evaluation (unless MANUAL_OVERRIDE)
        if self.state_machine is not None and self._SystemState is not None:
            if self.state_machine.state != self._SystemState.MANUAL_OVERRIDE:
                should_guard, reason = self.state_machine.evaluate_guard_mode(
                    nodes, gap, self.synthetic_sensor, ingestor=self.ingestor
                )
                if should_guard:
                    self.state_machine.set_state(self._SystemState.GUARD_MODE, reason)
                else:
                    self.state_machine.set_state(self._SystemState.OPTIMIZING, "")

            gap = self.state_machine.apply_state_to_gap(gap)
            gap.raw_metrics["system_state"] = self.state_machine.state.value

        return gap

    # ------------------------------------------------------------------
    # Full agent optimization path: telemetry -> target_duty
    # ------------------------------------------------------------------

    def optimize_control(
        self,
        owner_id: str,
        node_id: str,
        temp_c: float,
        fan_rpm: float,
        gpu_power_w: float,
        cpu_temp_c: Optional[float],
        max_fan_rpm: float,
        last_commanded_duty: Optional[float],
        peak_power_w: Optional[float],
        calibration_profile_dict: Optional[dict],
    ) -> dict:
        """Full agent optimization path. Returns {target_duty, delta, source, ...}.

        This is the core of _optimize_control_sync from api/main.py, extracted
        so both cloud API and gateway can use the same logic.
        """
        # Per-node calibration profile update
        node_cp = None
        if calibration_profile_dict is not None:
            try:
                existing_cp = self._store.get_calibration_profile(node_id)
                if existing_cp is None or (
                    hasattr(existing_cp, "temp_mean_c")
                    and abs(existing_cp.temp_mean_c - calibration_profile_dict.get("temp_mean_c", 0)) > 0.1
                ):
                    new_cp = CalibrationProfile(**{
                        k: v for k, v in calibration_profile_dict.items()
                        if k in CalibrationProfile.__dataclass_fields__
                    })
                    # Floor guards: agent may send stale thresholds from old code
                    new_cp.active_compute_trigger_w = max(50.0, new_cp.active_compute_trigger_w)
                    new_cp.cpu_safe_ceiling_c = max(55.0, new_cp.cpu_safe_ceiling_c)
                    hw = new_cp.gpu_hw_thermal_limit_c
                    if hw > 0:
                        new_cp.spike_trigger_temp_c = max(
                            hw - 8.0, min(hw * 0.92, new_cp.spike_trigger_temp_c)
                        )
                    # Cap spike hold floor: should not exceed fan_ceiling (max observed RPM)
                    if new_cp.fan_ceiling_rpm > 0 and new_cp.spike_hold_fan_floor_rpm > new_cp.fan_ceiling_rpm:
                        new_cp.spike_hold_fan_floor_rpm = new_cp.fan_ceiling_rpm
                    self._store.set_calibration_profile(node_id, new_cp)
                    logger.info(
                        "[PREDICTOR] Node %s calibration profile updated "
                        "(temp_mean=%.1f°C, temp_stdev=%.2f°C)",
                        node_id, new_cp.temp_mean_c, new_cp.temp_stdev_c,
                    )
            except Exception as exc:
                logger.debug("Failed to update calibration profile for node %s: %s", node_id, exc)

        node_cp = self._store.get_calibration_profile(node_id)

        # Auto-enrollment: create default profile for new nodes
        if node_cp is None:
            node_cp = CalibrationProfile()
            self._store.set_calibration_profile(node_id, node_cp)
            logger.info("[AUTO-ENROLL] Node %s created with default CalibrationProfile", node_id)

        # Build node time series from thermal history + current snapshot
        nodes = self._build_nodes_for_agent_control(
            owner_id, node_id, temp_c, fan_rpm, gpu_power_w, peak_power_w,
        )
        if len(nodes) < 2:
            return self._fallback_curve(temp_c, source="fallback_simple")

        try:
            # Apply telemetry filters + derivative enrichment
            self._apply_filters_and_enrichment(nodes)

            # Per-node spike detection
            now_ts = time.time()
            self._store.append_spike(node_id, now_ts, float(temp_c))
            spike_hist = self._store.get_spike_history(node_id)
            peaks: List[float] = [t for _, t in spike_hist]

            spike_trigger_c = None
            spike_hold_dur = None
            if node_cp is not None:
                spike_trigger_c = getattr(node_cp, "spike_trigger_temp_c", None)
                spike_hold_dur = getattr(node_cp, "spike_hold_duration_s", None)
            if spike_trigger_c is None and len(peaks) >= 5:
                import numpy as _np
                spike_trigger_c = float(_np.percentile(peaks, 90)) + 5.0
                spike_trigger_c = max(42.0, min(80.0, spike_trigger_c))

            session_key = f"{owner_id}:{node_id}"
            record_spike_if_hot(
                session_key,
                max(peaks) if peaks else float(temp_c),
                spike_trigger_temp_c=spike_trigger_c,
                spike_hold_duration_s=spike_hold_dur,
            )

            # Run optimization with state machine
            gap = self.run_with_state_machine(
                nodes,
                thermal_session_key=session_key,
                policy_context={"rated_max_fan_rpm": float(max_fan_rpm)},
                current_cpu_temp_c=cpu_temp_c,
                calibration_profile=node_cp,
                node_id=node_id,
            )

            # Convert recommended_cooling_delta to target_duty
            delta = gap.recommended_cooling_delta
            logger.warning(
                "[CONTROL_DEBUG] node=%s delta=%.4f fan_rpm=%.0f target_rpm=%.0f "
                "target_duty=%d reasoning=%s",
                node_id, delta, fan_rpm, fan_rpm * (1.0 + delta),
                int(round((fan_rpm * (1.0 + delta) / max(max_fan_rpm, 1000)) * 100)),
                (gap.reasoning_log or ["none"])[-1][:120],
            )
            current_rpm = fan_rpm
            max_rpm = max(max_fan_rpm, 1000.0)
            target_rpm = current_rpm * (1.0 + delta)
            target_rpm = max(0.0, min(max_rpm, target_rpm))
            target_duty = int(round((target_rpm / max_rpm) * 100.0))
            target_duty = max(0, min(100, target_duty))
            rm = gap.raw_metrics or {}

            # Failure posture analysis
            failure_payload = self._compute_failure_posture(
                nodes, target_duty, last_commanded_duty, fan_rpm, max_rpm,
            )
            # Update target_duty if posture applied boost/floor
            if isinstance(failure_payload, tuple):
                target_duty, failure_payload = failure_payload

            # Align reported RPM with final duty
            target_rpm = (target_duty / 100.0) * max_rpm

            return {
                "target_duty": target_duty,
                "recommended_cooling_delta": delta,
                "target_rpm": round(target_rpm, 0),
                "source": "optimization_brain",
                "policy_soft_floor_rpm": rm.get("policy_soft_floor_rpm_target"),
                "policy_floor_forced_after_layers": rm.get("policy_floor_forced_after_layers"),
                "policy_capacity_rpm": rm.get("policy_capacity_rpm"),
                "failure_posture": failure_payload,
            }

        except Exception as e:
            logger.warning("Agent optimize control failed: %s", e)
            return self._fallback_curve(temp_c, source="fallback_error")

    # ------------------------------------------------------------------
    # Debug / introspection accessors
    # ------------------------------------------------------------------

    def get_calibration_profiles(self) -> Dict[str, CalibrationProfile]:
        """Return all calibration profiles."""
        return self._store.list_calibration_profiles()

    def get_calibration_detail(self, node_id: str) -> Optional[dict]:
        """Return detailed calibration state for a single node."""
        cp = self._store.get_calibration_profile(node_id)
        if cp is None:
            return None
        return cp.to_dict() if hasattr(cp, "to_dict") else vars(cp)

    def get_safety_state(self, owner_id: str, node_id: str) -> dict:
        """Return per-node safety state."""
        from core.optimization.spike_hold_state import _until, _lock as _spike_lock
        session_key = f"{owner_id}:{node_id}"
        now = time.time()
        with _spike_lock:
            hold_until = _until.get(session_key, 0.0)
        hold_active = now < hold_until
        hold_remaining_s = max(0.0, hold_until - now) if hold_active else 0.0
        hist = self._store.get_spike_history(node_id)
        return {
            "node_id": node_id,
            "session_key": session_key,
            "spike_hold_active": hold_active,
            "spike_hold_remaining_s": round(hold_remaining_s, 1),
            "recent_temps": [{"ts": ts, "temp_c": t} for ts, t in hist[-10:]],
        }

    # ------------------------------------------------------------------
    # Maintenance mode
    # ------------------------------------------------------------------

    def set_maintenance_mode(self, node_id: str, enabled: bool) -> None:
        with self._internal_lock:
            if enabled:
                self._maintenance_mode_ids.add(node_id)
            else:
                self._maintenance_mode_ids.discard(node_id)

    def get_maintenance_mode_ids(self) -> set:
        with self._internal_lock:
            return set(self._maintenance_mode_ids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_nodes_for_agent_control(
        self,
        owner_id: str,
        node_id: str,
        temp_c: float,
        fan_rpm: float,
        gpu_power_w: float,
        peak_power_w: Optional[float],
        max_points: int = 30,
    ) -> List[BaseNode]:
        """Build ServerNode time series from thermal history + current snapshot."""
        history = self._store.get_thermal_history(owner_id, max_points=max_points - 1)
        nodes: List[BaseNode] = []
        now_ts = time.time()

        for row in history:
            out = self._history_row(row)
            ts = out[0]
            pilot_temp = out[1]
            pilot_rpm = out[3]
            pilot_gpu_pwr = out[7]
            pilot_peak_pwr = out[9] if len(out) > 9 else None
            thermal = float(pilot_temp) if pilot_temp is not None else 0.0
            power = float(pilot_gpu_pwr) if pilot_gpu_pwr is not None else 50.0
            cooling = float(pilot_rpm) if pilot_rpm is not None else 2000.0
            util = min(100.0, max(0.0, (power / 200.0) * 50.0)) if power else 0.0
            nodes.append(ServerNode(
                thermal_input=thermal,
                power_draw=power,
                cooling_output=cooling,
                utilization=util,
                node_id=node_id,
                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                source="thermal_history",
                peak_power_w=float(pilot_peak_pwr) if pilot_peak_pwr is not None else None,
            ))

        # Append current snapshot
        util = min(100.0, max(0.0, (gpu_power_w / 200.0) * 50.0))
        nodes.append(ServerNode(
            thermal_input=temp_c,
            power_draw=gpu_power_w,
            cooling_output=fan_rpm,
            utilization=util,
            node_id=node_id,
            timestamp=datetime.fromtimestamp(now_ts, tz=timezone.utc),
            source="agent",
            peak_power_w=peak_power_w,
        ))
        return nodes

    @staticmethod
    def _history_row(row: tuple) -> tuple:
        """Normalize thermal history row across legacy/new formats."""
        ts = float(row[0])
        pilot = float(row[1])
        baseline = float(row[2])
        pilot_rpm = float(row[3]) if len(row) > 3 and row[3] is not None else None
        baseline_rpm = float(row[4]) if len(row) > 4 and row[4] is not None else None
        pilot_cpu = float(row[5]) if len(row) > 5 and row[5] is not None else None
        baseline_cpu = float(row[6]) if len(row) > 6 and row[6] is not None else None
        pilot_gpu_pwr = float(row[7]) if len(row) > 7 and row[7] is not None else None
        baseline_gpu_pwr = float(row[8]) if len(row) > 8 and row[8] is not None else None
        pilot_peak_pwr = float(row[9]) if len(row) > 9 and row[9] is not None else None
        baseline_peak_pwr = float(row[10]) if len(row) > 10 and row[10] is not None else None
        return (ts, pilot, baseline, pilot_rpm, baseline_rpm, pilot_cpu, baseline_cpu,
                pilot_gpu_pwr, baseline_gpu_pwr, pilot_peak_pwr, baseline_peak_pwr)

    def _apply_filters_and_enrichment(self, nodes: List[BaseNode]) -> None:
        """Apply telemetry filters and derivative enrichment to nodes."""
        try:
            from core.ingestion import apply_telemetry_filters
            apply_telemetry_filters(nodes, self.sanity_check_filter, self.moving_avg_filter)
        except ImportError:
            pass
        if self.derivative_enricher is not None:
            self.derivative_enricher.enrich(nodes)

    @staticmethod
    def _fallback_curve(temp_c: float, source: str = "fallback_simple") -> dict:
        """Simple linear fan curve when optimization can't run."""
        delta_temp = temp_c - 65.0
        if delta_temp < -15:
            duty = 25
        elif delta_temp < -5:
            duty = 40
        elif delta_temp < 0:
            duty = 55
        elif delta_temp < 8:
            duty = 70
        elif delta_temp < 15:
            duty = 85
        else:
            duty = 100
        return {
            "target_duty": duty,
            "recommended_cooling_delta": 0.0,
            "source": source,
        }

    @staticmethod
    def _compute_failure_posture(
        nodes: List[BaseNode],
        target_duty: int,
        last_commanded_duty: Optional[float],
        fan_rpm: float,
        max_fan_rpm: float,
    ) -> Any:
        """Compute failure posture (stale history, fan slippage, saturation).

        Returns either:
        - (adjusted_duty, payload_dict) if posture was computed
        - dict with error info if posture computation failed
        """
        try:
            from core.safety.telemetry_failure_policy import build_posture_from_nodes
            adjusted_duty, fail_posture = build_posture_from_nodes(
                nodes,
                target_duty=target_duty,
                last_commanded_duty=last_commanded_duty,
                fan_rpm=float(fan_rpm),
                max_fan_rpm=float(max_fan_rpm),
                now=datetime.now(timezone.utc),
            )
            payload = {
                "reasons": fail_posture.reasons,
                "stale_history": fail_posture.stale_history,
                "stale_age_sec": round(fail_posture.stale_age_sec, 1),
                "fan_slippage": fail_posture.fan_slippage,
                "slippage_detail": fail_posture.slippage_detail or None,
                "saturation_suspected": fail_posture.saturation_suspected,
                "saturation_detail": fail_posture.saturation_detail or None,
                "duty_boost_applied": fail_posture.duty_boost,
                "duty_floor_applied": fail_posture.duty_floor,
            }
            return (adjusted_duty, payload)
        except Exception as e:
            logger.debug("failure posture skipped: %s", e)
            return {"reasons": [], "note": "failure_posture_unavailable"}
