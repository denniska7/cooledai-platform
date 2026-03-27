"""Policy floors, power-slope pre-ramp, and proactive cooling behavior."""

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from core.hal.base_node import ServerNode
from core.optimization.cooling_safety_policy import (
    ACTIVE_COMPUTE_POWER_THRESHOLD_W,
    POWER_SLOPE_PRE_RAMP_MODERATE_W_S,
    merge_policy_floors,
    sustained_active_compute_signal,
    parse_policy_context,
)
from core.optimization.optimizer import PowerCostOptimizer
from core.optimization.optimization_brain import OptimizationBrain, apply_guardrails, EfficiencyGap
from core.optimization.spike_hold_state import (
    clear_spike_hold,
    record_spike_if_hot,
    spike_hold_floor_rpm,
)


def _nodes_power_ramp_cool_temp(n: int = 12) -> list:
    """Rising GPU power, temperature still below target — predictive temp may lag."""
    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    nodes = []
    for i in range(n):
        t = base + timedelta(seconds=i)
        nodes.append(
            ServerNode(
                thermal_input=38.0 + i * 0.05,
                power_draw=30.0 + i * 8.0,  # strong positive power slope
                cooling_output=1800.0,
                utilization=40.0,
                node_id="gpu-1",
                timestamp=t,
                source="test",
            )
        )
    return nodes


class TestCoolingSafetyPolicy(unittest.TestCase):
    def tearDown(self) -> None:
        clear_spike_hold("test-session")

    def test_spike_hold_session_key_matches_brain(self):
        clear_spike_hold("test-session")
        import time

        record_spike_if_hot("test-session", 50.0, time.time())
        # Capacity uses rated max (7000 default) even when node telemetry shows ~1800 RPM.
        floor = spike_hold_floor_rpm("test-session", 7000.0, time.time())
        self.assertGreater(floor, 2700.0)
        brain = OptimizationBrain(target_temp=65.0)
        nodes = _nodes_power_ramp_cool_temp(14)
        gap = brain.analyze(nodes, thermal_session_key="test-session")
        self.assertGreaterEqual(gap.raw_metrics.get("spike_hold_min_rpm_request", 0), 2700.0)

    def test_sustained_active_compute(self):
        p = np.array([35.0, 39.0, 41.0, 45.0, 50.0])
        self.assertTrue(sustained_active_compute_signal(p))
        self.assertFalse(sustained_active_compute_signal(np.array([10.0, 12.0])))

    def test_merge_floors_active_and_spike(self):
        mx = 7000.0
        f1 = merge_policy_floors(
            mx,
            power_draw_w=ACTIVE_COMPUTE_POWER_THRESHOLD_W + 5,
            sustained_active=False,
            spike_hold_min_rpm=0.0,
        )
        self.assertGreaterEqual(f1, 2400.0)
        f2 = merge_policy_floors(
            mx,
            power_draw_w=10.0,
            sustained_active=False,
            spike_hold_min_rpm=2800.0,
        )
        self.assertGreaterEqual(f2, 2800.0)

    def test_merge_stuck_low_telemetry_uses_rated_capacity(self):
        """If max cooling in history == stuck-low RPM, rated max must lift the active floor."""
        f = merge_policy_floors(
            1932.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=ACTIVE_COMPUTE_POWER_THRESHOLD_W + 10,
            sustained_active=False,
            spike_hold_min_rpm=0.0,
        )
        self.assertGreaterEqual(f, 2400.0)

    def test_parse_spike_context(self):
        self.assertEqual(parse_policy_context(None), 0.0)
        self.assertEqual(parse_policy_context({}), 0.0)
        self.assertEqual(
            parse_policy_context({"spike_hold_active": True, "spike_hold_min_rpm": 2700.0}),
            2700.0,
        )

    def test_soft_floor_raises_low_fan_under_gpu_load(self):
        gap = EfficiencyGap(recommended_cooling_delta=-0.3)
        r = apply_guardrails(
            gap,
            current_thermal=48.0,
            current_cooling=1932.0,
            power_draw=120.0,
            max_cooling=1932.0,
            sustained_active_compute=True,
            spike_hold_min_rpm=0.0,
            rated_max_fan_rpm=7000.0,
        )
        proposed = 1932.0 * (1.0 + r.recommended_cooling_delta)
        self.assertGreaterEqual(proposed, 2400.0)
        self.assertFalse(r.emergency_mode)

    def test_power_slope_pre_ramp_respects_thermal_margin(self):
        """With large thermal margin (27°C), pre-ramp should NOT trigger even with steep power slope.

        Phase 6.2: margin check means pre-ramp only fires within 5°C of target.
        Temps ~38°C vs target=65°C → margin=27°C → no pre-ramp → optimizer returns
        a reduction candidate. Guardrails may still lift the final delta due to the
        active-compute fan floor (1800 RPM < 2500 RPM floor), but the reasoning
        should NOT mention "temp rising" — proving pre-ramp was correctly blocked.
        """
        brain = OptimizationBrain(target_temp=65.0)
        gap = brain.analyze(_nodes_power_ramp_cool_temp(14), policy_context={})
        # Power slope is steep (≈8 W/s) — confirm it was detected
        self.assertGreaterEqual(
            gap.raw_metrics.get("predicted_power_slope_w_per_s", 0.0),
            POWER_SLOPE_PRE_RAMP_MODERATE_W_S - 0.1,  # float tolerance
        )
        # Pre-ramp did NOT fire: reasoning should NOT contain "temp rising" or "Increasing cooling"
        reasoning = " ".join(gap.reasoning_log or []).lower()
        self.assertNotIn(
            "temp rising",
            reasoning,
            "Pre-ramp should NOT trigger with 27°C margin — but reasoning mentions temp rising",
        )

    def test_optimizer_safety_net_positive_delta(self):
        opt = PowerCostOptimizer()
        # Cool temp, not "rising" by temp alone; high power + slope triggers safety net.
        # Phase 6.2: safety net threshold raised to 5.0 W/s → use 6.0 to exceed it.
        # Minimum delta lowered from 0.06 to 0.03.
        r = opt.optimize_for_min_power(
            current_thermal=40.0,
            target_temp=65.0,
            max_safe_temp=80.0,
            current_cooling=2000.0,
            max_cooling=7000.0,
            over_provisioning=0.4,
            temp_rising=False,
            oscillation=False,
            power_slope_w_per_s=6.0,
            sustained_active_compute=True,
            current_power_w=ACTIVE_COMPUTE_POWER_THRESHOLD_W + 10,
        )
        self.assertGreaterEqual(r.recommended_delta, 0.03)


if __name__ == "__main__":
    unittest.main()
