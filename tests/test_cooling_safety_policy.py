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

        record_spike_if_hot("test-session", 80.0, time.time())  # Above 75°C threshold
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
        """Phase 6.1: With large thermal margin (>5°C), power slope alone should NOT
        force temp_rising=True — the brain should still recommend reducing cooling."""
        brain = OptimizationBrain(target_temp=65.0)
        # Temp ~38°C, target 65°C → 27°C margin → pre-ramp should NOT fire
        gap = brain.analyze(_nodes_power_ramp_cool_temp(14), policy_context={})
        self.assertLessEqual(
            gap.recommended_cooling_delta,
            0.0,
            "With 27°C thermal margin, brain should recommend reduction (not increase) "
            "despite rising power slope — thermal inertia provides ample safety.",
        )

    def test_optimizer_safety_net_positive_delta(self):
        opt = PowerCostOptimizer()
        # Phase 6.1: Safety net threshold raised to 5.0 W/s.
        # With 25°C margin and moderate power slope (6.0 W/s, above new threshold),
        # safety net should still trigger to ensure a positive delta.
        r = opt.optimize_for_min_power(
            current_thermal=40.0,
            target_temp=65.0,
            max_safe_temp=80.0,
            current_cooling=2000.0,
            max_cooling=7000.0,
            over_provisioning=0.4,
            temp_rising=False,
            oscillation=False,
            power_slope_w_per_s=6.0,  # Above new SAFETY_NET threshold (5.0)
            sustained_active_compute=True,
            current_power_w=ACTIVE_COMPUTE_POWER_THRESHOLD_W + 10,
        )
        self.assertGreaterEqual(r.recommended_delta, 0.03)  # New SAFETY_NET_MIN_DELTA


if __name__ == "__main__":
    unittest.main()
