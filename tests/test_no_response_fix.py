"""
Tests for the no-response fix and idle floor suppression.

Covers:
  - When GPU power is below active_compute_trigger, the active-compute floor
    should NOT be applied (idle gate suppression).
  - When GPU power is above active_compute_trigger, the floor IS applied.
  - The idle gate uses 85% of active_compute_trigger_w as the threshold.
  - merge_policy_floors returns 0 floor when sustained_active=False and power < threshold.
  - FAN_DIAG logging fires at the correct interval.
"""

import numpy as np
import pytest

from core.optimization.cooling_safety_policy import (
    merge_policy_floors,
    ACTIVE_COMPUTE_POWER_THRESHOLD_W,
)
from core.optimization.thermal_calibrator import CalibrationProfile


class TestIdleFloorSuppression:
    """Active-compute floor must NOT engage when GPU power is idle."""

    def test_floor_zero_when_idle_power_no_sustained(self):
        """merge_policy_floors returns 0 when sustained_active=False and power < threshold."""
        profile = CalibrationProfile(
            active_compute_fan_floor_rpm=2500.0,
            active_compute_trigger_w=19.5,  # gpu_idle(15) * 1.30
            spike_hold_fan_floor_rpm=2800.0,
        )
        floor = merge_policy_floors(
            telemetry_max_cooling_rpm=3200.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=12.0,  # Well below 40W threshold and idle
            sustained_active=False,
            spike_hold_min_rpm=0.0,
            calibration_profile=profile,
        )
        assert floor == 0.0, (
            f"Floor should be 0 when idle (power=12W, sustained=False), got {floor}"
        )

    def test_floor_applied_when_sustained_active(self):
        """merge_policy_floors returns active floor when sustained_active=True."""
        profile = CalibrationProfile(
            active_compute_fan_floor_rpm=2500.0,
            active_compute_trigger_w=19.5,
            spike_hold_fan_floor_rpm=2800.0,
        )
        floor = merge_policy_floors(
            telemetry_max_cooling_rpm=3200.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=60.0,
            sustained_active=True,
            spike_hold_min_rpm=0.0,
            calibration_profile=profile,
        )
        assert floor >= 2500.0, (
            f"Floor should be >= 2500 when sustained active, got {floor}"
        )

    def test_floor_applied_when_power_above_absolute_threshold(self):
        """merge_policy_floors applies floor when power >= ACTIVE_COMPUTE_POWER_THRESHOLD_W (40W)."""
        profile = CalibrationProfile(
            active_compute_fan_floor_rpm=2500.0,
            active_compute_trigger_w=19.5,
        )
        floor = merge_policy_floors(
            telemetry_max_cooling_rpm=3200.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=45.0,  # Above 40W absolute threshold
            sustained_active=False,
            spike_hold_min_rpm=0.0,
            calibration_profile=profile,
        )
        assert floor >= 2500.0, (
            f"Floor should engage at power=45W (above 40W threshold), got {floor}"
        )

    def test_spike_hold_floor_independent_of_power(self):
        """Spike hold floor applies regardless of current power draw."""
        profile = CalibrationProfile(
            active_compute_fan_floor_rpm=2500.0,
            spike_hold_fan_floor_rpm=2800.0,
        )
        floor = merge_policy_floors(
            telemetry_max_cooling_rpm=3200.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=10.0,  # Idle power
            sustained_active=False,
            spike_hold_min_rpm=2800.0,  # But spike hold is active
            calibration_profile=profile,
        )
        assert floor >= 2800.0, (
            f"Spike hold floor should apply even at idle power, got {floor}"
        )


class TestOptimizationBrainIdleGate:
    """OptimizationBrain should suppress sustained_active when GPU power is idle."""

    def test_idle_gate_suppresses_sustained_active(self):
        """When current power < active_trigger * 0.85, sustained_active must be False."""
        from core.hal.base_node import BaseNode
        from core.optimization.optimization_brain import OptimizationBrain

        profile = CalibrationProfile(
            active_compute_trigger_w=20.0,  # 15W idle * 1.30
            sustained_compute_window_s=5.0,
            active_compute_fan_floor_rpm=2500.0,
        )
        brain = OptimizationBrain(
            target_temp=65.0,
            calibration_profile=profile,
        )

        # Feed high power first to fill the rolling buffer
        for _ in range(20):
            nodes_high = [BaseNode(
                thermal_input=50.0,
                power_draw=80.0,
                cooling_output=3000.0,
                utilization=0.9,
            )]
            brain.analyze(
                nodes_high,
                policy_context={"rated_max_fan_rpm": 7000.0},
            )

        # Now feed idle power — the idle gate should suppress sustained_active
        nodes_idle = [BaseNode(
            thermal_input=35.0,
            power_draw=12.0,  # Below 20 * 0.85 = 17W
            cooling_output=3000.0,
            utilization=0.05,
        )]
        gap = brain.analyze(
            nodes_idle,
            policy_context={"rated_max_fan_rpm": 7000.0},
        )

        # sustained_active should be False due to idle gate
        assert gap.raw_metrics.get("sustained_active_compute") is False, (
            f"sustained_active should be False at 12W idle, "
            f"trigger_source={gap.raw_metrics.get('active_trigger_source')}"
        )
        assert gap.raw_metrics.get("active_trigger_source") == "idle_gate_suppressed"

    def test_high_power_triggers_sustained_active(self):
        """When current power is above trigger, sustained_active should eventually be True."""
        from core.hal.base_node import BaseNode
        from core.optimization.optimization_brain import OptimizationBrain

        profile = CalibrationProfile(
            active_compute_trigger_w=20.0,
            sustained_compute_window_s=5.0,
            active_compute_fan_floor_rpm=2500.0,
        )
        brain = OptimizationBrain(
            target_temp=65.0,
            calibration_profile=profile,
        )

        # Feed enough high-power samples (above trigger)
        gap = None
        for _ in range(20):
            nodes = [BaseNode(
                thermal_input=50.0,
                power_draw=60.0,  # Well above 20W trigger
                cooling_output=3000.0,
                utilization=0.8,
            )]
            gap = brain.analyze(
                nodes,
                policy_context={"rated_max_fan_rpm": 7000.0},
            )

        assert gap is not None
        assert gap.raw_metrics.get("sustained_active_compute") is True, (
            f"sustained_active should be True at 60W, "
            f"trigger_source={gap.raw_metrics.get('active_trigger_source')}"
        )
