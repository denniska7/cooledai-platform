"""
Tests for telemetry fixes A–G applied to optimization_brain, base_node,
base_collector, and cooledai_agent.

FIX A: Peak-hold power buffer (3-second max for spike detection)
FIX B: Rolling window for sustained compute (12-second mean)
FIX C: Spike/floor commands bypass hysteresis deadband
FIX D: Power trigger fires independently of temp/confidence
FIX E: peak_power_w field on BaseNode, TelemetryObject, ThermalSnapshot
FIX F: Minimum 15 samples before sustained compute decisions
FIX G: peak_power_w, no_response_event, active_trigger_source in raw_metrics
"""

import time
from unittest import mock

import numpy as np
import pytest

from core.hal.base_node import BaseNode, ServerNode
from gateway.collectors.base_collector import TelemetryObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nodes(n=1, thermal=40.0, power=50.0, cooling=3000.0, util=50.0,
                peak_power=None, rate=None):
    """Build a list of ServerNodes for brain.analyze()."""
    nodes = []
    for i in range(n):
        node = ServerNode(
            thermal_input=thermal,
            power_draw=power,
            cooling_output=cooling,
            utilization=util,
            node_id=f"node_{i}",
            peak_power_w=peak_power,
            thermal_input_rate=rate,
        )
        nodes.append(node)
    return nodes


def _make_brain(**kw):
    """Build an OptimizationBrain with default dependencies."""
    from core.optimization.optimization_brain import OptimizationBrain
    return OptimizationBrain(**kw)


# ---------------------------------------------------------------------------
# FIX E: peak_power_w on BaseNode and TelemetryObject
# ---------------------------------------------------------------------------

class TestFixE_PeakPowerField:
    """peak_power_w field on BaseNode and TelemetryObject."""

    def test_base_node_has_peak_power_w(self):
        node = ServerNode(power_draw=100.0, peak_power_w=150.0)
        assert node.peak_power_w == 150.0

    def test_base_node_peak_power_in_dict(self):
        node = ServerNode(power_draw=100.0, peak_power_w=150.0)
        d = node.to_normalized_dict()
        assert d["peak_power_w"] == 150.0

    def test_base_node_peak_power_absent_when_none(self):
        node = ServerNode(power_draw=100.0)
        d = node.to_normalized_dict()
        assert "peak_power_w" not in d

    def test_telemetry_object_has_peak_power_w(self):
        t = TelemetryObject(power_draw=0.1, peak_power_w=0.15)
        assert t.peak_power_w == 0.15

    def test_telemetry_object_dict_includes_peak(self):
        t = TelemetryObject(power_draw=0.1, peak_power_w=0.15)
        d = t.to_dict()
        assert d["peak_power_w"] == 0.15

    def test_telemetry_object_peak_none_default(self):
        t = TelemetryObject(power_draw=0.1)
        assert t.peak_power_w is None


# ---------------------------------------------------------------------------
# FIX A: Peak-hold power buffer in OptimizationBrain
# ---------------------------------------------------------------------------

class TestFixA_PeakHoldBuffer:
    """Peak-hold buffer tracks 3-second max power."""

    def test_peak_power_in_raw_metrics(self):
        brain = _make_brain()
        nodes = _make_nodes(power=80.0)
        gap = brain.analyze(nodes)
        assert "peak_power_w" in gap.raw_metrics
        assert gap.raw_metrics["peak_power_w"] >= 80.0

    def test_peak_holds_across_calls(self):
        brain = _make_brain()
        # First call: high power
        gap1 = brain.analyze(_make_nodes(power=200.0))
        # Second call: low power (but within 3s window — peak should still be 200)
        gap2 = brain.analyze(_make_nodes(power=20.0))
        assert gap2.raw_metrics["peak_power_w"] >= 200.0

    def test_peak_decays_after_window(self):
        brain = _make_brain()
        # Simulate time passing beyond the 3s window
        brain.analyze(_make_nodes(power=200.0))
        # Manually age out the buffer
        brain._peak_power_times = [time.monotonic() - 10.0]
        brain._peak_power_buffer = [200.0]
        gap = brain.analyze(_make_nodes(power=20.0))
        # After pruning, peak should be current power (20.0)
        assert gap.raw_metrics["peak_power_w"] == pytest.approx(20.0, abs=1.0)


# ---------------------------------------------------------------------------
# FIX B: Rolling 12-second mean for sustained compute
# ---------------------------------------------------------------------------

class TestFixB_RollingMean:
    """12-second rolling mean replaces consecutive counter."""

    def test_sustained_active_uses_rolling_mean(self):
        brain = _make_brain()
        # Feed 20 samples of high power (> default threshold 35W)
        for i in range(20):
            brain._sustained_power_history.append(100.0)
            brain._sustained_power_times.append(time.monotonic() + i * 0.1)
        nodes = _make_nodes(power=100.0)
        gap = brain.analyze(nodes)
        assert gap.raw_metrics["sustained_active_compute"] is True

    def test_low_power_not_sustained(self):
        brain = _make_brain()
        nodes = _make_nodes(power=5.0)
        # Feed enough low-power samples
        for i in range(20):
            brain._sustained_power_history.append(5.0)
            brain._sustained_power_times.append(time.monotonic() + i * 0.1)
        gap = brain.analyze(nodes)
        # Should not trigger sustained compute at 5W
        assert gap.raw_metrics.get("active_trigger_source") in ("none", "legacy_tail") or not gap.raw_metrics["sustained_active_compute"]


# ---------------------------------------------------------------------------
# FIX C: Spike/floor commands bypass hysteresis
# ---------------------------------------------------------------------------

class TestFixC_HysteresisBypass:
    """Spike/floor commands bypass hysteresis deadband."""

    def test_hysteresis_bypassed_when_floor_active(self):
        from core.optimization.optimization_brain import (
            MechanicalLongevityLayer,
            EfficiencyGap,
        )
        layer = MechanicalLongevityLayer()
        gap = EfficiencyGap()
        gap.recommended_cooling_delta = 0.03  # Below 5% deadband
        gap.raw_metrics["policy_soft_floor_rpm_target"] = 3000.0
        # Delta is positive and floor is active → should bypass hysteresis
        gap = layer.apply_hysteresis(gap)
        assert gap.recommended_cooling_delta == 0.03  # Not zeroed out
        assert not gap.hysteresis_hold

    def test_hysteresis_still_applies_without_floor(self):
        from core.optimization.optimization_brain import (
            MechanicalLongevityLayer,
            EfficiencyGap,
        )
        layer = MechanicalLongevityLayer()
        gap = EfficiencyGap()
        gap.recommended_cooling_delta = 0.03  # Below 5% deadband
        # No floor active
        gap = layer.apply_hysteresis(gap)
        assert gap.recommended_cooling_delta == 0.0
        assert gap.hysteresis_hold

    def test_hysteresis_not_bypassed_for_negative_delta(self):
        from core.optimization.optimization_brain import (
            MechanicalLongevityLayer,
            EfficiencyGap,
        )
        layer = MechanicalLongevityLayer()
        gap = EfficiencyGap()
        gap.recommended_cooling_delta = -0.03  # Small negative delta
        gap.raw_metrics["policy_soft_floor_rpm_target"] = 3000.0
        # Floor is active but delta is negative (reducing) → hysteresis should apply
        gap = layer.apply_hysteresis(gap)
        assert gap.recommended_cooling_delta == 0.0
        assert gap.hysteresis_hold


# ---------------------------------------------------------------------------
# FIX D: Power trigger fires independently
# ---------------------------------------------------------------------------

class TestFixD_IndependentPowerTrigger:
    """Peak power independently triggers active compute."""

    def test_high_peak_power_triggers_without_history(self):
        brain = _make_brain()
        # Single very high power sample should trigger via peak (1.5× threshold)
        # Default threshold ~35W, so 1.5× = 52.5W. Use 200W.
        nodes = _make_nodes(power=200.0)
        gap = brain.analyze(nodes)
        assert gap.raw_metrics["sustained_active_compute"] is True
        assert gap.raw_metrics["active_trigger_source"] == "peak_power_3s"


# ---------------------------------------------------------------------------
# FIX F: Minimum 15 samples
# ---------------------------------------------------------------------------

class TestFixF_MinSamples:
    """Sustained compute needs minimum 15 samples."""

    def test_fewer_than_15_samples_no_rolling_trigger(self):
        brain = _make_brain()
        # Feed only 10 samples of high power (above threshold but < 15 samples)
        for i in range(10):
            brain._sustained_power_history.append(100.0)
            brain._sustained_power_times.append(time.monotonic() + i * 0.1)
        # Power is moderate (not high enough for peak trigger at 1.5×)
        nodes = _make_nodes(power=45.0)
        gap = brain.analyze(nodes)
        # Rolling mean trigger should NOT fire (< 15 samples)
        # But legacy tail or peak might. Check trigger source is not rolling_12s_mean.
        source = gap.raw_metrics.get("active_trigger_source", "none")
        if source == "rolling_12s_mean":
            # Verify we had >= 15 samples in history
            assert len(brain._sustained_power_history) >= 15


# ---------------------------------------------------------------------------
# FIX G: New telemetry fields in raw_metrics
# ---------------------------------------------------------------------------

class TestFixG_TelemetryFields:
    """raw_metrics includes peak_power_w, no_response_event, active_trigger_source."""

    def test_all_new_fields_present(self):
        brain = _make_brain()
        gap = brain.analyze(_make_nodes())
        assert "peak_power_w" in gap.raw_metrics
        assert "active_trigger_source" in gap.raw_metrics
        assert "no_response_event" in gap.raw_metrics

    def test_no_response_event_when_delta_zero_and_active(self):
        brain = _make_brain()
        # Pre-fill history to trigger sustained_active
        for i in range(20):
            brain._sustained_power_history.append(200.0)
            brain._sustained_power_times.append(time.monotonic() + i * 0.1)
        nodes = _make_nodes(power=200.0, thermal=40.0, cooling=5000.0)
        gap = brain.analyze(nodes)
        # If delta ends up 0 despite sustained_active, it's a no_response_event
        if abs(gap.recommended_cooling_delta) < 1e-6 and gap.raw_metrics["sustained_active_compute"]:
            assert gap.raw_metrics["no_response_event"] is True

    def test_active_trigger_source_values(self):
        brain = _make_brain()
        gap = brain.analyze(_make_nodes(power=5.0))
        assert gap.raw_metrics["active_trigger_source"] in (
            "none", "peak_power_3s", "rolling_12s_mean", "legacy_tail",
            "idle_gate_suppressed",
        )

    def test_no_response_streak_increments(self):
        brain = _make_brain()
        # Run two cycles where delta stays zero (cool node, sustained compute false)
        gap1 = brain.analyze(_make_nodes(power=5.0, thermal=30.0))
        gap2 = brain.analyze(_make_nodes(power=5.0, thermal=30.0))
        # Without sustained_active, no_response_event should be False
        assert gap2.raw_metrics["no_response_event"] is False
        assert gap2.raw_metrics["no_response_streak"] == 0


# ---------------------------------------------------------------------------
# Agent ThermalSnapshot peak_gpu_power_w
# ---------------------------------------------------------------------------

class TestAgentSnapshot:
    """ThermalSnapshot tracks peak GPU power."""

    def test_snapshot_peak_power(self):
        from scripts.cooledai_agent import ThermalSnapshot
        snap = ThermalSnapshot()
        snap.gpu_power_w = [60.0, 80.0, 70.0]
        snap.compute_max()
        assert snap.peak_gpu_power_w == 80.0

    def test_snapshot_no_gpu_power(self):
        from scripts.cooledai_agent import ThermalSnapshot
        snap = ThermalSnapshot()
        snap.compute_max()
        assert snap.peak_gpu_power_w is None
