"""
Unit tests for ControlService — the shared optimization pipeline.

Uses InMemoryStateStore exclusively (no Redis, no network). Fast and
deterministic. Tests the core optimization logic that both cloud API
and edge gateway share.
"""

import sys
import time
import threading
import unittest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.optimization.in_memory_store import InMemoryStateStore
from core.optimization.control_service import ControlService
from core.optimization.thermal_calibrator import CalibrationProfile
from core.optimization.optimization_brain import EfficiencyGap
from core.hal.base_node import ServerNode, BaseNode
from core.optimization.spike_hold_state import clear_spike_hold


def _make_store() -> InMemoryStateStore:
    return InMemoryStateStore()


def _make_service(store=None) -> ControlService:
    """Create a ControlService with fresh defaults (gateway path)."""
    if store is None:
        store = _make_store()
    return ControlService(store=store)


def _make_nodes(n=30, temp=55.0, power=80.0, rpm=3000.0, node_id="test-node") -> list:
    """Generate a list of synthetic ServerNodes for brain input."""
    nodes = []
    now = time.time()
    for i in range(n):
        nodes.append(ServerNode(
            thermal_input=temp + (i * 0.1),
            power_draw=power,
            cooling_output=rpm,
            utilization=40.0,
            node_id=node_id,
            timestamp=datetime.fromtimestamp(now - (n - i) * 3, tz=timezone.utc),
            source="test",
        ))
    return nodes


class TestOptimizeControlBasic(unittest.TestCase):
    """Test ControlService.optimize_control returns valid results."""

    def test_returns_valid_target_duty(self):
        """optimize_control returns a dict with target_duty in 0-100."""
        store = _make_store()
        svc = _make_service(store)
        # Populate thermal history so we get >1 node
        now = time.time()
        for i in range(30):
            store.append_thermal_history("owner-1", (
                now - (30 - i) * 3,
                55.0, 55.0, 3000.0, 3000.0, None, None, 80.0, 80.0, None, None,
            ))

        result = svc.optimize_control(
            owner_id="owner-1",
            node_id="node-001",
            temp_c=55.0,
            fan_rpm=3000.0,
            gpu_power_w=80.0,
            cpu_temp_c=45.0,
            max_fan_rpm=7000.0,
            last_commanded_duty=None,
            peak_power_w=None,
            calibration_profile_dict=None,
        )

        self.assertIn("target_duty", result)
        self.assertIn("source", result)
        duty = result["target_duty"]
        self.assertGreaterEqual(duty, 0)
        self.assertLessEqual(duty, 100)

    def test_returns_valid_source_field(self):
        """source field is one of optimization_brain, fallback_simple, fallback_error."""
        store = _make_store()
        svc = _make_service(store)
        now = time.time()
        for i in range(30):
            store.append_thermal_history("owner-1", (
                now - (30 - i) * 3,
                55.0, 55.0, 3000.0, 3000.0, None, None, 80.0, 80.0, None, None,
            ))

        result = svc.optimize_control(
            owner_id="owner-1", node_id="node-001",
            temp_c=55.0, fan_rpm=3000.0, gpu_power_w=80.0,
            cpu_temp_c=None, max_fan_rpm=7000.0,
            last_commanded_duty=None, peak_power_w=None,
            calibration_profile_dict=None,
        )
        self.assertIn(result["source"], ("optimization_brain", "fallback_simple", "fallback_error"))


class TestFallbackCurve(unittest.TestCase):
    """Test fallback when insufficient thermal history."""

    def test_fallback_with_no_history(self):
        """No thermal history -> fallback_simple with duty based on temp."""
        svc = _make_service()
        result = svc.optimize_control(
            owner_id="owner-1", node_id="node-new",
            temp_c=50.0, fan_rpm=2000.0, gpu_power_w=60.0,
            cpu_temp_c=None, max_fan_rpm=7000.0,
            last_commanded_duty=None, peak_power_w=None,
            calibration_profile_dict=None,
        )
        self.assertEqual(result["source"], "fallback_simple")

    def test_fallback_hot_temp_high_duty(self):
        """Very hot temp -> duty = 100."""
        svc = _make_service()
        result = svc.optimize_control(
            owner_id="owner-1", node_id="node-hot",
            temp_c=85.0, fan_rpm=2000.0, gpu_power_w=200.0,
            cpu_temp_c=None, max_fan_rpm=7000.0,
            last_commanded_duty=None, peak_power_w=None,
            calibration_profile_dict=None,
        )
        self.assertEqual(result["target_duty"], 100)

    def test_fallback_cool_temp_low_duty(self):
        """Very cool temp -> duty = 25."""
        svc = _make_service()
        result = svc.optimize_control(
            owner_id="owner-1", node_id="node-cool",
            temp_c=40.0, fan_rpm=2000.0, gpu_power_w=10.0,
            cpu_temp_c=None, max_fan_rpm=7000.0,
            last_commanded_duty=None, peak_power_w=None,
            calibration_profile_dict=None,
        )
        self.assertEqual(result["target_duty"], 25)


class TestAutoEnrollment(unittest.TestCase):
    """Test that new nodes get a default CalibrationProfile on first contact."""

    def test_new_node_gets_default_profile(self):
        """First POST from unknown node creates default CalibrationProfile."""
        store = _make_store()
        svc = _make_service(store)

        # No profile exists yet
        self.assertIsNone(store.get_calibration_profile("node-brand-new"))

        # Call optimize (will use fallback since no history, but should enroll)
        svc.optimize_control(
            owner_id="owner-1", node_id="node-brand-new",
            temp_c=55.0, fan_rpm=2000.0, gpu_power_w=60.0,
            cpu_temp_c=None, max_fan_rpm=7000.0,
            last_commanded_duty=None, peak_power_w=None,
            calibration_profile_dict=None,
        )

        # Now profile exists
        cp = store.get_calibration_profile("node-brand-new")
        self.assertIsNotNone(cp)
        self.assertIsInstance(cp, CalibrationProfile)

    def test_provided_profile_overrides_default(self):
        """Agent-provided calibration_profile_dict updates the stored profile."""
        store = _make_store()
        svc = _make_service(store)

        profile_dict = {
            "temp_mean_c": 62.5,
            "temp_stdev_c": 3.2,
            "fan_idle_rpm": 1500.0,
            "fan_ceiling_rpm": 5000.0,
        }

        svc.optimize_control(
            owner_id="owner-1", node_id="node-with-profile",
            temp_c=62.0, fan_rpm=3000.0, gpu_power_w=80.0,
            cpu_temp_c=None, max_fan_rpm=7000.0,
            last_commanded_duty=None, peak_power_w=None,
            calibration_profile_dict=profile_dict,
        )

        cp = store.get_calibration_profile("node-with-profile")
        self.assertIsNotNone(cp)
        self.assertAlmostEqual(cp.temp_mean_c, 62.5, places=1)


class TestRunWithStateMachine(unittest.TestCase):
    """Test ControlService.run_with_state_machine returns EfficiencyGap."""

    def test_returns_efficiency_gap(self):
        """run_with_state_machine returns an EfficiencyGap dataclass."""
        svc = _make_service()
        nodes = _make_nodes(30, temp=55.0, power=80.0)
        gap = svc.run_with_state_machine(nodes)
        self.assertIsInstance(gap, EfficiencyGap)
        self.assertIsNotNone(gap.recommended_cooling_delta)

    def test_emergency_mode_at_high_temp(self):
        """T >= 80C should trigger emergency mode (guardrails)."""
        svc = _make_service()
        nodes = _make_nodes(30, temp=82.0, power=200.0, rpm=7000.0)
        gap = svc.run_with_state_machine(nodes)
        self.assertTrue(gap.emergency_mode)

    def test_normal_temp_no_emergency(self):
        """T = 55C should NOT trigger emergency mode."""
        svc = _make_service()
        nodes = _make_nodes(30, temp=55.0, power=80.0)
        gap = svc.run_with_state_machine(nodes)
        self.assertFalse(gap.emergency_mode)


class TestPerNodeIsolation(unittest.TestCase):
    """Test that ControlService maintains per-node state isolation."""

    def test_calibration_profiles_per_node(self):
        """Different nodes get different calibration profiles."""
        store = _make_store()
        svc = _make_service(store)

        cp_hot = CalibrationProfile(temp_mean_c=90.0, temp_stdev_c=5.0)
        cp_cold = CalibrationProfile(temp_mean_c=35.0, temp_stdev_c=2.0)

        store.set_calibration_profile("hot-node", cp_hot)
        store.set_calibration_profile("cold-node", cp_cold)

        hot = store.get_calibration_profile("hot-node")
        cold = store.get_calibration_profile("cold-node")

        self.assertAlmostEqual(hot.temp_mean_c, 90.0)
        self.assertAlmostEqual(cold.temp_mean_c, 35.0)
        self.assertIsNot(hot, cold)

    def test_spike_history_per_node(self):
        """Spike history is isolated per node_id."""
        store = _make_store()
        now = time.time()

        store.append_spike("node-A", now, 95.0)
        store.append_spike("node-A", now + 1, 96.0)
        store.append_spike("node-B", now, 35.0)

        hist_a = store.get_spike_history("node-A")
        hist_b = store.get_spike_history("node-B")

        self.assertEqual(len(hist_a), 2)
        self.assertEqual(len(hist_b), 1)
        self.assertAlmostEqual(hist_a[0][1], 95.0)
        self.assertAlmostEqual(hist_b[0][1], 35.0)


class TestMaintenanceMode(unittest.TestCase):
    """Test maintenance mode management via ControlService."""

    def test_set_and_get_maintenance_mode(self):
        svc = _make_service()
        self.assertEqual(svc.get_maintenance_mode_ids(), set())

        svc.set_maintenance_mode("node-001", True)
        self.assertIn("node-001", svc.get_maintenance_mode_ids())

        svc.set_maintenance_mode("node-001", False)
        self.assertNotIn("node-001", svc.get_maintenance_mode_ids())

    def test_maintenance_nodes_skipped_in_optimization(self):
        """Maintenance mode nodes are filtered from active_nodes."""
        svc = _make_service()
        svc.set_maintenance_mode("maint-node", True)

        nodes = _make_nodes(10, node_id="maint-node")
        nodes += _make_nodes(10, node_id="active-node")

        gap = svc.run_with_state_machine(nodes)
        # Should not crash; maint-node excluded from optimization
        self.assertIsInstance(gap, EfficiencyGap)


class TestDebugAccessors(unittest.TestCase):
    """Test ControlService debug/introspection methods."""

    def test_get_calibration_profiles_empty(self):
        svc = _make_service()
        self.assertEqual(svc.get_calibration_profiles(), {})

    def test_get_calibration_detail_returns_dict(self):
        store = _make_store()
        svc = _make_service(store)
        cp = CalibrationProfile(temp_mean_c=55.0)
        store.set_calibration_profile("node-X", cp)

        detail = svc.get_calibration_detail("node-X")
        self.assertIsNotNone(detail)
        self.assertIn("temp_mean_c", detail)

    def test_get_calibration_detail_missing_returns_none(self):
        svc = _make_service()
        self.assertIsNone(svc.get_calibration_detail("nonexistent"))

    def test_get_safety_state(self):
        """get_safety_state returns dict with required fields."""
        store = _make_store()
        svc = _make_service(store)
        state = svc.get_safety_state("owner-1", "node-001")
        self.assertIn("node_id", state)
        self.assertIn("session_key", state)
        self.assertIn("spike_hold_active", state)
        self.assertIn("spike_hold_remaining_s", state)
        self.assertIn("recent_temps", state)
        self.assertEqual(state["session_key"], "owner-1:node-001")


class TestConcurrentAccess(unittest.TestCase):
    """Test thread safety of ControlService."""

    def test_concurrent_optimize_no_crash(self):
        """10 threads calling optimize_control simultaneously don't crash."""
        store = _make_store()
        svc = _make_service(store)
        now = time.time()
        # Pre-populate history for all nodes
        for i in range(10):
            owner = f"owner-{i}"
            for j in range(30):
                store.append_thermal_history(owner, (
                    now - (30 - j) * 3,
                    55.0 + i, 55.0, 3000.0, 3000.0, None, None, 80.0, 80.0, None, None,
                ))

        results = [None] * 10
        errors = [None] * 10

        def _optimize(idx):
            try:
                clear_spike_hold(f"owner-{idx}:node-{idx:03d}")
                results[idx] = svc.optimize_control(
                    owner_id=f"owner-{idx}",
                    node_id=f"node-{idx:03d}",
                    temp_c=55.0 + idx,
                    fan_rpm=3000.0,
                    gpu_power_w=80.0,
                    cpu_temp_c=None,
                    max_fan_rpm=7000.0,
                    last_commanded_duty=None,
                    peak_power_w=None,
                    calibration_profile_dict=None,
                )
            except Exception as e:
                errors[idx] = str(e)

        threads = [threading.Thread(target=_optimize, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for i in range(10):
            self.assertIsNone(errors[i], f"Thread {i} failed: {errors[i]}")
            self.assertIsNotNone(results[i], f"Thread {i} returned None")
            self.assertIn("target_duty", results[i])


class TestInMemoryStateStore(unittest.TestCase):
    """Test InMemoryStateStore protocol compliance."""

    def test_thermal_history_ordering(self):
        """get_thermal_history returns entries sorted by timestamp."""
        store = _make_store()
        store.append_thermal_history("o", (300.0, 55.0, 55.0))
        store.append_thermal_history("o", (100.0, 50.0, 50.0))
        store.append_thermal_history("o", (200.0, 52.0, 52.0))

        hist = store.get_thermal_history("o", max_points=3)
        timestamps = [r[0] for r in hist]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_spike_history_max_30(self):
        """Spike history capped at 30 entries."""
        store = _make_store()
        for i in range(50):
            store.append_spike("node", float(i), float(i))
        hist = store.get_spike_history("node")
        self.assertLessEqual(len(hist), 30)


if __name__ == "__main__":
    unittest.main()
