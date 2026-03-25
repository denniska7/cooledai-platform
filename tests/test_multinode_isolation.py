"""
Task 0.3: Multi-node isolation tests.

Verifies that spike hold state, calibration profiles, and safety policies
are properly isolated between nodes.
"""

import threading
import time
import unittest

from core.optimization.spike_hold_state import (
    record_spike_if_hot,
    spike_hold_floor_rpm,
    clear_spike_hold,
)
from core.optimization.cooling_safety_policy import (
    sustained_active_compute_signal,
    policy_capacity_rpm,
    active_compute_fan_floor_rpm,
    merge_policy_floors,
)
import numpy as np


class TestSpikeHoldIsolation(unittest.TestCase):
    """Node A spike hold must NOT affect Node B's optimization."""

    def setUp(self):
        # Use composite keys matching the new pattern: owner_id:node_id
        self.node_a = "owner-test:node-001"
        self.node_b = "owner-test:node-002"

    def tearDown(self):
        clear_spike_hold(self.node_a)
        clear_spike_hold(self.node_b)

    def test_spike_hold_isolation(self):
        """Node A spike hold must NOT affect Node B."""
        # Record a spike on node A (95°C, well above 42°C threshold)
        record_spike_if_hot(self.node_a, 95.0, spike_trigger_temp_c=42.0, spike_hold_duration_s=240)

        # Node A should be in spike hold (floor RPM > 0)
        floor_a = spike_hold_floor_rpm(self.node_a, 7000)
        self.assertGreater(floor_a, 0.0, "Node A should have active spike hold")

        # Node B should NOT be in spike hold
        floor_b = spike_hold_floor_rpm(self.node_b, 7000)
        self.assertEqual(floor_b, 0.0, "Node B should NOT have spike hold from Node A's spike")

    def test_spike_hold_expiry(self):
        """Spike hold should expire independently per node."""
        # Record spike on node A with very short duration
        record_spike_if_hot(self.node_a, 95.0, spike_trigger_temp_c=42.0, spike_hold_duration_s=0.1)

        # Initially active
        floor_a = spike_hold_floor_rpm(self.node_a, 7000)
        self.assertGreater(floor_a, 0.0)

        # Wait for expiry
        time.sleep(0.2)

        # Should be expired
        floor_a = spike_hold_floor_rpm(self.node_a, 7000)
        self.assertEqual(floor_a, 0.0, "Node A spike hold should have expired")

    def test_independent_spike_triggers(self):
        """Each node can independently trigger and clear spike holds."""
        record_spike_if_hot(self.node_a, 95.0, spike_trigger_temp_c=42.0, spike_hold_duration_s=60)
        record_spike_if_hot(self.node_b, 50.0, spike_trigger_temp_c=42.0, spike_hold_duration_s=60)

        # Node A: high temp -> spike hold active
        floor_a = spike_hold_floor_rpm(self.node_a, 7000)
        self.assertGreater(floor_a, 0.0)

        # Node B: 50°C > 42°C threshold -> also active
        floor_b = spike_hold_floor_rpm(self.node_b, 7000)
        self.assertGreater(floor_b, 0.0)

        # Clear only node A
        clear_spike_hold(self.node_a)
        floor_a = spike_hold_floor_rpm(self.node_a, 7000)
        self.assertEqual(floor_a, 0.0, "Node A should be cleared")

        # Node B still active
        floor_b = spike_hold_floor_rpm(self.node_b, 7000)
        self.assertGreater(floor_b, 0.0, "Node B should still be in spike hold")


class TestConcurrentSpikeIsolation(unittest.TestCase):
    """Concurrent threads triggering spikes on different nodes."""

    def setUp(self):
        self.num_nodes = 10
        self.session_keys = [f"owner-test:node-{i:03d}" for i in range(self.num_nodes)]

    def tearDown(self):
        for key in self.session_keys:
            clear_spike_hold(key)

    def test_concurrent_spike_isolation(self):
        """10 concurrent threads, each triggering spikes — verify independence."""
        errors = []

        def spike_worker(idx: int):
            key = self.session_keys[idx]
            # Even-indexed nodes get hot (95°C), odd-indexed stay cool (30°C)
            temp = 95.0 if idx % 2 == 0 else 30.0
            try:
                record_spike_if_hot(key, temp, spike_trigger_temp_c=42.0, spike_hold_duration_s=60)
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [threading.Thread(target=spike_worker, args=(i,)) for i in range(self.num_nodes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Errors during concurrent spikes: {errors}")

        # Verify: even-indexed nodes have spike hold, odd-indexed do not
        for idx in range(self.num_nodes):
            key = self.session_keys[idx]
            floor = spike_hold_floor_rpm(key, 7000)
            if idx % 2 == 0:
                self.assertGreater(floor, 0.0, f"Node {idx} (hot) should have spike hold")
            else:
                self.assertEqual(floor, 0.0, f"Node {idx} (cool) should NOT have spike hold")


class TestSafetyPolicyPerNode(unittest.TestCase):
    """Verify cooling_safety_policy.py functions are pure/stateless."""

    def test_sustained_active_compute_signal_stateless(self):
        """Calling with different params should not leave residual state."""
        high_power = np.array([100.0, 120.0, 110.0, 115.0, 130.0])
        low_power = np.array([5.0, 3.0, 4.0, 2.0, 6.0])

        # Call with high power first
        result_high = sustained_active_compute_signal(high_power)
        self.assertTrue(result_high, "High power should signal sustained active compute")

        # Call with low power — should NOT be affected by previous call
        result_low = sustained_active_compute_signal(low_power)
        self.assertFalse(result_low, "Low power should NOT signal sustained active compute")

        # Call with high power again — verify same result
        result_high2 = sustained_active_compute_signal(high_power)
        self.assertTrue(result_high2, "Second high-power call should match first")

    def test_policy_capacity_rpm_stateless(self):
        """policy_capacity_rpm is pure function."""
        cap1 = policy_capacity_rpm(3000.0, 7000.0)
        cap2 = policy_capacity_rpm(1500.0, 5000.0)
        cap3 = policy_capacity_rpm(3000.0, 7000.0)

        self.assertEqual(cap1, cap3, "Same inputs should yield same output")
        self.assertNotEqual(cap1, cap2, "Different inputs should yield different outputs")

    def test_merge_policy_floors_no_side_effects(self):
        """merge_policy_floors does not modify any global state."""
        floor1 = merge_policy_floors(
            3000.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=100.0,
            sustained_active=True,
            spike_hold_min_rpm=0.0,
        )
        floor2 = merge_policy_floors(
            3000.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=5.0,
            sustained_active=False,
            spike_hold_min_rpm=0.0,
        )
        # The floors should differ based on inputs
        self.assertGreater(floor1, floor2, "Active compute floor should be higher than idle floor")


if __name__ == "__main__":
    unittest.main()
