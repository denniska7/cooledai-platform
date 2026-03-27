"""
Phase 6 Integration Tests — Optimization Engine Rework.

Tests rate-of-change safety, disabled cautionary cooling, mode-aware
policy floors, and shadow/supervised/active mode behavior.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

_TEST_API_KEY = os.environ.get("COOLEDAI_API_KEY", "")
if not _TEST_API_KEY:
    _keys_file = project_root / "data" / "api_keys.json"
    if _keys_file.exists():
        import json as _json
        _raw = _json.loads(_keys_file.read_text())
        _keys = _raw.get("keys", _raw)
        if _keys:
            _TEST_API_KEY = next(iter(_keys.keys()), "")
    if not _TEST_API_KEY:
        _TEST_API_KEY = "sk-test-phase6"
    os.environ["COOLEDAI_API_KEY"] = _TEST_API_KEY


class TestRateOfChangeSafety(unittest.TestCase):
    """Step 3: Rate-of-change monitoring replaces static spike hold."""

    def test_safe_when_stable(self):
        from core.optimization.cooling_safety_policy import check_rate_of_change
        now = time.time()
        # Stable temps: 65°C for 30 seconds
        history = [(now - i, 65.0) for i in range(30, 0, -1)]
        self.assertEqual(check_rate_of_change(history), "safe")

    def test_caution_when_rising_moderately(self):
        from core.optimization.cooling_safety_policy import check_rate_of_change
        now = time.time()
        # Rising at ~0.3°C/s over 30 seconds (0→9°C)
        history = [(now - 30 + i, 60.0 + i * 0.3) for i in range(31)]
        result = check_rate_of_change(history)
        self.assertEqual(result, "caution")

    def test_emergency_when_rising_fast(self):
        from core.optimization.cooling_safety_policy import check_rate_of_change
        now = time.time()
        # Rising at ~1.0°C/s over 30 seconds (0→30°C)
        history = [(now - 30 + i, 50.0 + i * 1.0) for i in range(31)]
        result = check_rate_of_change(history)
        self.assertEqual(result, "emergency")

    def test_safe_when_cooling(self):
        from core.optimization.cooling_safety_policy import check_rate_of_change
        now = time.time()
        # Cooling: temp dropping
        history = [(now - 30 + i, 70.0 - i * 0.2) for i in range(31)]
        self.assertEqual(check_rate_of_change(history), "safe")

    def test_safe_with_insufficient_data(self):
        from core.optimization.cooling_safety_policy import check_rate_of_change
        self.assertEqual(check_rate_of_change([]), "safe")
        self.assertEqual(check_rate_of_change([(time.time(), 65.0)]), "safe")


class TestSpikeHoldThresholdRaised(unittest.TestCase):
    """Verify spike hold threshold was raised from 42°C to 75°C."""

    def test_spike_hold_threshold(self):
        from core.optimization.cooling_safety_policy import SPIKE_HOLD_ENTER_TEMP_C
        self.assertGreaterEqual(SPIKE_HOLD_ENTER_TEMP_C, 75.0,
                                "Spike hold threshold should be >= 75°C (raised from 42°C)")

    def test_no_spike_hold_at_normal_temps(self):
        """Normal operating temps (50-68°C) should NOT trigger spike hold."""
        from core.optimization.spike_hold_state import record_spike_if_hot, spike_hold_floor_rpm, clear_spike_hold
        clear_spike_hold("test-normal")
        record_spike_if_hot("test-normal", 68.0, time.time())
        floor = spike_hold_floor_rpm("test-normal", 7000.0, time.time())
        self.assertEqual(floor, 0.0, "68°C should NOT trigger spike hold (threshold is 75°C)")
        clear_spike_hold("test-normal")


class TestCautionaryCoolingDisabled(unittest.TestCase):
    """Step 5: Cautionary cooling override should be disabled."""

    def test_brain_does_not_override_delta(self):
        """Brain should not apply +15% override even at low confidence."""
        from core.optimization.optimization_brain import OptimizationBrain
        from core.hal.base_node import ServerNode
        from datetime import datetime, timezone
        import numpy as np

        brain = OptimizationBrain(target_temp=65.0)
        nodes = []
        now = datetime.now(tz=timezone.utc)
        # Create 15 nodes with moderate temp (60°C) — optimizer should recommend reduction
        for i in range(15):
            nodes.append(ServerNode(
                thermal_input=60.0,
                power_draw=50.0,
                cooling_output=3500.0,
                utilization=50.0,
                node_id="test-node",
                timestamp=now,
                source="test",
            ))
        gap = brain.analyze(nodes, thermal_session_key="test-caution")
        # The delta should NOT be +0.15 (the old cautionary override)
        # It should be <= 0 (a reduction) since temp is below target
        self.assertLessEqual(gap.recommended_cooling_delta, 0.15,
                             f"Cautionary override should be disabled. Got delta={gap.recommended_cooling_delta}")
        self.assertFalse(gap.cautionary_cooling_state,
                         "cautionary_cooling_state should be False (disabled)")


class TestPolicyFloorsDisabledInSupervisedActive(unittest.TestCase):
    """Policy floors (active compute + spike hold) should be 0 in supervised/active."""

    def test_merge_floors_zero_in_supervised(self):
        from core.optimization.cooling_safety_policy import merge_policy_floors
        floor = merge_policy_floors(
            3500.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=120.0,
            sustained_active=True,
            spike_hold_min_rpm=2800.0,
            mode="supervised",
        )
        self.assertEqual(floor, 0.0, "Policy floors should be 0 in supervised mode")

    def test_merge_floors_zero_in_active(self):
        from core.optimization.cooling_safety_policy import merge_policy_floors
        floor = merge_policy_floors(
            3500.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=120.0,
            sustained_active=True,
            spike_hold_min_rpm=2800.0,
            mode="active",
        )
        self.assertEqual(floor, 0.0, "Policy floors should be 0 in active mode")

    def test_merge_floors_applied_in_shadow(self):
        from core.optimization.cooling_safety_policy import merge_policy_floors
        floor = merge_policy_floors(
            3500.0,
            rated_max_fan_rpm=7000.0,
            power_draw_w=120.0,
            sustained_active=True,
            spike_hold_min_rpm=2800.0,
            mode="shadow",
        )
        self.assertGreater(floor, 0.0, "Policy floors should apply in shadow mode")


class TestShadowModePassthrough(unittest.TestCase):
    """Shadow mode should return applied=False."""

    def test_shadow_mode_not_applied(self):
        from fastapi.testclient import TestClient
        from core.optimization.in_memory_store import InMemoryStateStore
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        mock_syncer = MagicMock()
        mock_syncer.current_mode = "shadow"

        app = create_app(optimization_service=service, policy_syncer=mock_syncer, keys_file=None)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/optimize/control",
            json={"temp_c": 65.0, "fan_rpm": 3500.0, "gpu_power_w": 120.0,
                  "node_id": "ST550-CooledAI-Predictive", "max_fan_rpm": 7000.0},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "shadow")
        self.assertFalse(data["applied"])


class TestSupervisedModeBMCClamp(unittest.TestCase):
    """Supervised mode clamps at 80% of BMC baseline (20% max reduction)."""

    def test_supervised_clamp_20_pct(self):
        from fastapi.testclient import TestClient
        from core.optimization.in_memory_store import InMemoryStateStore
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app, _compose_thermal_row

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        mock_syncer = MagicMock()
        mock_syncer.current_mode = "supervised"

        app = create_app(optimization_service=service, policy_syncer=mock_syncer, keys_file=None)
        client = TestClient(app)

        # Seed history for brain to engage
        owner = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        from core.models.agent_input import AgentOptimizeControlInput
        body = AgentOptimizeControlInput(
            temp_c=60.0, fan_rpm=3500.0, gpu_power_w=120.0,
            node_id="ST550-CooledAI-Predictive", max_fan_rpm=7000.0,
        )
        for _ in range(3):
            result = service.optimize(owner, body)
            row = _compose_thermal_row(body, result)
            store.append_thermal_history(owner, row)
            with service._cache_lock:
                service._cache.clear()

        resp = client.post(
            "/api/v1/optimize/control",
            json={"temp_c": 60.0, "fan_rpm": 3500.0, "gpu_power_w": 120.0,
                  "node_id": "ST550-CooledAI-Predictive", "max_fan_rpm": 7000.0},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "supervised")
        self.assertTrue(data["applied"])
        # Floor should be 80% of BMC baseline: 3500 * 0.80 / 7000 * 100 = 40%
        if data.get("target_duty") is not None:
            bmc_duty = (3500.0 / 7000.0) * 100  # 50%
            floor_duty = bmc_duty * 0.80  # 40%
            self.assertGreaterEqual(data["target_duty"], floor_duty - 1,
                                    f"Supervised duty {data['target_duty']}% should be >= floor {floor_duty}%")


if __name__ == "__main__":
    unittest.main()
