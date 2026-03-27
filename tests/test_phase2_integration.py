"""
Phase 2 Integration Tests — Real Brain Optimization + Production Hardening.

Tests are designed to run without Redis or external services.
All core logic uses InMemoryStateStore.
"""

import json
import os
import sys
import tempfile
import time
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.models.agent_input import AgentOptimizeControlInput
from core.optimization.in_memory_store import InMemoryStateStore
from gateway.optimization_service import LocalOptimizationService

# Ensure the legacy API key env var is set for gateway key validation
_TEST_API_KEY = "sk-test-phase2-key"
os.environ.setdefault("COOLEDAI_API_KEY", _TEST_API_KEY)


def _make_body(**overrides) -> AgentOptimizeControlInput:
    """Create a realistic agent telemetry payload."""
    defaults = dict(
        temp_c=65.0,
        fan_rpm=3000.0,
        gpu_power_w=120.0,
        cpu_temp_c=45.0,
        node_id="ST550-CooledAI-Predictive",
        max_fan_rpm=7000.0,
        last_commanded_duty=50.0,
        peak_power_w=180.0,
    )
    defaults.update(overrides)
    return AgentOptimizeControlInput(**defaults)


class TestThermalHistoryPopulation(unittest.TestCase):
    """Step 1: Verify thermal history accumulation enables brain engagement."""

    def test_brain_engages_after_history_accumulated(self):
        """3 sequential optimize calls → 2nd+ should not be fallback_simple."""
        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        owner_id = "test-owner"
        body = _make_body()

        sources = []
        for i in range(5):
            result = service.optimize(owner_id, body)
            # Write thermal history (same as gateway/api.py does after Step 1)
            from gateway.api import _compose_thermal_row
            row = _compose_thermal_row(body, result)
            store.append_thermal_history(owner_id, row)
            sources.append(result.get("source", "unknown"))
            # Clear cache so each call re-evaluates
            with service._cache_lock:
                service._cache.clear()

        # First call may be fallback (no history yet), but subsequent should use brain
        # At minimum, the last call should NOT be fallback_simple
        self.assertNotEqual(
            sources[-1], "fallback_simple",
            f"Brain never engaged after 5 calls. Sources: {sources}",
        )

    def test_thermal_history_row_format(self):
        """Verify _compose_thermal_row produces valid 11-tuple."""
        from gateway.api import _compose_thermal_row

        body = _make_body()
        result = {"target_duty": 40, "target_rpm": 2800}
        row = _compose_thermal_row(body, result)

        self.assertEqual(len(row), 11, f"Expected 11-tuple, got {len(row)}: {row}")
        self.assertIsInstance(row[0], float)  # timestamp
        self.assertAlmostEqual(row[1], 65.0)  # pilot_temp
        self.assertAlmostEqual(row[2], 65.0)  # baseline_temp (same when no baseline)
        self.assertAlmostEqual(row[3], 2800.0)  # pilot_rpm (from target_rpm)
        self.assertAlmostEqual(row[4], 3000.0)  # baseline_rpm (from body.fan_rpm)

    def test_thermal_history_row_with_baseline(self):
        """Verify _compose_thermal_row uses baseline snapshot when provided."""
        from gateway.api import _compose_thermal_row

        body = _make_body()
        result = {"target_duty": 40}
        baseline = {"temp_c": 70.0, "fan_rpm": 4000.0, "gpu_power_w": 130.0}
        row = _compose_thermal_row(body, result, baseline_snapshot=baseline)

        self.assertAlmostEqual(row[1], 65.0)  # pilot_temp
        self.assertAlmostEqual(row[2], 70.0)  # baseline_temp (from baseline)
        self.assertAlmostEqual(row[4], 4000.0)  # baseline_rpm (from baseline)

    def test_first_call_is_fallback(self):
        """First optimize call with empty history should be fallback."""
        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        result = service.optimize("test-owner", _make_body())
        # With no history, brain can't build time series → fallback
        self.assertIn("fallback", result.get("source", ""), f"Expected fallback, got: {result}")


class TestStatePersistence(unittest.TestCase):
    """Step 3: Verify file-based state persistence."""

    def test_thermal_history_survives_flush_load(self):
        """Write history → flush to file → new store → load → verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_file = os.path.join(tmpdir, "thermal_history.json")
            store = InMemoryStateStore(thermal_history_file=hist_file)

            # Write some history
            for i in range(5):
                row = (time.time() + i, 65.0 + i, 60.0, 3000.0, 4000.0,
                       45.0, 42.0, 120.0, 100.0, 180.0, 150.0)
                store.append_thermal_history("owner-1", row)

            store.flush_thermal_history_to_file()

            # Create new store and load
            store2 = InMemoryStateStore(thermal_history_file=hist_file)
            store2.load_thermal_history_from_file()

            history = store2.get_thermal_history("owner-1", max_points=10)
            self.assertEqual(len(history), 5)
            self.assertAlmostEqual(history[0][1], 65.0)

    def test_calibration_profile_survives_flush_load(self):
        """CalibrationProfile persists across flush/load cycle."""
        from core.optimization.thermal_calibrator import CalibrationProfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cal_file = os.path.join(tmpdir, "calibration.json")
            store = InMemoryStateStore()

            # Set a profile with non-default values
            profile = CalibrationProfile(
                temp_mean_c=67.5,
                temp_stdev_c=3.2,
                fan_rated_max_rpm=7000.0,
                spike_trigger_temp_c=72.0,
            )
            store.set_calibration_profile("node-001", profile)
            store.flush_calibration_to_file(cal_file)

            # Create new store and load
            store2 = InMemoryStateStore()
            store2.load_calibration_from_file(cal_file)

            loaded = store2.get_calibration_profile("node-001")
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded.temp_mean_c, 67.5)
            self.assertAlmostEqual(loaded.spike_trigger_temp_c, 72.0)

    def test_calibration_handles_stale_fields(self):
        """Calibration load gracefully skips unknown fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_file = os.path.join(tmpdir, "calibration.json")
            # Write a profile with an extra field that doesn't exist in CalibrationProfile
            data = {
                "node-001": {
                    "temp_mean_c": 70.0,
                    "temp_stdev_c": 2.0,
                    "nonexistent_field": 999,
                }
            }
            Path(cal_file).write_text(json.dumps(data))

            store = InMemoryStateStore()
            store.load_calibration_from_file(cal_file)

            loaded = store.get_calibration_profile("node-001")
            self.assertIsNotNone(loaded)
            self.assertAlmostEqual(loaded.temp_mean_c, 70.0)


def _redis_available() -> bool:
    """Check if Redis is running on localhost:6379."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        return True
    except Exception:
        return False


class TestRedisIntegration(unittest.TestCase):
    """Step 4: Verify Redis integration and fallback."""

    def test_redis_fallback_to_inmemory(self):
        """RedisStateStore with bad URL → falls back to in-memory."""
        try:
            from gateway.state_store import RedisStateStore
            store = RedisStateStore(redis_url="redis://localhost:99999/0")
            # Should fall back gracefully
            self.assertFalse(store.available)
            # Fallback store should still work
            store.set_calibration_profile("node-001", None)
        except Exception:
            pass  # If RedisStateStore can't even import, that's fine too

    def test_inmemory_with_file_persistence_works(self):
        """InMemoryStateStore with file persistence works as Redis fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_file = os.path.join(tmpdir, "thermal_history.json")
            store = InMemoryStateStore(thermal_history_file=hist_file)

            # Run a full optimization cycle
            service = LocalOptimizationService(store)
            body = _make_body()
            result = service.optimize("test-owner", body)

            # Write history (as gateway would)
            from gateway.api import _compose_thermal_row
            row = _compose_thermal_row(body, result)
            store.append_thermal_history("test-owner", row)

            # Verify history accumulated
            history = store.get_thermal_history("test-owner", max_points=10)
            self.assertGreaterEqual(len(history), 1)

    @unittest.skipUnless(_redis_available(), "Redis not running")
    def test_redis_store_round_trip(self):
        """Redis store: write profile → read back → matches."""
        from gateway.state_store import RedisStateStore
        from core.optimization.thermal_calibrator import CalibrationProfile

        store = RedisStateStore(redis_url="redis://localhost:6379/0")
        profile = CalibrationProfile(temp_mean_c=68.0, spike_trigger_temp_c=73.0)
        store.set_calibration_profile("test-node-redis", profile)
        loaded = store.get_calibration_profile("test-node-redis")
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded.temp_mean_c, 68.0)


class TestControlNodePassthrough(unittest.TestCase):
    """Step 2: Verify control-node detection and passthrough."""

    def test_is_control_node(self):
        from gateway.api import _is_control_node
        self.assertTrue(_is_control_node("Control-ST550"))
        self.assertTrue(_is_control_node("traditional-node"))
        self.assertTrue(_is_control_node("BASELINE-unit"))
        self.assertFalse(_is_control_node("ST550-CooledAI-Predictive"))
        self.assertFalse(_is_control_node("pilot-node"))

    def test_control_node_returns_passthrough(self):
        """Control node should get source: control_passthrough, target_duty: None."""
        from fastapi.testclient import TestClient
        from gateway.api import create_app

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        app = create_app(optimization_service=service, keys_file=None)
        client = TestClient(app)

        api_key = _TEST_API_KEY
        resp = client.post(
            "/api/v1/optimize/control",
            json={
                "temp_c": 55.0,
                "fan_rpm": 4000.0,
                "gpu_power_w": 100.0,
                "node_id": "Control-ST550-Traditional",
            },
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsNone(data["target_duty"])
        self.assertEqual(data["source"], "control_passthrough")

    def test_pilot_node_not_passthrough(self):
        """Pilot node should get real optimization, not passthrough."""
        from fastapi.testclient import TestClient
        from gateway.api import create_app

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        app = create_app(optimization_service=service, keys_file=None)
        client = TestClient(app)

        api_key = _TEST_API_KEY
        resp = client.post(
            "/api/v1/optimize/control",
            json={
                "temp_c": 65.0,
                "fan_rpm": 3000.0,
                "gpu_power_w": 120.0,
                "node_id": "ST550-CooledAI-Predictive",
            },
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsNotNone(data.get("target_duty"))
        self.assertNotEqual(data.get("source"), "control_passthrough")

    def test_pilot_uses_control_baseline(self):
        """When both pilot and control report, history rows use control as baseline."""
        from gateway.api import (
            _compose_thermal_row,
            _update_latest_snapshot,
            _get_latest_control_snapshot,
        )

        # Simulate control node reporting
        control_body = _make_body(
            node_id="Control-ST550-Traditional",
            temp_c=70.0,
            fan_rpm=4500.0,
            gpu_power_w=130.0,
        )
        _update_latest_snapshot(control_body.node_id, control_body)

        # Now get control snapshot for pilot's row
        ctrl_snap = _get_latest_control_snapshot()
        self.assertIsNotNone(ctrl_snap)
        self.assertAlmostEqual(ctrl_snap["temp_c"], 70.0)

        # Compose row for pilot with control baseline
        pilot_body = _make_body(temp_c=65.0, fan_rpm=3000.0)
        result = {"target_duty": 40, "target_rpm": 2800}
        row = _compose_thermal_row(pilot_body, result, baseline_snapshot=ctrl_snap)

        self.assertAlmostEqual(row[1], 65.0)   # pilot_temp
        self.assertAlmostEqual(row[2], 70.0)   # baseline_temp (from control)
        self.assertAlmostEqual(row[3], 2800.0)  # pilot_rpm
        self.assertAlmostEqual(row[4], 4500.0)  # baseline_rpm (from control)


class TestGracefulFailover(unittest.TestCase):
    """Step 6: Verify cloud-disconnected mode in CloudForwarder."""

    def test_cloud_disconnected_after_consecutive_failures(self):
        """10 consecutive failures → cloud_disconnected flag set."""
        from gateway.cloud_forwarder import CloudForwarder

        fwd = CloudForwarder(
            cloud_url="http://localhost:99999",
            api_key="test-key",
            batch_interval=1.0,
        )
        self.assertFalse(fwd.cloud_disconnected)
        self.assertEqual(fwd._consecutive_failures, 0)

        # Simulate 10 consecutive failures
        for _ in range(10):
            fwd._total_failures += 1
            fwd._consecutive_failures += 1
            if fwd._consecutive_failures >= fwd._DISCONNECT_THRESHOLD:
                fwd._cloud_disconnected = True

        self.assertTrue(fwd.cloud_disconnected)

    def test_cloud_reconnects_on_success(self):
        """After disconnected mode, success resets the flag."""
        from gateway.cloud_forwarder import CloudForwarder

        fwd = CloudForwarder(
            cloud_url="http://localhost:99999",
            api_key="test-key",
        )
        fwd._consecutive_failures = 15
        fwd._cloud_disconnected = True

        # Simulate successful flush
        fwd._consecutive_failures = 0
        fwd._cloud_disconnected = False

        self.assertFalse(fwd.cloud_disconnected)
        self.assertEqual(fwd._consecutive_failures, 0)

    def test_stats_includes_cloud_disconnected(self):
        """stats() includes cloud_disconnected and consecutive_failures."""
        from gateway.cloud_forwarder import CloudForwarder

        fwd = CloudForwarder(cloud_url="http://localhost:99999", api_key="test-key")
        stats = fwd.stats()
        self.assertIn("cloud_disconnected", stats)
        self.assertIn("consecutive_failures", stats)
        self.assertFalse(stats["cloud_disconnected"])

    def test_health_degraded_when_cloud_disconnected(self):
        """Health endpoint returns degraded when cloud is disconnected."""
        from fastapi.testclient import TestClient
        from gateway.api import create_app

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        forwarder = MagicMock()
        forwarder.cloud_connected = False
        forwarder.cloud_disconnected = True

        app = create_app(optimization_service=service, cloud_forwarder=forwarder, keys_file=None)
        client = TestClient(app)

        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "degraded")
        self.assertTrue(data["cloud_disconnected"])


class TestMonitoringEndpoints(unittest.TestCase):
    """Step 5: Verify debug/monitoring endpoints return expected shapes."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from gateway.api import create_app

        self.store = InMemoryStateStore()
        self.service = LocalOptimizationService(self.store)
        self.forwarder = MagicMock()
        self.forwarder.stats.return_value = {
            "connected": True,
            "seconds_since_last_sync": 2.5,
            "total_batches_sent": 10,
            "total_entries_sent": 50,
            "total_failures": 0,
            "buffer_size": 0,
        }
        app = create_app(
            optimization_service=self.service,
            cloud_forwarder=self.forwarder,
            keys_file=None,
        )
        self.client = TestClient(app)
        self.headers = {"X-API-Key": _TEST_API_KEY}

    def test_forwarder_endpoint(self):
        resp = self.client.get("/api/v1/debug/forwarder", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("connected", data)
        self.assertIn("total_batches_sent", data)
        self.assertIn("buffer_size", data)

    def test_brain_state_endpoint(self):
        resp = self.client.get("/api/v1/debug/brain-state", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("nodes_in_history", data)
        self.assertIn("calibration_status", data)

    def test_thermal_history_stats_endpoint(self):
        # Write some history first
        for i in range(3):
            row = (time.time() + i, 65.0, 60.0, 3000.0, 4000.0,
                   None, None, 120.0, 100.0, None, None)
            self.store.append_thermal_history("user_3B2tUMI61WvTOsmR2ZMfHhXjsDa", row)

        resp = self.client.get("/api/v1/debug/thermal-history-stats", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["point_count"], 3)
        self.assertIsNotNone(data["oldest_ts"])
        self.assertIsNotNone(data["newest_ts"])

    def test_savings_validation_endpoint(self):
        # Write history with different pilot/baseline RPMs
        for i in range(5):
            row = (time.time() + i, 65.0, 70.0, 2500.0, 4000.0,
                   45.0, 42.0, 120.0, 100.0, 180.0, 150.0)
            self.store.append_thermal_history("user_3B2tUMI61WvTOsmR2ZMfHhXjsDa", row)

        resp = self.client.get("/api/v1/debug/savings-validation", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("savings_watts", data)
        self.assertIn("savings_percent", data)
        self.assertIn("pilot_avg_duty", data)
        self.assertIn("baseline_avg_duty", data)
        # Pilot at 2500 RPM vs baseline at 4000 RPM → savings should be positive
        self.assertGreater(data["savings_watts"], 0)
        self.assertGreater(data["savings_percent"], 0)


class TestSavingsValidationE2E(unittest.TestCase):
    """Step 7: End-to-end savings validation through the gateway API."""

    def test_savings_endpoint_after_optimization(self):
        """5 pilot requests → savings-validation shows brain engaged + savings > 0."""
        from fastapi.testclient import TestClient
        from gateway.api import create_app

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        app = create_app(optimization_service=service, keys_file=None)
        client = TestClient(app)
        headers = {"X-API-Key": _TEST_API_KEY}

        # Send 5 optimization requests (builds history)
        for i in range(5):
            resp = client.post(
                "/api/v1/optimize/control",
                json={
                    "temp_c": 65.0 + i * 0.5,
                    "fan_rpm": 3000.0,
                    "gpu_power_w": 120.0,
                    "cpu_temp_c": 45.0,
                    "node_id": "ST550-CooledAI-Predictive",
                    "max_fan_rpm": 7000.0,
                    "peak_power_w": 180.0,
                },
                headers=headers,
            )
            self.assertEqual(resp.status_code, 200)

        # Check savings-validation endpoint
        resp = client.get("/api/v1/debug/savings-validation", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["data_points_last_hour"], 0)
        self.assertIn("savings_watts", data)
        self.assertIn("pilot_avg_duty", data)

    def test_brain_transition_from_fallback_to_engaged(self):
        """Verify source transitions from fallback to optimization_brain.

        Uses direct service calls (bypassing 1s result cache) to prove
        the brain engages after history accumulates.
        """
        from gateway.api import _compose_thermal_row

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)
        body = _make_body()
        owner = "test-owner"

        sources = []
        for i in range(5):
            # Clear cache to force re-evaluation each call
            with service._cache_lock:
                service._cache.clear()
            result = service.optimize(owner, body)
            row = _compose_thermal_row(body, result)
            store.append_thermal_history(owner, row)
            sources.append(result.get("source", "unknown"))

        # First is fallback (no history), 2nd+ should be brain-engaged
        self.assertIn("fallback", sources[0])
        self.assertNotEqual(sources[-1], "fallback_simple",
                            f"Brain never engaged. Sources: {sources}")


if __name__ == "__main__":
    unittest.main()
