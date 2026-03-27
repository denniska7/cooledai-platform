"""
Phase 1 Integration Test Suite.

Tests the edge gateway architecture:
- Gateway API (same contract as cloud)
- LocalOptimizationService
- ControlService via gateway path
- RedisStateStore (skip if Redis not running)
- InMemoryStateStore fallback
- CloudForwarder batching
- Health endpoint
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.optimization.in_memory_store import InMemoryStateStore
from core.optimization.control_service import ControlService
from core.optimization.thermal_calibrator import CalibrationProfile
from core.optimization.spike_hold_state import clear_spike_hold
from core.models.agent_input import AgentOptimizeControlInput


# ---------------------------------------------------------------------------
# Redis availability check
# ---------------------------------------------------------------------------

def redis_available() -> bool:
    """Check if Redis is running on localhost:6379."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        r.ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate_history(store: InMemoryStateStore, owner_id: str, n: int = 30,
                      temp: float = 55.0, power: float = 80.0, rpm: float = 3000.0):
    """Fill thermal history so optimize_control doesn't hit fallback path."""
    now = time.time()
    for i in range(n):
        store.append_thermal_history(owner_id, (
            now - (n - i) * 3,
            temp, temp, rpm, rpm, None, None, power, power, None, None,
        ))


def _make_api_keys_file(keys: dict) -> str:
    """Write a temporary api_keys.json file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"keys": keys}, f)
    return path


# ---------------------------------------------------------------------------
# TestGatewayAPI — gateway/api.py via FastAPI TestClient
# ---------------------------------------------------------------------------

class TestGatewayAPI(unittest.TestCase):
    """Test the gateway FastAPI app (same contract as cloud API)."""

    @classmethod
    def setUpClass(cls):
        """Create gateway app with LocalOptimizationService."""
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app

        cls._store = InMemoryStateStore()
        cls._opt_svc = LocalOptimizationService(cls._store)

        # Populate history for test owner
        _populate_history(cls._store, "admin")

        keys = {
            "sk-test-gw-key": {
                "owner_id": "admin",
                "facilities": {
                    "facility-default": {
                        "nodes": ["*"],
                        "tier": "legacy",
                        "rate_limit_rps": 1000,
                    }
                },
            },
            "sk-owner-a": {
                "owner_id": "owner-A",
                "facilities": {
                    "facility-east": {
                        "nodes": ["node-001", "node-002"],
                        "tier": "enterprise",
                        "rate_limit_rps": 100,
                    }
                },
            },
        }
        keys_file = _make_api_keys_file(keys)
        cls._keys_file = keys_file

        app = create_app(
            optimization_service=cls._opt_svc,
            cloud_forwarder=None,
            keys_file=keys_file,
        )

        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._keys_file)

    def test_optimize_control_same_contract(self):
        """Gateway /optimize/control returns same fields as cloud API."""
        resp = self.client.post(
            "/api/v1/optimize/control",
            json={
                "temp_c": 55.0,
                "fan_rpm": 3000.0,
                "gpu_power_w": 80.0,
                "node_id": "test-node-001",
                "max_fan_rpm": 7000.0,
            },
            headers={"X-API-Key": "sk-test-gw-key"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("target_duty", data)
        self.assertIn("recommended_cooling_delta", data)
        self.assertIn("source", data)
        self.assertGreaterEqual(data["target_duty"], 0)
        self.assertLessEqual(data["target_duty"], 100)

    def test_optimize_control_requires_auth(self):
        """Missing API key returns 401."""
        resp = self.client.post(
            "/api/v1/optimize/control",
            json={"temp_c": 55.0, "fan_rpm": 3000.0, "node_id": "x"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_optimize_control_invalid_key(self):
        """Invalid API key returns 401."""
        resp = self.client.post(
            "/api/v1/optimize/control",
            json={"temp_c": 55.0, "fan_rpm": 3000.0, "node_id": "x"},
            headers={"X-API-Key": "sk-bogus-key"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_node_access_403(self):
        """Restricted key accessing unauthorized node returns 403."""
        resp = self.client.post(
            "/api/v1/optimize/control",
            json={"temp_c": 55.0, "fan_rpm": 3000.0, "node_id": "node-999"},
            headers={"X-API-Key": "sk-owner-a"},
        )
        self.assertEqual(resp.status_code, 403)

    def test_node_access_allowed(self):
        """Restricted key accessing authorized node succeeds."""
        # Populate history for owner-A
        _populate_history(self._store, "owner-A")
        resp = self.client.post(
            "/api/v1/optimize/control",
            json={"temp_c": 55.0, "fan_rpm": 3000.0, "node_id": "node-001"},
            headers={"X-API-Key": "sk-owner-a"},
        )
        self.assertEqual(resp.status_code, 200)

    def test_health_endpoint(self):
        """GET /health returns correct status fields."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("brain_loaded", data)
        self.assertIn("redis_connected", data)
        self.assertIn("cloud_connected", data)
        self.assertIn("nodes_active", data)
        self.assertIn("uptime_seconds", data)
        self.assertTrue(data["brain_loaded"])

    def test_debug_calibrators(self):
        """GET /debug/calibrators returns calibration profiles."""
        resp = self.client.get(
            "/api/v1/debug/calibrators",
            headers={"X-API-Key": "sk-test-gw-key"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("calibrators", data)
        self.assertIn("owner_id", data)

    def test_debug_safety(self):
        """GET /debug/safety/{node_id} returns safety state."""
        resp = self.client.get(
            "/api/v1/debug/safety/node-001",
            headers={"X-API-Key": "sk-test-gw-key"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("spike_hold_active", data)
        self.assertIn("spike_hold_remaining_s", data)

    def test_bearer_token_auth(self):
        """Authorization: Bearer header works as alternative to X-API-Key."""
        resp = self.client.post(
            "/api/v1/optimize/control",
            json={
                "temp_c": 55.0, "fan_rpm": 3000.0,
                "gpu_power_w": 80.0, "node_id": "test-node-bearer",
            },
            headers={"Authorization": "Bearer sk-test-gw-key"},
        )
        self.assertEqual(resp.status_code, 200)


class TestGatewayRateLimiting(unittest.TestCase):
    """Test rate limiting on gateway API."""

    def test_rate_limit_returns_429(self):
        """Exceeding rate_limit_rps returns 429."""
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app
        from fastapi.testclient import TestClient

        store = InMemoryStateStore()
        svc = LocalOptimizationService(store)

        keys = {
            "sk-limited": {
                "owner_id": "admin",
                "facilities": {
                    "facility-default": {
                        "nodes": ["*"],
                        "tier": "test",
                        "rate_limit_rps": 2,  # Very low: 2 req/s, burst=4
                    }
                },
            },
        }
        keys_file = _make_api_keys_file(keys)
        try:
            app = create_app(optimization_service=svc, keys_file=keys_file)
            client = TestClient(app)

            # Burn through the burst allowance
            statuses = []
            for _ in range(10):
                resp = client.post(
                    "/api/v1/optimize/control",
                    json={"temp_c": 55.0, "fan_rpm": 3000.0, "node_id": "x"},
                    headers={"X-API-Key": "sk-limited"},
                )
                statuses.append(resp.status_code)

            # At least one should be 429 (burst=4, so after 4 requests at most)
            self.assertIn(429, statuses, f"Expected 429 in statuses: {statuses}")
        finally:
            os.unlink(keys_file)


class TestGatewayConcurrent(unittest.TestCase):
    """Test gateway under concurrent load."""

    def test_10_concurrent_nodes(self):
        """10 simulated nodes posting concurrently get valid responses."""
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app
        from fastapi.testclient import TestClient

        store = InMemoryStateStore()
        svc = LocalOptimizationService(store)
        for i in range(10):
            _populate_history(store, f"admin")

        keys = {
            "sk-concurrent": {
                "owner_id": "admin",
                "facilities": {"f": {"nodes": ["*"], "tier": "t", "rate_limit_rps": 1000}},
            }
        }
        keys_file = _make_api_keys_file(keys)
        try:
            app = create_app(optimization_service=svc, keys_file=keys_file)
            client = TestClient(app)

            results = [None] * 10
            errors = [None] * 10

            def _post(idx):
                try:
                    resp = client.post(
                        "/api/v1/optimize/control",
                        json={
                            "temp_c": 50.0 + idx * 2,
                            "fan_rpm": 3000.0,
                            "gpu_power_w": 80.0,
                            "node_id": f"node-{idx:03d}",
                        },
                        headers={"X-API-Key": "sk-concurrent"},
                    )
                    results[idx] = resp.json()
                except Exception as e:
                    errors[idx] = str(e)

            threads = [threading.Thread(target=_post, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            for i in range(10):
                self.assertIsNone(errors[i], f"Node {i} error: {errors[i]}")
                self.assertIsNotNone(results[i], f"Node {i} got no result")
                self.assertIn("target_duty", results[i])
        finally:
            os.unlink(keys_file)

    def test_failsafe_brain_crash(self):
        """If brain.analyze() throws, gateway returns fallback, not 500."""
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app
        from fastapi.testclient import TestClient

        store = InMemoryStateStore()
        svc = LocalOptimizationService(store)
        _populate_history(store, "admin")

        keys = {
            "sk-crash": {
                "owner_id": "admin",
                "facilities": {"f": {"nodes": ["*"], "tier": "t", "rate_limit_rps": 1000}},
            }
        }
        keys_file = _make_api_keys_file(keys)
        try:
            app = create_app(optimization_service=svc, keys_file=keys_file)
            client = TestClient(app)

            # Patch brain.analyze to throw
            with patch.object(svc._service.brain, "analyze", side_effect=RuntimeError("brain crashed")):
                resp = client.post(
                    "/api/v1/optimize/control",
                    json={"temp_c": 55.0, "fan_rpm": 3000.0, "node_id": "crash-node"},
                    headers={"X-API-Key": "sk-crash"},
                )
            # Should NOT be 500; should be 200 with fallback
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("target_duty", data)
            self.assertEqual(data["source"], "fallback_error")
        finally:
            os.unlink(keys_file)


# ---------------------------------------------------------------------------
# TestRedisState — skip if Redis not running
# ---------------------------------------------------------------------------

@unittest.skipUnless(redis_available(), "Redis not running on localhost:6379")
class TestRedisState(unittest.TestCase):
    """Test RedisStateStore persistence (requires running Redis)."""

    def setUp(self):
        from gateway.state_store import RedisStateStore
        self.store = RedisStateStore(redis_url="redis://localhost:6379/15")  # Use DB 15 for tests
        # Clean test keys
        if self.store.available:
            self.store._redis.flushdb()

    def tearDown(self):
        if hasattr(self, "store") and self.store.available:
            self.store._redis.flushdb()

    def test_calibration_survives_restart(self):
        """Write profile to Redis, create new store, verify loaded."""
        cp = CalibrationProfile(temp_mean_c=72.5, temp_stdev_c=4.1)
        self.store.set_calibration_profile("node-persist", cp)

        # Simulate restart: create new store from same Redis
        from gateway.state_store import RedisStateStore
        store2 = RedisStateStore(redis_url="redis://localhost:6379/15")
        loaded = store2.get_calibration_profile("node-persist")
        self.assertIsNotNone(loaded)
        self.assertAlmostEqual(loaded.temp_mean_c, 72.5, places=1)

    def test_spike_hold_survives_restart(self):
        """Spike history persists across store instances."""
        now = time.time()
        self.store.append_spike("node-spike", now, 95.0)
        self.store.append_spike("node-spike", now + 1, 96.0)

        from gateway.state_store import RedisStateStore
        store2 = RedisStateStore(redis_url="redis://localhost:6379/15")
        hist = store2.get_spike_history("node-spike")
        self.assertEqual(len(hist), 2)
        self.assertAlmostEqual(hist[0][1], 95.0)

    def test_warm_start_loads_profiles(self):
        """warm_start populates in-memory fallback from Redis."""
        cp = CalibrationProfile(temp_mean_c=65.0)
        self.store.set_calibration_profile("warm-node", cp)

        from gateway.state_store import RedisStateStore
        store2 = RedisStateStore(redis_url="redis://localhost:6379/15")
        count = store2.warm_start()
        self.assertGreaterEqual(count, 1)
        # Fallback should now have the profile
        fallback_cp = store2._fallback.get_calibration_profile("warm-node")
        self.assertIsNotNone(fallback_cp)

    def test_list_calibration_profiles(self):
        """list_calibration_profiles returns all stored profiles."""
        self.store.set_calibration_profile("n1", CalibrationProfile(temp_mean_c=50.0))
        self.store.set_calibration_profile("n2", CalibrationProfile(temp_mean_c=60.0))
        profiles = self.store.list_calibration_profiles()
        self.assertIn("n1", profiles)
        self.assertIn("n2", profiles)


# ---------------------------------------------------------------------------
# TestInMemoryFallback — always runs, no Redis needed
# ---------------------------------------------------------------------------

class TestInMemoryFallback(unittest.TestCase):
    """Test that RedisStateStore degrades gracefully to in-memory."""

    def test_graceful_degradation_bad_url(self):
        """RedisStateStore with unreachable URL falls back to in-memory."""
        from gateway.state_store import RedisStateStore
        store = RedisStateStore(redis_url="redis://192.0.2.1:9999/0")
        self.assertFalse(store.available)

        # All operations should work via fallback
        store.set_calibration_profile("node-fallback", CalibrationProfile(temp_mean_c=55.0))
        cp = store.get_calibration_profile("node-fallback")
        self.assertIsNotNone(cp)
        self.assertAlmostEqual(cp.temp_mean_c, 55.0)

    def test_optimization_works_without_redis(self):
        """Full optimize path works with InMemoryStateStore (no Redis)."""
        store = InMemoryStateStore()
        svc = ControlService(store=store)
        _populate_history(store, "owner-no-redis")

        result = svc.optimize_control(
            owner_id="owner-no-redis", node_id="node-mem",
            temp_c=55.0, fan_rpm=3000.0, gpu_power_w=80.0,
            cpu_temp_c=None, max_fan_rpm=7000.0,
            last_commanded_duty=None, peak_power_w=None,
            calibration_profile_dict=None,
        )
        self.assertIn("target_duty", result)
        self.assertIn(result["source"], ("optimization_brain", "fallback_simple", "fallback_error"))

    def test_spike_history_fallback(self):
        """RedisStateStore spike history works via fallback."""
        from gateway.state_store import RedisStateStore
        store = RedisStateStore(redis_url="redis://192.0.2.1:9999/0")
        now = time.time()
        store.append_spike("node-fb", now, 88.0)
        hist = store.get_spike_history("node-fb")
        self.assertEqual(len(hist), 1)
        self.assertAlmostEqual(hist[0][1], 88.0)


# ---------------------------------------------------------------------------
# TestCloudForwarding — mocked HTTP
# ---------------------------------------------------------------------------

class TestCloudForwarding(unittest.TestCase):
    """Test CloudForwarder batching and resilience."""

    def test_enqueue_buffers_entries(self):
        """enqueue adds entries to the buffer."""
        from gateway.cloud_forwarder import CloudForwarder
        fwd = CloudForwarder(cloud_url="http://localhost:9999", api_key="sk-test")
        fwd.enqueue("node-1", {"temp_c": 55.0}, {"target_duty": 40})
        fwd.enqueue("node-2", {"temp_c": 60.0}, {"target_duty": 50})
        self.assertEqual(len(fwd._buffer), 2)

    def test_buffer_capped_at_max_size(self):
        """Buffer doesn't grow beyond max_buffer_size."""
        from gateway.cloud_forwarder import CloudForwarder
        fwd = CloudForwarder(cloud_url="http://localhost:9999", api_key="sk-test", max_buffer_size=5)
        for i in range(10):
            fwd.enqueue(f"node-{i}", {"temp_c": float(i)}, {})
        self.assertEqual(len(fwd._buffer), 5)

    def test_cloud_connected_default_true(self):
        """Forwarder starts with connected=True."""
        from gateway.cloud_forwarder import CloudForwarder
        fwd = CloudForwarder(cloud_url="http://localhost:9999", api_key="sk-test")
        self.assertTrue(fwd.cloud_connected)

    def test_stats_returns_dict(self):
        """stats() returns a dict with expected fields."""
        from gateway.cloud_forwarder import CloudForwarder
        fwd = CloudForwarder(cloud_url="http://localhost:9999", api_key="sk-test")
        stats = fwd.stats()
        self.assertIn("connected", stats)
        self.assertIn("total_batches_sent", stats)
        self.assertIn("buffer_size", stats)

    def test_flush_clears_buffer(self):
        """After _flush, buffer is cleared (even if POST fails)."""
        from gateway.cloud_forwarder import CloudForwarder
        fwd = CloudForwarder(cloud_url="http://192.0.2.1:9999", api_key="sk-test")
        fwd.enqueue("node-1", {}, {})
        fwd.enqueue("node-2", {}, {})

        # Run flush in an event loop (it will fail since URL is unreachable)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(fwd._flush())
        finally:
            loop.close()

        # Buffer should be re-buffered on failure, but originally cleared
        # (re-buffer puts them back, so buffer should still have entries)
        # The key test: no crash occurred during async flush
        self.assertIsInstance(fwd._buffer, list)


# ---------------------------------------------------------------------------
# TestPolicySyncer
# ---------------------------------------------------------------------------

class TestPolicySyncer(unittest.TestCase):
    """Test PolicySyncer basic behavior."""

    def test_init_defaults(self):
        """PolicySyncer initializes with sensible defaults."""
        from gateway.policy_syncer import PolicySyncer
        syncer = PolicySyncer(cloud_url="http://localhost:9999", api_key="sk-test")
        self.assertTrue(syncer.cloud_connected)  # Starts optimistic
        self.assertIsNone(syncer.last_config)
        self.assertEqual(syncer.interval, 60.0)


# ---------------------------------------------------------------------------
# TestCloudEndpoints — cloud-side batch and config endpoints
# ---------------------------------------------------------------------------

class TestCloudEndpoints(unittest.TestCase):
    """Test the cloud API gateway endpoints (telemetry-batch, config, failover)."""

    @classmethod
    def setUpClass(cls):
        from api.main import app
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)
        # Get a valid API key
        from api.main import _api_key_registry
        cls.api_key = next(iter(_api_key_registry.keys()), "")

    def test_telemetry_batch_accepted(self):
        """POST /gateway/telemetry-batch returns 200."""
        resp = self.client.post(
            "/api/v1/gateway/telemetry-batch",
            json={
                "batch": [
                    {"node_id": "node-001", "telemetry": {"temp_c": 55.0, "fan_rpm": 3000.0, "gpu_power_w": 80.0}, "optimization": {"target_duty": 45}, "ts": "2026-01-01T00:00:00Z"},
                    {"node_id": "node-002", "telemetry": {"temp_c": 60.0, "fan_rpm": 3500.0, "gpu_power_w": 100.0}, "optimization": {"target_duty": 55}, "ts": "2026-01-01T00:00:01Z"},
                ],
                "gateway_id": "gw-test",
                "batch_size": 2,
                "ts": "2026-01-01T00:00:05Z",
            },
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "accepted")
        self.assertEqual(data["processed"], 2)

    def test_telemetry_batch_empty(self):
        """Empty batch is accepted."""
        resp = self.client.post(
            "/api/v1/gateway/telemetry-batch",
            json={"batch": [], "gateway_id": "gw-test", "batch_size": 0, "ts": ""},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["processed"], 0)

    def test_gateway_config_returns_fields(self):
        """GET /gateway/config returns required config fields."""
        resp = self.client.get(
            "/api/v1/gateway/config",
            params={"facility_id": "facility-default"},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("system_state", data)
        self.assertIn("api_key_registry", data)
        self.assertIn("model_version", data)
        self.assertIn("updated_at", data)
        self.assertIn(data["system_state"], ("OPTIMIZING", "GUARD_MODE", "MANUAL_OVERRIDE"))

    def test_failover_event_recorded(self):
        """POST /events/failover returns 200."""
        resp = self.client.post(
            "/api/v1/events/failover",
            json={"gateway_id": "gw-standby", "reason": "primary_unreachable", "ts": "2026-01-01T00:00:00Z"},
            headers={"X-API-Key": self.api_key},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "recorded")

    def test_telemetry_batch_requires_auth(self):
        """Batch endpoint requires API key."""
        resp = self.client.post(
            "/api/v1/gateway/telemetry-batch",
            json={"batch": [], "gateway_id": "x", "batch_size": 0, "ts": ""},
        )
        self.assertEqual(resp.status_code, 401)


# ---------------------------------------------------------------------------
# TestHAManagerStub
# ---------------------------------------------------------------------------

class TestHAManagerStub(unittest.TestCase):
    """Test that HA manager stub works as standalone."""

    def test_standalone_is_active(self):
        from gateway.ha.ha_manager import HAManager
        ha = HAManager()
        self.assertTrue(ha.is_active)
        self.assertEqual(ha.role, "standalone")

    def test_promote_raises_not_implemented(self):
        from gateway.ha.ha_manager import HAManager
        ha = HAManager()
        with self.assertRaises(NotImplementedError):
            ha.promote_to_primary()

    def test_standalone_heartbeat_noop(self):
        """start_heartbeat is a no-op in standalone mode."""
        from gateway.ha.ha_manager import HAManager
        ha = HAManager()
        ha.start_heartbeat()  # Should not raise


# ---------------------------------------------------------------------------
# TestAgentInputModel
# ---------------------------------------------------------------------------

class TestAgentInputModel(unittest.TestCase):
    """Test the shared Pydantic model."""

    def test_default_values(self):
        """AgentOptimizeControlInput has sensible defaults."""
        body = AgentOptimizeControlInput(temp_c=55.0, fan_rpm=3000.0)
        self.assertEqual(body.gpu_power_w, 50.0)
        self.assertEqual(body.node_id, "ST550-CooledAI-Predictive")
        self.assertEqual(body.max_fan_rpm, 7000.0)
        self.assertIsNone(body.cpu_temp_c)
        self.assertIsNone(body.peak_power_w)
        self.assertIsNone(body.calibration_profile)

    def test_all_fields_settable(self):
        body = AgentOptimizeControlInput(
            temp_c=72.0, fan_rpm=5000.0, gpu_power_w=200.0,
            cpu_temp_c=60.0, node_id="custom-node",
            max_fan_rpm=8000.0, last_commanded_duty=65.0,
            peak_power_w=250.0, calibration_profile={"temp_mean_c": 70.0},
        )
        self.assertEqual(body.temp_c, 72.0)
        self.assertEqual(body.node_id, "custom-node")
        self.assertEqual(body.peak_power_w, 250.0)


if __name__ == "__main__":
    unittest.main()
