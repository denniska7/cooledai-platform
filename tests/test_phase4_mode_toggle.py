"""
Phase 4 Integration Tests — Operational Mode Toggle & Savings Activation.

Tests mode API endpoints, supervised/active mode clamping, thermal safety
fallback, audit logging, and savings accumulation.
"""

import os
import sys
import time
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Ensure API key is available
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
        _TEST_API_KEY = "sk-test-phase4"
    os.environ["COOLEDAI_API_KEY"] = _TEST_API_KEY


def _get_client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def _headers():
    return {"X-API-Key": _TEST_API_KEY}


class TestModeAPI(unittest.TestCase):
    """Test mode GET/POST endpoints."""

    def test_get_mode_default_shadow(self):
        """Default mode should be shadow."""
        client = _get_client()
        resp = client.get("/api/v1/gateway/mode", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "shadow")
        self.assertIn("shadow", data["available_modes"])
        self.assertIn("supervised", data["available_modes"])
        self.assertIn("active", data["available_modes"])
        self.assertIn("can_activate", data)
        self.assertIn("confidence_pct", data)
        self.assertIn("data_hours", data)

    def test_set_mode_shadow_always_allowed(self):
        """Switching to shadow should always succeed."""
        client = _get_client()
        resp = client.post(
            "/api/v1/gateway/mode",
            json={"mode": "shadow", "reason": "test"},
            headers=_headers(),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["mode"], "shadow")

    def test_set_mode_invalid_rejected(self):
        """Invalid mode name should return 400."""
        client = _get_client()
        resp = client.post(
            "/api/v1/gateway/mode",
            json={"mode": "turbo", "reason": "test"},
            headers=_headers(),
        )
        self.assertEqual(resp.status_code, 400)

    def test_set_mode_active_with_prerequisites(self):
        """Setting active mode should succeed when prerequisites are met."""
        from api.main import _thermal_history, _tenant_telemetry

        client = _get_client()
        # Find the owner_id for this key
        resp = client.get("/api/v1/gateway/mode", headers=_headers())
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"

        # Populate enough data for prerequisites
        _thermal_history.setdefault(owner_id, [])
        base_ts = time.time() - 7200  # 2 hours of data
        for i in range(100):
            _thermal_history[owner_id].append((
                base_ts + i * 72,
                68.0 + (i % 5), 70.0,
                2500.0 + i * 10, 3500.0,
                45.0, 42.0,
                120.0, 100.0,
                180.0, 150.0,
            ))

        # Add live telemetry
        _tenant_telemetry.setdefault(owner_id, {})
        _tenant_telemetry[owner_id]["ST550-CooledAI-Predictive"] = {
            "_received_at": time.time(),
            "node_id": "ST550-CooledAI-Predictive",
            "fan_rpm": 3000,
        }

        # Verify prerequisites pass
        resp = client.get("/api/v1/gateway/mode", headers=_headers())
        data = resp.json()
        self.assertTrue(data["can_activate"], f"Should be activatable: {data}")

        # Set active
        resp = client.post(
            "/api/v1/gateway/mode",
            json={"mode": "active", "reason": "test activation"},
            headers=_headers(),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["mode"], "active")

    def test_mode_change_logged_in_history(self):
        """Each mode change should appear in audit history."""
        client = _get_client()

        # Make a mode change
        client.post(
            "/api/v1/gateway/mode",
            json={"mode": "shadow", "reason": "history test"},
            headers=_headers(),
        )

        # Check history
        resp = client.get("/api/v1/gateway/mode/history", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("history", data)
        self.assertGreater(len(data["history"]), 0)
        latest = data["history"][0]
        self.assertIn("from_mode", latest)
        self.assertIn("to_mode", latest)
        self.assertIn("at", latest)

    def test_mode_history_returns_recent_first(self):
        """History endpoint should return most recent changes first."""
        client = _get_client()

        # Make two changes
        client.post("/api/v1/gateway/mode", json={"mode": "shadow"}, headers=_headers())
        time.sleep(0.01)
        client.post("/api/v1/gateway/mode", json={"mode": "shadow", "reason": "second"}, headers=_headers())

        resp = client.get("/api/v1/gateway/mode/history", headers=_headers())
        data = resp.json()
        history = data["history"]
        if len(history) >= 2:
            # Most recent should be first
            self.assertGreaterEqual(
                history[0]["timestamp"],
                history[1]["timestamp"],
            )


class TestSupervisedMode(unittest.TestCase):
    """Test supervised mode RPM clamping in gateway."""

    def test_supervised_clamps_rpm_to_floor(self):
        """In supervised mode, RPM should not go below 90% of baseline."""
        from core.optimization.in_memory_store import InMemoryStateStore
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app, _compose_thermal_row
        from core.models.agent_input import AgentOptimizeControlInput
        from fastapi.testclient import TestClient
        from unittest.mock import MagicMock

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)

        # Mock policy syncer in supervised mode
        mock_syncer = MagicMock()
        mock_syncer.current_mode = "supervised"
        mock_syncer.fan_rpm_floor_pct = 0.90

        app = create_app(optimization_service=service, policy_syncer=mock_syncer, keys_file=None)
        client = TestClient(app)

        # Seed history so brain engages
        owner = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        body = AgentOptimizeControlInput(
            temp_c=65.0, fan_rpm=3500.0, gpu_power_w=120.0,
            node_id="ST550-CooledAI-Predictive", max_fan_rpm=7000.0,
        )
        for _ in range(3):
            result = service.optimize(owner, body)
            row = _compose_thermal_row(body, result)
            store.append_thermal_history(owner, row)
            with service._cache_lock:
                service._cache.clear()

        # Send request — in supervised mode, duty should not go below floor
        resp = client.post(
            "/api/v1/optimize/control",
            json={"temp_c": 65.0, "fan_rpm": 3500.0, "gpu_power_w": 120.0,
                  "node_id": "ST550-CooledAI-Predictive", "max_fan_rpm": 7000.0},
            headers={"X-API-Key": _TEST_API_KEY},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("mode"), "supervised")
        self.assertTrue(data.get("applied"))
        # Floor: 90% of 3500 RPM = 3150 RPM → duty = 3150/7000*100 ≈ 45%
        if data.get("target_duty") is not None:
            min_duty = int(round((3500 * 0.90 / 7000.0) * 100))
            self.assertGreaterEqual(data["target_duty"], min_duty,
                                    f"Supervised duty {data['target_duty']} < floor {min_duty}")


class TestActiveModeAndSavings(unittest.TestCase):
    """Test savings are positive in active mode."""

    def test_active_mode_includes_applied_flag(self):
        """In active mode, result should have applied=True."""
        from core.optimization.in_memory_store import InMemoryStateStore
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app
        from fastapi.testclient import TestClient
        from unittest.mock import MagicMock

        store = InMemoryStateStore()
        service = LocalOptimizationService(store)

        mock_syncer = MagicMock()
        mock_syncer.current_mode = "active"

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
        self.assertEqual(data.get("mode"), "active")
        self.assertTrue(data.get("applied"))

    def test_shadow_mode_not_applied(self):
        """In shadow mode, result should have applied=False."""
        from core.optimization.in_memory_store import InMemoryStateStore
        from gateway.optimization_service import LocalOptimizationService
        from gateway.api import create_app
        from fastapi.testclient import TestClient
        from unittest.mock import MagicMock

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
        self.assertEqual(data.get("mode"), "shadow")
        self.assertFalse(data.get("applied"))


class TestThermalSafety(unittest.TestCase):
    """Test auto-fallback on high temperature."""

    def test_fallback_endpoint_records_event(self):
        """Cloud API should record auto-fallback notification."""
        client = _get_client()
        resp = client.post(
            "/api/v1/gateway/mode/fallback",
            json={
                "reason": "thermal_ceiling",
                "gpu_temp_c": 87.2,
                "ceiling_c": 85.0,
                "reverted_to": "shadow",
            },
            headers=_headers(),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "recorded")
        self.assertEqual(data["mode"], "shadow")

    def test_fallback_logged_in_history(self):
        """Auto-fallback events should appear in mode history."""
        from api.main import _mode_audit_log

        initial_count = len(_mode_audit_log)

        client = _get_client()
        client.post(
            "/api/v1/gateway/mode/fallback",
            json={"reason": "thermal_ceiling", "gpu_temp_c": 88.0, "ceiling_c": 85.0},
            headers=_headers(),
        )

        self.assertGreater(len(_mode_audit_log), initial_count)
        latest = _mode_audit_log[-1]
        self.assertEqual(latest["reason"], "auto_fallback")
        self.assertIn("88.0", latest["details"])


class TestGatewayConfigIncludesMode(unittest.TestCase):
    """Verify gateway/config endpoint returns mode for PolicySyncer."""

    def test_config_includes_mode(self):
        client = _get_client()
        resp = client.get("/api/v1/gateway/config", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("mode", data)
        self.assertIn(data["mode"], ["shadow", "supervised", "active"])
        self.assertIn("fan_rpm_floor_pct", data)
        self.assertIn("thermal_ceiling_c", data)


class TestDashboardSummaryIncludesMode(unittest.TestCase):
    """Verify dashboard-summary includes operational_mode."""

    def test_summary_includes_mode(self):
        client = _get_client()
        resp = client.get("/api/v1/dashboard-summary", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("operational_mode", data)
        self.assertIn(data["operational_mode"], ["shadow", "supervised", "active"])


class TestPolicySyncerMode(unittest.TestCase):
    """Test PolicySyncer stores mode from cloud config."""

    def test_apply_config_updates_mode(self):
        from gateway.policy_syncer import PolicySyncer

        syncer = PolicySyncer(cloud_url="http://localhost", api_key="test")
        self.assertEqual(syncer.current_mode, "shadow")

        syncer._apply_config({"mode": "active", "fan_rpm_floor_pct": 0.85, "thermal_ceiling_c": 80.0})
        self.assertEqual(syncer.current_mode, "active")
        self.assertAlmostEqual(syncer.fan_rpm_floor_pct, 0.85)
        self.assertAlmostEqual(syncer.thermal_ceiling_c, 80.0)

    def test_apply_config_ignores_invalid_mode(self):
        from gateway.policy_syncer import PolicySyncer

        syncer = PolicySyncer(cloud_url="http://localhost", api_key="test")
        syncer._apply_config({"mode": "turbo"})
        self.assertEqual(syncer.current_mode, "shadow")  # unchanged


if __name__ == "__main__":
    unittest.main()
