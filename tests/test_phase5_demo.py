"""
Phase 5 Integration Tests — Demo-Ready Portal & Savings Validation.

Tests new endpoints (activity-feed, savings-projection) and projected savings
calculations in the stats endpoint.
"""

import os
import sys
import time
import unittest
from pathlib import Path

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
        _TEST_API_KEY = "sk-test-phase5"
    os.environ["COOLEDAI_API_KEY"] = _TEST_API_KEY


def _get_client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def _headers():
    return {"X-API-Key": _TEST_API_KEY}


def _seed_history(owner_id: str, count: int = 10):
    """Seed thermal history with rows where pilot_rpm < baseline_rpm."""
    from api.main import _thermal_history
    _thermal_history.setdefault(owner_id, [])
    for i in range(count):
        _thermal_history[owner_id].append((
            time.time() - i * 5,
            68.0 + (i % 3), 70.0,  # pilot_temp, baseline_temp
            2500.0, 3500.0,  # pilot_rpm < baseline_rpm → savings
            45.0, 42.0,
            120.0, 100.0,
            180.0, 150.0,
        ))


class TestActivityFeed(unittest.TestCase):
    """Verify /api/v1/activity-feed endpoint."""

    def test_activity_feed_returns_entries(self):
        """Activity feed should return entries list."""
        client = _get_client()
        resp = client.get("/api/v1/activity-feed?limit=10", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("entries", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["entries"], list)

    def test_activity_feed_limit_capped(self):
        """Activity feed should not return more than limit."""
        client = _get_client()
        resp = client.get("/api/v1/activity-feed?limit=5", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertLessEqual(len(data["entries"]), 5)


class TestSavingsProjection(unittest.TestCase):
    """Verify /api/v1/savings-projection endpoint."""

    def test_projection_returns_fields(self):
        """Savings projection should return all required fields."""
        from api.main import _thermal_history
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        _seed_history(owner_id)

        client = _get_client()
        resp = client.get("/api/v1/savings-projection", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for field in [
            "mode", "avg_power_reduction_watts", "recommendations_with_savings",
            "projected_annual_usd_saved", "projected_monthly_usd_saved",
            "projected_co2_avoided_kg", "data_hours",
        ]:
            self.assertIn(field, data, f"Missing field: {field}")

    def test_projection_positive_when_pilot_lower(self):
        """When pilot RPM < baseline RPM, projection should show savings."""
        from api.main import _thermal_history
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        _seed_history(owner_id, 20)

        client = _get_client()
        resp = client.get("/api/v1/savings-projection", headers=_headers())
        data = resp.json()
        self.assertGreater(data["avg_power_reduction_watts"], 0)
        self.assertGreater(data["projected_annual_usd_saved"], 0)
        self.assertGreater(data["projected_co2_avoided_kg"], 0)


class TestStatsProjectedSavings(unittest.TestCase):
    """Verify /api/v1/stats includes projected savings fields."""

    def test_stats_has_projected_fields(self):
        """Stats response should include projected savings."""
        from api.main import _thermal_history
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        _seed_history(owner_id)

        client = _get_client()
        resp = client.get("/api/v1/stats", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("projected_savings_usd", data)
        self.assertIn("projected_monthly_usd", data)
        self.assertIn("projected_power_reduction_watts", data)
        self.assertIn("brain_recommendations_count", data)

    def test_stats_projected_positive_with_savings_history(self):
        """Projected savings > 0 when history has pilot_rpm < baseline_rpm."""
        from api.main import _thermal_history
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        _seed_history(owner_id, 20)

        client = _get_client()
        resp = client.get("/api/v1/stats", headers=_headers())
        data = resp.json()
        self.assertGreater(data["projected_savings_usd"], 0)
        self.assertGreater(data["brain_recommendations_count"], 0)


class TestDashboardSummaryProjected(unittest.TestCase):
    """Verify dashboard-summary includes projected values."""

    def test_summary_has_projected_fields(self):
        from api.main import _thermal_history
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"
        _seed_history(owner_id)

        client = _get_client()
        resp = client.get("/api/v1/dashboard-summary", headers=_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("projected_monthly_usd", data)
        self.assertIn("projected_annual_usd", data)
        self.assertIn("operational_mode", data)


class TestFanAffinityLawMath(unittest.TestCase):
    """Verify fan affinity law calculations are correct."""

    def test_fan_power_watts(self):
        """P = (RPM / 7000)^3 * 12W * 6 fans."""
        from api.main import _fan_power_watts
        # At 3500 RPM: (3500/7000)^3 * 12 * 6 = 0.125 * 72 = 9.0W
        self.assertAlmostEqual(_fan_power_watts(3500), 9.0, places=1)
        # At 7000 RPM: (7000/7000)^3 * 12 * 6 = 72.0W
        self.assertAlmostEqual(_fan_power_watts(7000), 72.0, places=1)
        # At 0 RPM: 0W
        self.assertAlmostEqual(_fan_power_watts(0), 0.0)

    def test_savings_math(self):
        """Savings = baseline_power - pilot_power."""
        from api.main import _fan_power_watts
        pilot_w = _fan_power_watts(2500)   # ~1.83W
        baseline_w = _fan_power_watts(3500)  # ~9.0W
        savings = baseline_w - pilot_w
        self.assertGreater(savings, 5.0)  # Should save ~7.2W


if __name__ == "__main__":
    unittest.main()
