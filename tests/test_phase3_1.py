"""
Phase 3.1 Integration Tests — Multi-Metric Chart Fix & Optimizer RPM Correction.

Tests bucket aggregation includes all metrics, savings uses brain target_rpm,
and SSL context is present in CloudForwarder.
"""

import os
import sys
import time
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Ensure API key is available for test auth
_TEST_API_KEY = os.environ.get("COOLEDAI_API_KEY", "")
if not _TEST_API_KEY:
    # Load from data/api_keys.json if available
    _keys_file = project_root / "data" / "api_keys.json"
    if _keys_file.exists():
        import json
        _keys = json.loads(_keys_file.read_text())
        _raw = _keys.get("keys", _keys)
        if _raw:
            _TEST_API_KEY = next(iter(_raw.keys()), "")
    if not _TEST_API_KEY:
        _TEST_API_KEY = "sk-test-phase31"
    os.environ["COOLEDAI_API_KEY"] = _TEST_API_KEY


def _make_row(ts=None, pilot_temp=68.0, baseline_temp=70.0,
              pilot_rpm=2500.0, baseline_rpm=3500.0,
              pilot_cpu=45.0, baseline_cpu=42.0,
              pilot_gpu_pwr=120.0, baseline_gpu_pwr=100.0,
              pilot_peak_pwr=180.0, baseline_peak_pwr=150.0):
    return (
        ts or time.time(),
        pilot_temp, baseline_temp,
        pilot_rpm, baseline_rpm,
        pilot_cpu, baseline_cpu,
        pilot_gpu_pwr, baseline_gpu_pwr,
        pilot_peak_pwr, baseline_peak_pwr,
    )


class TestThermalHistoryBucketsAllMetrics(unittest.TestCase):
    """Bug A: Buckets must include fan RPM, CPU temp, GPU power, peak power."""

    def test_buckets_include_all_metrics(self):
        """Aggregated buckets should include all 11-tuple metrics."""
        from fastapi.testclient import TestClient
        from api.main import app, _thermal_history

        client = TestClient(app)
        api_key = _TEST_API_KEY
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"

        # Insert rows in same hour
        base_ts = time.time() - 60  # 1 minute ago (same hour bucket)
        _thermal_history.setdefault(owner_id, [])
        for i in range(3):
            _thermal_history[owner_id].append(_make_row(
                ts=base_ts + i * 5,
                pilot_rpm=2500 + i * 100,
                baseline_rpm=3500 + i * 50,
                pilot_cpu=40.0 + i,
                baseline_cpu=38.0 + i,
                pilot_gpu_pwr=120.0 + i * 10,
                baseline_gpu_pwr=100.0 + i * 5,
                pilot_peak_pwr=180.0 + i * 5,
                baseline_peak_pwr=150.0 + i * 5,
            ))

        resp = client.get(
            "/api/v1/thermal-history?hours=1&mode=aggregated",
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        buckets = data.get("buckets", [])
        self.assertGreater(len(buckets), 0, "Should have at least 1 bucket")

        bucket = buckets[-1]  # Most recent bucket
        # All metric fields should be present
        for field in ["pilot", "baseline", "pilot_fan_rpm", "baseline_fan_rpm",
                      "pilot_cpu_temp", "baseline_cpu_temp",
                      "pilot_gpu_power_w", "baseline_gpu_power_w"]:
            self.assertIn(field, bucket, f"Bucket missing field: {field}")
            self.assertIsNotNone(bucket[field], f"Bucket field {field} is None")

    def test_buckets_average_correctly(self):
        """Multiple rows in same hour should be averaged."""
        from fastapi.testclient import TestClient
        from api.main import app, _thermal_history

        client = TestClient(app)
        api_key = _TEST_API_KEY
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"

        # Clear and insert fresh rows in same hour
        _thermal_history[owner_id] = []
        base_ts = time.time() - 30
        # Insert 3 rows with fan RPMs: 2000, 3000, 4000 → avg should be 3000
        for rpm in [2000.0, 3000.0, 4000.0]:
            _thermal_history[owner_id].append(_make_row(
                ts=base_ts,
                pilot_rpm=rpm,
                baseline_rpm=rpm + 500,
            ))

        resp = client.get(
            "/api/v1/thermal-history?hours=1&mode=aggregated",
            headers={"X-API-Key": api_key},
        )
        data = resp.json()
        buckets = data.get("buckets", [])
        self.assertGreater(len(buckets), 0, f"No buckets returned. Response: {data}")
        # The averaged pilot_fan_rpm should be 3000.0
        bucket = buckets[-1]
        self.assertAlmostEqual(bucket["pilot_fan_rpm"], 3000.0, delta=1.0)


class TestSavingsUsesBrainTargetRpm(unittest.TestCase):
    """Bug B: Savings should use brain's target_rpm, not actual measured RPM."""

    def test_batch_ingestion_uses_target_rpm(self):
        """When batch includes optimization.target_rpm, savings should reflect it."""
        from fastapi.testclient import TestClient
        from api.main import app, _thermal_history

        client = TestClient(app)
        api_key = _TEST_API_KEY

        # Send batch where target_rpm (2800) < fan_rpm (3500)
        resp = client.post(
            "/api/v1/gateway/telemetry-batch",
            json={
                "batch": [
                    {
                        "node_id": "ST550-CooledAI-Predictive",
                        "telemetry": {
                            "temp_c": 68.0,
                            "fan_rpm": 3500.0,  # Actual measured RPM
                            "gpu_power_w": 120.0,
                            "cpu_temp_c": 45.0,
                        },
                        "optimization": {
                            "target_rpm": 2800.0,  # Brain recommends lower
                            "target_duty": 40,
                            "source": "optimization_brain",
                        },
                        "ts": "2026-03-26T00:00:00Z",
                    }
                ],
                "gateway_id": "test-gw",
                "batch_size": 1,
                "ts": "2026-03-26T00:00:00Z",
            },
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(resp.status_code, 200, f"Batch failed: {resp.text}")

        # Find the row we just inserted (match by temp_c=68.0 and optimization data)
        found_row = None
        for oid, rows in _thermal_history.items():
            for row in reversed(rows):
                if abs(float(row[1]) - 68.0) < 0.1 and abs(float(row[3]) - 2800.0) < 1.0:
                    found_row = row
                    break
            if found_row:
                break
        self.assertIsNotNone(found_row, "Batch row with target_rpm=2800 not found in _thermal_history")

        latest = found_row
        # Index 3 = pilot_rpm (should be brain's 2800, not actual 3500)
        pilot_rpm = float(latest[3])
        self.assertAlmostEqual(pilot_rpm, 2800.0, delta=1.0,
                               msg=f"pilot_rpm should be brain's target (2800), got {pilot_rpm}")
        # Index 4 = baseline_rpm (should be actual 3500)
        baseline_rpm = float(latest[4])
        self.assertAlmostEqual(baseline_rpm, 3500.0, delta=1.0)

    def test_savings_positive_when_target_below_baseline(self):
        """When brain recommends lower RPM, delta_w should be positive."""
        from fastapi.testclient import TestClient
        from api.main import app, _thermal_history

        client = TestClient(app)
        api_key = _TEST_API_KEY
        owner_id = "user_3B2tUMI61WvTOsmR2ZMfHhXjsDa"

        # Insert rows where pilot_rpm (brain target) < baseline_rpm
        _thermal_history[owner_id] = []
        for i in range(5):
            _thermal_history[owner_id].append(_make_row(
                ts=time.time() - i * 5,
                pilot_rpm=2800.0,     # Brain's recommendation
                baseline_rpm=3500.0,  # Control/BMC default
            ))

        resp = client.get(
            "/api/v1/savings-chart?hours=1",
            headers={"X-API-Key": api_key},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        points = data.get("points", [])
        self.assertGreater(len(points), 0)
        # delta_w should be positive (baseline > optimized)
        total_delta = sum(p["delta_w"] for p in points)
        self.assertGreater(total_delta, 0, f"delta_w should be positive, got {total_delta}")


class TestSSLContextPresent(unittest.TestCase):
    """Bug C: CloudForwarder must use certifi SSL context."""

    def test_ssl_context_defined(self):
        """_SSL_CTX should be non-None when certifi is installed."""
        try:
            import certifi  # noqa: F401
            from gateway.cloud_forwarder import _SSL_CTX
            self.assertIsNotNone(_SSL_CTX, "SSL context should be set when certifi is installed")
        except ImportError:
            self.skipTest("certifi not installed")


class TestConfidenceThresholdLowered(unittest.TestCase):
    """Verify confidence threshold is 0.55 for 2-node PoC."""

    def test_threshold_is_055(self):
        from core.optimization.confidence_estimator import CAUTIONARY_CONFIDENCE_THRESHOLD
        self.assertEqual(CAUTIONARY_CONFIDENCE_THRESHOLD, 0.55)


if __name__ == "__main__":
    unittest.main()
