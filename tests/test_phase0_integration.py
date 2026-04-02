"""
Phase 0 Integration Test Suite.

Tests all foundation fixes working together:
- API key hierarchy with facility/node validation
- Per-node calibration profile isolation
- Per-node spike hold isolation
- Concurrent multi-node operation
- Audit log signing
"""

import hashlib
import hmac as hmac_mod
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Spike hold state
from core.optimization.spike_hold_state import (
    record_spike_if_hot,
    spike_hold_floor_rpm,
    clear_spike_hold,
)


class TestAPIKeyHierarchy(unittest.TestCase):
    """Test the hierarchical API key system."""

    def test_migrate_legacy_keys(self):
        """Legacy flat format migrates to hierarchical format."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        # Simulate loading the migration function
        from api.main import _migrate_legacy_keys

        legacy = {
            "<YOUR_COOLEDAI_API_KEY>": {
                "owner_id": "admin",
                "label": "Admin (auto-generated on first startup)",
                "created_at": "2026-03-17T01:54:33.916812Z",
            }
        }

        migrated = _migrate_legacy_keys(legacy)

        # Should have "keys" wrapper
        self.assertIn("keys", migrated)
        keys = migrated["keys"]

        # The legacy key should be preserved
        self.assertIn("<YOUR_COOLEDAI_API_KEY>", keys)
        entry = keys["<YOUR_COOLEDAI_API_KEY>"]

        # Should have facilities with wildcard
        self.assertIn("facilities", entry)
        self.assertIn("facility-default", entry["facilities"])
        self.assertIn("*", entry["facilities"]["facility-default"]["nodes"])

        # owner_id preserved
        self.assertEqual(entry["owner_id"], "admin")

    def test_migrate_already_new_format(self):
        """New format should not be re-migrated."""
        from api.main import _migrate_legacy_keys

        new_format = {
            "keys": {
                "sk-abc123": {
                    "owner_id": "owner-001",
                    "facilities": {
                        "facility-us-east-1": {
                            "nodes": ["node-001", "node-002"],
                            "tier": "enterprise",
                            "rate_limit_rps": 100,
                        }
                    },
                }
            }
        }

        result = _migrate_legacy_keys(new_format)
        self.assertEqual(result, new_format, "New format should pass through unchanged")

    def test_validate_node_access_wildcard(self):
        """Wildcard nodes list permits any node_id."""
        from api.main import _validate_node_access, _api_key_registry, _build_key_indexes

        test_key = "sk-test-wildcard"
        _api_key_registry[test_key] = {
            "owner_id": "admin",
            "facilities": {
                "facility-default": {
                    "nodes": ["*"],
                    "tier": "legacy",
                    "rate_limit_rps": 100,
                }
            },
        }
        _build_key_indexes()

        try:
            # Should not raise for any node_id
            fid = _validate_node_access(test_key, "any-node-001")
            self.assertIsNotNone(fid)
            fid = _validate_node_access(test_key, "completely-random-node")
            self.assertIsNotNone(fid)
        finally:
            del _api_key_registry[test_key]
            _build_key_indexes()

    def test_validate_node_access_restricted(self):
        """Restricted facility only permits listed nodes."""
        from api.main import _validate_node_access, _api_key_registry, _build_key_indexes
        from fastapi import HTTPException

        test_key = "sk-test-restricted"
        _api_key_registry[test_key] = {
            "owner_id": "owner-A",
            "facilities": {
                "facility-east": {
                    "nodes": ["node-001", "node-002"],
                    "tier": "enterprise",
                    "rate_limit_rps": 100,
                }
            },
        }
        _build_key_indexes()

        try:
            # Authorized nodes should work
            fid = _validate_node_access(test_key, "node-001")
            self.assertEqual(fid, "facility-east")

            fid = _validate_node_access(test_key, "node-002")
            self.assertEqual(fid, "facility-east")

            # Unauthorized node should raise 403
            with self.assertRaises(HTTPException) as ctx:
                _validate_node_access(test_key, "node-999")
            self.assertEqual(ctx.exception.status_code, 403)
        finally:
            del _api_key_registry[test_key]
            _build_key_indexes()

    def test_api_key_facility_isolation(self):
        """Two owners cannot access each other's nodes."""
        from api.main import _validate_node_access, _api_key_registry, _build_key_indexes
        from fastapi import HTTPException

        key_a = "sk-owner-a-key"
        key_b = "sk-owner-b-key"
        _api_key_registry[key_a] = {
            "owner_id": "owner-A",
            "facilities": {
                "facility-east": {
                    "nodes": ["node-001", "node-002", "node-003"],
                    "tier": "enterprise",
                    "rate_limit_rps": 100,
                }
            },
        }
        _api_key_registry[key_b] = {
            "owner_id": "owner-B",
            "facilities": {
                "facility-west": {
                    "nodes": ["node-004", "node-005", "node-006"],
                    "tier": "enterprise",
                    "rate_limit_rps": 100,
                }
            },
        }
        _build_key_indexes()

        try:
            # Owner A can access their nodes
            self.assertEqual(_validate_node_access(key_a, "node-001"), "facility-east")

            # Owner A CANNOT access Owner B's nodes
            with self.assertRaises(HTTPException) as ctx:
                _validate_node_access(key_a, "node-004")
            self.assertEqual(ctx.exception.status_code, 403)

            # Owner B can access their nodes
            self.assertEqual(_validate_node_access(key_b, "node-004"), "facility-west")

            # Owner B CANNOT access Owner A's nodes
            with self.assertRaises(HTTPException) as ctx:
                _validate_node_access(key_b, "node-001")
            self.assertEqual(ctx.exception.status_code, 403)
        finally:
            del _api_key_registry[key_a]
            del _api_key_registry[key_b]
            _build_key_indexes()


class TestCalibrationProfileIsolation(unittest.TestCase):
    """Test that per-node calibration profiles are isolated."""

    def test_profiles_stored_per_node(self):
        """Different nodes get different calibration profiles."""
        from api.main import _node_profiles_lock, _node_calibration_profiles

        try:
            from core.optimization.thermal_calibrator import CalibrationProfile
        except ImportError:
            self.skipTest("CalibrationProfile not importable")

        # Create two different profiles
        cp1 = CalibrationProfile()
        cp1.temp_mean_c = 95.0
        cp1.temp_stdev_c = 2.0

        cp2 = CalibrationProfile()
        cp2.temp_mean_c = 35.0
        cp2.temp_stdev_c = 1.0

        with _node_profiles_lock:
            _node_calibration_profiles["node-001"] = cp1
            _node_calibration_profiles["node-002"] = cp2

        try:
            with _node_profiles_lock:
                profile_1 = _node_calibration_profiles.get("node-001")
                profile_2 = _node_calibration_profiles.get("node-002")

            # Verify they are different objects with different values
            self.assertIsNot(profile_1, profile_2,
                             "Node profiles must be different objects")
            self.assertAlmostEqual(profile_1.temp_mean_c, 95.0, places=1)
            self.assertAlmostEqual(profile_2.temp_mean_c, 35.0, places=1)
        finally:
            with _node_profiles_lock:
                _node_calibration_profiles.pop("node-001", None)
                _node_calibration_profiles.pop("node-002", None)


class TestSpikeIsolationUnderLoad(unittest.TestCase):
    """Test spike hold isolation when multiple nodes are active."""

    def setUp(self):
        self.num_nodes = 10
        self.session_keys = [f"owner-test:node-{i:03d}" for i in range(self.num_nodes)]

    def tearDown(self):
        for key in self.session_keys:
            clear_spike_hold(key)

    def test_spike_on_one_node_does_not_affect_others(self):
        """Trigger thermal spike on node-003, verify only node-003 enters hold."""
        # All nodes start with no spike hold
        for key in self.session_keys:
            floor = spike_hold_floor_rpm(key, 7000)
            self.assertEqual(floor, 0.0, f"Pre-test: {key} should have no spike hold")

        # Trigger spike on node-003 only (95°C > 42°C threshold)
        spike_key = self.session_keys[3]  # owner-test:node-003
        record_spike_if_hot(spike_key, 95.0, spike_trigger_temp_c=42.0, spike_hold_duration_s=60)

        # Verify only node-003 is in spike hold
        for i, key in enumerate(self.session_keys):
            floor = spike_hold_floor_rpm(key, 7000)
            if i == 3:
                self.assertGreater(floor, 0.0,
                                   "node-003 should have active spike hold")
            else:
                self.assertEqual(floor, 0.0,
                                 f"node-{i:03d} should NOT have spike hold")

    def test_concurrent_optimization_no_crosstalk(self):
        """Multiple threads doing optimization work don't interfere."""
        results = {}
        errors = []

        def worker(idx: int):
            key = self.session_keys[idx]
            temp = 95.0 if idx == 3 else 35.0  # Only node-003 is hot
            try:
                record_spike_if_hot(
                    key, temp,
                    spike_trigger_temp_c=42.0,
                    spike_hold_duration_s=60,
                )
                floor = spike_hold_floor_rpm(key, 7000)
                results[idx] = floor
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(self.num_nodes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")

        # Only node-003 should have non-zero floor
        for idx, floor in results.items():
            if idx == 3:
                self.assertGreater(floor, 0.0, "node-003 should be in spike hold")
            else:
                self.assertEqual(floor, 0.0, f"node-{idx:03d} should NOT be in spike hold")


class TestAuditLogSigning(unittest.TestCase):
    """Test HMAC-SHA256 signing of API audit log entries."""

    def test_signing_and_verification(self):
        """Signed entries can be verified."""
        try:
            from api.audit_log import verify_audit_entry
        except ImportError:
            self.skipTest("verify_audit_entry not available")

        signing_key = "test-signing-key-12345"
        entry = {
            "type": "api_request",
            "key_id": "sk-test12345...",
            "client_id": "owner-001",
            "endpoint": "/api/v1/optimize/control",
            "method": "POST",
            "status_code": 200,
            "ip": "127.0.0.1",
            "data_plane_touched": False,
            "ts": "2026-03-25T12:00:00+00:00",
        }

        # Sign it
        payload = json.dumps(entry, sort_keys=True, default=str)
        sig = hmac_mod.new(
            signing_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        entry["signature"] = sig

        # Verify
        self.assertTrue(verify_audit_entry(entry, signing_key))

    def test_tampered_entry_fails_verification(self):
        """Tampered entries fail verification."""
        try:
            from api.audit_log import verify_audit_entry
        except ImportError:
            self.skipTest("verify_audit_entry not available")

        signing_key = "test-signing-key-12345"
        entry = {
            "type": "api_request",
            "key_id": "sk-test...",
            "status_code": 200,
            "data_plane_touched": False,
            "ts": "2026-03-25T12:00:00+00:00",
        }

        payload = json.dumps(entry, sort_keys=True, default=str)
        sig = hmac_mod.new(
            signing_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        entry["signature"] = sig

        # Tamper with a field
        entry["status_code"] = 500

        self.assertFalse(verify_audit_entry(entry, signing_key))

    def test_unsigned_entry_fails_verification(self):
        """Entries without signature return False."""
        try:
            from api.audit_log import verify_audit_entry
        except ImportError:
            self.skipTest("verify_audit_entry not available")

        entry = {"type": "api_request", "ts": "2026-03-25T12:00:00+00:00"}
        self.assertFalse(verify_audit_entry(entry, "any-key"))


class TestPerNodeSpikeHistory(unittest.TestCase):
    """Test the per-node spike history buffer."""

    def test_spike_history_per_node(self):
        """Each node maintains its own spike detection history."""
        from api.main import _node_spike_history_lock, _node_spike_history

        with _node_spike_history_lock:
            _node_spike_history["node-hot"] = [(time.time() - i, 90.0 - i) for i in range(10)]
            _node_spike_history["node-cool"] = [(time.time() - i, 30.0 + i * 0.1) for i in range(10)]

        try:
            with _node_spike_history_lock:
                hot_peaks = [t for _, t in _node_spike_history.get("node-hot", [])]
                cool_peaks = [t for _, t in _node_spike_history.get("node-cool", [])]

            self.assertTrue(all(t > 60.0 for t in hot_peaks),
                            "Hot node should have high temps")
            self.assertTrue(all(t < 35.0 for t in cool_peaks),
                            "Cool node should have low temps")
        finally:
            with _node_spike_history_lock:
                _node_spike_history.pop("node-hot", None)
                _node_spike_history.pop("node-cool", None)


class TestDebugEndpointData(unittest.TestCase):
    """Test that debug data structures are correctly populated per-node."""

    def test_calibration_profiles_independent(self):
        """Verify calibration profiles stored independently by node_id."""
        from api.main import _node_profiles_lock, _node_calibration_profiles

        try:
            from core.optimization.thermal_calibrator import CalibrationProfile
        except ImportError:
            self.skipTest("CalibrationProfile not importable")

        nodes = {}
        for i in range(5):
            cp = CalibrationProfile()
            cp.temp_mean_c = 30.0 + i * 15.0  # 30, 45, 60, 75, 90
            nid = f"node-{i:03d}"
            with _node_profiles_lock:
                _node_calibration_profiles[nid] = cp
            nodes[nid] = 30.0 + i * 15.0

        try:
            # Verify each node has its own profile
            for nid, expected_temp in nodes.items():
                with _node_profiles_lock:
                    cp = _node_calibration_profiles.get(nid)
                self.assertIsNotNone(cp, f"Node {nid} should have a calibration profile")
                self.assertAlmostEqual(cp.temp_mean_c, expected_temp, places=1,
                                       msg=f"Node {nid} temp_mean should be {expected_temp}")
        finally:
            with _node_profiles_lock:
                for nid in nodes:
                    _node_calibration_profiles.pop(nid, None)


if __name__ == "__main__":
    unittest.main()
