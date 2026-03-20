"""Tests for core.safety.telemetry_failure_policy."""
import unittest
from datetime import datetime, timedelta, timezone

from core.safety.telemetry_failure_policy import (
    apply_failure_adjustments,
    assess_fan_slippage,
    assess_history_staleness,
    assess_temp_rise_at_max_cooling,
    build_posture_from_nodes,
)


class TestStaleness(unittest.TestCase):
    def test_fresh_not_stale(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = [now - timedelta(seconds=10)]
        stale, age = assess_history_staleness(ts, now=now, max_age_sec=120)
        self.assertFalse(stale)
        self.assertLess(age, 120)

    def test_stale(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = [now - timedelta(seconds=500)]
        stale, age = assess_history_staleness(ts, now=now, max_age_sec=120)
        self.assertTrue(stale)
        self.assertGreaterEqual(age, 500)


class TestFanSlip(unittest.TestCase):
    def test_no_cmd_skips(self):
        slip, _ = assess_fan_slippage(None, 2000, 7000)
        self.assertFalse(slip)

    def test_slip_high_cmd_low_tach(self):
        slip, msg = assess_fan_slippage(90.0, 1500, 7000)
        self.assertTrue(slip)
        self.assertIn("slip", msg)


class TestSaturation(unittest.TestCase):
    def test_rising_at_max(self):
        temps = [40.0, 40.1, 40.2, 40.35, 40.5]
        sat, _ = assess_temp_rise_at_max_cooling(temps, 98)
        self.assertTrue(sat)

    def test_low_duty_skips(self):
        temps = [40.0, 40.5, 41.0, 41.5, 42.0]
        sat, _ = assess_temp_rise_at_max_cooling(temps, 50)
        self.assertFalse(sat)


class TestApplyAdjustments(unittest.TestCase):
    def test_stale_boosts(self):
        out, p = apply_failure_adjustments(
            40, stale_history=True, stale_age_sec=200, fan_slip=False, saturation=False
        )
        self.assertGreaterEqual(out, 70)
        self.assertIn("stale_telemetry_window", p.reasons)


class DummyNode:
    def __init__(self, ts, thermal):
        self.timestamp = ts
        self.thermal_input = thermal


class TestBuildFromNodes(unittest.TestCase):
    def test_slip_increases_duty(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        nodes = [DummyNode(now - timedelta(seconds=i), 45.0) for i in range(5, 0, -1)]
        duty, p = build_posture_from_nodes(
            nodes,
            target_duty=50,
            last_commanded_duty=90.0,
            fan_rpm=800.0,  # ~11% implied vs 90% cmd → slippage
            max_fan_rpm=7000.0,
            now=now,
        )
        self.assertTrue(p.fan_slippage)
        self.assertGreater(duty, 50)


if __name__ == "__main__":
    unittest.main()
