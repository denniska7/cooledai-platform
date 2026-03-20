"""Tests for TempPredictor trend fit, confidence, and extrapolation guards."""

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from core.hal.base_node import ServerNode
from core.optimization.predictor import (
    MAX_EXTRAP_SLOPE_C_PER_S,
    PREDICTION_HORIZON_S,
    TempPredictor,
    _weighted_linear_trend,
)


def _nodes_ramp(start_c: float, n: int, dt_s: float = 1.0) -> list:
    """Linear temp/power ramp like HWiNFO CSV (1 Hz)."""
    base = datetime(2026, 1, 26, tzinfo=timezone.utc)
    nodes = []
    for i in range(n):
        t = base + timedelta(seconds=i * dt_s)
        temp = start_c + i * 1.0
        pwr = 80.0 + i * 5.0
        rpm = 1200.0 + i * 50.0
        nodes.append(
            ServerNode(
                thermal_input=temp,
                power_draw=pwr,
                cooling_output=rpm,
                utilization=40.0,
                node_id="test-node",
                timestamp=t,
                source="test",
            )
        )
    return nodes


class TestTempPredictor(unittest.TestCase):
    def test_weighted_trend_matches_clean_ramp(self):
        t = np.arange(10, dtype=float)
        y = 20.0 + 0.5 * t
        slope, intercept, r2, rmse = _weighted_linear_trend(t, y)
        self.assertAlmostEqual(slope, 0.5, delta=0.02)
        self.assertGreater(r2, 0.99)
        self.assertLess(rmse, 0.05)

    def test_predictor_ramp_extrapolates_t10(self):
        nodes = _nodes_ramp(45.0, 12)
        r = TempPredictor().predict(nodes)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.predicted_temp_t10, r.predicted_temp + 10.0, delta=0.5)
        self.assertGreater(r.confidence, 0.55)
        self.assertAlmostEqual(r.temp_trajectory_slope, 1.0, delta=0.05)
        self.assertAlmostEqual(r.extrapolation_temp_slope_c_per_s, 1.0, delta=0.05)

    def test_spike_slope_clamped(self):
        base = datetime(2026, 1, 26, tzinfo=timezone.utc)
        nodes = []
        for i in range(8):
            t = base + timedelta(seconds=i)
            nodes.append(
                ServerNode(
                    thermal_input=50.0 + i * 0.1,
                    power_draw=100.0,
                    cooling_output=2000.0,
                    utilization=40.0,
                    node_id="test-node",
                    timestamp=t,
                    source="test",
                )
            )
        nodes[-1].thermal_input = 95.0
        r = TempPredictor().predict(nodes)
        self.assertIsNotNone(r)
        max_jump = max(0.5, MAX_EXTRAP_SLOPE_C_PER_S * PREDICTION_HORIZON_S)
        self.assertLessEqual(
            abs(r.predicted_temp_t10 - r.predicted_temp),
            max_jump + 0.01,
        )

    def test_ewma_filter_constant_time(self):
        from core.ingestion.telemetry_smoothing import EwmaTelemetryFilter

        f = EwmaTelemetryFilter(alpha=0.5)
        base = datetime(2026, 1, 26, tzinfo=timezone.utc)
        n = ServerNode(
            thermal_input=60.0,
            power_draw=100.0,
            cooling_output=2000.0,
            utilization=50.0,
            node_id="n1",
            timestamp=base,
            source="t",
        )
        for _ in range(20):
            n.thermal_input = 60.0 + np.random.randn() * 2
            f.apply([n])
        self.assertGreater(n.thermal_input, 55.0)
        self.assertLess(n.thermal_input, 65.0)


if __name__ == "__main__":
    unittest.main()
