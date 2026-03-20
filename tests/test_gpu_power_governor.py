"""Unit tests for GPU power governor curve."""
import unittest

from core.optimization.gpu_power_governor import (
    GpuPowerEnvelope,
    compute_all_targets,
    compute_target_power_w,
)


class TestComputeTargetPower(unittest.TestCase):
    def setUp(self) -> None:
        self.env = GpuPowerEnvelope(index=0, min_w=30.0, default_w=100.0, max_w=120.0)

    def test_cool_full_tdp(self) -> None:
        w = compute_target_power_w(
            50.0, self.env, temp_full_power_c=62, temp_soft_start_c=72, temp_hard_c=82
        )
        self.assertEqual(w, 100.0)

    def test_hot_reduces(self) -> None:
        w = compute_target_power_w(
            85.0, self.env, temp_full_power_c=62, temp_soft_start_c=72, temp_hard_c=82
        )
        self.assertLess(w, 100.0)
        self.assertGreaterEqual(w, 55.0)  # min_fraction 0.55 * default


class TestComputeAll(unittest.TestCase):
    def test_two_gpus(self) -> None:
        envs = [
            GpuPowerEnvelope(0, 25, 80, 90),
            GpuPowerEnvelope(1, 25, 80, 90),
        ]
        pairs = compute_all_targets(
            [60.0, 80.0],
            envs,
            temp_full_power_c=62,
            temp_soft_start_c=72,
            temp_hard_c=82,
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0], 0)
        self.assertEqual(pairs[0][1], 80)
        self.assertLess(pairs[1][1], 80)


if __name__ == "__main__":
    unittest.main()
