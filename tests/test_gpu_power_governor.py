"""Unit tests for GPU power governor — Phase 1 (free functions) + Phase 2 (GPUPowerGovernor class)."""
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from core.optimization.gpu_power_governor import (
    GPUPowerGovernor,
    GpuPowerEnvelope,
    compute_all_targets,
    compute_target_power_w,
    reset_all_gpus_to_default,
    set_gpu_power_limit_w,
)


# ===================================================================
# Phase 1: Free-function tests (existing, preserved)
# ===================================================================


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

    def test_at_full_boundary(self) -> None:
        w = compute_target_power_w(
            62.0, self.env, temp_full_power_c=62, temp_soft_start_c=72, temp_hard_c=82
        )
        self.assertEqual(w, 100.0)

    def test_at_soft_boundary(self) -> None:
        w = compute_target_power_w(
            72.0, self.env, temp_full_power_c=62, temp_soft_start_c=72, temp_hard_c=82
        )
        floor = max(30.0, 100.0 * 0.55)
        mid = (100.0 + floor) / 2.0
        self.assertAlmostEqual(w, mid, places=1)

    def test_at_hard_boundary(self) -> None:
        w = compute_target_power_w(
            82.0, self.env, temp_full_power_c=62, temp_soft_start_c=72, temp_hard_c=82
        )
        floor = max(30.0, 100.0 * 0.55)
        self.assertAlmostEqual(w, floor, places=1)

    def test_very_hot(self) -> None:
        """Above hard ceiling, should be at floor."""
        w = compute_target_power_w(
            100.0, self.env, temp_full_power_c=62, temp_soft_start_c=72, temp_hard_c=82
        )
        floor = max(30.0, 100.0 * 0.55)
        self.assertAlmostEqual(w, floor, places=1)


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


# ===================================================================
# Phase 2: GPUPowerGovernor class tests
# ===================================================================


def _make_envs(n: int = 1, default_w: float = 200.0) -> list:
    return [
        GpuPowerEnvelope(index=i, min_w=50.0, default_w=default_w, max_w=300.0)
        for i in range(n)
    ]


def _make_governor(
    n_gpus: int = 1,
    performance_mode: bool = True,
    calibration_window_s: float = 0.0,  # skip calibration by default for most tests
    **kwargs,
) -> GPUPowerGovernor:
    """Helper to create a governor with sensible test defaults."""
    envs = _make_envs(n_gpus)
    return GPUPowerGovernor(
        envs,
        performance_mode=performance_mode,
        calibration_window_s=calibration_window_s,
        dry_run=True,
        **kwargs,
    )


class TestGPUGovernorInit(unittest.TestCase):
    """Basic construction and properties."""

    def test_init_with_envelopes(self) -> None:
        gov = _make_governor(n_gpus=2)
        self.assertEqual(len(gov.envelopes), 2)
        self.assertTrue(gov.performance_mode)

    def test_from_env_no_gpus(self) -> None:
        """from_env returns None when no GPUs found."""
        result = GPUPowerGovernor.from_env(envelopes=[], dry_run=True)
        self.assertIsNone(result)

    @patch.dict(os.environ, {
        "COOLEDAI_GPU_PERF_MODE": "false",
        "COOLEDAI_GPU_GOV_CAL_WINDOW_S": "60",
    })
    def test_from_env_reads_env_vars(self) -> None:
        envs = _make_envs(1)
        gov = GPUPowerGovernor.from_env(envelopes=envs, dry_run=True)
        self.assertIsNotNone(gov)
        self.assertFalse(gov.performance_mode)
        self.assertAlmostEqual(gov._calibration_window_s, 60.0)


# -------------------------------------------------------------------
# Profile-driven threshold computation (Step 2)
# -------------------------------------------------------------------

class TestProfileDrivenThresholds(unittest.TestCase):
    """update_from_profile should remap temp bands from CalibrationProfile."""

    def _make_profile(self, mean=55.0, stdev=3.0, p90=70.0, spike=65.0):
        p = MagicMock()
        p.temp_mean_c = mean
        p.temp_stdev_c = stdev
        p.temp_p90_c = p90
        p.spike_trigger_temp_c = spike
        return p

    def test_basic_profile_update(self) -> None:
        gov = _make_governor()
        old_full = gov._temp_full_c
        old_soft = gov._temp_soft_c
        old_hard = gov._temp_hard_c

        profile = self._make_profile(mean=55.0, stdev=3.0, p90=70.0, spike=65.0)
        gov.update_from_profile(profile)

        # full = mean + 0.5*stdev = 55 + 1.5 = 56.5
        self.assertAlmostEqual(gov._temp_full_c, 56.5, places=1)
        # soft = spike_trigger = 65.0
        self.assertAlmostEqual(gov._temp_soft_c, 65.0, places=1)
        # hard = p90 = 70.0
        self.assertAlmostEqual(gov._temp_hard_c, 70.0, places=1)

    def test_profile_ensures_ordering(self) -> None:
        """full < soft < hard must always hold, even with bad profile values."""
        gov = _make_governor()
        # Give a profile where spike < full (inverted)
        profile = self._make_profile(mean=70.0, stdev=2.0, p90=72.0, spike=60.0)
        gov.update_from_profile(profile)
        # full = 70 + 1 = 71, spike=60 < full so soft = full + 5 = 76
        # p90=72 < soft=76 so hard = soft + 5 = 81
        self.assertLess(gov._temp_full_c, gov._temp_soft_c)
        self.assertLess(gov._temp_soft_c, gov._temp_hard_c)

    def test_profile_with_zero_mean_ignored(self) -> None:
        """Profile with zero mean is not applied."""
        gov = _make_governor()
        old_full = gov._temp_full_c
        profile = self._make_profile(mean=0.0)
        gov.update_from_profile(profile)
        self.assertEqual(gov._temp_full_c, old_full)

    def test_profile_clamped_within_sane_range(self) -> None:
        """Extreme temps get clamped to 30-95°C range."""
        gov = _make_governor()
        profile = self._make_profile(mean=95.0, stdev=10.0, p90=100.0, spike=98.0)
        gov.update_from_profile(profile)
        self.assertLessEqual(gov._temp_full_c, 85.0)
        self.assertLessEqual(gov._temp_soft_c, 90.0)
        self.assertLessEqual(gov._temp_hard_c, 95.0)


# -------------------------------------------------------------------
# Hysteresis preventing rapid changes (Step 3)
# -------------------------------------------------------------------

class TestHysteresis(unittest.TestCase):
    """Governor should not apply tiny watt changes or change too frequently."""

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_watts_hysteresis_blocks_small_change(self, mock_set) -> None:
        gov = _make_governor(min_change_w=10.0)
        # First tick — applies
        gov.tick([80.0], [150.0])
        first_call_count = mock_set.call_count

        # Tiny temp change — should NOT apply again (within 10W hysteresis)
        gov._last_tick = 0  # bypass time hysteresis
        gov.tick([80.1], [150.0])
        self.assertEqual(mock_set.call_count, first_call_count)

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_time_hysteresis_blocks_rapid_ticks(self, mock_set) -> None:
        gov = _make_governor(min_interval_s=60.0)
        gov.tick([85.0], [150.0])
        first_count = mock_set.call_count

        # Immediate second tick — blocked by time hysteresis
        gov.tick([40.0], [150.0])  # big temp change, but too soon
        self.assertEqual(mock_set.call_count, first_count)


# -------------------------------------------------------------------
# Auto-calibration window (Step 4)
# -------------------------------------------------------------------

class TestAutoCalibration(unittest.TestCase):
    """Governor should observe for calibration_window_s before adjusting."""

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_calibration_blocks_early_adjustments(self, mock_set) -> None:
        gov = _make_governor(calibration_window_s=300.0)
        self.assertFalse(gov.calibrated)

        # Tick during calibration — no adjustment applied
        results = gov.tick([90.0], [180.0])
        self.assertEqual(len(results), 0)
        mock_set.assert_not_called()
        self.assertFalse(gov.calibrated)

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_calibration_completes_after_window(self, mock_set) -> None:
        gov = _make_governor(calibration_window_s=0.01)
        # Feed some power samples
        gov._check_calibration(100.0)
        gov._check_calibration(200.0)
        time.sleep(0.02)
        gov._check_calibration(150.0)

        self.assertTrue(gov.calibrated)

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_calibration_computes_idle_peak(self, mock_set) -> None:
        gov = _make_governor(calibration_window_s=0.01)
        # Simulate power samples
        for p in [50, 60, 70, 80, 90, 100, 110, 120, 130, 180]:
            gov._check_calibration(float(p))
        time.sleep(0.02)
        gov._check_calibration(100.0)

        self.assertTrue(gov.calibrated)
        self.assertGreater(gov._observed_idle_w, 0)
        self.assertGreater(gov._observed_peak_w, gov._observed_idle_w)

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_min_allowed_limit_after_calibration(self, mock_set) -> None:
        """After calibration, min_allowed = max(nvidia_min, idle * 1.10)."""
        gov = _make_governor(calibration_window_s=0.01)
        for _ in range(20):
            gov._check_calibration(100.0)  # all samples at 100W
        time.sleep(0.02)
        gov._check_calibration(100.0)

        self.assertTrue(gov.calibrated)
        # idle ~100W, so min_allowed = max(50.0, 100*1.10) = 110.0
        cal_min = gov._min_allowed_limit_w.get(0)
        self.assertIsNotNone(cal_min)
        self.assertAlmostEqual(cal_min, 110.0, delta=5.0)


# -------------------------------------------------------------------
# Reset to defaults on shutdown
# -------------------------------------------------------------------

class TestResetToDefaults(unittest.TestCase):
    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_reset_calls_nvidia_smi_for_each_gpu(self, mock_set) -> None:
        gov = _make_governor(n_gpus=2)
        gov.reset_to_defaults()
        self.assertEqual(mock_set.call_count, 2)
        # Should restore to default_w (200.0)
        for call in mock_set.call_args_list:
            self.assertEqual(call[0][1], 200)

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_reset_clears_last_limits(self, mock_set) -> None:
        gov = _make_governor()
        gov._last_limits[0] = 150.0
        gov.reset_to_defaults()
        self.assertEqual(len(gov._last_limits), 0)


# -------------------------------------------------------------------
# Workload parity guard (Step 5)
# -------------------------------------------------------------------

class TestParityGuard(unittest.TestCase):
    """In performance_mode, GPU stays at full TDP when thermally healthy."""

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_perf_mode_blocks_reduction_when_cool(self, mock_set) -> None:
        """When temp < spike_trigger in performance_mode, no reduction."""
        gov = _make_governor(performance_mode=True)
        # Set spike trigger via profile
        profile = MagicMock()
        profile.temp_mean_c = 55.0
        profile.temp_stdev_c = 3.0
        profile.temp_p90_c = 75.0
        profile.spike_trigger_temp_c = 68.0
        gov.update_from_profile(profile)

        # Temp 60°C is below spike_trigger 68°C — parity guard should keep full TDP
        results = gov.tick([60.0], [150.0])
        # Should either not change (at default) or change to default
        for idx, w, reason in results:
            self.assertEqual(w, 200.0)  # default_w
            self.assertEqual(reason, "parity_guard")

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_perf_mode_allows_reduction_when_hot(self, mock_set) -> None:
        """When temp > spike_trigger, reduction is allowed even in perf mode."""
        gov = _make_governor(performance_mode=True)
        profile = MagicMock()
        profile.temp_mean_c = 55.0
        profile.temp_stdev_c = 3.0
        profile.temp_p90_c = 75.0
        profile.spike_trigger_temp_c = 68.0
        gov.update_from_profile(profile)

        # Temp 80°C > spike_trigger — should reduce
        results = gov.tick([80.0], [180.0])
        if results:
            _, w, reason = results[0]
            self.assertLess(w, 200.0)
            self.assertEqual(reason, "temperature_band")

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_efficiency_mode_reduces_when_cool(self, mock_set) -> None:
        """Without performance_mode, reduction allowed even below spike_trigger."""
        gov = _make_governor(performance_mode=False)
        profile = MagicMock()
        profile.temp_mean_c = 55.0
        profile.temp_stdev_c = 3.0
        profile.temp_p90_c = 75.0
        profile.spike_trigger_temp_c = 68.0
        gov.update_from_profile(profile)

        # Temp 60°C is in blend range (full=56.5, soft=68, hard=75)
        # so target < default
        results = gov.tick([60.0], [180.0])
        if results:
            _, w, reason = results[0]
            self.assertLessEqual(w, 200.0)


# -------------------------------------------------------------------
# Idle detection
# -------------------------------------------------------------------

class TestIdleDetection(unittest.TestCase):
    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_idle_gpu_gets_full_tdp(self, mock_set) -> None:
        """GPU drawing < 15% of default should stay at default (idle)."""
        gov = _make_governor(performance_mode=False)
        # Power = 20W, which is < 200 * 0.15 = 30W
        results = gov.tick([80.0], [20.0])
        for idx, w, reason in results:
            self.assertEqual(w, 200.0)
            self.assertEqual(reason, "idle")


# -------------------------------------------------------------------
# nvidia-smi mocking for set/reset
# -------------------------------------------------------------------

class TestNvidiaSmiMocked(unittest.TestCase):
    """Verify nvidia-smi subprocess calls are properly mocked."""

    @patch("core.optimization.gpu_power_governor.subprocess.check_call")
    def test_set_power_limit_calls_nvidia_smi(self, mock_cc) -> None:
        result = set_gpu_power_limit_w(0, 180)
        mock_cc.assert_called_once()
        cmd = mock_cc.call_args[0][0]
        self.assertEqual(cmd[0], "nvidia-smi")
        self.assertIn("-pl", cmd)
        self.assertIn("180", cmd)

    @patch("core.optimization.gpu_power_governor.subprocess.check_call")
    def test_set_power_limit_dry_run(self, mock_cc) -> None:
        result = set_gpu_power_limit_w(0, 180, dry_run=True)
        self.assertTrue(result)
        mock_cc.assert_not_called()

    @patch("core.optimization.gpu_power_governor.subprocess.check_call",
           side_effect=FileNotFoundError)
    def test_set_power_limit_no_nvidia_smi(self, mock_cc) -> None:
        result = set_gpu_power_limit_w(0, 180)
        self.assertFalse(result)

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_reset_all_gpus(self, mock_set) -> None:
        envs = _make_envs(3, default_w=250.0)
        reset_all_gpus_to_default(envs, dry_run=False)
        self.assertEqual(mock_set.call_count, 3)
        for call in mock_set.call_args_list:
            self.assertEqual(call[0][1], 250)


# -------------------------------------------------------------------
# Structured logging
# -------------------------------------------------------------------

class TestGPUGovLogging(unittest.TestCase):
    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_maybe_log_emits_every_10s(self, mock_set) -> None:
        gov = _make_governor()
        gov._last_log_time = 0  # force log
        with self.assertLogs("cooledai.gpu_gov", level="INFO") as cm:
            gov.tick([85.0], [180.0])
        gpu_gov_logs = [l for l in cm.output if "[GPU_GOV]" in l]
        self.assertGreater(len(gpu_gov_logs), 0)

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_log_contains_required_fields(self, mock_set) -> None:
        gov = _make_governor()
        gov._last_log_time = 0
        # First tick to get past calibration
        gov.tick([85.0], [180.0])
        # Force log time reset so second tick also logs
        gov._last_log_time = 0
        gov._last_tick = 0
        with self.assertLogs("cooledai.gpu_gov", level="INFO") as cm:
            gov.tick([85.0], [180.0])
        gpu_gov_logs = [l for l in cm.output if "[GPU_GOV]" in l and "gpu_temp=" in l]
        self.assertGreater(len(gpu_gov_logs), 0, "Expected [GPU_GOV] log with gpu_temp= field")
        log_line = gpu_gov_logs[0]
        for field in ["gpu_temp=", "current_limit=", "default_limit=", "reduction=", "reason="]:
            self.assertIn(field, log_line)


if __name__ == "__main__":
    unittest.main()
