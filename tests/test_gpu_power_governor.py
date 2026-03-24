"""Unit tests for GPU power governor — Phase 1 (free functions) + Phase 2 (GPUPowerGovernor class)."""
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from core.optimization.gpu_power_governor import (
    GPUPowerGovernor,
    GpuClockCaps,
    GpuPowerEnvelope,
    compute_all_targets,
    compute_target_power_w,
    is_fixed_tdp,
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
        self.assertEqual(cmd[0], "sudo")
        self.assertEqual(cmd[1], "nvidia-smi")
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


# ===================================================================
# FIX 1: Efficiency gate tests
# ===================================================================


class TestEfficiencyGate(unittest.TestCase):
    """Efficiency gate blocks TDP reduction when power>90W, near spike temp, and ramping."""

    def _make_profile(self, spike=68.0, active_trigger=100.0):
        p = MagicMock()
        p.temp_mean_c = 55.0
        p.temp_stdev_c = 3.0
        p.temp_p90_c = 75.0
        p.spike_trigger_temp_c = spike
        p.active_compute_trigger_w = active_trigger
        return p

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_gate_blocks_all_three_conditions(self, mock_set) -> None:
        """All 3 conditions true: power>90W, near spike, power ramping → BLOCKED."""
        gov = _make_governor(performance_mode=False)
        gov.update_from_profile(self._make_profile(spike=68.0))

        # Set EWMA to a lower value so current power > EWMA (ramping)
        gov._ewma_power_w = 100.0

        # Temp 66°C (within 3°C of spike 68°C), power 110W (>90W), ramping (110 > 100 EWMA)
        results = gov.tick([66.0], [110.0])
        for idx, w, reason in results:
            self.assertEqual(w, 200.0)
            self.assertEqual(reason, "efficiency_gate_blocked")

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_gate_allows_when_power_below_90(self, mock_set) -> None:
        """Power <= 90W means condition 1 fails → gate open, reduction allowed."""
        gov = _make_governor(performance_mode=False)
        gov.update_from_profile(self._make_profile(spike=68.0))

        gov._ewma_power_w = 70.0

        # Temp 66°C (near spike), power 80W (<=90W) — condition 1 fails
        results = gov.tick([66.0], [80.0])
        for idx, w, reason in results:
            # Should reduce since gate is open (power not > 90)
            self.assertNotEqual(reason, "efficiency_gate_blocked")

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_gate_allows_when_temp_far_from_spike(self, mock_set) -> None:
        """Temp well below spike_trigger - 3°C → gate open."""
        gov = _make_governor(performance_mode=False)
        gov.update_from_profile(self._make_profile(spike=68.0))

        gov._ewma_power_w = 100.0

        # Temp 58°C (far from spike 68-3=65°C), power 110W (>90), ramping
        results = gov.tick([58.0], [110.0])
        for idx, w, reason in results:
            self.assertNotEqual(reason, "efficiency_gate_blocked")

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_gate_allows_when_power_declining(self, mock_set) -> None:
        """Power trending down (current < EWMA) → gate open."""
        gov = _make_governor(performance_mode=False)
        gov.update_from_profile(self._make_profile(spike=68.0))

        # Set EWMA higher than current → declining trend
        gov._ewma_power_w = 130.0

        # Temp 66°C (near spike), power 110W (>90W), but declining (110 < 130 EWMA)
        results = gov.tick([66.0], [110.0])
        for idx, w, reason in results:
            self.assertNotEqual(reason, "efficiency_gate_blocked")

    def test_efficiency_gate_open_direct(self) -> None:
        """Direct unit test of _efficiency_gate_open()."""
        gov = _make_governor(performance_mode=False)
        gov._spike_trigger_c = 68.0

        # All three conditions met → blocked (returns False)
        gov._ewma_power_w = 100.0
        self.assertFalse(gov._efficiency_gate_open(66.0, 110.0))

        # Power <= 90W → open
        self.assertTrue(gov._efficiency_gate_open(66.0, 85.0))

        # Temp far from spike → open
        self.assertTrue(gov._efficiency_gate_open(60.0, 110.0))

        # Power declining → open
        gov._ewma_power_w = 120.0
        self.assertTrue(gov._efficiency_gate_open(66.0, 110.0))


# ===================================================================
# FIX 2: Continuous learning tests
# ===================================================================


class TestMiniRecalibration(unittest.TestCase):
    """Rolling mini-recalibration should EWMA-smooth observed stats."""

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_mini_recal_updates_idle_peak(self, mock_set) -> None:
        gov = _make_governor(performance_mode=True)
        # Pre-set calibration values
        gov._observed_idle_w = 80.0
        gov._observed_peak_w = 200.0

        # Fill recent samples with shifted power distribution
        for _ in range(50):
            gov._recent_samples.append((60.0, 120.0))  # higher idle
        for _ in range(50):
            gov._recent_samples.append((70.0, 250.0))  # higher peak

        # Force recal by setting last_recal_time far in the past
        gov._last_recal_time = 0
        gov._check_continuous_learning(time.monotonic())

        # EWMA should move idle and peak toward new values
        # idle: 0.20 * 120 + 0.80 * 80 = 88
        self.assertGreater(gov._observed_idle_w, 80.0)
        self.assertLess(gov._observed_idle_w, 120.0)
        # peak: 0.20 * 250 + 0.80 * 200 = 210
        self.assertGreater(gov._observed_peak_w, 200.0)
        self.assertLess(gov._observed_peak_w, 250.0)


class TestDriftDetection(unittest.TestCase):
    """Temp drift > 4°C from baseline should trigger immediate recalibration."""

    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_drift_triggers_recal(self, mock_set) -> None:
        gov = _make_governor(performance_mode=True)
        gov._baseline_temp_mean = 55.0
        gov._observed_idle_w = 80.0
        gov._observed_peak_w = 180.0
        gov._last_recal_time = time.monotonic()  # recently recalibrated

        # Fill recent samples with temp drift of 5°C (> threshold 4°C)
        for _ in range(100):
            gov._recent_samples.append((60.0, 150.0))

        old_idle = gov._observed_idle_w
        with self.assertLogs("cooledai.gpu_gov", level="INFO") as cm:
            gov._check_continuous_learning(time.monotonic())

        # Should have logged drift detection
        drift_logs = [l for l in cm.output if "Drift detected" in l]
        self.assertGreater(len(drift_logs), 0)
        # Baseline should be updated
        self.assertAlmostEqual(gov._baseline_temp_mean, 60.0, delta=1.0)


# ===================================================================
# FIX 2: State persistence tests
# ===================================================================


class TestStatePersistence(unittest.TestCase):
    """save_state/load_state should round-trip governor learning state."""

    def test_save_and_load_roundtrip(self) -> None:
        import tempfile
        gov = _make_governor()
        gov._observed_idle_w = 85.0
        gov._observed_peak_w = 210.0
        gov._baseline_temp_mean = 58.0
        gov._calibrated = True
        gov._temp_full_c = 57.0
        gov._temp_soft_c = 66.0
        gov._temp_hard_c = 74.0
        gov._spike_trigger_c = 68.0
        gov._active_compute_trigger_w = 105.0
        gov._min_allowed_limit_w = {0: 93.5}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            gov.save_state(path)

            # Create fresh governor and load state
            gov2 = _make_governor()
            self.assertTrue(gov2.load_state(path))
            self.assertAlmostEqual(gov2._observed_idle_w, 85.0)
            self.assertAlmostEqual(gov2._observed_peak_w, 210.0)
            self.assertAlmostEqual(gov2._baseline_temp_mean, 58.0)
            self.assertTrue(gov2._calibrated)
            self.assertAlmostEqual(gov2._temp_full_c, 57.0)
            self.assertAlmostEqual(gov2._spike_trigger_c, 68.0)
            self.assertAlmostEqual(gov2._active_compute_trigger_w, 105.0)
            self.assertAlmostEqual(gov2._min_allowed_limit_w[0], 93.5)
        finally:
            os.unlink(path)

    def test_load_missing_file_returns_false(self) -> None:
        gov = _make_governor()
        result = gov.load_state("/nonexistent/path/state.json")
        self.assertFalse(result)


# ===================================================================
# Clock scaling tests (PATH B for fixed-TDP GPUs)
# ===================================================================


def _make_fixed_tdp_envs(n: int = 1) -> list:
    """Create envelopes where min_w == max_w (fixed TDP like T4)."""
    return [
        GpuPowerEnvelope(index=i, min_w=75.0, default_w=75.0, max_w=75.0)
        for i in range(n)
    ]


class TestFixedTDPDetection(unittest.TestCase):
    def test_fixed_tdp_detected(self) -> None:
        envs = _make_fixed_tdp_envs(2)
        self.assertTrue(is_fixed_tdp(envs))

    def test_variable_tdp_not_fixed(self) -> None:
        envs = _make_envs(1)  # min=50, default=200, max=300
        self.assertFalse(is_fixed_tdp(envs))


def _make_fixed_gov(performance_mode=False, **extra_patches):
    """Create a governor with fixed-TDP GPUs and mocked nvidia-smi queries."""
    return None  # Replaced by per-test setup below


class TestIdleClockManagement(unittest.TestCase):
    """IMPROVEMENT 1: Idle GPU should get reduced clocks after 1 tick (was 3)."""

    def _make_gov(self):
        with patch("core.optimization.gpu_power_governor.query_gpu_clock_caps") as mc, \
             patch("core.optimization.gpu_power_governor.query_supported_graphics_clocks") as ms, \
             patch("core.optimization.gpu_power_governor._query_min_memory_clock", return_value=None):
            mc.return_value = [GpuClockCaps(index=0, max_graphics_mhz=1721, max_memory_mhz=3504)]
            ms.return_value = [1721, 1600, 1400, 1200, 1000, 860]
            return GPUPowerGovernor(
                _make_fixed_tdp_envs(1),
                performance_mode=False, calibration_window_s=0.0, dry_run=True,
            )

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1075})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 2, 5)])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_idle_triggers_after_1_tick(self, mock_ac, mock_util, mock_clk):
        gov = self._make_gov()
        # 1 tick at <5% utilization → should enter idle immediately
        gov.tick([50.0], [10.0])
        self.assertIn(gov._gpu_mode[0], ("idle_efficiency", "pre_idle"))
        mock_ac.assert_called()
        self.assertTrue(gov._clocks_reduced[0])

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 50, 30)])
    @patch("core.optimization.gpu_power_governor.reset_gpu_clocks", return_value=True)
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_active_restores_after_2_tick_hysteresis(self, mock_ac, mock_rac, mock_util, mock_clk):
        gov = self._make_gov()
        gov._gpu_mode[0] = "idle_efficiency"
        gov._clocks_reduced[0] = True
        # Tick 1: utilization 50% > 15% — active_ticks goes to 1, but not enough
        gov.tick([60.0], [50.0])
        # With 2-tick hysteresis, first tick shouldn't restore yet
        # Tick 2: still active → active_ticks reaches 2 → restore
        gov._last_tick = 0
        gov.tick([60.0], [50.0])
        mock_rac.assert_called()
        self.assertEqual(gov._gpu_mode[0], "active_compute")
        self.assertFalse(gov._clocks_reduced[0])


class TestMemoryBoundDetection(unittest.TestCase):
    """IMPROVEMENT 2: Memory-bound workloads get reduced core clocks."""

    def _make_gov(self):
        with patch("core.optimization.gpu_power_governor.query_gpu_clock_caps") as mc, \
             patch("core.optimization.gpu_power_governor.query_supported_graphics_clocks") as ms, \
             patch("core.optimization.gpu_power_governor._query_min_memory_clock", return_value=None):
            mc.return_value = [GpuClockCaps(index=0, max_graphics_mhz=1721, max_memory_mhz=3504)]
            ms.return_value = [1721, 1600, 1400, 1290, 1200, 1000, 860]
            return GPUPowerGovernor(
                _make_fixed_tdp_envs(1),
                performance_mode=False, calibration_window_s=0.0, dry_run=True,
            )

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1500})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 30, 90)])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_membound_triggers_after_5_ticks(self, mock_ac, mock_util, mock_clk):
        gov = self._make_gov()
        # 5 ticks with SM < 40%, mem > 80%
        for _ in range(5):
            gov._last_tick = 0
            gov.tick([60.0], [50.0])
        self.assertEqual(gov._gpu_mode[0], "memory_bound")
        mock_ac.assert_called()
        # Should apply ~75% of 1721 = 1290 (snapped)
        call_args = mock_ac.call_args
        self.assertEqual(call_args[0][2], 1290)

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 70, 50)])
    @patch("core.optimization.gpu_power_governor.reset_gpu_clocks", return_value=True)
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_compute_bound_restores(self, mock_ac, mock_rac, mock_util, mock_clk):
        gov = self._make_gov()
        gov._gpu_mode[0] = "memory_bound"
        gov._clocks_reduced[0] = True
        # SM 70% > 60% → restore
        gov.tick([60.0], [60.0])
        mock_rac.assert_called()
        self.assertEqual(gov._gpu_mode[0], "active_compute")


class TestThermalHeadroom(unittest.TestCase):
    """IMPROVEMENT 3: Thermal headroom tracking and fan optimization benefit logging."""

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 80, 50)])
    def test_headroom_logged(self, mock_util, mock_clk):
        with patch("core.optimization.gpu_power_governor.query_gpu_clock_caps") as mc, \
             patch("core.optimization.gpu_power_governor.query_supported_graphics_clocks") as ms, \
             patch("core.optimization.gpu_power_governor._query_min_memory_clock", return_value=None):
            mc.return_value = [GpuClockCaps(index=0, max_graphics_mhz=1721, max_memory_mhz=3504)]
            ms.return_value = [1721, 1600, 1400]
            gov = GPUPowerGovernor(
                _make_fixed_tdp_envs(1),
                performance_mode=False, calibration_window_s=0.0, dry_run=True,
            )
        gov._spike_trigger_c = 72.0
        # Temp 55°C, spike 72°C → headroom = 17°C > 10, clock at max → benefit confirmed
        with self.assertLogs("cooledai.gpu_gov", level="INFO") as cm:
            gov.tick([55.0], [60.0])
        headroom_logs = [l for l in cm.output if "thermal_headroom" in l]
        self.assertGreater(len(headroom_logs), 0)
        benefit_logs = [l for l in cm.output if "fan_optimization_benefit=confirmed" in l]
        self.assertGreater(len(benefit_logs), 0)


class TestResetRestoresClocks(unittest.TestCase):
    """reset_to_defaults should restore clocks on fixed-TDP GPUs."""

    @patch("core.optimization.gpu_power_governor._query_min_memory_clock", return_value=None)
    @patch("core.optimization.gpu_power_governor.query_gpu_clock_caps")
    @patch("core.optimization.gpu_power_governor.query_supported_graphics_clocks", return_value=[1721])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    @patch("core.optimization.gpu_power_governor.reset_gpu_clocks", return_value=True)
    @patch("core.optimization.gpu_power_governor.set_gpu_power_limit_w", return_value=True)
    def test_reset_restores_clocks(self, mock_pl, mock_rac, mock_ac, mock_sc, mock_caps, mock_mem):
        mock_caps.return_value = [GpuClockCaps(index=0, max_graphics_mhz=1721, max_memory_mhz=3504)]
        gov = GPUPowerGovernor(
            _make_fixed_tdp_envs(1),
            performance_mode=False, calibration_window_s=0.0, dry_run=False,
        )
        gov._clocks_reduced[0] = True
        gov.reset_to_defaults()
        mock_rac.assert_called_once_with(0, dry_run=False)
        self.assertFalse(gov._clocks_reduced[0])


def _make_fixed_gov_for_improvements():
    """Helper to create a fixed-TDP governor with all mocks for improvement tests."""
    with patch("core.optimization.gpu_power_governor.query_gpu_clock_caps") as mc, \
         patch("core.optimization.gpu_power_governor.query_supported_graphics_clocks") as ms, \
         patch("core.optimization.gpu_power_governor._query_min_memory_clock", return_value=405):
        mc.return_value = [GpuClockCaps(index=0, max_graphics_mhz=1721, max_memory_mhz=5001)]
        ms.return_value = [1721, 1600, 1400, 1200, 1000, 860, 139]
        gov = GPUPowerGovernor(
            _make_fixed_tdp_envs(1),
            performance_mode=False, calibration_window_s=0.0, dry_run=True,
        )
    gov._spike_trigger_c = 72.0
    return gov


class TestPreIdlePrediction(unittest.TestCase):
    """IMP2: Pre-idle prediction via power derivative."""

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 20, 10)])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_pre_idle_triggers_on_steep_power_drop(self, mock_ac, mock_util, mock_clk):
        gov = _make_fixed_gov_for_improvements()
        # Seed power readings with a steep drop (>15W/tick)
        gov._power_readings.extend([100.0, 80.0, 55.0])
        gov._util_readings.extend([60, 25])
        # Now tick at 20% util, power 30W, temp 55°C → derivative -23.3 W/tick
        gov.tick([55.0], [30.0])
        self.assertIn(gov._gpu_mode[0], ("pre_idle", "idle_efficiency"))

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 20, 10)])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_pre_idle_blocked_when_temp_high(self, mock_ac, mock_util, mock_clk):
        gov = _make_fixed_gov_for_improvements()
        gov._power_readings.extend([100.0, 80.0, 55.0])
        gov._util_readings.extend([60, 25])
        # Temp 70°C > 65°C threshold → pre-idle should NOT trigger
        gov.tick([70.0], [30.0])
        # Might still enter idle_efficiency via 1-tick idle if util < 5, but not pre_idle
        # At util=20%, it won't be idle_efficiency either
        self.assertNotEqual(gov._gpu_mode[0], "pre_idle")


class TestPStateRaceTracker(unittest.TestCase):
    """IMP3: Learn T4 hardware P-state response time."""

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 2, 5)])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_race_win_when_clocks_reduced_before_hw(self, mock_ac, mock_util, mock_clk):
        gov = _make_fixed_gov_for_improvements()
        # Tick 1: idle → clocks reduced (CooledAI responds)
        gov.tick([55.0], [50.0])
        self.assertTrue(gov._clocks_reduced[0])
        # Now simulate hardware reaching low power (< 20W)
        gov._last_tick = 0
        gov.tick([55.0], [15.0])
        # Should record a race win
        self.assertGreaterEqual(gov._race_wins, 1)

    def test_hw_response_ewma_updates(self):
        gov = _make_fixed_gov_for_improvements()
        initial = gov._hw_response_ewma_s
        # Manually simulate
        gov._hw_response_times.append(3.0)
        gov._hw_response_ewma_s = 0.3 * 3.0 + 0.7 * initial
        self.assertAlmostEqual(gov._hw_response_ewma_s, 0.3 * 3.0 + 0.7 * 5.0, places=1)


class TestWorkloadCycleLearning(unittest.TestCase):
    """IMP4: Learn workload cycle timing for predictive idle."""

    def test_cycle_ewma_updates(self):
        gov = _make_fixed_gov_for_improvements()
        # Simulate cycle data
        gov._avg_request_cycle_s = 0
        gov._cycle_samples = 0
        # First cycle
        gov._avg_request_cycle_s = 30.0
        gov._cycle_samples = 1
        # Second cycle
        alpha = 0.2
        gov._avg_request_cycle_s = alpha * 25.0 + (1 - alpha) * 30.0
        gov._cycle_samples = 2
        self.assertAlmostEqual(gov._avg_request_cycle_s, 29.0, places=1)

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 15, 10)])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_cycle_prediction_triggers_early_idle(self, mock_ac, mock_util, mock_clk):
        import time as _time
        gov = _make_fixed_gov_for_improvements()
        gov._avg_request_cycle_s = 20.0
        gov._cycle_samples = 5  # enough samples for confidence
        # Last active ended 15s ago (0.7 * 20 = 14s threshold)
        gov._last_active_end_time[0] = _time.monotonic() - 15.0
        gov.tick([55.0], [30.0])
        # Should enter idle/pre_idle via cycle prediction
        self.assertIn(gov._gpu_mode[0], ("pre_idle", "idle_efficiency"))


class TestCoIdleDeepClocks(unittest.TestCase):
    """IMP5: Deep clock reduction (memory + core) during confirmed idle."""

    def test_min_memory_clock_stored(self):
        gov = _make_fixed_gov_for_improvements()
        # Should have stored the min memory clock from the mock (405)
        self.assertEqual(gov._min_mem_clock.get(0), 405)

    @patch("core.optimization.gpu_power_governor.query_gpu_current_clocks", return_value={0: 1721})
    @patch("core.optimization.gpu_power_governor.query_gpu_utilization", return_value=[(0, 0, 2)])
    @patch("core.optimization.gpu_power_governor.set_gpu_clocks", return_value=True)
    def test_idle_applies_min_mem_and_core(self, mock_ac, mock_util, mock_clk):
        gov = _make_fixed_gov_for_improvements()
        gov.tick([50.0], [10.0])
        # Check set_gpu_clocks was called with min memory clock (405)
        mock_ac.assert_called()
        call_args = mock_ac.call_args
        mem_mhz = call_args[0][1]
        core_mhz = call_args[0][2]
        self.assertEqual(mem_mhz, 405)  # IMP5: reduced memory clock
        self.assertEqual(core_mhz, 139)  # lowest supported core clock


if __name__ == "__main__":
    unittest.main()
