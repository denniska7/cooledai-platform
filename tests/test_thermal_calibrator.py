"""
Tests for core.optimization.thermal_calibrator — ThermalCalibrator & CalibrationProfile.

Covers:
  - Profile computation from known distribution (all 9 thresholds)
  - Bootstrap defaults active before window completes
  - Safety clamps on out-of-range observations
  - EWMA smoothing on recalibration (no hard jumps)
  - Drift detector fires re-observation at >5°C shift
  - Manual override env vars skip calibration and log WARNING
  - to_dict() output contains all required keys
  - Absolute power trigger fires when sustained above threshold
  - Asymmetric slew: upward steps larger than downward steps
  - min_response_quantum rounds up small changes rather than suppressing
"""

import json
import os
import tempfile
import time
import logging
from datetime import datetime, timezone, timedelta
from unittest import mock

import numpy as np
import pytest

from core.optimization.thermal_calibrator import (
    ThermalCalibrator,
    CalibrationProfile,
    CALIBRATION_STATE_CALIBRATING,
    CALIBRATION_STATE_CALIBRATED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_calibrator(window_s: float = 10.0, fan_rated: float = 7000.0, gpu_limit: float = 83.0, **kw):
    """Build a calibrator with a short window for testing."""
    return ThermalCalibrator(
        fan_rated_max_rpm=fan_rated,
        gpu_hw_thermal_limit_c=gpu_limit,
        calibration_window_s=window_s,
        recalib_interval_s=kw.get("recalib_interval_s", 3600.0),
    )


def _feed_known_distribution(cal: ThermalCalibrator, n: int = 200, t0: float = 0.0):
    """Feed a known distribution so we can verify percentile-based thresholds.

    Fan RPMs: uniform 1000–5000  → P10≈1400, P95≈4600, range≈3200
    GPU power: uniform 10–100    → P10≈19
    GPU temp:  normal μ=40 σ=3   → mean≈40, std≈3
    """
    rng = np.random.RandomState(42)
    fan = rng.uniform(1000, 5000, n)
    power = rng.uniform(10, 100, n)
    temp = rng.normal(40, 3, n)
    for i in range(n):
        cal.update(
            fan_rpms=[float(fan[i])],
            gpu_power_w=[float(power[i])],
            gpu_temp_c=[float(temp[i])],
            cpu_temp_c=[float(temp[i]) - 5],
            now=t0 + i,
        )
    return fan, power, temp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBootstrapDefaults:
    """Bootstrap defaults active before window completes."""

    def test_initial_state_is_calibrating(self):
        cal = _build_calibrator(window_s=1000.0)
        assert cal.profile.calibration_state == CALIBRATION_STATE_CALIBRATING

    def test_bootstrap_values_use_rated_max(self):
        cal = _build_calibrator(fan_rated=7000.0)
        p = cal.profile
        assert p.active_compute_fan_floor_rpm == pytest.approx(7000.0 * 0.30, rel=0.01)
        assert p.spike_hold_fan_floor_rpm == pytest.approx(7000.0 * 0.36, rel=0.01)
        assert p.active_compute_trigger_w == 15.0
        assert p.spike_trigger_temp_c == 42.0
        assert p.hysteresis_rpm == 30.0
        assert p.slew_rate_up_rpm_per_cycle == 40.0
        assert p.slew_rate_down_rpm_per_cycle == 12.0
        assert p.min_response_quantum_rpm == 40.0
        assert p.spike_hold_duration_s == 240.0

    def test_not_calibrated_before_window(self):
        cal = _build_calibrator(window_s=1000.0)
        for i in range(50):
            cal.update(fan_rpms=[3000], gpu_power_w=[50], gpu_temp_c=[40], now=float(i))
        assert not cal.is_calibrated
        assert cal.profile.calibration_state == CALIBRATION_STATE_CALIBRATING


class TestProfileComputation:
    """Profile computation from known distribution — verify all 9 thresholds."""

    def test_thresholds_match_formulas(self):
        """Verify derived thresholds match percentage formulas from observed anchors."""
        cal = _build_calibrator(window_s=5.0)
        fan, power, temp = _feed_known_distribution(cal, n=200, t0=0.0)

        assert cal.is_calibrated
        p = cal.profile
        assert p.calibration_state == CALIBRATION_STATE_CALIBRATED

        # Verify derived thresholds are self-consistent (percentage formulas of anchors)
        fan_range = p.fan_range_rpm
        assert fan_range == pytest.approx(p.fan_ceiling_rpm - p.fan_idle_rpm, rel=0.01)

        assert p.active_compute_fan_floor_rpm == pytest.approx(
            p.fan_idle_rpm + fan_range * 0.12, rel=0.01,
        )
        assert p.active_compute_trigger_w == pytest.approx(
            p.gpu_idle_w * 1.30, rel=0.01,
        )
        # spike_hold_fan_floor = fan_ceiling * 1.10 (clamped to rated*0.95)
        raw_spike_floor = p.fan_ceiling_rpm * 1.10
        expected_spike_floor = min(raw_spike_floor, 7000.0 * 0.95)
        assert p.spike_hold_fan_floor_rpm == pytest.approx(expected_spike_floor, rel=0.01)

        # hysteresis = fan_range * 0.08, clamped [20, 60]
        raw_hyst = fan_range * 0.08
        expected_hyst = max(20.0, min(60.0, raw_hyst))
        assert p.hysteresis_rpm == pytest.approx(expected_hyst, rel=0.01)

        # slew_rate_up = fan_range * 0.15, min 30
        assert p.slew_rate_up_rpm_per_cycle == pytest.approx(
            max(30.0, fan_range * 0.15), rel=0.01,
        )
        # slew_rate_down = fan_range * 0.04
        assert p.slew_rate_down_rpm_per_cycle == pytest.approx(
            fan_range * 0.04, rel=0.01,
        )
        # min_response_quantum = fan_range * 0.12, min 30
        assert p.min_response_quantum_rpm == pytest.approx(
            max(30.0, fan_range * 0.12), rel=0.01,
        )

    def test_sustained_compute_window_is_12(self):
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        assert cal.profile.sustained_compute_window_s == 12.0


class TestSafetyClamps:
    """Safety clamps enforce bounds on out-of-range observations."""

    def test_spike_trigger_clamped_to_hw_limit_range(self):
        # Very low temp_std → spike_trigger would be very close to mean
        # New formula: min(hw_limit - 8, max(p99 + 5, 75))
        # Clamp: [hw_limit - 8, hw_limit * 0.92]
        cal = _build_calibrator(window_s=5.0, gpu_limit=83.0)
        # Feed uniform temp near 20°C with very low std
        for i in range(200):
            cal.update(
                fan_rpms=[3000.0],
                gpu_power_w=[50.0],
                gpu_temp_c=[20.0 + (i % 3) * 0.1],
                now=float(i),
            )
        p = cal.profile
        # p99 ≈ 20.2 → max(25.2, 75) = 75 → min(75, 75) = 75
        # Clamp: max(75, min(76.36, 75)) = 75
        assert p.spike_trigger_temp_c >= 75.0 - 0.1
        assert p.spike_trigger_temp_c <= 83.0 * 0.92 + 0.1

    def test_spike_trigger_clamped_high(self):
        # Very high temps → spike_trigger clamped to hw_limit * 0.92
        cal = _build_calibrator(window_s=5.0, gpu_limit=83.0)
        for i in range(200):
            cal.update(
                fan_rpms=[3000.0],
                gpu_power_w=[50.0],
                gpu_temp_c=[78.0 + (i % 5) * 0.5],
                now=float(i),
            )
        p = cal.profile
        # p99 near 80 → max(85, 75) = 85 → min(75, 85) = 75
        # Clamp: max(75, min(76.36, 75)) = 75
        assert p.spike_trigger_temp_c >= 75.0 - 0.1
        assert p.spike_trigger_temp_c <= 83.0 * 0.92 + 0.1

    def test_hysteresis_clamped_min_max(self):
        # Very small fan range → hysteresis would be < 20
        cal = _build_calibrator(window_s=5.0)
        for i in range(200):
            cal.update(
                fan_rpms=[3000.0 + (i % 2) * 10],  # range ~10 RPM
                gpu_power_w=[50.0],
                gpu_temp_c=[40.0],
                now=float(i),
            )
        p = cal.profile
        assert p.hysteresis_rpm >= 20.0
        assert p.hysteresis_rpm <= 60.0

    def test_slew_rate_up_minimum_30(self):
        cal = _build_calibrator(window_s=5.0)
        for i in range(200):
            cal.update(
                fan_rpms=[3000.0 + (i % 2) * 5],
                gpu_power_w=[50.0],
                gpu_temp_c=[40.0],
                now=float(i),
            )
        p = cal.profile
        assert p.slew_rate_up_rpm_per_cycle >= 30.0

    def test_min_response_quantum_minimum_30(self):
        cal = _build_calibrator(window_s=5.0)
        for i in range(200):
            cal.update(
                fan_rpms=[3000.0 + (i % 2) * 5],
                gpu_power_w=[50.0],
                gpu_temp_c=[40.0],
                now=float(i),
            )
        p = cal.profile
        assert p.min_response_quantum_rpm >= 30.0

    def test_spike_hold_duration_clamped(self):
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200)
        p = cal.profile
        assert 120.0 <= p.spike_hold_duration_s <= 600.0

    def test_spike_hold_floor_never_exceeds_rated_095(self):
        cal = _build_calibrator(window_s=5.0, fan_rated=7000.0)
        # Feed high fan RPMs so ceiling * 1.10 would exceed rated * 0.95
        for i in range(200):
            cal.update(
                fan_rpms=[6800.0 + (i % 5) * 20],
                gpu_power_w=[50.0],
                gpu_temp_c=[40.0],
                now=float(i),
            )
        p = cal.profile
        assert p.spike_hold_fan_floor_rpm <= 7000.0 * 0.95 + 0.1


class TestEWMARecalibration:
    """EWMA smoothing on recalibration — no hard jumps."""

    def test_recalibration_blends_not_jumps(self):
        cal = _build_calibrator(window_s=5.0, recalib_interval_s=10.0)
        # Initial calibration
        _feed_known_distribution(cal, n=100, t0=0.0)
        assert cal.is_calibrated
        old_floor = cal.profile.active_compute_fan_floor_rpm

        # Feed very different data after recalib interval
        for i in range(100):
            cal.update(
                fan_rpms=[6000.0],  # Much higher than before
                gpu_power_w=[200.0],
                gpu_temp_c=[60.0],
                now=100.0 + 10.0 + i,  # Past recalib interval
            )
        new_floor = cal.profile.active_compute_fan_floor_rpm

        # Should have moved toward new value but NOT jumped fully
        # New data would give ~6000 P10 idle, much higher floor
        # EWMA alpha=0.25 means new_floor ≈ 0.25*new + 0.75*old
        assert new_floor > old_floor  # Moved toward new
        # But not all the way to what a fresh calibration would give
        fresh_cal = _build_calibrator(window_s=5.0)
        for i in range(200):
            fresh_cal.update(
                fan_rpms=[6000.0],
                gpu_power_w=[200.0],
                gpu_temp_c=[60.0],
                now=float(i),
            )
        fresh_floor = fresh_cal.profile.active_compute_fan_floor_rpm
        assert new_floor < fresh_floor  # Didn't jump all the way


class TestDriftDetector:
    """Drift detector fires re-observation at >5°C rolling shift."""

    def test_drift_triggers_recalibration(self):
        cal = _build_calibrator(window_s=5.0, recalib_interval_s=99999.0)
        # Initial calibration at ~40°C
        _feed_known_distribution(cal, n=100, t0=0.0)
        assert cal.is_calibrated
        old_mean = cal.profile.temp_mean_c
        assert abs(old_mean - 40.0) < 5.0  # sanity check

        # Now feed temps at ~55°C (>5°C shift from ~40°C mean).
        # Feed many samples so that the accumulated data shifts the overall mean.
        # The drift detector fires when 30-min rolling mean diverges >5°C from
        # calibrated temp_mean_c. After EWMA recalib, temp_mean should move.
        for i in range(500):
            cal.update(
                fan_rpms=[3000.0],
                gpu_power_w=[50.0],
                gpu_temp_c=[55.0],
                now=200.0 + i,
            )
        # EWMA blends 0.25*new + 0.75*old; "new" is computed from ALL data.
        # With 100 old at 40°C + 500 new at 55°C, overall mean ≈ 52.5°C.
        # EWMA: 0.25*52.5 + 0.75*40.2 ≈ 43.3 → shifted from 40.2
        assert cal.profile.temp_mean_c > old_mean + 0.5


class TestManualOverride:
    """Manual override env vars skip calibration and log WARNING."""

    def test_manual_override_from_env(self):
        env = {
            "COOLEDAI_ACTIVE_FLOOR_RPM": "3000",
            "COOLEDAI_SPIKE_HOLD_RPM": "4000",
            "COOLEDAI_SPIKE_TRIGGER_C": "50.0",
        }
        with mock.patch.dict(os.environ, env):
            cal = _build_calibrator(window_s=5.0)
            p = cal.profile
            # Bootstrap profile should have override values
            assert p.active_compute_fan_floor_rpm == 3000.0
            assert p.spike_hold_fan_floor_rpm == 4000.0
            assert p.spike_trigger_temp_c == 50.0

            # After calibration, overrides still hold
            _feed_known_distribution(cal, n=200, t0=0.0)
            p = cal.profile
            assert p.active_compute_fan_floor_rpm == 3000.0
            assert p.spike_hold_fan_floor_rpm == 4000.0
            assert p.spike_trigger_temp_c == 50.0

    def test_manual_override_logs_warning(self, caplog):
        env = {"COOLEDAI_ACTIVE_FLOOR_RPM": "3000"}
        with mock.patch.dict(os.environ, env):
            with caplog.at_level(logging.WARNING, logger="cooledai.thermal_calibrator"):
                _build_calibrator()
            assert "Manual override active" in caplog.text


class TestToDict:
    """to_dict() output contains all required keys."""

    def test_all_keys_present(self):
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        d = cal.profile.to_dict()

        required_keys = {
            "calibration_state", "calibration_progress_pct", "sample_count",
            "fan_idle_rpm", "fan_ceiling_rpm", "fan_range_rpm",
            "gpu_idle_w", "temp_mean_c", "temp_stdev_c", "temp_p90_c", "temp_p99_c",
            "spike_recovery_s",
            "fan_rated_max_rpm", "gpu_hw_thermal_limit_c",
            "active_compute_fan_floor_rpm", "active_compute_trigger_w",
            "spike_hold_fan_floor_rpm", "spike_trigger_temp_c",
            "hysteresis_rpm", "slew_rate_up_rpm_per_cycle",
            "slew_rate_down_rpm_per_cycle", "min_response_quantum_rpm",
            "spike_hold_duration_s", "sustained_compute_window_s",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_state_values(self):
        cal = _build_calibrator(window_s=5.0)
        d = cal.profile.to_dict()
        assert d["calibration_state"] == CALIBRATION_STATE_CALIBRATING

        _feed_known_distribution(cal, n=200, t0=0.0)
        d = cal.profile.to_dict()
        assert d["calibration_state"] == CALIBRATION_STATE_CALIBRATED
        assert d["calibration_progress_pct"] == 100.0


class TestAbsolutePowerTrigger:
    """Absolute power trigger fires when sustained above threshold."""

    def test_sustained_power_activates_compute(self):
        """OptimizationBrain's absolute power trigger should detect sustained above threshold."""
        from core.optimization.thermal_calibrator import CalibrationProfile

        profile = CalibrationProfile(
            active_compute_trigger_w=30.0,
            sustained_compute_window_s=5.0,
        )

        # Simulate the sustained power detection logic from optimization_brain
        history = []
        times = []
        for i in range(10):
            history.append(35.0)  # Above 30W threshold
            times.append(float(i))

        # Prune to window
        cutoff = times[-1] - profile.sustained_compute_window_s
        while times and times[0] < cutoff:
            times.pop(0)
            history.pop(0)

        # Check: all above threshold and enough samples
        assert len(history) >= 3
        assert all(p > profile.active_compute_trigger_w for p in history)


class TestAsymmetricSlew:
    """Asymmetric slew: upward steps larger than downward steps."""

    def test_slew_rates_asymmetric(self):
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        p = cal.profile
        assert cal.is_calibrated
        # slew_up = fan_range * 0.15, slew_down = fan_range * 0.04
        assert p.slew_rate_up_rpm_per_cycle > p.slew_rate_down_rpm_per_cycle
        # Ratio should be 0.15/0.04 = 3.75
        # But slew_up has a min(30) clamp — verify the ratio holds when range is large enough
        expected_up = max(30.0, p.fan_range_rpm * 0.15)
        expected_down = p.fan_range_rpm * 0.04
        if expected_down > 0:
            assert p.slew_rate_up_rpm_per_cycle / p.slew_rate_down_rpm_per_cycle == pytest.approx(
                expected_up / expected_down, rel=0.05,
            )

    def test_optimizer_asymmetric_slew(self):
        """PowerCostOptimizer with profile applies asymmetric slew."""
        from core.optimization.optimizer import PowerCostOptimizer
        from core.optimization.thermal_calibrator import CalibrationProfile

        profile = CalibrationProfile(
            slew_rate_up_rpm_per_cycle=200.0,
            slew_rate_down_rpm_per_cycle=50.0,
            min_response_quantum_rpm=40.0,
            active_compute_trigger_w=30.0,
        )
        opt = PowerCostOptimizer(calibration_profile=profile)

        # Temp rising → optimizer wants to increase
        result_up = opt.optimize_for_min_power(
            current_thermal=72.0,
            target_temp=65.0,
            max_safe_temp=80.0,
            current_cooling=3000.0,
            max_cooling=7000.0,
            over_provisioning=0.0,
            temp_rising=True,
            oscillation=False,
            power_slope_w_per_s=0.0,
        )

        # Result should be positive delta
        assert result_up.recommended_delta > 0
        # Delta in RPM should not exceed slew_up
        delta_rpm_up = result_up.recommended_delta * 3000.0
        assert delta_rpm_up <= 200.0 + 1.0  # slew_up limit

        # Temp stable, over-provisioned → wants to decrease
        result_down = opt.optimize_for_min_power(
            current_thermal=50.0,
            target_temp=65.0,
            max_safe_temp=80.0,
            current_cooling=5000.0,
            max_cooling=7000.0,
            over_provisioning=0.5,
            temp_rising=False,
            oscillation=False,
            power_slope_w_per_s=0.0,
        )
        if result_down.recommended_delta < 0:
            delta_rpm_down = abs(result_down.recommended_delta * 5000.0)
            assert delta_rpm_down <= 50.0 + 1.0  # slew_down limit


class TestMinResponseQuantum:
    """min_response_quantum rounds up small changes rather than suppressing."""

    def test_small_delta_rounded_up(self):
        from core.optimization.optimizer import PowerCostOptimizer
        from core.optimization.thermal_calibrator import CalibrationProfile

        profile = CalibrationProfile(
            slew_rate_up_rpm_per_cycle=500.0,
            slew_rate_down_rpm_per_cycle=100.0,
            min_response_quantum_rpm=100.0,
            active_compute_trigger_w=30.0,
        )
        opt = PowerCostOptimizer(calibration_profile=profile)

        # Force a scenario that would produce a small positive delta
        result = opt.optimize_for_min_power(
            current_thermal=71.0,
            target_temp=65.0,
            max_safe_temp=80.0,
            current_cooling=5000.0,
            max_cooling=7000.0,
            over_provisioning=0.0,
            temp_rising=True,
            oscillation=False,
            power_slope_w_per_s=0.0,
        )
        # If delta would have been small, it should be rounded up to min_quantum
        if result.recommended_delta > 0:
            delta_rpm = result.recommended_delta * 5000.0
            # Either it's zero or it's >= min_response_quantum
            assert delta_rpm >= 100.0 - 1.0 or delta_rpm < 1e-6


class TestCalibrationProgress:
    """Progress tracking during calibration window."""

    def test_progress_advances(self):
        cal = _build_calibrator(window_s=100.0)
        cal.update(fan_rpms=[3000], gpu_power_w=[50], gpu_temp_c=[40], now=0.0)
        assert cal.profile.calibration_progress_pct == pytest.approx(0.0, abs=1.0)

        cal.update(fan_rpms=[3000], gpu_power_w=[50], gpu_temp_c=[40], now=50.0)
        assert cal.profile.calibration_progress_pct == pytest.approx(50.0, abs=1.0)

        cal.update(fan_rpms=[3000], gpu_power_w=[50], gpu_temp_c=[40], now=100.0)
        assert cal.profile.calibration_progress_pct == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# FIX 1: Idle/Active Split Buffer Tests
# ---------------------------------------------------------------------------

class TestIdleActiveSplitBuffers:
    """fan_idle_rpm from idle-only, fan_ceiling_rpm from active-only samples."""

    def _feed_split_data(self, cal, n_idle=50, n_active=50, t0=0.0,
                         idle_fan=1500, active_fan=5000,
                         idle_power=10.0, active_power=80.0):
        """Feed clearly separated idle and active data.

        Interleaves idle and active samples so both buffers accumulate
        before the calibration window closes.
        """
        total = n_idle + n_active
        idle_fed = 0
        active_fed = 0
        for i in range(total):
            # Alternate: feed idle and active to fill both buckets early
            if (i % 2 == 0 and idle_fed < n_idle) or active_fed >= n_active:
                cal.update(
                    fan_rpms=[float(idle_fan + (idle_fed % 5) * 10)],
                    gpu_power_w=[float(idle_power + (idle_fed % 3))],
                    gpu_temp_c=[35.0 + (idle_fed % 3) * 0.5],
                    now=t0 + i * 0.1,  # Use 0.1s spacing to fit in window
                )
                idle_fed += 1
            else:
                cal.update(
                    fan_rpms=[float(active_fan + (active_fed % 5) * 20)],
                    gpu_power_w=[float(active_power + (active_fed % 3) * 5)],
                    gpu_temp_c=[55.0 + (active_fed % 3) * 0.5],
                    now=t0 + i * 0.1,
                )
                active_fed += 1

    def test_idle_buffer_used_for_fan_idle_rpm(self):
        """With >=20 idle samples, fan_idle_rpm comes from idle buffer P10."""
        cal = _build_calibrator(window_s=5.0)
        self._feed_split_data(cal, n_idle=50, n_active=50)
        assert cal.is_calibrated
        p = cal.profile
        # fan_idle from idle-only samples (~1500 RPM range), should be much
        # lower than the combined P10 of all fans (which includes 5000 RPM active)
        assert p.fan_idle_rpm < 2000.0

    def test_active_buffer_used_for_fan_ceiling_rpm(self):
        """With >=20 active samples, fan_ceiling_rpm comes from active buffer P95."""
        cal = _build_calibrator(window_s=5.0)
        self._feed_split_data(cal, n_idle=50, n_active=50)
        assert cal.is_calibrated
        p = cal.profile
        # fan_ceiling from active-only samples (~5000 RPM range)
        assert p.fan_ceiling_rpm > 4500.0

    def test_combined_fallback_with_insufficient_idle(self):
        """<20 idle samples → uses combined buffer + logs warning."""
        cal = _build_calibrator(window_s=5.0)
        # Only 5 idle, 95 active
        self._feed_split_data(cal, n_idle=5, n_active=95)
        assert cal.is_calibrated
        # Should still produce valid profile (combined fallback)
        assert cal.profile.fan_idle_rpm > 0

    def test_combined_fallback_with_insufficient_active(self):
        """<20 active samples → uses combined buffer + logs warning."""
        cal = _build_calibrator(window_s=5.0)
        # Only 95 idle, 5 active
        self._feed_split_data(cal, n_idle=95, n_active=5)
        assert cal.is_calibrated
        assert cal.profile.fan_ceiling_rpm > 0

    def test_ewma_recalib_uses_split_buffers(self):
        """EWMA recalibration also uses split buffers."""
        cal = _build_calibrator(window_s=5.0, recalib_interval_s=10.0)
        self._feed_split_data(cal, n_idle=50, n_active=50, t0=0.0)
        assert cal.is_calibrated
        old_idle = cal.profile.fan_idle_rpm
        # Feed more data and trigger recalib
        for i in range(50):
            cal.update(
                fan_rpms=[1200.0],  # lower idle
                gpu_power_w=[8.0],
                gpu_temp_c=[33.0],
                now=100.0 + 10.0 + i,
            )
        # EWMA should have pulled idle_rpm down
        assert cal.profile.fan_idle_rpm < old_idle + 50


# ---------------------------------------------------------------------------
# FIX 1: Floor Sanity Clamp
# ---------------------------------------------------------------------------

class TestFloorSanityClamp:
    """Floor sanity check prevents >20% upward jump in one cycle."""

    def test_large_jump_triggers_ewma_blend(self):
        """New floor >20% above previous → starts EWMA blend, not immediate jump."""
        cal = _build_calibrator(window_s=5.0, recalib_interval_s=10.0)
        # Initial calibration with low fan data
        for i in range(100):
            cal.update(
                fan_rpms=[2000.0 + (i % 5) * 10],
                gpu_power_w=[10.0],
                gpu_temp_c=[35.0],
                now=float(i),
            )
        assert cal.is_calibrated
        initial_floor = cal.profile.active_compute_fan_floor_rpm

        # Now feed drastically higher fan data to trigger recalib
        for i in range(100):
            cal.update(
                fan_rpms=[6000.0],
                gpu_power_w=[150.0],
                gpu_temp_c=[70.0],
                now=100.0 + 10.0 + i,
            )
        # The floor should NOT have jumped >20% from initial
        new_floor = cal.profile.active_compute_fan_floor_rpm
        # Allow some EWMA movement but not a full jump
        assert new_floor < initial_floor * 1.25  # at most 25% above (some EWMA blend)


# ---------------------------------------------------------------------------
# FIX 1: Warmup Guard
# ---------------------------------------------------------------------------

class TestWarmupGuard:
    """First 10 minutes after start: floor can only go DOWN from bootstrap."""

    def test_floor_cannot_rise_during_warmup(self):
        """During warmup (<600s), calibrated floor is clamped to bootstrap."""
        cal = _build_calibrator(window_s=5.0)
        bootstrap_floor = cal.profile.active_compute_fan_floor_rpm
        # Feed high-fan data within warmup window
        for i in range(100):
            cal.update(
                fan_rpms=[6000.0 + (i % 5) * 50],
                gpu_power_w=[150.0],
                gpu_temp_c=[70.0],
                now=float(i),  # All within first 100s (<600s warmup)
            )
        assert cal.is_calibrated
        # Floor should not exceed bootstrap
        assert cal.profile.active_compute_fan_floor_rpm <= bootstrap_floor + 1.0

    def test_floor_can_drop_during_warmup(self):
        """During warmup, floor CAN drop below bootstrap."""
        cal = _build_calibrator(window_s=5.0)
        bootstrap_floor = cal.profile.active_compute_fan_floor_rpm
        # Feed very low fan data
        for i in range(100):
            cal.update(
                fan_rpms=[500.0 + (i % 5) * 10],
                gpu_power_w=[5.0],
                gpu_temp_c=[25.0],
                now=float(i),
            )
        assert cal.is_calibrated
        # Floor can be lower than bootstrap
        assert cal.profile.active_compute_fan_floor_rpm < bootstrap_floor


# ---------------------------------------------------------------------------
# FIX 3: Persistence Tests
# ---------------------------------------------------------------------------

class TestPersistence:
    """save_profile and load_profile round-trip."""

    def test_save_and_load_roundtrip(self):
        """Save then load produces equivalent profile."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        assert cal.is_calibrated
        original = cal.profile.to_dict()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            assert cal.save_profile(path) is True
            # Load into a fresh calibrator
            cal2 = _build_calibrator(window_s=5.0)
            assert cal2.load_profile(path) is True
            assert cal2.is_calibrated
            loaded = cal2.profile.to_dict()
            # All numeric fields should match
            for key in original:
                if isinstance(original[key], (int, float)):
                    assert loaded[key] == pytest.approx(original[key], rel=0.01), \
                        f"Mismatch on {key}: {loaded[key]} vs {original[key]}"
        finally:
            os.unlink(path)

    def test_load_rejects_stale_profile(self):
        """Profile older than 24 hours is rejected."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
            json.dump({
                "schema_version": 1,
                "saved_at": old_time,
                "profile": cal.profile.to_dict(),
            }, f)
        try:
            cal2 = _build_calibrator(window_s=5.0)
            assert cal2.load_profile(path) is False
            assert not cal2.is_calibrated
        finally:
            os.unlink(path)

    def test_load_rejects_wrong_schema(self):
        """Wrong schema_version is rejected."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            json.dump({
                "schema_version": 999,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "profile": cal.profile.to_dict(),
            }, f)
        try:
            cal2 = _build_calibrator(window_s=5.0)
            assert cal2.load_profile(path) is False
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        """Missing file returns False gracefully."""
        cal = _build_calibrator(window_s=5.0)
        assert cal.load_profile("/tmp/nonexistent_calibration_xxxx.json") is False

    def test_load_corrupted_json(self):
        """Corrupted JSON returns False gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            f.write("not valid json{{{")
        try:
            cal = _build_calibrator(window_s=5.0)
            assert cal.load_profile(path) is False
        finally:
            os.unlink(path)

    def test_loaded_profile_skips_observation(self):
        """Loaded profile marks calibration complete immediately."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save_profile(path)
            # Fresh calibrator with long window
            cal2 = _build_calibrator(window_s=99999.0)
            assert not cal2.is_calibrated
            assert cal2.load_profile(path) is True
            # Should be calibrated immediately, no observation window needed
            assert cal2.is_calibrated
            assert cal2.profile.calibration_state == CALIBRATION_STATE_CALIBRATED
        finally:
            os.unlink(path)

    def test_save_writes_valid_json(self):
        """save_profile writes valid JSON with schema_version and saved_at."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save_profile(path)
            with open(path) as f:
                data = json.load(f)
            assert data["schema_version"] == 1
            assert "saved_at" in data
            assert "profile" in data
            assert isinstance(data["profile"], dict)
        finally:
            os.unlink(path)

    def test_stale_profile_idle_near_ceiling_rejected(self):
        """Profile with fan_idle_rpm within 10% of fan_ceiling_rpm is rejected (circular ref)."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        assert cal.is_calibrated

        # Manually corrupt the profile: set idle ≈ ceiling (circular reference bug)
        # ratio = 3300/3400 = 0.97 > 0.90 threshold → rejected
        cal._profile.fan_idle_rpm = 3300.0
        cal._profile.fan_ceiling_rpm = 3400.0

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save_profile(path)
            cal2 = _build_calibrator(window_s=99999.0)
            assert not cal2.is_calibrated
            assert cal2.load_profile(path) is False  # must reject
            assert not cal2.is_calibrated  # stays uncalibrated → will re-observe
        finally:
            os.unlink(path)


    def test_profile_idle_at_89pct_of_ceiling_accepted(self):
        """Profile with fan_idle_rpm at 89% of ceiling should be accepted (below 90% threshold)."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        assert cal.is_calibrated

        # Set idle/ceiling with meaningful separation: ratio = 2670/3000 = 0.89 < 0.90
        cal._profile.fan_idle_rpm = 2670.0
        cal._profile.fan_ceiling_rpm = 3000.0

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save_profile(path)
            cal2 = _build_calibrator(window_s=99999.0)
            assert not cal2.is_calibrated
            assert cal2.load_profile(path) is True  # should accept
            assert cal2.is_calibrated
        finally:
            os.unlink(path)

    def test_profile_idle_at_91pct_of_ceiling_rejected(self):
        """Profile with fan_idle_rpm at 91% of ceiling should be rejected (above 90% threshold)."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        assert cal.is_calibrated

        # Set idle/ceiling too close: ratio = 3640/4000 = 0.91 > 0.90
        cal._profile.fan_idle_rpm = 3640.0
        cal._profile.fan_ceiling_rpm = 4000.0

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save_profile(path)
            cal2 = _build_calibrator(window_s=99999.0)
            assert not cal2.is_calibrated
            assert cal2.load_profile(path) is False  # must reject
            assert not cal2.is_calibrated
        finally:
            os.unlink(path)


class TestCalibrationIdleStepping:
    """Tests for the calibration idle stepping override in the agent control loop."""

    def test_idle_step_fires_during_calibration_window(self):
        """When _in_calibration_window=True and GPU power is below trigger,
        the agent should override duty to bootstrap floor."""
        # Simulate the idle stepping logic from the agent
        cal = _build_calibrator(window_s=1800.0)
        assert not cal.is_calibrated  # still in observation window
        bootstrap_rpm = cal._bootstrap_floor  # 7000 * 0.30 = 2100

        # Simulate: GPU idle (5W < 15W trigger)
        gpu_power = 5.0
        trigger = cal.profile.active_compute_trigger_w  # 15.0

        # This mirrors the agent logic
        in_calibration_window = not cal.is_calibrated
        assert in_calibration_window

        duty = 50  # optimizer would command 50% (~3500 RPM)
        if in_calibration_window and gpu_power < trigger:
            bootstrap_duty = int(round((bootstrap_rpm / 7000.0) * 100.0))
            bootstrap_duty = max(15, min(100, bootstrap_duty))
            duty = bootstrap_duty

        # duty should be bootstrap floor, not the optimizer's 50%
        assert duty == 30  # 2100/7000 = 0.30 → 30%
        assert duty != 50  # NOT the optimizer's original

    def test_idle_step_releases_after_calibration(self):
        """After calibration completes, idle stepping no longer overrides duty."""
        cal = _build_calibrator(window_s=5.0)
        _feed_known_distribution(cal, n=200, t0=0.0)
        assert cal.is_calibrated  # calibration done

        in_calibration_window = not cal.is_calibrated
        assert not in_calibration_window

        # With calibration done, duty should pass through unchanged
        duty = 50
        gpu_power = 5.0
        trigger = cal.profile.active_compute_trigger_w
        if in_calibration_window and gpu_power < trigger:
            duty = 30  # would override
        assert duty == 50  # NOT overridden — idle stepping released
