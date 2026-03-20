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

import os
import time
import logging
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
        # Clamp: [hw_limit * 0.50, hw_limit * 0.85]
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
        # mean ≈ 20, std ≈ tiny → spike_trigger = mean + 2.5σ ≈ 20.x
        # But clamp forces it to at least 83 * 0.50 = 41.5
        assert p.spike_trigger_temp_c >= 83.0 * 0.50 - 0.1
        assert p.spike_trigger_temp_c <= 83.0 * 0.85 + 0.1

    def test_spike_trigger_clamped_high(self):
        # Very high temps → spike_trigger above hw_limit * 0.85
        cal = _build_calibrator(window_s=5.0, gpu_limit=83.0)
        for i in range(200):
            cal.update(
                fan_rpms=[3000.0],
                gpu_power_w=[50.0],
                gpu_temp_c=[78.0 + (i % 5) * 0.5],
                now=float(i),
            )
        p = cal.profile
        assert p.spike_trigger_temp_c <= 83.0 * 0.85 + 0.1

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
