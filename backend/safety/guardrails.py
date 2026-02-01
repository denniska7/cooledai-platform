"""
CooledAI Mechanical Resonance Guard - Vibration Protection

Prevents fan RPM from operating in 'danger zones' where servers may vibrate
or resonate, causing mechanical stress and premature failure.

If AI recommends RPM within a resonance zone, the logic snaps to the nearest
safe boundary outside the zone.
"""

import logging
from typing import List, Tuple

_logger = logging.getLogger("cooledai.guardrails.resonance")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)

# RPM ranges where mechanical resonance can occur (low, high) - avoid these zones
# Typical server fan resonance zones based on blade/vendor specs
RESONANCE_ZONES: List[Tuple[float, float]] = [
    (1200, 1350),   # Zone 1: common 1U/2U server resonance
    (2800, 3000),   # Zone 2: high-RPM blade resonance
]

# Offset from zone boundary to safe RPM (stay this far from zone edges)
RESONANCE_SAFETY_MARGIN = 10.0  # RPM


def _get_zone_boundaries(zone: Tuple[float, float]) -> Tuple[float, float]:
    """Return (low_safe, high_safe) boundaries for a zone."""
    low, high = zone
    return (low - RESONANCE_SAFETY_MARGIN, high + RESONANCE_SAFETY_MARGIN)


def snap_rpm_to_safe_boundary(
    rpm: float,
    resonance_zones: List[Tuple[float, float]] = None,
) -> float:
    """
    Snap RPM to nearest safe boundary if it falls within a resonance zone.

    If the recommended RPM is inside a danger zone (e.g., 1200-1350 or 2800-3000),
    returns the nearest safe RPM outside the zone (e.g., 1190 or 1360 for first zone).

    Args:
        rpm: Recommended RPM from AI
        resonance_zones: Optional override for RESONANCE_ZONES

    Returns:
        Safe RPM (unchanged if not in zone, else snapped to nearest boundary)
    """
    zones = resonance_zones or RESONANCE_ZONES
    for zone in zones:
        low, high = zone
        if low <= rpm <= high:
            low_safe, high_safe = _get_zone_boundaries(zone)
            # Snap to nearest boundary
            if rpm - low_safe <= high_safe - rpm:
                snapped = low_safe
            else:
                snapped = high_safe
            _logger.warning(
                "Mechanical Resonance Guard: RPM %.0f in danger zone (%.0f-%.0f). "
                "Snapping to safe boundary %.0f RPM.",
                rpm, low, high, snapped,
            )
            return snapped
    return rpm


def apply_resonance_guard_to_cooling_delta(
    current_cooling_rpm: float,
    recommended_cooling_delta: float,
    max_cooling_rpm: float = 3000.0,
) -> float:
    """
    Apply resonance guard when converting delta to RPM.

    Computes proposed_rpm = current_cooling_rpm * (1 + delta), then snaps
    to safe boundary if in resonance zone. Returns the adjusted delta that
    would produce the safe RPM.

    Args:
        current_cooling_rpm: Current fan/cooling RPM
        recommended_cooling_delta: AI's recommended delta (-1 to 1)
        max_cooling_rpm: Max cooling capacity (for delta=1.0)

    Returns:
        Adjusted recommended_cooling_delta (may differ if resonance snap applied)
    """
    if current_cooling_rpm <= 0:
        return recommended_cooling_delta
    # proposed_rpm from delta: linear interpolation 0 -> current, 1 -> max
    proposed_rpm = current_cooling_rpm + recommended_cooling_delta * (max_cooling_rpm - current_cooling_rpm)
    # Simpler: proposed = current * (1 + delta) with cap at max
    proposed_rpm = min(
        current_cooling_rpm * (1.0 + recommended_cooling_delta),
        max_cooling_rpm,
    )
    proposed_rpm = max(proposed_rpm, 0)
    safe_rpm = snap_rpm_to_safe_boundary(proposed_rpm)
    if abs(safe_rpm - proposed_rpm) < 0.5:
        return recommended_cooling_delta
    # Back-calculate delta for safe_rpm: safe_rpm = current * (1 + d) => d = safe_rpm/current - 1
    if current_cooling_rpm > 0:
        adjusted_delta = (safe_rpm / current_cooling_rpm) - 1.0
        return max(-1.0, min(1.0, adjusted_delta))
    return recommended_cooling_delta
