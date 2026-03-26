"""
In-memory state store for CooledAI optimization pipeline.

Wraps the per-node dict+lock patterns from api/main.py into a clean
StateBackend interface. Used by both the cloud API and the edge gateway
(as a fallback when Redis is unavailable).
"""

import json
import threading
import logging
from dataclasses import asdict as dc_asdict, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from core.optimization.thermal_calibrator import CalibrationProfile

logger = logging.getLogger("cooledai.state_store")


# ---------------------------------------------------------------------------
# StateBackend protocol — pluggable backend for ControlService
# ---------------------------------------------------------------------------

@runtime_checkable
class StateBackend(Protocol):
    """Interface for per-node state persistence.

    Implementations: InMemoryStateStore (this file), RedisStateStore (gateway).
    """

    # Calibration profiles
    def get_calibration_profile(self, node_id: str) -> Optional[CalibrationProfile]: ...
    def set_calibration_profile(self, node_id: str, profile: CalibrationProfile) -> None: ...
    def list_calibration_profiles(self) -> Dict[str, CalibrationProfile]: ...

    # Spike history
    def append_spike(self, node_id: str, ts: float, temp_c: float) -> None: ...
    def get_spike_history(self, node_id: str, max_points: int = 30) -> List[Tuple[float, float]]: ...

    # Thermal history
    def append_thermal_history(self, owner_id: str, row: tuple) -> None: ...
    def get_thermal_history(self, owner_id: str, max_points: int = 30) -> List[tuple]: ...


# ---------------------------------------------------------------------------
# InMemoryStateStore
# ---------------------------------------------------------------------------

class InMemoryStateStore:
    """Thread-safe in-memory state store with optional JSON persistence for thermal history.

    Can accept pre-existing dict objects (shared_*) so api/main.py can share
    its module-level state with the store during the Phase 1 transition.
    When shared dicts are provided, the store operates on the SAME objects.
    """

    def __init__(
        self,
        thermal_history_file: Optional[str] = None,
        max_thermal_history_points: int = 60 * 24 * 60 * 12,  # 60 days at 5s interval
        shared_profiles: Optional[Dict[str, Any]] = None,
        shared_profiles_lock: Optional[threading.Lock] = None,
        shared_spikes: Optional[Dict[str, list]] = None,
        shared_spikes_lock: Optional[threading.Lock] = None,
        shared_history: Optional[Dict[str, list]] = None,
        shared_history_lock: Optional[threading.Lock] = None,
    ):
        # Calibration profiles: node_id -> CalibrationProfile
        self._profiles_lock = shared_profiles_lock or threading.Lock()
        self._profiles: Dict[str, Any] = shared_profiles if shared_profiles is not None else {}

        # Spike history: node_id -> [(ts, temp_c), ...]
        self._spikes_lock = shared_spikes_lock or threading.Lock()
        self._spikes: Dict[str, list] = shared_spikes if shared_spikes is not None else {}

        # Thermal history: owner_id -> [row_tuple, ...]
        self._history_lock = shared_history_lock or threading.Lock()
        self._history: Dict[str, list] = shared_history if shared_history is not None else {}
        self._max_history = max_thermal_history_points

        # Optional disk persistence for thermal history
        self._history_file = Path(thermal_history_file) if thermal_history_file else None

    # --- Calibration profiles ---

    def get_calibration_profile(self, node_id: str) -> Optional[CalibrationProfile]:
        with self._profiles_lock:
            return self._profiles.get(node_id)

    def set_calibration_profile(self, node_id: str, profile: CalibrationProfile) -> None:
        with self._profiles_lock:
            self._profiles[node_id] = profile

    def list_calibration_profiles(self) -> Dict[str, CalibrationProfile]:
        with self._profiles_lock:
            return dict(self._profiles)

    # --- Spike history ---

    def append_spike(self, node_id: str, ts: float, temp_c: float) -> None:
        with self._spikes_lock:
            hist = self._spikes.setdefault(node_id, [])
            hist.append((ts, temp_c))
            if len(hist) > 30:
                self._spikes[node_id] = hist[-30:]

    def get_spike_history(self, node_id: str, max_points: int = 30) -> List[Tuple[float, float]]:
        with self._spikes_lock:
            return list(self._spikes.get(node_id, []))[-max_points:]

    # --- Thermal history ---

    def append_thermal_history(self, owner_id: str, row: tuple) -> None:
        with self._history_lock:
            hist = self._history.setdefault(owner_id, [])
            hist.append(row)
            if len(hist) > self._max_history:
                hist.pop(0)

    def get_thermal_history(self, owner_id: str, max_points: int = 30) -> List[tuple]:
        with self._history_lock:
            hist = self._history.get(owner_id, [])
            return sorted(hist, key=lambda x: x[0])[-max_points:]

    def get_thermal_history_raw(self, owner_id: str) -> list:
        """Return the raw history list reference (for backward compat with api/main.py)."""
        with self._history_lock:
            return self._history.setdefault(owner_id, [])

    def get_all_thermal_history(self) -> Dict[str, list]:
        """Return the full thermal history dict (for backward compat with api/main.py)."""
        with self._history_lock:
            return self._history

    # --- Persistence helpers (used by cloud API for disk backup) ---

    def load_thermal_history_from_file(self, path: Optional[str] = None) -> None:
        """Load thermal history from JSON file on disk."""
        fpath = Path(path) if path else self._history_file
        if not fpath or not fpath.exists():
            return
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
            with self._history_lock:
                for owner_id, rows in raw.items():
                    self._history[owner_id] = [tuple(r) for r in rows]
            logger.info("Loaded thermal history from %s (%d owners)", fpath, len(raw))
        except Exception as e:
            logger.warning("Failed to load thermal history from %s: %s", fpath, e)

    def flush_thermal_history_to_file(self, path: Optional[str] = None) -> None:
        """Flush thermal history to JSON file on disk."""
        fpath = Path(path) if path else self._history_file
        if not fpath:
            return
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = fpath.with_suffix(".tmp")
            with self._history_lock:
                data = {k: [list(r) for r in v] for k, v in self._history.items()}
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(fpath)
        except Exception as e:
            logger.warning("Failed to flush thermal history to %s: %s", fpath, e)

    def load_calibration_from_file(self, path: str) -> None:
        """Load calibration profiles from JSON file on disk."""
        fpath = Path(path)
        if not fpath.exists():
            return
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
            valid_fields = set(CalibrationProfile.__dataclass_fields__)
            with self._profiles_lock:
                for node_id, d in raw.items():
                    try:
                        filtered = {k: v for k, v in d.items() if k in valid_fields}
                        self._profiles[node_id] = CalibrationProfile(**filtered)
                    except Exception as exc:
                        logger.debug("Skipped stale calibration for %s: %s", node_id, exc)
            logger.info("Loaded calibration profiles from %s (%d nodes)", fpath, len(raw))
        except Exception as e:
            logger.warning("Failed to load calibration profiles from %s: %s", fpath, e)

    def flush_calibration_to_file(self, path: str) -> None:
        """Flush calibration profiles to JSON file on disk (atomic write)."""
        fpath = Path(path)
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = fpath.with_suffix(".tmp")
            with self._profiles_lock:
                data = {}
                for node_id, profile in self._profiles.items():
                    try:
                        data[node_id] = dc_asdict(profile)
                    except Exception:
                        pass
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(fpath)
        except Exception as e:
            logger.warning("Failed to flush calibration profiles to %s: %s", fpath, e)
