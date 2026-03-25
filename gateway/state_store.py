"""
Redis-backed state store for the CooledAI edge gateway.

Provides persistence across gateway restarts and enables HA replication.
Falls back gracefully to InMemoryStateStore if Redis is unavailable.

Key design:
  calibration_profiles   Hash   node_id -> CalibrationProfile JSON   No TTL
  spike_hold:{key}       String Expiry timestamp                     Auto-expire
  safety_contexts        Hash   node_id -> SafetyContext JSON         No TTL
  telemetry:{node_id}    List   Ring buffer (100 entries via LTRIM)   No TTL
  gateway:config         Hash   Cloud-synced config                   60s refresh
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from core.optimization.in_memory_store import InMemoryStateStore
from core.optimization.thermal_calibrator import CalibrationProfile

logger = logging.getLogger("cooledai.gateway.state_store")


class RedisStateStore:
    """Redis-backed state store with in-memory fallback.

    Implements the same interface as InMemoryStateStore (StateBackend protocol).
    If Redis is unavailable at init or at runtime, all operations fall back
    to an in-memory store with a warning log.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._fallback = InMemoryStateStore()
        self._redis = None
        self._available = False

        try:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._available = True
            logger.info("RedisStateStore: connected to %s", redis_url)
        except ImportError:
            logger.warning("RedisStateStore: redis package not installed — using in-memory fallback")
        except Exception as e:
            logger.warning("RedisStateStore: Redis unavailable (%s) — using in-memory fallback", e)

    # --- Calibration Profiles ---

    def get_calibration_profile(self, node_id: str) -> Optional[CalibrationProfile]:
        if self._available:
            try:
                data = self._redis.hget("calibration_profiles", node_id)
                if data:
                    d = json.loads(data)
                    return CalibrationProfile(**{
                        k: v for k, v in d.items()
                        if k in CalibrationProfile.__dataclass_fields__
                    })
            except Exception as e:
                logger.debug("Redis get_calibration_profile failed: %s", e)
        return self._fallback.get_calibration_profile(node_id)

    def set_calibration_profile(self, node_id: str, profile: CalibrationProfile) -> None:
        # Always write to fallback (in-memory)
        self._fallback.set_calibration_profile(node_id, profile)
        # Write-through to Redis
        if self._available:
            try:
                data = profile.to_dict() if hasattr(profile, "to_dict") else vars(profile)
                self._redis.hset("calibration_profiles", node_id, json.dumps(data, default=str))
            except Exception as e:
                logger.debug("Redis set_calibration_profile failed: %s", e)

    def list_calibration_profiles(self) -> Dict[str, CalibrationProfile]:
        if self._available:
            try:
                all_data = self._redis.hgetall("calibration_profiles")
                result = {}
                for node_id, raw in all_data.items():
                    try:
                        d = json.loads(raw)
                        result[node_id] = CalibrationProfile(**{
                            k: v for k, v in d.items()
                            if k in CalibrationProfile.__dataclass_fields__
                        })
                    except Exception:
                        pass
                return result
            except Exception as e:
                logger.debug("Redis list_calibration_profiles failed: %s", e)
        return self._fallback.list_calibration_profiles()

    # --- Spike History ---

    def append_spike(self, node_id: str, ts: float, temp_c: float) -> None:
        self._fallback.append_spike(node_id, ts, temp_c)
        if self._available:
            try:
                entry = json.dumps({"ts": ts, "temp_c": temp_c})
                key = f"spike_history:{node_id}"
                self._redis.lpush(key, entry)
                self._redis.ltrim(key, 0, 29)  # Keep last 30
            except Exception as e:
                logger.debug("Redis append_spike failed: %s", e)

    def get_spike_history(self, node_id: str, max_points: int = 30) -> List[Tuple[float, float]]:
        if self._available:
            try:
                key = f"spike_history:{node_id}"
                raw = self._redis.lrange(key, 0, max_points - 1)
                result = []
                for item in reversed(raw):  # LPUSH stores newest first; reverse for chronological
                    d = json.loads(item)
                    result.append((d["ts"], d["temp_c"]))
                return result
            except Exception as e:
                logger.debug("Redis get_spike_history failed: %s", e)
        return self._fallback.get_spike_history(node_id, max_points)

    # --- Thermal History ---

    def append_thermal_history(self, owner_id: str, row: tuple) -> None:
        self._fallback.append_thermal_history(owner_id, row)
        if self._available:
            try:
                key = f"thermal_history:{owner_id}"
                self._redis.lpush(key, json.dumps(list(row), default=str))
                self._redis.ltrim(key, 0, 99)  # Keep last 100 for gateway ring buffer
            except Exception as e:
                logger.debug("Redis append_thermal_history failed: %s", e)

    def get_thermal_history(self, owner_id: str, max_points: int = 30) -> List[tuple]:
        if self._available:
            try:
                key = f"thermal_history:{owner_id}"
                raw = self._redis.lrange(key, 0, max_points - 1)
                result = []
                for item in reversed(raw):
                    result.append(tuple(json.loads(item)))
                return sorted(result, key=lambda x: x[0])
            except Exception as e:
                logger.debug("Redis get_thermal_history failed: %s", e)
        return self._fallback.get_thermal_history(owner_id, max_points)

    # --- Telemetry Ring Buffer (gateway-specific) ---

    def push_telemetry(self, node_id: str, entry: dict, max_entries: int = 100) -> None:
        """Store recent telemetry for trend analysis."""
        if self._available:
            try:
                key = f"telemetry:{node_id}"
                self._redis.lpush(key, json.dumps(entry, default=str))
                self._redis.ltrim(key, 0, max_entries - 1)
            except Exception as e:
                logger.debug("Redis push_telemetry failed: %s", e)

    def get_recent_telemetry(self, node_id: str, count: int = 100) -> list:
        """Get last N telemetry entries for a node."""
        if self._available:
            try:
                key = f"telemetry:{node_id}"
                raw = self._redis.lrange(key, 0, count - 1)
                return [json.loads(x) for x in raw]
            except Exception as e:
                logger.debug("Redis get_recent_telemetry failed: %s", e)
        return []

    # --- Pub/Sub for coordination events ---

    def publish_event(self, channel: str, event: dict) -> None:
        """Publish coordination events (calibration waves, firmware updates, emergency stops)."""
        if self._available:
            try:
                self._redis.publish(channel, json.dumps(event, default=str))
            except Exception as e:
                logger.debug("Redis publish_event failed: %s", e)

    def subscribe(self, channel: str):
        """Subscribe to coordination events. Returns a pubsub object."""
        if self._available:
            try:
                pubsub = self._redis.pubsub()
                pubsub.subscribe(channel)
                return pubsub
            except Exception as e:
                logger.debug("Redis subscribe failed: %s", e)
        return None

    # --- Config storage ---

    def save_config(self, key: str, value: dict) -> None:
        if self._available:
            try:
                self._redis.hset("gateway:config", key, json.dumps(value, default=str))
            except Exception as e:
                logger.debug("Redis save_config failed: %s", e)

    def load_config(self, key: str) -> Optional[dict]:
        if self._available:
            try:
                data = self._redis.hget("gateway:config", key)
                return json.loads(data) if data else None
            except Exception as e:
                logger.debug("Redis load_config failed: %s", e)
        return None

    # --- Health ---

    def ping(self) -> bool:
        if self._available:
            try:
                return self._redis.ping()
            except Exception:
                self._available = False
                return False
        return False

    @property
    def available(self) -> bool:
        return self._available

    # --- Warm start: load all profiles from Redis on startup ---

    def warm_start(self) -> int:
        """Load all calibration profiles from Redis into in-memory fallback.

        Returns number of profiles loaded. Call on gateway startup to avoid
        the 30-minute recalibration penalty after a restart.
        """
        if not self._available:
            return 0
        try:
            profiles = self.list_calibration_profiles()
            for node_id, cp in profiles.items():
                self._fallback.set_calibration_profile(node_id, cp)
            logger.info("RedisStateStore: warm start loaded %d calibration profiles", len(profiles))
            return len(profiles)
        except Exception as e:
            logger.warning("RedisStateStore: warm start failed: %s", e)
            return 0
