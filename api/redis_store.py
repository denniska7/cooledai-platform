"""
Redis-backed state store for the CooledAI cloud API (Railway).

Provides persistence for all dashboard state across container restarts.
Falls back gracefully to in-memory dicts if Redis is unavailable.

Key schema:
  cloud:thermal_history:{owner_id}      List   Rolling thermal data (max 2000)
  cloud:telemetry:{owner_id}:{node_id}  String JSON snapshot (300s TTL)
  cloud:accumulators:{owner_id}         Hash   Cumulative savings
  cloud:activity_feed                   List   Brain recommendation log (max 200)
  cloud:brain_rec:{node_id}             String JSON last recommendation
  cloud:gateway_mode                    String JSON mode + timestamp
  cloud:agent_config:{node_id}          String JSON per-node config
  cloud:maintenance_nodes               Set    Node IDs in maintenance
  cloud:cooling_prev:{unit_id}          String JSON previous cooling state
"""

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("cooledai.cloud.redis_store")


class CloudRedisStore:
    """Redis-backed state store with in-memory fallback for the cloud API."""

    def __init__(self, redis_url: str | None = None):
        self._redis = None
        self._available = False

        # In-memory fallback state
        self._mem_thermal_history: Dict[str, list] = {}
        self._mem_telemetry: Dict[str, Dict[str, dict]] = {}
        self._mem_accumulators: Dict[str, dict] = {}
        self._mem_activity_feed: list = []
        self._mem_brain_rec: Dict[str, dict] = {}
        self._mem_gateway_mode: Dict[str, Any] = {}
        self._mem_agent_configs: Dict[str, dict] = {}
        self._mem_maintenance: set = set()
        self._mem_cooling_prev: Dict[str, Any] = {}
        self._mem_lock = threading.Lock()

        url = redis_url or os.environ.get("REDIS_URL")
        if not url:
            logger.info("CloudRedisStore: No REDIS_URL — using in-memory fallback")
            return

        try:
            import redis as redis_lib
            self._redis = redis_lib.Redis.from_url(url, decode_responses=True)
            self._redis.ping()
            self._available = True
            logger.info("CloudRedisStore: connected to Redis")
        except ImportError:
            logger.warning("CloudRedisStore: redis package not installed — in-memory fallback")
        except Exception as e:
            logger.warning("CloudRedisStore: Redis unavailable (%s) — in-memory fallback", e)

    # ── Thermal History ──────────────────────────────────────────────────

    def append_thermal_history(self, owner_id: str, row: tuple) -> None:
        if self._available:
            try:
                key = f"cloud:thermal_history:{owner_id}"
                self._redis.lpush(key, json.dumps(list(row), default=str))
                self._redis.ltrim(key, 0, 1999)
                return
            except Exception as e:
                logger.debug("Redis append_thermal_history failed: %s", e)
        with self._mem_lock:
            history = self._mem_thermal_history.setdefault(owner_id, [])
            history.append(row)
            if len(history) > 2000:
                history.pop(0)

    def get_thermal_history(self, owner_id: str, max_points: int = 2000) -> list:
        if self._available:
            try:
                key = f"cloud:thermal_history:{owner_id}"
                raw = self._redis.lrange(key, 0, max_points - 1)
                result = [tuple(json.loads(item)) for item in reversed(raw)]
                return sorted(result, key=lambda x: x[0])
            except Exception as e:
                logger.debug("Redis get_thermal_history failed: %s", e)
        with self._mem_lock:
            return sorted(list(self._mem_thermal_history.get(owner_id, [])), key=lambda x: x[0])

    def get_thermal_history_count(self, owner_id: str) -> int:
        if self._available:
            try:
                return self._redis.llen(f"cloud:thermal_history:{owner_id}")
            except Exception as e:
                logger.debug("Redis get_thermal_history_count failed: %s", e)
        with self._mem_lock:
            return len(self._mem_thermal_history.get(owner_id, []))

    # ── Live Telemetry Snapshots ─────────────────────────────────────────

    def set_telemetry_snapshot(self, owner_id: str, node_id: str, data: dict) -> None:
        if self._available:
            try:
                key = f"cloud:telemetry:{owner_id}:{node_id}"
                self._redis.set(key, json.dumps(data, default=str), ex=300)
                return
            except Exception as e:
                logger.debug("Redis set_telemetry_snapshot failed: %s", e)
        with self._mem_lock:
            self._mem_telemetry.setdefault(owner_id, {})[node_id] = data

    def get_telemetry_snapshot(self, owner_id: str, node_id: str) -> Optional[dict]:
        if self._available:
            try:
                key = f"cloud:telemetry:{owner_id}:{node_id}"
                raw = self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.debug("Redis get_telemetry_snapshot failed: %s", e)
        with self._mem_lock:
            return self._mem_telemetry.get(owner_id, {}).get(node_id)

    def get_all_telemetry(self, owner_id: str) -> Dict[str, dict]:
        if self._available:
            try:
                prefix = f"cloud:telemetry:{owner_id}:"
                result = {}
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{prefix}*", count=100)
                    for key in keys:
                        node_id = key[len(prefix):]
                        raw = self._redis.get(key)
                        if raw:
                            result[node_id] = json.loads(raw)
                    if cursor == 0:
                        break
                return result
            except Exception as e:
                logger.debug("Redis get_all_telemetry failed: %s", e)
        with self._mem_lock:
            return dict(self._mem_telemetry.get(owner_id, {}))

    def get_all_telemetry_all_owners(self) -> Dict[str, dict]:
        """Get all telemetry across all owners (for admin/discovery endpoints)."""
        if self._available:
            try:
                prefix = "cloud:telemetry:"
                result = {}
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{prefix}*", count=200)
                    for key in keys:
                        raw = self._redis.get(key)
                        if raw:
                            result[key[len(prefix):]] = json.loads(raw)
                    if cursor == 0:
                        break
                return result
            except Exception as e:
                logger.debug("Redis get_all_telemetry_all_owners failed: %s", e)
        with self._mem_lock:
            all_nodes = {}
            for owner_nodes in self._mem_telemetry.values():
                all_nodes.update(owner_nodes)
            return all_nodes

    # ── Savings Accumulators ─────────────────────────────────────────────

    def set_accumulator(self, owner_id: str, data: dict) -> None:
        if self._available:
            try:
                key = f"cloud:accumulators:{owner_id}"
                self._redis.set(key, json.dumps(data, default=str))
                return
            except Exception as e:
                logger.debug("Redis set_accumulator failed: %s", e)
        with self._mem_lock:
            self._mem_accumulators[owner_id] = data

    def get_accumulator(self, owner_id: str) -> Optional[dict]:
        if self._available:
            try:
                raw = self._redis.get(f"cloud:accumulators:{owner_id}")
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.debug("Redis get_accumulator failed: %s", e)
        with self._mem_lock:
            return self._mem_accumulators.get(owner_id)

    def get_all_accumulators(self) -> Dict[str, dict]:
        if self._available:
            try:
                prefix = "cloud:accumulators:"
                result = {}
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{prefix}*", count=100)
                    for key in keys:
                        raw = self._redis.get(key)
                        if raw:
                            result[key[len(prefix):]] = json.loads(raw)
                    if cursor == 0:
                        break
                return result
            except Exception as e:
                logger.debug("Redis get_all_accumulators failed: %s", e)
        with self._mem_lock:
            return dict(self._mem_accumulators)

    # ── Activity Feed ────────────────────────────────────────────────────

    def push_activity(self, entry: dict) -> None:
        if self._available:
            try:
                self._redis.lpush("cloud:activity_feed", json.dumps(entry, default=str))
                self._redis.ltrim("cloud:activity_feed", 0, 199)
                return
            except Exception as e:
                logger.debug("Redis push_activity failed: %s", e)
        with self._mem_lock:
            self._mem_activity_feed.insert(0, entry)
            if len(self._mem_activity_feed) > 200:
                self._mem_activity_feed.pop()

    def get_activity_feed(self, limit: int = 50) -> list:
        if self._available:
            try:
                raw = self._redis.lrange("cloud:activity_feed", 0, limit - 1)
                return [json.loads(item) for item in raw]
            except Exception as e:
                logger.debug("Redis get_activity_feed failed: %s", e)
        with self._mem_lock:
            return list(self._mem_activity_feed[:limit])

    def get_activity_feed_total(self) -> int:
        if self._available:
            try:
                return self._redis.llen("cloud:activity_feed")
            except Exception as e:
                logger.debug("Redis get_activity_feed_total failed: %s", e)
        with self._mem_lock:
            return len(self._mem_activity_feed)

    # ── Brain Recommendations ────────────────────────────────────────────

    def set_brain_recommendation(self, node_id: str, rec: dict) -> None:
        if self._available:
            try:
                self._redis.set(f"cloud:brain_rec:{node_id}", json.dumps(rec, default=str))
                return
            except Exception as e:
                logger.debug("Redis set_brain_recommendation failed: %s", e)
        with self._mem_lock:
            self._mem_brain_rec[node_id] = rec

    def get_all_brain_recommendations(self) -> Dict[str, dict]:
        if self._available:
            try:
                prefix = "cloud:brain_rec:"
                result = {}
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=f"{prefix}*", count=100)
                    for key in keys:
                        raw = self._redis.get(key)
                        if raw:
                            result[key[len(prefix):]] = json.loads(raw)
                    if cursor == 0:
                        break
                return result
            except Exception as e:
                logger.debug("Redis get_all_brain_recommendations failed: %s", e)
        with self._mem_lock:
            return dict(self._mem_brain_rec)

    # ── Gateway Mode ─────────────────────────────────────────────────────

    def set_gateway_mode(self, mode: str, ts: Any) -> None:
        if self._available:
            try:
                self._redis.set("cloud:gateway_mode", json.dumps({"mode": mode, "ts": ts}, default=str))
                return
            except Exception as e:
                logger.debug("Redis set_gateway_mode failed: %s", e)
        with self._mem_lock:
            self._mem_gateway_mode = {"mode": mode, "ts": ts}

    def get_gateway_mode(self) -> Dict[str, Any]:
        if self._available:
            try:
                raw = self._redis.get("cloud:gateway_mode")
                return json.loads(raw) if raw else {}
            except Exception as e:
                logger.debug("Redis get_gateway_mode failed: %s", e)
        with self._mem_lock:
            return dict(self._mem_gateway_mode)

    # ── Agent Configs ────────────────────────────────────────────────────

    def set_agent_config(self, node_id: str, config: dict) -> None:
        if self._available:
            try:
                self._redis.set(f"cloud:agent_config:{node_id}", json.dumps(config, default=str))
                return
            except Exception as e:
                logger.debug("Redis set_agent_config failed: %s", e)
        with self._mem_lock:
            self._mem_agent_configs[node_id] = config

    def get_agent_config(self, node_id: str) -> Optional[dict]:
        if self._available:
            try:
                raw = self._redis.get(f"cloud:agent_config:{node_id}")
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.debug("Redis get_agent_config failed: %s", e)
        with self._mem_lock:
            return self._mem_agent_configs.get(node_id)

    # ── Maintenance Mode ─────────────────────────────────────────────────

    def add_maintenance_node(self, node_id: str) -> None:
        if self._available:
            try:
                self._redis.sadd("cloud:maintenance_nodes", node_id)
                return
            except Exception as e:
                logger.debug("Redis add_maintenance_node failed: %s", e)
        with self._mem_lock:
            self._mem_maintenance.add(node_id)

    def remove_maintenance_node(self, node_id: str) -> None:
        if self._available:
            try:
                self._redis.srem("cloud:maintenance_nodes", node_id)
                return
            except Exception as e:
                logger.debug("Redis remove_maintenance_node failed: %s", e)
        with self._mem_lock:
            self._mem_maintenance.discard(node_id)

    def is_maintenance(self, node_id: str) -> bool:
        if self._available:
            try:
                return bool(self._redis.sismember("cloud:maintenance_nodes", node_id))
            except Exception as e:
                logger.debug("Redis is_maintenance failed: %s", e)
        with self._mem_lock:
            return node_id in self._mem_maintenance

    def get_maintenance_nodes(self) -> set:
        if self._available:
            try:
                return self._redis.smembers("cloud:maintenance_nodes")
            except Exception as e:
                logger.debug("Redis get_maintenance_nodes failed: %s", e)
        with self._mem_lock:
            return set(self._mem_maintenance)

    # ── Cooling Unit Previous State ──────────────────────────────────────

    def set_cooling_previous_state(self, state: Dict[str, Any]) -> None:
        if self._available:
            try:
                self._redis.set("cloud:cooling_prev", json.dumps(state, default=str))
                return
            except Exception as e:
                logger.debug("Redis set_cooling_previous_state failed: %s", e)
        with self._mem_lock:
            self._mem_cooling_prev = state

    def get_cooling_previous_state(self) -> Dict[str, Any]:
        if self._available:
            try:
                raw = self._redis.get("cloud:cooling_prev")
                return json.loads(raw) if raw else {}
            except Exception as e:
                logger.debug("Redis get_cooling_previous_state failed: %s", e)
        with self._mem_lock:
            return dict(self._mem_cooling_prev)

    # ── Health ───────────────────────────────────────────────────────────

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
