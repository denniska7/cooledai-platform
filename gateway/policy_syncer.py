"""
Policy syncer for the CooledAI edge gateway.

Pulls configuration updates from the cloud API every 60 seconds:
- API key registry updates
- System state (OPTIMIZING / GUARD_MODE / MANUAL_OVERRIDE)
- Maintenance mode node IDs
- Safety overrides

If the cloud is unreachable, the gateway continues with the last-known
configuration indefinitely.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cooledai.gateway.syncer")


class PolicySyncer:
    """Pull config from cloud every N seconds."""

    def __init__(
        self,
        cloud_url: str,
        api_key: str,
        interval: float = 60.0,
        facility_id: str = "facility-default",
    ):
        self.cloud_url = cloud_url.rstrip("/")
        self.api_key = api_key
        self.interval = interval
        self.facility_id = facility_id

        self._last_sync = 0.0
        self._last_config: Optional[dict] = None
        self._connected = True

        # Operational mode (synced from cloud)
        self.current_mode: str = "shadow"
        self.fan_rpm_floor_pct: float = 0.90
        self.thermal_ceiling_c: float = 85.0

    async def sync_loop(self) -> None:
        """Background task on Thread 4's asyncio loop."""
        logger.info("PolicySyncer: started (sync every %.0fs from %s)", self.interval, self.cloud_url)
        while True:
            await asyncio.sleep(self.interval)
            await self._sync()

    async def _sync(self) -> None:
        """Pull config from cloud."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.cloud_url}/api/v1/gateway/config",
                    params={"facility_id": self.facility_id},
                    headers={"X-API-Key": self.api_key},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        config = await resp.json()
                        self._last_config = config
                        self._last_sync = time.time()
                        self._connected = True
                        self._apply_config(config)
                    else:
                        logger.warning("PolicySyncer: config pull failed (status=%d)", resp.status)
        except ImportError:
            logger.debug("PolicySyncer: aiohttp not available, skipping sync")
        except Exception as e:
            logger.debug("PolicySyncer: sync failed: %s", e)
            self._connected = False

    def _apply_config(self, config: dict) -> None:
        """Apply pulled configuration locally."""
        # Update operational mode
        new_mode = config.get("mode")
        if new_mode and new_mode in ("shadow", "supervised", "active"):
            if new_mode != self.current_mode:
                logger.info("PolicySyncer: Mode changed %s → %s", self.current_mode, new_mode)
            self.current_mode = new_mode
        self.fan_rpm_floor_pct = float(config.get("fan_rpm_floor_pct", 0.90))
        self.thermal_ceiling_c = float(config.get("thermal_ceiling_c", 85.0))

        # Update local API key registry if present
        key_registry = config.get("api_key_registry")
        if key_registry:
            try:
                project_root = Path(__file__).resolve().parent.parent
                keys_file = project_root / "data" / "api_keys.json"
                keys_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = keys_file.with_suffix(".tmp")
                tmp.write_text(json.dumps({"keys": key_registry}, indent=2), encoding="utf-8")
                tmp.replace(keys_file)
                logger.debug("PolicySyncer: Updated local API key registry (%d keys)", len(key_registry))
            except Exception as e:
                logger.warning("PolicySyncer: Failed to update local keys: %s", e)

    @property
    def cloud_connected(self) -> bool:
        return self._connected

    @property
    def last_config(self) -> Optional[dict]:
        return self._last_config

    @property
    def seconds_since_last_sync(self) -> float:
        if self._last_sync == 0:
            return float("inf")
        return time.time() - self._last_sync
