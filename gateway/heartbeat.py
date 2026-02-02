"""
CooledAI Heartbeat - Agent Health Signal

Sends agent CPU/RAM health to portal every 60 seconds.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime

_logger = logging.getLogger("cooledai.heartbeat")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)

HEARTBEAT_INTERVAL_SEC = 60


def get_agent_health() -> Dict[str, Any]:
    """Get agent CPU and RAM health."""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": round(cpu_percent, 2),
            "ram_percent": round(mem.percent, 2),
            "ram_used_mb": round(mem.used / (1024 * 1024), 2),
            "ram_total_mb": round(mem.total / (1024 * 1024), 2),
            "agent_id": os.environ.get("COOLEDAI_AGENT_ID", "default"),
        }
    except ImportError:
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": 0,
            "ram_percent": 0,
            "ram_used_mb": 0,
            "ram_total_mb": 0,
            "agent_id": os.environ.get("COOLEDAI_AGENT_ID", "default"),
            "error": "psutil not installed",
        }
    except Exception as e:
        _logger.warning("Heartbeat: Failed to get health: %s", e)
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": 0,
            "ram_percent": 0,
            "ram_used_mb": 0,
            "ram_total_mb": 0,
            "agent_id": "default",
            "error": str(e),
        }
