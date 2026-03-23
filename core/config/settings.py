"""
CooledAI Settings — single source of truth for production configuration.

Loads ``configs/cooledai.yaml``, validates against a schema, and exposes
typed properties.  Environment variables override YAML values using the
pattern ``COOLEDAI_<SECTION>_<KEY>`` (e.g. ``COOLEDAI_SERVER_CLIENT_NAME``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("cooledai.config")

_REQUIRED_SECTIONS = {"server", "hardware", "features", "alerts"}

# Fields that MUST be present in certain sections
_REQUIRED_FIELDS: Dict[str, set] = {
    "server": {"client_name", "site", "rack"},
    "hardware": {"gpu_count", "max_fan_rpm"},
}

# gpu_model is explicitly OPTIONAL and never used in optimization logic
_OPTIONAL_FIELDS = {"gpu_model"}


class ConfigValidationError(Exception):
    """Raised when cooledai.yaml is invalid or missing required fields."""


class Settings:
    """Validated, typed wrapper around cooledai.yaml."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._raw: Dict[str, Any] = {}
        self._path = config_path or self._find_config()
        if self._path:
            self._load(self._path)
        else:
            log.info("[CONFIG] No cooledai.yaml found — using defaults + env vars")
            self._raw = self._defaults()
        self._apply_env_overrides()

    # ── Loading ──────────────────────────────────────────────────

    @staticmethod
    def _find_config() -> Optional[str]:
        """Search common locations for cooledai.yaml."""
        candidates = [
            Path(os.environ.get("COOLEDAI_CONFIG", "")),
            Path("/etc/cooledai/cooledai.yaml"),
            Path.cwd() / "configs" / "cooledai.yaml",
            Path(__file__).resolve().parents[2] / "configs" / "cooledai.yaml",
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
        return None

    def _load(self, path: str) -> None:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            log.warning("[CONFIG] PyYAML not installed — using defaults")
            self._raw = self._defaults()
            return
        with open(path, encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh) or {}
        log.info("[CONFIG] Loaded %s", path)

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        return {
            "server": {"client_name": "", "site": "", "rack": "", "tls_cert_path": "", "require_https": False},
            "hardware": {"gpu_count": 2, "gpu_model": "", "max_fan_rpm": 7000},
            "features": {"gpu_governor": True, "xcc_control": True, "cpu_thermal_guard": True},
            "alerts": {"warning_temp_c": 65, "critical_temp_c": 85, "shutdown_temp_c": 95, "webhook_url": "", "email": ""},
            "calibration": {"profile_backup_path": "/var/lib/cooledai/calibration_backup.json"},
            "audit": {"log_path": "/var/log/cooledai/audit.log", "retention_days": 90},
            "auth": {"key_expiry_days": 90, "key_warning_days": 75, "rate_limit_per_minute": 60},
            "telemetry": {"strict_whitelist": False},
        }

    def _apply_env_overrides(self) -> None:
        """Env vars like COOLEDAI_ALERTS_WARNING_TEMP_C override YAML."""
        prefix = "COOLEDAI_"
        for key, val in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section, field = parts
            if section in self._raw and isinstance(self._raw[section], dict):
                # Coerce types
                existing = self._raw[section].get(field)
                if isinstance(existing, bool):
                    self._raw[section][field] = val.lower() in ("1", "true", "yes")
                elif isinstance(existing, int):
                    try:
                        self._raw[section][field] = int(val)
                    except ValueError:
                        pass
                elif isinstance(existing, float):
                    try:
                        self._raw[section][field] = float(val)
                    except ValueError:
                        pass
                else:
                    self._raw[section][field] = val

    # ── Validation ───────────────────────────────────────────────

    def validate(self, *, strict: bool = True) -> None:
        """Validate config.  Raises ConfigValidationError on failure."""
        errors = []
        for section in _REQUIRED_SECTIONS:
            if section not in self._raw:
                errors.append(f"Missing required section: [{section}]")
                continue
            for field in _REQUIRED_FIELDS.get(section, set()):
                val = self._raw[section].get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    if strict:
                        errors.append(f"[{section}].{field} is required")
        if errors:
            raise ConfigValidationError(
                "Invalid cooledai.yaml:\n  " + "\n  ".join(errors)
            )

    # ── Accessors ────────────────────────────────────────────────

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._raw.get(section, {}).get(key, default)

    @property
    def client_name(self) -> str:
        return self.get("server", "client_name", "")

    @property
    def site(self) -> str:
        return self.get("server", "site", "")

    @property
    def tls_cert_path(self) -> str:
        return self.get("server", "tls_cert_path", "")

    @property
    def require_https(self) -> bool:
        return self.get("server", "require_https", False)

    @property
    def gpu_count(self) -> int:
        return self.get("hardware", "gpu_count", 2)

    @property
    def max_fan_rpm(self) -> int:
        return self.get("hardware", "max_fan_rpm", 7000)

    @property
    def warning_temp_c(self) -> float:
        return self.get("alerts", "warning_temp_c", 65.0)

    @property
    def critical_temp_c(self) -> float:
        return self.get("alerts", "critical_temp_c", 85.0)

    @property
    def shutdown_temp_c(self) -> float:
        return self.get("alerts", "shutdown_temp_c", 95.0)

    @property
    def webhook_url(self) -> str:
        return self.get("alerts", "webhook_url", "")

    @property
    def alert_email(self) -> str:
        return self.get("alerts", "email", "")

    @property
    def calibration_backup_path(self) -> str:
        return self.get("calibration", "profile_backup_path",
                        "/var/lib/cooledai/calibration_backup.json")

    @property
    def audit_log_path(self) -> str:
        return self.get("audit", "log_path", "/var/log/cooledai/audit.log")

    @property
    def audit_retention_days(self) -> int:
        return self.get("audit", "retention_days", 90)

    @property
    def key_expiry_days(self) -> int:
        return self.get("auth", "key_expiry_days", 90)

    @property
    def key_warning_days(self) -> int:
        return self.get("auth", "key_warning_days", 75)

    @property
    def rate_limit_per_minute(self) -> int:
        return self.get("auth", "rate_limit_per_minute", 60)

    @property
    def strict_whitelist(self) -> bool:
        return self.get("telemetry", "strict_whitelist", False)

    @property
    def features(self) -> Dict[str, bool]:
        return self._raw.get("features", {})
