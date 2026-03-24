"""
CooledAI XCC Fan Controller — Lenovo XClarity Controller Fan Control

Controls server fans via the Lenovo XCC BMC through multiple paths:
  1. Redfish PATCH /redfish/v1/Chassis/1/Thermal  (standard Redfish)
  2. Redfish POST  Managers/1/Actions/Oem/LenovoManager.SetFanControlMode
  3. Lenovo OEM fan policy endpoints
  4. IPMI over LAN (ipmitool -I lanplus) — works even without /dev/ipmi0

The ST550 BMC firmware 1.90 returns 405 on Redfish PATCH, so IPMI over LAN
is the primary write path.  Redfish is still used for monitoring (reads).

Environment variables:
  XCC_BMC_HOST  — XCC hostname/IP  (e.g. "169.254.95.118")
  XCC_BMC_USER  — Redfish/IPMI username  (default: "USERID")
  XCC_BMC_PASS  — Redfish/IPMI password  (default: "***REDACTED_BMC_PASS***")

Usage::

    from core.hardware.xcc_fan_controller import XCCFanController

    ctrl = XCCFanController()
    if ctrl.available:
        ctrl.set_manual_fan_percent(60)
    # On shutdown:
    ctrl.restore_auto_control()
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_log = logging.getLogger("cooledai.xcc_fan")


class _CredentialScrubFilter(logging.Filter):
    """Mask XCC BMC credentials in log output."""

    _secrets: List[str] = []

    @classmethod
    def register_secrets(cls, *values: str) -> None:
        cls._secrets = [v for v in values if v and len(v) > 2]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for secret in self._secrets:
            if secret in msg:
                record.msg = str(record.msg).replace(secret, "***")
                if record.args:
                    record.args = tuple(
                        str(a).replace(secret, "***") if isinstance(a, str) else a
                        for a in (record.args if isinstance(record.args, tuple) else (record.args,))
                    )
        return True


_log.addFilter(_CredentialScrubFilter())


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


# ---------------------------------------------------------------------------
# SSL context (self-signed XCC certs)
# ---------------------------------------------------------------------------

_SSL_CTX: Optional[ssl.SSLContext] = None


def _ssl_ctx() -> ssl.SSLContext:
    global _SSL_CTX
    if _SSL_CTX is None:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        _SSL_CTX = ctx
    return _SSL_CTX


# ---------------------------------------------------------------------------
# XCC Fan Controller
# ---------------------------------------------------------------------------

class XCCFanController:
    """Controls Lenovo XCC/BMC fans via Redfish and IPMI over LAN.

    Tries multiple paths in order:
      1. Redfish PATCH Chassis/1/Thermal  (standard)
      2. Redfish POST Managers/1/Actions/Oem/LenovoManager.SetFanControlMode
      3. Lenovo OEM PATCH endpoints
      4. IPMI over LAN (ipmitool -I lanplus) — Lenovo OEM 0x3a, then Dell 0x30

    Falls back gracefully when the BMC does not support a given method.
    """

    # Redfish paths to attempt for fan control, in priority order
    THERMAL_PATH = "/redfish/v1/Chassis/1/Thermal/"
    OEM_FAN_ACTION_PATH = "/redfish/v1/Managers/1/Actions/Oem/LenovoManager.SetFanControlMode"
    OEM_FAN_PATH = "/redfish/v1/Managers/1/Oem/Lenovo/FanControl"
    BIOS_PENDING_PATH = "/redfish/v1/Systems/1/Bios/Pending"

    # Control method constants
    METHOD_REDFISH_THERMAL = "redfish_thermal"
    METHOD_REDFISH_OEM_ACTION = "redfish_oem_action"
    METHOD_REDFISH_OEM = "redfish_oem"
    METHOD_IPMI_LAN = "ipmi_lan"
    METHOD_NONE = "none"

    def __init__(
        self,
        host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout_s: float = 3.0,
    ):
        self._host = host or _env("XCC_BMC_HOST")
        self._user = username or _env("XCC_BMC_USER", "USERID")
        self._pass = password or _env("XCC_BMC_PASS", "***REDACTED_BMC_PASS***")
        self._timeout = timeout_s
        self._available = False
        self._manual_active = False
        self._working_method: str = self.METHOD_NONE  # Cache the method that works
        self._ipmi_lan_available = False  # IPMI over LAN connectivity confirmed
        self._ipmi_lan_variant: str = ""  # "lenovo_0x3a" or "dell_0x30"

        # Register secrets for log scrubbing
        _CredentialScrubFilter.register_secrets(self._pass, self._user, self._host)

        # Build auth header for Redfish
        cred = f"{self._user}:{self._pass}".encode()
        self._auth = f"Basic {base64.b64encode(cred).decode()}"

        if self._host:
            self._base_url = f"https://{self._host}"
            self._available = self._probe()
        else:
            _log.info("[XCC_FAN] No XCC_BMC_HOST configured — XCC fan control disabled.")
            self._base_url = ""

    @property
    def available(self) -> bool:
        """True if XCC is reachable (Redfish or IPMI LAN)."""
        return self._available

    @property
    def manual_active(self) -> bool:
        """True if we have issued a manual fan command (need to restore on shutdown)."""
        return self._manual_active

    @property
    def working_method(self) -> str:
        """The control method confirmed working during probe."""
        return self._working_method

    @property
    def ipmi_lan_available(self) -> bool:
        """True if IPMI over LAN responded to chassis status check."""
        return self._ipmi_lan_available

    # -----------------------------------------------------------------------
    # HTTP helpers (stdlib, no requests dependency)
    # -----------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Issue HTTP request. Returns (status_code, response_dict_or_None)."""
        url = self._base_url + path
        headers = {
            "Authorization": self._auth,
            "Content-Type": "application/json",
        }
        data = json.dumps(body).encode() if body is not None else None
        # ── HMAC request signing (optional) ──
        hmac_secret = os.environ.get("COOLEDAI_XCC_HMAC_SECRET", "")
        if hmac_secret:
            import hashlib, hmac as _hmac
            ts = str(int(time.time()))
            msg = f"{method}{path}{ts}".encode()
            sig = _hmac.new(hmac_secret.encode(), msg, hashlib.sha256).hexdigest()
            headers["X-CooledAI-Timestamp"] = ts
            headers["X-CooledAI-Signature"] = sig
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=self._timeout) as resp:
                resp_body = resp.read().decode()
                try:
                    parsed = json.loads(resp_body) if resp_body else {}
                except json.JSONDecodeError:
                    parsed = {}
                return (resp.status, parsed)
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode()[:500]
            except Exception:
                pass
            return (exc.code, {"error": body_text})
        except Exception as exc:
            _log.debug("[XCC_FAN] Request %s %s failed: %s", method, path, exc)
            return (0, None)

    def _get(self, path: str) -> tuple:
        return self._request("GET", path)

    def _patch(self, path: str, body: Dict[str, Any]) -> tuple:
        return self._request("PATCH", path, body)

    def _post(self, path: str, body: Dict[str, Any]) -> tuple:
        return self._request("POST", path, body)

    # -----------------------------------------------------------------------
    # IPMI over LAN helpers
    # -----------------------------------------------------------------------

    def _ipmi_cmd(self, raw_args: List[str], timeout: float = 3.0) -> tuple:
        """Run ipmitool -I lanplus command.  Returns (success: bool, stdout: str, stderr: str).

        Password is passed via IPMI_PASSWORD env var to avoid exposure in
        process listings (``ps aux``).
        """
        cmd = [
            "ipmitool", "-I", "lanplus",
            "-H", self._host,
            "-U", self._user,
            "-E",  # read password from IPMI_PASSWORD env var
            "-e", "",  # escape char (empty = disable)
        ] + raw_args
        env = {**os.environ, "IPMI_PASSWORD": self._pass}
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return (result.returncode == 0, result.stdout.strip(), result.stderr.strip())
        except FileNotFoundError:
            _log.debug("[XCC_FAN] ipmitool not found — IPMI over LAN unavailable.")
            return (False, "", "ipmitool not found")
        except subprocess.TimeoutExpired:
            _log.debug("[XCC_FAN] ipmitool timed out after %.1fs", timeout)
            return (False, "", "timeout")
        except Exception as exc:
            _log.debug("[XCC_FAN] ipmitool failed: %s", exc)
            return (False, "", str(exc))

    def _probe_ipmi_lan(self) -> bool:
        """Test IPMI over LAN connectivity via 'chassis status'."""
        ok, stdout, stderr = self._ipmi_cmd(["chassis", "status"], timeout=self._timeout)
        if ok and "System Power" in stdout:
            _log.info("[XCC_FAN] IPMI over LAN reachable at %s (chassis status OK)", self._host)
            return True
        _log.info("[XCC_FAN] IPMI over LAN not reachable at %s: %s", self._host, stderr or stdout)
        return False

    def _try_ipmi_lan_fan_set(self, pct: int) -> bool:
        """Set fan speed via IPMI over LAN.

        Tries Lenovo OEM NetFn 0x3a first, then Dell-style 0x30.
        """
        hex_pct = hex(max(0, min(100, pct)))

        # Try cached variant first if we already know which works
        if self._ipmi_lan_variant == "lenovo_0x3a":
            return self._ipmi_lan_lenovo_set(hex_pct)
        elif self._ipmi_lan_variant == "dell_0x30":
            return self._ipmi_lan_dell_set(hex_pct)

        # Probe both — try Lenovo OEM 0x3a first
        if self._ipmi_lan_lenovo_set(hex_pct):
            self._ipmi_lan_variant = "lenovo_0x3a"
            return True

        # Fall back to Dell-style 0x30 0x30
        if self._ipmi_lan_dell_set(hex_pct):
            self._ipmi_lan_variant = "dell_0x30"
            return True

        return False

    def _ipmi_lan_lenovo_set(self, hex_pct: str) -> bool:
        """Lenovo OEM: raw 0x3a 0x07 0x01 {pct}."""
        # First enable manual mode
        ok, stdout, stderr = self._ipmi_cmd(["raw", "0x3a", "0x07", "0x01", "0x00"])
        if not ok:
            _log.debug("[XCC_FAN] Lenovo 0x3a manual mode failed: %s", stderr)
            return False

        # Set fan speed
        ok, stdout, stderr = self._ipmi_cmd(["raw", "0x3a", "0x07", "0x01", hex_pct])
        if ok:
            _log.info("[XCC_FAN] method=ipmi_lan variant=lenovo_0x3a fan_pct=%s — OK", hex_pct)
            return True
        _log.debug("[XCC_FAN] Lenovo 0x3a fan set failed: %s", stderr)
        return False

    def _ipmi_lan_dell_set(self, hex_pct: str) -> bool:
        """Dell-style: raw 0x30 0x30 0x02 0xff {pct}."""
        # Disable automatic fan control
        ok, stdout, stderr = self._ipmi_cmd(["raw", "0x30", "0x30", "0x01", "0x00"])
        if not ok:
            _log.debug("[XCC_FAN] Dell 0x30 manual mode failed: %s", stderr)
            return False

        # Set fan duty
        ok, stdout, stderr = self._ipmi_cmd(["raw", "0x30", "0x30", "0x02", "0xff", hex_pct])
        if ok:
            _log.info("[XCC_FAN] method=ipmi_lan variant=dell_0x30 fan_pct=%s — OK", hex_pct)
            return True
        _log.debug("[XCC_FAN] Dell 0x30 fan set failed: %s", stderr)
        return False

    def _restore_ipmi_lan_auto(self) -> bool:
        """Re-enable automatic fan control via IPMI over LAN."""
        if self._ipmi_lan_variant == "lenovo_0x3a":
            # Lenovo: set mode back to auto (0x3a 0x07 0x00 = auto)
            ok, _, stderr = self._ipmi_cmd(["raw", "0x3a", "0x07", "0x00", "0x00"])
            if ok:
                _log.info("[XCC_FAN] IPMI LAN auto restore via Lenovo 0x3a — OK")
                return True
            _log.debug("[XCC_FAN] Lenovo 0x3a auto restore failed: %s", stderr)

        if self._ipmi_lan_variant == "dell_0x30" or not self._ipmi_lan_variant:
            # Dell: re-enable automatic (0x30 0x30 0x01 0x01)
            ok, _, stderr = self._ipmi_cmd(["raw", "0x30", "0x30", "0x01", "0x01"])
            if ok:
                _log.info("[XCC_FAN] IPMI LAN auto restore via Dell 0x30 — OK")
                return True
            _log.debug("[XCC_FAN] Dell 0x30 auto restore failed: %s", stderr)

        return False

    # -----------------------------------------------------------------------
    # Probe / init
    # -----------------------------------------------------------------------

    def _probe(self) -> bool:
        """Test XCC connectivity and find a working fan control path."""
        # --- Redfish probe ---
        status, data = self._get("/redfish/v1/")
        if status != 200:
            _log.warning(
                "[XCC_FAN] XCC Redfish at %s not reachable (status=%d) — trying IPMI LAN...",
                self._host, status,
            )
        else:
            _log.info("[XCC_FAN] XCC reachable at %s — probing fan control paths...", self._host)

            # Probe Redfish PATCH on Thermal
            th_status, thermal = self._get(self.THERMAL_PATH)
            if th_status == 200 and thermal:
                fans = thermal.get("Fans", [])
                _log.info(
                    "[XCC_FAN] Thermal endpoint OK — %d fans detected, checking PATCH support...",
                    len(fans),
                )
                if fans:
                    fan_id = fans[0].get("MemberId", "0")
                    current = fans[0].get("Reading", 50)
                    test_status, _ = self._patch(self.THERMAL_PATH, {
                        "Fans": [{"MemberId": fan_id, "Reading": current}]
                    })
                    if test_status in (200, 204):
                        self._working_method = self.METHOD_REDFISH_THERMAL
                        _log.info("[XCC_FAN] PATCH %s supported (status=%d) — direct Redfish fan control!",
                                 self.THERMAL_PATH, test_status)
                        return True
                    _log.info("[XCC_FAN] PATCH %s returned %d — trying alternate paths...",
                             self.THERMAL_PATH, test_status)

            # Probe Redfish POST OEM Actions (LenovoManager.SetFanControlMode)
            test_status, resp = self._post(self.OEM_FAN_ACTION_PATH, {
                "FanControlMode": "Manual", "FanSpeed": 50,
            })
            if test_status in (200, 204):
                self._working_method = self.METHOD_REDFISH_OEM_ACTION
                _log.info("[XCC_FAN] POST %s supported (status=%d) — OEM action fan control!",
                         self.OEM_FAN_ACTION_PATH, test_status)
                # Immediately restore to auto since we just set manual during probe
                self._post(self.OEM_FAN_ACTION_PATH, {"FanControlMode": "Automatic"})
                return True
            elif test_status in (400, 404, 405):
                _log.info("[XCC_FAN] POST %s returned %d — not supported on this firmware.",
                         self.OEM_FAN_ACTION_PATH, test_status)
            elif test_status > 0:
                _log.info("[XCC_FAN] POST %s returned %d", self.OEM_FAN_ACTION_PATH, test_status)

            # Probe OEM fan control PATCH endpoint
            oem_status, oem_data = self._get(self.OEM_FAN_PATH)
            if oem_status == 200:
                test_status, _ = self._patch(self.OEM_FAN_PATH, {})
                if test_status in (200, 204, 400):
                    self._working_method = self.METHOD_REDFISH_OEM
                    _log.info("[XCC_FAN] OEM fan endpoint available at %s", self.OEM_FAN_PATH)
                    return True

        # --- IPMI over LAN probe ---
        self._ipmi_lan_available = self._probe_ipmi_lan()
        if self._ipmi_lan_available:
            self._working_method = self.METHOD_IPMI_LAN
            _log.info("[XCC_FAN] IPMI over LAN confirmed — will use ipmitool -I lanplus for fan control.")
            return True

        # --- Monitoring only (Redfish reads work even if writes don't) ---
        if status == 200:
            _log.info(
                "[XCC_FAN] No writable fan control path found — XCC available for monitoring only. "
                "Fan control will rely on BIOS operating mode."
            )
            return True

        _log.warning("[XCC_FAN] Neither Redfish nor IPMI LAN reachable — fan control disabled.")
        return False

    # -----------------------------------------------------------------------
    # Fan control (unified)
    # -----------------------------------------------------------------------

    def set_manual_fan_percent(self, pct: int) -> bool:
        """Set all fans to a specific percentage (0-100).

        Tries methods in priority order: Redfish PATCH → Redfish OEM Action →
        Redfish OEM PATCH → IPMI over LAN.

        Returns True if the command was accepted by the BMC.
        """
        pct = max(0, min(100, pct))

        if not self._available:
            _log.debug("[XCC_FAN] Not available — skipping set_manual_fan_percent(%d)", pct)
            return False

        # Try cached working method first
        success = False
        method_used = self.METHOD_NONE

        if self._working_method == self.METHOD_REDFISH_THERMAL:
            success = self._set_via_thermal_patch(pct)
            if success:
                method_used = self.METHOD_REDFISH_THERMAL

        elif self._working_method == self.METHOD_REDFISH_OEM_ACTION:
            success = self._set_via_oem_action(pct)
            if success:
                method_used = self.METHOD_REDFISH_OEM_ACTION

        elif self._working_method == self.METHOD_REDFISH_OEM:
            success = self._set_via_oem(pct)
            if success:
                method_used = self.METHOD_REDFISH_OEM

        elif self._working_method == self.METHOD_IPMI_LAN:
            success = self._try_ipmi_lan_fan_set(pct)
            if success:
                method_used = self.METHOD_IPMI_LAN

        # If cached method failed, try all methods in order
        if not success:
            for try_method, try_fn in [
                (self.METHOD_REDFISH_THERMAL, lambda: self._set_via_thermal_patch(pct)),
                (self.METHOD_REDFISH_OEM_ACTION, lambda: self._set_via_oem_action(pct)),
                (self.METHOD_REDFISH_OEM, lambda: self._set_via_oem(pct)),
                (self.METHOD_IPMI_LAN, lambda: self._try_ipmi_lan_fan_set(pct)),
            ]:
                if try_method == self._working_method:
                    continue  # already tried
                if try_fn():
                    success = True
                    method_used = try_method
                    self._working_method = try_method  # update cache
                    break

        if success:
            self._manual_active = True
            _log.info("[XCC_FAN] method=%s fan_pct=%d — SUCCESS", method_used, pct)
        else:
            _log.warning("[XCC_FAN] method=none fan_pct=%d — ALL methods failed", pct)

        return success

    def _set_via_thermal_patch(self, pct: int) -> bool:
        """Standard Redfish: PATCH Chassis/1/Thermal with fan readings."""
        status, thermal = self._get(self.THERMAL_PATH)
        if status != 200 or not thermal:
            return False

        fans = thermal.get("Fans", [])
        if not fans:
            return False

        fan_patch = []
        for fan in fans:
            member_id = fan.get("MemberId", fan.get("Name", "0"))
            fan_patch.append({"MemberId": str(member_id), "Reading": pct})

        patch_status, resp = self._patch(self.THERMAL_PATH, {"Fans": fan_patch})
        if patch_status in (200, 204):
            return True

        # If 405, try individual fan URIs
        if patch_status == 405:
            for fan in fans:
                fan_uri = fan.get("@odata.id")
                if fan_uri:
                    s, _ = self._patch(fan_uri, {"Reading": pct})
                    if s in (200, 204):
                        return True

        return False

    def _set_via_oem_action(self, pct: int) -> bool:
        """Lenovo OEM Action: POST Managers/1/Actions/Oem/LenovoManager.SetFanControlMode."""
        status, resp = self._post(self.OEM_FAN_ACTION_PATH, {
            "FanControlMode": "Manual",
            "FanSpeed": pct,
        })
        if status in (200, 204):
            return True
        return False

    def _set_via_oem(self, pct: int) -> bool:
        """Lenovo OEM: PATCH Managers/1/Oem/Lenovo/FanControl."""
        body = {"FanSpeedPercent": pct, "Mode": "Manual"}
        status, resp = self._patch(self.OEM_FAN_PATH, body)
        if status in (200, 204):
            return True
        return False

    def restore_auto_control(self) -> bool:
        """Re-enable automatic fan management on the XCC.

        Called on clean shutdown or when target drops below threshold.
        """
        if not self._available:
            return False

        if not self._manual_active:
            _log.debug("[XCC_FAN] No manual override was active — nothing to restore.")
            return True

        success = False

        # Restore via the method that was used to set
        if self._working_method == self.METHOD_IPMI_LAN:
            success = self._restore_ipmi_lan_auto()

        if not success and self._working_method == self.METHOD_REDFISH_OEM_ACTION:
            status, _ = self._post(self.OEM_FAN_ACTION_PATH, {"FanControlMode": "Automatic"})
            if status in (200, 204):
                _log.info("[XCC_FAN] Restored auto via OEM action POST (status=%d).", status)
                success = True

        if not success:
            # Try OEM PATCH restore
            status, _ = self._patch(self.OEM_FAN_PATH, {"Mode": "Automatic"})
            if status in (200, 204):
                _log.info("[XCC_FAN] Restored automatic fan control via OEM endpoint (status=%d).", status)
                success = True

        if not success:
            # Try thermal PATCH with empty fans
            status, _ = self._patch(self.THERMAL_PATH, {"Fans": []})
            if status in (200, 204):
                _log.info("[XCC_FAN] Restored auto via thermal PATCH empty fans (status=%d).", status)
                success = True

        if not success and self._ipmi_lan_available:
            # Last resort: IPMI LAN auto restore
            success = self._restore_ipmi_lan_auto()

        if success:
            self._manual_active = False
        else:
            _log.warning("[XCC_FAN] Could not restore automatic fan control — BMC may require manual intervention.")

        return success

    def read_fan_speeds(self) -> List[Dict[str, Any]]:
        """Read current fan speeds from XCC Thermal endpoint.

        Returns list of dicts: [{"name": "Fan 1", "reading_pct": 56, "status": "OK"}, ...]
        """
        if not self._available:
            return []

        status, thermal = self._get(self.THERMAL_PATH)
        if status != 200 or not thermal:
            return []

        result = []
        for fan in thermal.get("Fans", []):
            result.append({
                "name": fan.get("Name", fan.get("MemberId", "?")),
                "reading_pct": fan.get("Reading"),
                "units": fan.get("ReadingUnits", "Percent"),
                "status": fan.get("Status", {}).get("Health", "?"),
            })
        return result
