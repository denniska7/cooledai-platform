#!/usr/bin/env python3
"""
CooledAI Watchdog Sidecar - Standalone API Heartbeat + Redfish Fail-Safe

Zero dependencies: Python stdlib only (urllib, json, ssl, base64, time).

Behavior:
- Heartbeats the main API every 5 seconds.
- If API does not respond for 30 seconds, takes over Redfish and forces
  all fans to Fail-Safe High (70% RPM) until connection returns.
- Runs indefinitely; suitable for systemd/supervisor.

Environment:
  COOLEDAI_API_URL        - API base URL (default: https://proactive-creativity-production.up.railway.app)
  COOLEDAI_API_TIMEOUT    - Seconds until takeover (default: 30)
  COOLEDAI_REDFISH_URLS   - Comma-separated Redfish base URLs (e.g. https://192.168.1.10)
  COOLEDAI_REDFISH_USER   - Redfish username
  COOLEDAI_REDFISH_PASS   - Redfish password
  COOLEDAI_FAIL_SAFE_PCT  - Fan percent in fail-safe (default: 70)
  COOLEDAI_VERIFY_SSL     - 0 to disable SSL verify for self-signed (default: 1)
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request


# --- Config ---

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


# --- HTTP (stdlib only) ---

def _http_get(url: str, timeout: float = 5.0, verify_ssl: bool = True) -> bool:
    """GET url, return True if 2xx."""
    ctx = ssl.create_default_context()
    if not verify_ssl:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _redfish_patch_cooling(
    base_url: str,
    user: str,
    password: str,
    cooling_percent: int,
    verify_ssl: bool = True,
    timeout: float = 5.0,
) -> bool:
    """
    PATCH Redfish Chassis/1/Thermal with CoolingPercent.
    Returns True if 2xx.
    """
    url = base_url.rstrip("/") + "/redfish/v1/Chassis/1/Thermal"
    payload = json.dumps({"CoolingPercent": min(100, max(0, cooling_percent))}).encode("utf-8")
    ctx = ssl.create_default_context()
    if not verify_ssl:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, data=payload, method="PATCH")
    req.add_header("Content-Type", "application/json")
    if user and password:
        auth = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


# --- Main Loop ---

def main() -> int:
    api_url = _env(
        "COOLEDAI_API_URL",
        "https://proactive-creativity-production.up.railway.app",
    )
    health_path = _env("COOLEDAI_HEALTH_PATH", "/health")
    api_timeout_sec = _env_int("COOLEDAI_API_TIMEOUT", 30)
    redfish_urls_str = _env("COOLEDAI_REDFISH_URLS", "")
    redfish_user = _env("COOLEDAI_REDFISH_USER", "")
    redfish_pass = _env("COOLEDAI_REDFISH_PASS", "")
    fail_safe_pct = _env_int("COOLEDAI_FAIL_SAFE_PCT", 70)
    verify_ssl = _env_bool("COOLEDAI_VERIFY_SSL", True)
    poll_interval = 5.0

    health_url = api_url.rstrip("/") + health_path
    redfish_urls = [u.strip() for u in redfish_urls_str.split(",") if u.strip()]

    if not redfish_urls:
        print("WARN: COOLEDAI_REDFISH_URLS empty; fail-safe will have no Redfish targets.", file=sys.stderr)

    last_ok = time.monotonic()
    in_fail_safe = False
    last_fail_safe_send = 0.0
    fail_safe_send_interval = 30.0  # Send 70% every 30s while in fail-safe

    print("Watchdog sidecar started.", file=sys.stderr)
    print(f"  API: {health_url}", file=sys.stderr)
    print(f"  Timeout: {api_timeout_sec}s", file=sys.stderr)
    print(f"  Redfish: {len(redfish_urls)} endpoint(s)", file=sys.stderr)
    print(f"  Fail-safe: {fail_safe_pct}% RPM", file=sys.stderr)

    while True:
        ok = _http_get(health_url, timeout=5.0, verify_ssl=verify_ssl)
        now = time.monotonic()

        if ok:
            last_ok = now
            if in_fail_safe:
                print("API recovered; exiting fail-safe.", file=sys.stderr)
                in_fail_safe = False
        else:
            elapsed = now - last_ok
            if elapsed >= api_timeout_sec:
                if not in_fail_safe:
                    print(
                        f"API unreachable for {elapsed:.0f}s; taking over Redfish. Fail-safe {fail_safe_pct}% RPM.",
                        file=sys.stderr,
                    )
                    in_fail_safe = True
                if now - last_fail_safe_send >= fail_safe_send_interval:
                    last_fail_safe_send = now
                    for url in redfish_urls:
                        try:
                            wrote = _redfish_patch_cooling(
                                url, redfish_user, redfish_pass, fail_safe_pct, verify_ssl=verify_ssl
                            )
                            if wrote:
                                print(f"  Sent {fail_safe_pct}% to {url}", file=sys.stderr)
                            else:
                                print(f"  FAIL to write {url}", file=sys.stderr)
                        except Exception as e:
                            print(f"  Error {url}: {e}", file=sys.stderr)

        time.sleep(poll_interval)


if __name__ == "__main__":
    sys.exit(main() or 0)
