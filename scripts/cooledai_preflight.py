#!/usr/bin/env python3
"""
CooledAI Preflight Check — validate configuration before first run.

Usage:
    python3 scripts/cooledai_preflight.py [--config configs/cooledai.yaml]

Checks:
  1. Validate cooledai.yaml against schema
  2. Test XCC BMC connectivity (Redfish GET /redfish/v1/)
  3. Check API health endpoint (GET /api/v1/health)
  4. Send one test telemetry record and verify 200 response

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check_config(config_path: str) -> bool:
    """Check 1: Validate cooledai.yaml."""
    print(f"\n[1/4] Validating config: {config_path}")
    try:
        from core.config.settings import Settings
        settings = Settings(config_path)
        settings.validate(strict=False)
        print(f"  Client: {settings.client_name or '(not set)'}")
        print(f"  Site:   {settings.site or '(not set)'}")
        print(f"  GPUs:   {settings.gpu_count}")
        print(f"  [{PASS}] Config is valid")
        return True
    except Exception as exc:
        print(f"  [{FAIL}] {exc}")
        return False


def check_xcc(config_path: str) -> bool:
    """Check 2: Test XCC BMC connectivity."""
    print("\n[2/4] Testing XCC BMC connectivity")
    host = os.environ.get("XCC_BMC_HOST", "")
    user = os.environ.get("XCC_BMC_USER", "USERID")
    passwd = os.environ.get("XCC_BMC_PASS", "")

    if not host:
        print(f"  [{FAIL}] XCC_BMC_HOST not set — skipping BMC check")
        print("  Set XCC_BMC_HOST in /etc/cooledai/agent.env or environment")
        return False

    url = f"https://{host}/redfish/v1/"
    print(f"  Connecting to {url} ...")

    try:
        import base64
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        cred = base64.b64encode(f"{user}:{passwd}".encode()).decode()
        req = Request(url, headers={"Authorization": f"Basic {cred}"})
        resp = urlopen(req, timeout=10, context=ctx)
        data = json.loads(resp.read())
        product = data.get("Product", data.get("Name", "Unknown"))
        print(f"  BMC Product: {product}")
        print(f"  [{PASS}] XCC BMC reachable")
        return True
    except Exception as exc:
        print(f"  [{FAIL}] Cannot reach BMC: {exc}")
        return False


def check_api_health(api_url: str, api_key: str) -> bool:
    """Check 3: GET /api/v1/health."""
    print("\n[3/4] Checking API health endpoint")
    url = f"{api_url.rstrip('/')}/api/v1/health"
    print(f"  GET {url}")

    try:
        req = Request(url, headers={"X-API-Key": api_key})
        resp = urlopen(req, timeout=10)
        data = json.loads(resp.read())
        status = data.get("status", "unknown")
        version = data.get("version", "unknown")
        print(f"  Status:  {status}")
        print(f"  Version: {version}")
        if status in ("healthy", "degraded"):
            print(f"  [{PASS}] API is responsive")
            return True
        else:
            print(f"  [{FAIL}] API status is '{status}'")
            return False
    except Exception as exc:
        print(f"  [{FAIL}] Cannot reach API: {exc}")
        return False


def check_telemetry(api_url: str, api_key: str, node_id: str) -> bool:
    """Check 4: Send one test telemetry record."""
    print("\n[4/4] Sending test telemetry record")
    url = f"{api_url.rstrip('/')}/api/v1/telemetry"

    payload = json.dumps({
        "node_id": node_id,
        "telemetry": [{
            "node_id": node_id,
            "gpu_temp_c": 42.0,
            "gpu_power_w": 50.0,
            "cpu_temp_c": 38.0,
            "fan_rpm": 3000,
            "timestamp_utc": "2026-01-01T00:00:00Z",
        }],
    }).encode("utf-8")

    try:
        req = Request(
            url,
            data=payload,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
        )
        resp = urlopen(req, timeout=10)
        status = resp.getcode()
        body = json.loads(resp.read())
        print(f"  Status: {status}")
        print(f"  Response: {body}")
        if status == 200:
            print(f"  [{PASS}] Telemetry accepted")
            return True
        else:
            print(f"  [{FAIL}] Unexpected status {status}")
            return False
    except Exception as exc:
        print(f"  [{FAIL}] Telemetry POST failed: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="CooledAI Preflight Check")
    parser.add_argument("--config", default="configs/cooledai.yaml")
    parser.add_argument("--api-url", default=os.environ.get(
        "COOLEDAI_API_URL", "https://proactive-creativity-production.up.railway.app"))
    parser.add_argument("--api-key", default=os.environ.get("COOLEDAI_API_KEY", ""))
    parser.add_argument("--node-id", default=os.environ.get("COOLEDAI_NODE_ID", "preflight-test"))
    args = parser.parse_args()

    print("=" * 60)
    print("  CooledAI Preflight Check")
    print("=" * 60)

    results = [
        ("Config validation", check_config(args.config)),
        ("XCC BMC connectivity", check_xcc(args.config)),
        ("API health", check_api_health(args.api_url, args.api_key)),
        ("Telemetry POST", check_telemetry(args.api_url, args.api_key, args.node_id)),
    ]

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        icon = PASS if passed else FAIL
        print(f"  [{icon}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks passed. CooledAI is ready for deployment.")
        sys.exit(0)
    else:
        print("One or more checks failed. Review errors above before deploying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
