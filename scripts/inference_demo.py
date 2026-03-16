#!/usr/bin/env python3
"""
CooledAI Inference Demo — AI Decision Report

Demonstrates the full predictive pipeline:
  1. Sends mock rack telemetry to POST /optimize
  2. Receives the AI's recommended cooling delta, power metrics, and reasoning
  3. Runs a local FOPDT prediction (60-second horizon) using the topology model
  4. Compares recommended power vs. 100% Emergency Blast

Usage:
    # With running backend (default http://localhost:8000):
    python scripts/inference_demo.py

    # Override URL or key:
    COOLEDAI_API_URL=http://localhost:8000 COOLEDAI_API_KEY=mykey python scripts/inference_demo.py
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to path so we can import core modules for local FOPDT
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("COOLEDAI_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("COOLEDAI_API_KEY", "")

# Try loading from backend/.env or root .env if key not in environment
if not API_KEY:
    for env_path in [PROJECT_ROOT / "backend" / ".env", PROJECT_ROOT / ".env"]:
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("COOLEDAI_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            API_KEY = val
                            break
            except Exception:
                pass
        if API_KEY:
            break

if not API_KEY:
    print("WARNING: No COOLEDAI_API_KEY found in environment or .env files.")
    print("         The backend will reject authenticated requests.")
    print("         Set COOLEDAI_API_KEY=<your-key> or add it to backend/.env\n")


def validate_api_key(key: str) -> bool:
    """Check that the key looks like a valid 32-char hex string (or any non-empty key)."""
    if not key:
        return False
    if re.fullmatch(r"[0-9a-fA-F]{32,}", key):
        return True
    # Accept any non-empty key (dev mode may use short strings like 'test123')
    return True


# ---------------------------------------------------------------------------
# Mock Rack Payload
# ---------------------------------------------------------------------------
# Scenario: A single server rack running hot.
#   - current_temp (thermal_input): 75°C — above the 65°C target
#   - cpu_utilization: 90% — heavy GPU/AI training workload
#   - power_draw: 45,000 W — high-density rack (e.g. 8× Blackwell GB200)
#   - cooling_output: 2100 RPM — fans are running but not at max (3000 RPM)
MOCK_PAYLOAD = [
    {
        "thermal_input": 75.0,
        "power_draw": 45000.0,
        "cooling_output": 2100.0,
        "utilization": 0.90,
        "node_id": "rack_01",
    }
]

# For the local FOPDT demo: simulate what happens over 60 seconds
CURRENT_TEMP = 75.0
CURRENT_RPM = 2100.0
MAX_RPM = 3000.0
RATED_FAN_POWER_W = 12000.0  # 12 kW per CRAC at 100%


# ---------------------------------------------------------------------------
# API Call
# ---------------------------------------------------------------------------
def call_optimize(payload: list, api_url: str, api_key: str) -> dict:
    """POST /optimize with the mock data and return the JSON response."""
    url = api_url.rstrip("/") + "/optimize"
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = api_key

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"\n  API Error: HTTP {e.code} — {e.reason}")
        if error_body:
            print(f"  Body: {error_body[:500]}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\n  Connection Error: {e.reason}")
        print(f"  Is the backend running at {api_url}?")
        print("  Start it with: cd backend && source venv/bin/activate && uvicorn main:app --reload")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Local FOPDT Prediction (60-second horizon)
# ---------------------------------------------------------------------------
def predict_fopdt_60s(
    current_temp: float,
    current_rpm: float,
    recommended_delta: float,
    max_rpm: float = 3000.0,
) -> dict:
    """
    Run a local FOPDT prediction to estimate temperature 60 seconds from now.

    Uses the topology module's Rack model with default calibration parameters.
    The recommended_delta is applied as a step change at t=0.
    """
    try:
        import numpy as np
        from core.models.topology import Rack, FOPDTParams

        rack = Rack(
            rack_id="rack_01",
            fopdt=FOPDTParams(
                tau=120.0,       # 120s time constant (typical for loaded rack)
                theta=15.0,      # 15s dead time (fan ramp to airflow effect)
                gain=-0.01,      # -0.01 °C per RPM increase
                thermal_mass=150_000.0,  # 150 kJ/K
            ),
        )

        proposed_rpm = min(max_rpm, current_rpm * (1.0 + recommended_delta))
        proposed_rpm = max(0.0, proposed_rpm)

        # Superposition: single step at t=0
        dt_eff = 60.0 - 0.0 - rack.theta  # effective time after dead time
        if dt_eff > 0:
            delta_cooling = proposed_rpm - current_rpm
            delta_T_ss = rack.gain * delta_cooling  # steady-state delta
            delta_T = delta_T_ss * (1.0 - np.exp(-dt_eff / rack.tau))
            predicted_temp = current_temp + delta_T
        else:
            # Still in dead-time window
            predicted_temp = current_temp
            delta_T = 0.0

        return {
            "predicted_temp_60s": round(predicted_temp, 2),
            "delta_T": round(delta_T, 2),
            "proposed_rpm": round(proposed_rpm, 0),
            "dead_time_s": rack.theta,
            "time_constant_s": rack.tau,
            "thermal_mass_jk": rack.fopdt.thermal_mass,
        }

    except ImportError:
        return {"error": "core.models.topology not available — run from project root"}


# ---------------------------------------------------------------------------
# Power Comparison: Recommended vs. Emergency Blast
# ---------------------------------------------------------------------------
def power_comparison(
    current_rpm: float,
    recommended_delta: float,
    max_rpm: float = 3000.0,
    rated_power_w: float = 12000.0,
) -> dict:
    """Compare fan power at recommended RPM vs. 100% Emergency Blast."""
    proposed_rpm = min(max_rpm, max(0.0, current_rpm * (1.0 + recommended_delta)))

    # Fan Affinity Law: P = P_rated × (N / N_rated)³
    power_proposed = rated_power_w * (proposed_rpm / max_rpm) ** 3
    power_blast = rated_power_w * (max_rpm / max_rpm) ** 3  # = rated_power_w
    power_current = rated_power_w * (current_rpm / max_rpm) ** 3

    savings_vs_blast = power_blast - power_proposed
    savings_pct = (savings_vs_blast / power_blast) * 100 if power_blast > 0 else 0

    return {
        "current_rpm": round(current_rpm, 0),
        "proposed_rpm": round(proposed_rpm, 0),
        "max_rpm": round(max_rpm, 0),
        "power_at_current_W": round(power_current, 0),
        "power_at_proposed_W": round(power_proposed, 0),
        "power_at_100pct_W": round(power_blast, 0),
        "savings_vs_blast_W": round(savings_vs_blast, 0),
        "savings_vs_blast_pct": round(savings_pct, 1),
    }


# ---------------------------------------------------------------------------
# Pretty Print
# ---------------------------------------------------------------------------
SEP = "═" * 66


def print_report(api_response: dict, fopdt: dict, power: dict) -> None:
    """Print the Decision Report to the terminal."""
    delta = api_response.get("recommended_cooling_delta", 0)
    raw = api_response.get("raw_metrics", {})
    recs = api_response.get("recommendations", [])

    print()
    print(SEP)
    print("  ❄️  CooledAI — AI Decision Report")
    print(SEP)

    # ── Input ──
    node = MOCK_PAYLOAD[0]
    print()
    print("  📡 Input Telemetry")
    print(f"     Rack:            {node['node_id']}")
    print(f"     Temperature:     {node['thermal_input']}°C  (target: 65°C)")
    print(f"     CPU Utilization: {node['utilization'] * 100:.0f}%")
    print(f"     Power Draw:      {node['power_draw'] / 1000:.1f} kW")
    print(f"     Current Fan RPM: {node['cooling_output']:.0f}")
    print()

    # ── FOPDT Prediction ──
    print("  🔮 FOPDT Physics Prediction (60-second horizon)")
    if "error" in fopdt:
        print(f"     {fopdt['error']}")
    else:
        print(f"     Predicted Temp @ T+60s:  {fopdt['predicted_temp_60s']}°C  (ΔT = {fopdt['delta_T']:+.2f}°C)")
        print(f"     Proposed Fan RPM:        {fopdt['proposed_rpm']:.0f}")
        print(f"     Dead Time (θ):           {fopdt['dead_time_s']}s  (delay before cooling effect)")
        print(f"     Time Constant (τ):       {fopdt['time_constant_s']}s  (thermal inertia)")
        print(f"     Thermal Mass (m·cₚ):     {fopdt['thermal_mass_jk']:,.0f} J/K")
    print()

    # ── AI Decision ──
    print("  🧠 Optimization Brain Decision")
    print(f"     Cooling Delta:           {delta:+.0%}")
    print(f"     Efficiency Score:        {api_response.get('efficiency_score', 0):.1f} / 100")
    print(f"     Over-Provisioning:       {api_response.get('over_provisioning_ratio', 0):.1%}")
    print(f"     Thermal Lag:             {api_response.get('thermal_lag_seconds', 0):.1f}s")
    print(f"     Prediction Confidence:   {api_response.get('prediction_confidence', 0):.0%}")
    if api_response.get("cautionary_cooling_state"):
        print("     ⚠️  Cautionary Cooling:  ACTIVE (confidence below threshold)")
    print()

    # ── Power-Cost Analysis (Fan Affinity Laws) ──
    print("  ⚡ Power-Cost Analysis (Fan Affinity Laws: P ∝ N³)")
    print(f"     Current Power ({power['current_rpm']:.0f} RPM):   {power['power_at_current_W']:>7,.0f} W")
    print(f"     Proposed Power ({power['proposed_rpm']:.0f} RPM):  {power['power_at_proposed_W']:>7,.0f} W")
    print(f"     Emergency Blast (100%):    {power['power_at_100pct_W']:>7,.0f} W")
    print(f"     ─────────────────────────────────────────")
    print(f"     Savings vs. Blast:         {power['savings_vs_blast_W']:>7,.0f} W  ({power['savings_vs_blast_pct']:.1f}%)")

    # API-reported power metrics (if available from optimizer)
    cooling_power = raw.get("cooling_power_w", 0)
    power_savings = raw.get("power_savings_w", 0)
    kwh_saved = raw.get("kwh_saved_per_hour", 0)
    if cooling_power or power_savings:
        print()
        print(f"     Zone Cooling Power:        {cooling_power:>7,.0f} W")
        print(f"     Power Savings (optimizer): {power_savings:>7,.0f} W")
        print(f"     kWh Saved / Hour:          {kwh_saved:>7.2f}")
    print()

    # ── Financial ──
    dollars_today = api_response.get("dollars_saved_today", 0)
    monthly_roi = api_response.get("estimated_monthly_roi", 0)
    if dollars_today or monthly_roi:
        print("  💰 Financial Estimate")
        print(f"     Dollars Saved Today:     ${dollars_today:,.2f}")
        print(f"     Estimated Monthly ROI:   ${monthly_roi:,.2f}")
        print()

    # ── Recommendations ──
    if recs:
        print("  📋 Recommendations")
        for r in recs[:5]:
            print(f"     • {r}")
        print()

    # ── Reasoning ──
    reasoning = api_response.get("diagnostic_reasoning", "")
    if reasoning:
        print("  💬 Diagnostic Reasoning")
        # Wrap long reasoning text
        words = reasoning.split()
        line = "     "
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(line)
                line = "     " + w
            else:
                line += " " + w if line.strip() else "     " + w
        if line.strip():
            print(line)
        print()

    # ── Key Validation ──
    print("  🔐 Authentication")
    if API_KEY:
        masked = API_KEY[:4] + "…" + API_KEY[-4:] if len(API_KEY) > 8 else "****"
        is_hex32 = bool(re.fullmatch(r"[0-9a-fA-F]{32}", API_KEY))
        fmt = "32-char hex ✓" if is_hex32 else f"{len(API_KEY)}-char string"
        print(f"     API Key:  {masked}  ({fmt})")
    else:
        print("     API Key:  (none — running unauthenticated)")
    print(f"     Endpoint: POST {API_URL}/optimize")
    print()
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print(SEP)
    print("  ❄️  CooledAI Inference Demo")
    print(SEP)
    print(f"\n  Backend:  {API_URL}")
    print(f"  API Key:  {'set (' + str(len(API_KEY)) + ' chars)' if API_KEY else 'not set'}")
    print(f"  Payload:  1 rack  |  75°C  |  90% CPU  |  45 kW\n")

    if not validate_api_key(API_KEY):
        print("  ⚠️  No valid API key. Set COOLEDAI_API_KEY in environment or .env\n")

    # 1. Call the API
    print("  → Sending POST /optimize ...", end="", flush=True)
    result = call_optimize(MOCK_PAYLOAD, API_URL, API_KEY)
    print(" ✓")

    # 2. Local FOPDT prediction
    print("  → Running FOPDT prediction (60s horizon) ...", end="", flush=True)
    delta = result.get("recommended_cooling_delta", 0)
    fopdt = predict_fopdt_60s(CURRENT_TEMP, CURRENT_RPM, delta, MAX_RPM)
    print(" ✓")

    # 3. Power comparison
    print("  → Computing Fan Affinity power costs ...", end="", flush=True)
    power = power_comparison(CURRENT_RPM, delta, MAX_RPM, RATED_FAN_POWER_W)
    print(" ✓")

    # 4. Print the report
    print_report(result, fopdt, power)


if __name__ == "__main__":
    main()
