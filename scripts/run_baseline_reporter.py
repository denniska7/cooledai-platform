#!/usr/bin/env python3
"""
Run CooledAI Baseline Reporter: record 24h of current-state snapshots (no adjustments)
and/or generate the comparison report (Current PUE vs Projected CooledAI PUE, CO2, rebates).

Usage:
  # Record simulated baseline snapshots every 5 min for 24h (for testing/demo)
  python scripts/run_baseline_reporter.py record --interval 5 --hours 24

  # Generate comparison report from existing baseline data
  python scripts/run_baseline_reporter.py report

  # Record one snapshot (e.g. from live metrics)
  python scripts/run_baseline_reporter.py snapshot --it-kw 120 --cooling-kw 40

Environment (optional):
  COOLEDAI_UTILITY_RATE    $/kWh (default 0.12)
  COOLEDAI_CO2_KG_PER_KWH  kg CO2 per kWh (default 0.4)
  COOLEDAI_REBATE_PER_KWH  $/kWh for rebate estimate (default 0.05)
  COOLEDAI_PROJECTED_PUE_IMPROVEMENT_PCT  default 12
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.baseline import BaselineReporter, init_baseline_db
from backend.baseline.baseline_storage import DEFAULT_DB_PATH, BASELINE_WINDOW_HOURS


def _reporter() -> BaselineReporter:
    db_path = Path(os.environ.get("COOLEDAI_BASELINE_DB", str(project_root / "data" / "baseline.db")))
    return BaselineReporter(
        db_path=db_path,
        utility_rate_per_kwh=float(os.environ.get("COOLEDAI_UTILITY_RATE", "0.12")),
        co2_kg_per_kwh=float(os.environ.get("COOLEDAI_CO2_KG_PER_KWH", "0.4")),
        rebate_per_kwh=float(os.environ.get("COOLEDAI_REBATE_PER_KWH", "0.05")),
        projected_pue_improvement_pct=float(os.environ.get("COOLEDAI_PROJECTED_PUE_IMPROVEMENT_PCT", "12")),
    )


def cmd_report(args) -> None:
    """Generate and print comparison report from existing baseline data."""
    init_baseline_db(_reporter().db_path)
    reporter = _reporter()
    report = reporter.generate_comparison_report(hours=args.hours)
    print("--- CooledAI Baseline Comparison Report ---")
    print(report.message)
    print()
    print("Current PUE:              ", report.current_pue)
    print("Projected CooledAI PUE:   ", report.projected_cooledai_pue)
    print("PUE improvement:           ", f"{report.pue_improvement_pct:.1f}%")
    print("Current total kWh (window):", report.current_total_kwh_24h)
    print("Projected total kWh:      ", report.projected_total_kwh_24h)
    print("kWh saved:                ", report.kwh_saved_24h)
    print("CO2 reduction:            ", f"{report.co2_reduction_kg:.0f} kg ({report.co2_reduction_tonnes:.4f} tonnes)")
    print("Estimated utility rebates:", f"${report.estimated_utility_rebates_usd:.2f}")
    print("Dollars saved (window):   ", f"${report.dollars_saved_24h:.2f}")
    print("Samples:                  ", report.sample_count)
    print("Window:                   ", report.start_ts.isoformat(), "→", report.end_ts.isoformat())
    if report.details:
        print("Details (rates):           ", report.details)


def cmd_record(args) -> None:
    """Record simulated baseline snapshots every interval minutes for the given hours."""
    import random
    reporter = _reporter()
    init_baseline_db(reporter.db_path)
    interval_sec = args.interval * 60
    end_time = time.time() + (args.hours * 3600)
    # Simulated facility: IT 100–150 kW, PUE ~1.3–1.5
    print(f"Recording baseline every {args.interval} min for {args.hours} h (simulated data)...")
    count = 0
    while time.time() < end_time:
        it_kw = 100 + random.uniform(0, 50)
        cooling_kw = it_kw * random.uniform(0.25, 0.45)
        reporter.record_snapshot(it_power_kw=it_kw, cooling_power_kw=cooling_kw)
        count += 1
        print(f"  Snapshot {count} recorded (IT={it_kw:.0f} kW, cooling={cooling_kw:.0f} kW)")
        if time.time() < end_time:
            time.sleep(interval_sec)
    print(f"Recorded {count} snapshots. Run 'report' to generate comparison.")


def cmd_snapshot(args) -> None:
    """Record a single baseline snapshot (e.g. from live metrics)."""
    reporter = _reporter()
    init_baseline_db(reporter.db_path)
    row_id = reporter.record_snapshot(
        it_power_kw=args.it_kw,
        cooling_power_kw=args.cooling_kw,
        pue=args.pue,
        total_facility_kw=args.total_kw,
    )
    print(f"Baseline snapshot recorded (id={row_id}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="CooledAI Baseline Reporter")
    sub = parser.add_subparsers(dest="command", required=True)
    # report
    p_report = sub.add_parser("report", help="Generate comparison report from stored baseline data")
    p_report.add_argument("--hours", type=float, default=BASELINE_WINDOW_HOURS, help="Window hours (default 24)")
    p_report.set_defaults(run=cmd_report)
    # record (simulated)
    p_record = sub.add_parser("record", help="Record simulated snapshots every N min for 24h (demo)")
    p_record.add_argument("--interval", type=int, default=5, help="Minutes between snapshots")
    p_record.add_argument("--hours", type=float, default=24, help="Total recording hours")
    p_record.set_defaults(run=cmd_record)
    # snapshot (one shot)
    p_snap = sub.add_parser("snapshot", help="Record one baseline snapshot")
    p_snap.add_argument("--it-kw", type=float, required=True, help="IT power kW")
    p_snap.add_argument("--cooling-kw", type=float, required=True, help="Cooling power kW")
    p_snap.add_argument("--pue", type=float, default=None, help="PUE (optional)")
    p_snap.add_argument("--total-kw", type=float, default=None, help="Total facility kW (optional)")
    p_snap.set_defaults(run=cmd_snapshot)
    #
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
