#!/usr/bin/env python3
"""
Cold-start FOPDT calibration from a JSON/JSONL telemetry file.

Use after collecting time-series with columns: unix time (seconds), temperature_C,
cooling_rpm (or duty×max_rpm), power_w.

Example JSONL (one object per line):
  {"t": 1700000000, "T": 48.2, "cooling": 3200, "power": 120}
  {"t": 1700000003, "T": 48.1, "cooling": 3800, "power": 122}

Generate samples by running the agent + workload so fans step occasionally, then
export history or paste API telemetry into this format.

Usage:
  python3 scripts/cold_start_fopdt_calibration.py --rack-id pilot-node-01 data/fopdt_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.models.topology import RackRegistry  # noqa: E402


def load_samples(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl" or "\n{" in text:
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out
    data = json.loads(text)
    if isinstance(data, list):
        return data
    raise SystemExit("JSON must be a list of samples or JSONL")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit FOPDT from telemetry file")
    ap.add_argument("file", type=Path, help="JSON array or JSONL of samples")
    ap.add_argument("--rack-id", default="default_rack", help="Rack / node id for calibration file")
    ap.add_argument("--min-samples", type=int, default=30)
    args = ap.parse_args()

    samples = load_samples(args.file)
    reg = RackRegistry()
    for row in samples:
        t = float(row["t"])
        T = float(row["T"])
        cooling = float(row.get("cooling", row.get("cooling_rpm", 0)))
        power = float(row.get("power", row.get("power_w", 0)))
        reg.add_telemetry(args.rack_id, t, T, cooling, power)

    ok = reg.calibrate_rack(args.rack_id, min_samples=args.min_samples)
    rack = reg.get_rack(args.rack_id)
    fp = rack.fopdt
    print(json.dumps({
        "calibrated": ok,
        "rack_id": args.rack_id,
        "tau_s": fp.tau,
        "theta_s": fp.theta,
        "gain": fp.gain,
        "n_samples": fp.n_samples,
        "calibrated_at": fp.calibrated_at,
    }, indent=2))
    if not ok:
        print(
            "\nTip: need more samples and clearer cooling steps "
            "(|Δcooling| large enough). See core/models/topology.py calibrate_rack.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
