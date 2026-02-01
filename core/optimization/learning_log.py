"""
CooledAI Learning Log - Feedback Loop for Model Tuning

Records Predicted Temp vs Actual Temp achieved after each command.
tune_model() adjusts Aggression Coefficients to minimize prediction error.
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Default log path
DEFAULT_LEARNING_LOG = "data/learning_log.csv"

# Aggression coefficients (tuned by tune_model)
# Higher = more aggressive cooling changes
AGGRESSION_BASE = 1.0
AGGRESSION_NEAR_LIMIT = 2.0  # When approaching MAX_SAFE_TEMP


@dataclass
class LearningEntry:
    """Single learning log entry."""
    timestamp: str
    predicted_temp: float
    actual_temp: float
    error: float
    recommended_delta: float
    current_thermal: float
    power_draw: float
    cooling_output: float


def _ensure_log_dir(path: str) -> None:
    """Ensure directory exists for log file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def log_prediction(
    predicted_temp: float,
    actual_temp: float,
    recommended_delta: float,
    current_thermal: float,
    power_draw: float,
    cooling_output: float,
    log_path: str = DEFAULT_LEARNING_LOG,
) -> None:
    """
    Record predicted vs actual temp for feedback loop.
    Call after applying a command and measuring actual result.
    """
    _ensure_log_dir(log_path)
    error = actual_temp - predicted_temp
    entry = LearningEntry(
        timestamp=datetime.now().isoformat(),
        predicted_temp=predicted_temp,
        actual_temp=actual_temp,
        error=error,
        recommended_delta=recommended_delta,
        current_thermal=current_thermal,
        power_draw=power_draw,
        cooling_output=cooling_output,
    )
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "predicted_temp", "actual_temp", "error",
                "recommended_delta", "current_thermal", "power_draw", "cooling_output",
            ])
        w.writerow([
            entry.timestamp,
            f"{entry.predicted_temp:.2f}",
            f"{entry.actual_temp:.2f}",
            f"{entry.error:.2f}",
            f"{entry.recommended_delta:.2f}",
            f"{entry.current_thermal:.2f}",
            f"{entry.power_draw:.2f}",
            f"{entry.cooling_output:.2f}",
        ])


def load_learning_log(log_path: str = DEFAULT_LEARNING_LOG) -> List[LearningEntry]:
    """Load learning log entries."""
    if not os.path.exists(log_path):
        return []
    entries = []
    with open(log_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                entries.append(LearningEntry(
                    timestamp=row.get("timestamp", ""),
                    predicted_temp=float(row.get("predicted_temp", 0)),
                    actual_temp=float(row.get("actual_temp", 0)),
                    error=float(row.get("error", 0)),
                    recommended_delta=float(row.get("recommended_delta", 0)),
                    current_thermal=float(row.get("current_thermal", 0)),
                    power_draw=float(row.get("power_draw", 0)),
                    cooling_output=float(row.get("cooling_output", 0)),
                ))
            except (ValueError, KeyError):
                continue
    return entries


def tune_model(
    log_path: str = DEFAULT_LEARNING_LOG,
    min_entries: int = 10,
) -> dict:
    """
    Analyze learning log and adjust Aggression Coefficients to minimize error.
    Returns dict with tuned coefficients.
    """
    entries = load_learning_log(log_path)
    if len(entries) < min_entries:
        return {
            "aggression_base": AGGRESSION_BASE,
            "aggression_near_limit": AGGRESSION_NEAR_LIMIT,
            "mae": 0.0,
            "entries_used": len(entries),
            "tuned": False,
        }

    errors = [e.error for e in entries]
    mae = sum(abs(e) for e in errors) / len(errors)

    # If we consistently over-predict (actual < predicted), we're too aggressive
    # If we under-predict (actual > predicted), we're too conservative
    mean_error = sum(errors) / len(errors)
    # Positive mean_error = we under-predicted (actual hotter) -> need more aggression
    # Negative = we over-predicted (actual cooler) -> reduce aggression

    # Simple tuning: scale aggression by 1 + 0.1 * sign(mean_error)
    aggression_adj = 1.0 + 0.1 * (1 if mean_error > 0.5 else -1 if mean_error < -0.5 else 0)
    aggression_base = max(0.5, min(2.0, AGGRESSION_BASE * aggression_adj))

    return {
        "aggression_base": aggression_base,
        "aggression_near_limit": max(aggression_base, AGGRESSION_NEAR_LIMIT),
        "mae": mae,
        "mean_error": mean_error,
        "entries_used": len(entries),
        "tuned": True,
    }
