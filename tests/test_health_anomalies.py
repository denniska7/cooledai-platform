"""Test GET /health/anomalies: severity score and failing component history."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.shadow.shadow_logs import init_shadow_db, log_failing_component_event, get_failing_component_events
from core.anomaly import severity_score_from_finding, FailingComponentFinding

def main():
    init_shadow_db()
    f = FailingComponentFinding(
        node_id="CRAC-01",
        message="Failing Component: CRAC-01 ...",
        power_draw_prev=1000,
        power_draw_now=1200,
        delta_t_prev=5.0,
        delta_t_now=4.0,
    )
    sev = severity_score_from_finding(f)
    print("severity_score", sev)
    assert 0 <= sev <= 100
    log_failing_component_event(
        f.node_id, f.message,
        f.power_draw_prev, f.power_draw_now,
        f.delta_t_prev, f.delta_t_now,
        sev,
    )
    events = get_failing_component_events(limit=10)
    print("count", len(events))
    assert len(events) >= 1
    assert "severity_score" in events[0]
    assert "node_id" in events[0]
    print("first", events[0])
    print("OK")

if __name__ == "__main__":
    main()
