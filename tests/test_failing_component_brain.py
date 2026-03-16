"""Test OptimizationBrain with FailingComponentFindings and InfluenceMap redistribution."""
from core.hal.base_node import CRAC_Unit, ServerNode
from core.ingestion.influence_map import InfluenceMap
from core.optimization.optimization_brain import OptimizationBrain
from core.anomaly import FailingComponentFinding

def test_failing_component_redistribution():
    im = InfluenceMap()
    im.set_influence("CRAC-01", "Zone-A", 0.6)
    im.set_influence("CRAC-02", "Zone-A", 0.5)

    crac1 = CRAC_Unit(thermal_input=28, power_draw=1100, cooling_output=1200, utilization=60, node_id="CRAC-01", supply_temp=23, return_temp=28)
    crac2 = CRAC_Unit(thermal_input=26, power_draw=800, cooling_output=1100, utilization=50, node_id="CRAC-02", supply_temp=22, return_temp=26)
    server = ServerNode(thermal_input=45, power_draw=200, cooling_output=0, utilization=70, node_id="server-1")
    nodes = [crac1, crac2, server]

    finding = FailingComponentFinding(node_id="CRAC-01", message="Failing Component: CRAC-01 ...", power_draw_prev=1000, power_draw_now=1100, delta_t_prev=5.0, delta_t_now=4.5)
    findings = [finding]

    brain = OptimizationBrain()
    gap = brain.analyze(nodes, influence_map=im, failing_component_findings=findings)

    assert "Reducing load on" in gap.diagnostic_reasoning
    assert "redistributing cooling to" in gap.diagnostic_reasoning
    assert "CRAC-01" in gap.diagnostic_reasoning
    assert "CRAC-02" in gap.diagnostic_reasoning or "other" in gap.diagnostic_reasoning.lower()
    assert gap.recommended_cooling_delta_by_unit.get("CRAC-01") < 0
    assert gap.recommended_cooling_delta_by_unit.get("CRAC-02") > 0
    assert gap.raw_metrics.get("redistribution_healthy_units") is not None
    print("diagnostic_reasoning:", gap.diagnostic_reasoning)
    print("recommended_cooling_delta_by_unit:", gap.recommended_cooling_delta_by_unit)
    print("OK")

if __name__ == "__main__":
    test_failing_component_redistribution()
