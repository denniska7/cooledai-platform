#!/usr/bin/env bash
# CPU idle / P-state coordination — DOCUMENTATION + MANUAL STEPS ONLY.
# CooledAI does NOT enable this automatically (latency / orchestrator risk).
#
# After GPU power governance is validated, site SRE may experiment:
#
#   sudo apt-get install -y linux-tools-common linux-tools-$(uname -r)
#   sudo cpupower frequency-info
#   # Example (DANGEROUS for latency-sensitive services — test on lab only):
#   # sudo cpupower frequency-set -g powersave
#
# Prefer per-cgroup or tuned profiles over global governors.
#
# Telemetry already includes CPU temp + GPU util; use portal / logs to correlate
# before changing idle policy.
#
echo "cpu_idle_coordination_stub: no changes applied. See docs/GPU_AND_CPU_POWER_PHASE2.md"
exit 0
