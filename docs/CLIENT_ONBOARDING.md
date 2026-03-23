# CooledAI — Client Onboarding Playbook

## Overview

This document defines the systematic process for onboarding a new client site onto the CooledAI thermal optimization platform. The goal is to collect all required information upfront so deployment is predictable and repeatable regardless of hardware vendor or data center configuration.

---

## Phase 1: Discovery Questionnaire

Send this to the client before any site work begins.

### 1.1 Facility & Environment

| # | Question | Why We Need It |
|---|----------|----------------|
| 1 | How many racks/nodes will CooledAI manage? | Scope licensing and agent deployment |
| 2 | What is the ambient air temperature range at the inlet? (summer peak, winter low) | Calibrates thermal baseline and sets safe operating bounds |
| 3 | What cooling infrastructure is in place? (CRAC units, in-row coolers, rear-door heat exchangers, liquid cooling loops) | Determines which cooling actuators we can optimize |
| 4 | Is there hot aisle / cold aisle containment? | Affects airflow modeling and sensor placement |
| 5 | What is the current measured PUE? (or best estimate) | Establishes the efficiency baseline we improve against |
| 6 | Are there any regulatory or compliance constraints on temperature ranges? (e.g., ASHRAE A1-A4 class) | Sets hard temperature ceilings the optimizer must never exceed |
| 7 | What is the facility's power capacity per rack? (kW) | Determines thermal headroom and alert thresholds |

### 1.2 Server Hardware

| # | Question | Why We Need It |
|---|----------|----------------|
| 8 | Server vendor and model? (e.g., Lenovo ThinkSystem ST550, Dell PowerEdge R760, HPE ProLiant DL380) | Determines BMC type, IPMI OEM commands, and Redfish API surface |
| 9 | BMC/IPMI firmware type and version? (e.g., Lenovo XCC v1.90, Dell iDRAC9 v7.x, HPE iLO6) | Fan control capabilities vary drastically by firmware version |
| 10 | Number and type of GPUs per node? (e.g., 2x NVIDIA Quadro P2000, 4x A100, 8x H100) | Determines power envelope, thermal mass, and GPU governor strategy |
| 11 | GPU TDP / default power limit per card? (watts) | Sets active compute trigger thresholds for calibration |
| 12 | Number and type of fans per node? (and rated max RPM if known) | Configures fan controller parameters (bootstrap floor, ceiling) |
| 13 | Current BIOS operating/thermal mode? (e.g., Efficiency_FavorPower, CustomMode, Performance, Acoustic) | Directly affects fan baseline — wrong mode can block optimization |
| 14 | Is liquid cooling present? If yes, what percentage of heat is liquid-captured? | Adjusts the air-side optimization target (e.g., 90% liquid / 10% air) |

### 1.3 BMC / Management Access

| # | Question | Why We Need It |
|---|----------|----------------|
| 15 | BMC/IPMI network IP address for each managed node | Agent connects to BMC for fan control and telemetry |
| 16 | BMC credentials (username / password) | Required for Redfish API and IPMI-over-LAN commands |
| 17 | Is IPMI-over-LAN enabled? | Primary fan control transport — must be enabled |
| 18 | Is Redfish API enabled and accessible? | Used for BIOS configuration, thermal readings, and health monitoring |
| 19 | Are there any firewall rules between the OS and BMC network? | Agent must reach BMC from the host OS |
| 20 | Is there a shared BMC/management VLAN or dedicated out-of-band network? | Network topology for agent configuration |

### 1.4 Operating System & Access

| # | Question | Why We Need It |
|---|----------|----------------|
| 21 | OS and version? (e.g., Ubuntu 24.04, RHEL 9, Rocky 9) | Package compatibility and systemd configuration |
| 22 | SSH access credentials (or key-based auth) | Agent deployment and management |
| 23 | Does the deployment user have sudo/root access? | Required for systemd service installation and ipmitool |
| 24 | Is `ipmitool` installed? (or can it be installed?) | Required for IPMI-over-LAN fan commands |
| 25 | Is `nvidia-smi` available? (for GPU nodes) | Required for GPU power/temp telemetry |
| 26 | Is there an existing DCIM, monitoring, or telemetry system? (e.g., Prometheus, Grafana, Nagios) | Integration points for dashboards and alerting |
| 27 | Network connectivity to CooledAI cloud API? (or air-gapped?) | Determines on-prem vs hybrid deployment model |

### 1.5 Workload Profile

| # | Question | Why We Need It |
|---|----------|----------------|
| 28 | What is the primary workload type? (AI training, inference, HPC, general compute, mixed) | Shapes the thermal prediction model — bursty inference vs sustained training have very different thermal profiles |
| 29 | Is the workload steady-state or highly variable? | Determines calibration window length and spike bypass thresholds |
| 30 | Are there known peak usage windows? (time of day, batch jobs, etc.) | Pre-cooling opportunity scheduling |
| 31 | What is the typical GPU utilization range? (idle %, peak %) | Sets idle/active detection thresholds |
| 32 | Are there any thermal incidents on record? (shutdowns, throttling events) | Calibrates risk tolerance and alert sensitivity |

### 1.6 Control vs Pilot Setup

| # | Question | Why We Need It |
|---|----------|----------------|
| 33 | Can we designate one node as "pilot" (CooledAI-optimized) and one as "control" (traditional cooling)? | A/B comparison is essential to prove savings |
| 34 | Are the pilot and control nodes identical hardware? | Ensures apples-to-apples comparison |
| 35 | Can we run an identical synthetic workload on both nodes during evaluation? | Standardized benchmark eliminates workload variability |
| 36 | What is the evaluation period? (recommended: minimum 7 days) | Enough cycles to capture daily ambient temperature variation |

---

## Phase 2: Pre-Deployment Validation

Complete these checks before installing the agent.

### 2.1 BMC Capability Assessment

Run on each target node to determine the fan control surface:

```
[ ] Redfish API reachable: curl -sk https://{bmc_ip}/redfish/v1/
[ ] Authentication works: curl -sk -u {user}:{pass} https://{bmc_ip}/redfish/v1/Chassis/1/Thermal
[ ] Fan readings available in Redfish Thermal endpoint
[ ] IPMI-over-LAN works: ipmitool -I lanplus -H {bmc_ip} -U {user} -P {pass} power status
[ ] Fan control command identified (vendor-specific):
    - Lenovo XCC:  raw 0x3a 0x07 0x01 {pct}
    - Dell iDRAC:  raw 0x30 0x30 0x01 0x00 (manual) + raw 0x30 0x30 0x02 0xff {pct}
    - HPE iLO:     Redfish PATCH /Thermal or OEM endpoint
    - Supermicro:  raw 0x30 0x70 0x66 0x01 {zone} {pct}
[ ] Fan control commands actually move fans (verify RPM change, not just return code!)
[ ] BIOS operating mode documented and set to allow fan override
```

### 2.2 Vendor-Specific Notes

| Vendor | BMC | Fan Control Method | Known Gotchas |
|--------|-----|-------------------|---------------|
| **Lenovo** | XCC | IPMI OEM 0x3a 0x07 | Commands may return success but BMC ignores them in Efficiency_FavorPower mode. Must set BIOS to CustomMode first. BIOS PATCH path is `/Bios/Pending` not `/Bios/Settings`. |
| **Dell** | iDRAC | IPMI 0x30 0x30 | Must disable "Third-Party PCIe Card Default Thermal Profile" in iDRAC. Manual mode must be explicitly enabled before setting speed. |
| **HPE** | iLO | Redfish PATCH or iLO fan profile | Fan zones vary by model. Some require iLO Advanced license. |
| **Supermicro** | IPMI | Raw 0x30 0x70 0x66 | Fan zones (CPU, peripheral, etc.) must be set independently. Full-speed mode override exists. |

### 2.3 Baseline Measurements

Before enabling optimization, capture 24-48 hours of baseline data:

```
[ ] Fan RPM at idle (no workload, all GPUs cold)
[ ] Fan RPM under sustained load (run benchmark for 1 hour)
[ ] GPU temperature at idle and under load
[ ] Ambient/inlet temperature
[ ] Power consumption at idle and under load (if PDU metering available)
[ ] Current PUE reading (if available)
```

---

## Phase 3: Agent Deployment

### 3.1 Installation Checklist

```
[ ] Create service account: useradd -r -s /sbin/nologin cooledai
[ ] Create config directory: mkdir -p /etc/cooledai
[ ] Deploy agent code to /opt/cooledai/
[ ] Write environment file /etc/cooledai/agent.env:
      XCC_BMC_HOST={bmc_ip}
      XCC_BMC_USER={user}
      XCC_BMC_PASS={pass}
      NODE_ROLE=pilot|control
      API_KEY={cooledai_cloud_key}
[ ] Set permissions: chmod 600 /etc/cooledai/agent.env
[ ] Install systemd service: cooledai-agent.service
[ ] Install ipmitool if missing
[ ] Verify nvidia-smi accessible
[ ] Delete any stale calibration profiles: rm -f /etc/cooledai/calibration_profile.json
[ ] Start agent: systemctl enable --now cooledai-agent
[ ] Verify agent logs: journalctl -u cooledai-agent -f
```

### 3.2 Critical First-Run Checks

The agent's first 30 minutes are the **calibration observation window**. During this time:

```
[ ] Agent starts fresh (no stale profile loaded)
[ ] Calibration window begins collecting idle and active fan samples
[ ] Idle stepping forces fans to bootstrap floor (~30% of rated max) when GPU is idle
[ ] Active samples collected when workload runs
[ ] After 30 min: profile derived with separate idle/active fan baselines
[ ] Profile saved with schema_version:1
[ ] Verify profile rejection guard: idle_rpm / ceiling_rpm must be < 0.90
```

### 3.3 Workload Simulator (for Evaluation)

Deploy the identical workload benchmark on both pilot and control nodes:

```bash
# On both nodes:
python3 thermal_workload_benchmark.py --model llama3.2:3b --seed 42

# Requirements:
# - Ollama installed and serving on ports 11434 (GPU0) and 11435 (GPU1)
# - Model pulled on both instances
# - Same --seed ensures identical workload patterns
```

For non-GPU nodes or different workloads, adapt the benchmark but ensure:
- Same script, same parameters on both nodes
- Mix of predictable and unpredictable load phases
- Includes idle periods (for fan floor measurement)
- Includes burst/spike periods (for response time measurement)

---

## Phase 4: Monitoring & Validation

### 4.1 Key Metrics to Track

| Metric | Source | Comparison Point |
|--------|--------|-----------------|
| Fan RPM (average, min, max) | Agent telemetry | Pilot vs Control |
| Fan power consumption | Derived from RPM (cubic law) | Pilot vs Control |
| GPU temperature | nvidia-smi via agent | Must stay within ASHRAE limits |
| Thermal response time (load spike → fan ramp) | Agent logs | Pilot should be faster (predictive) |
| No-response rate (% of cycles where optimizer doesn't act) | Agent logs | Target < 30% |
| PUE | Facility metering | Pilot rack vs Control rack |
| Thermal events (throttle/shutdown) | Agent alerts | Must be zero |

### 4.2 Success Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Fan energy reduction | > 15% vs control | Average fan RPM reduction × cubic power law |
| Temperature compliance | 100% within limits | Zero thermal events during evaluation |
| System reliability | 99.9% agent uptime | systemd restart count |
| Optimization response | < 10s to load change | Time from GPU power spike to fan target update |

### 4.3 Reporting

Generate a comparison report after the evaluation period:

```
1. Executive summary: energy saved, temperatures maintained
2. Side-by-side fan RPM timeseries (pilot vs control)
3. Side-by-side GPU temperature timeseries
4. Workload correlation analysis (GPU power → fan response)
5. Estimated annual savings (fan kWh × electricity rate)
6. Recommendation: proceed to full deployment or adjust
```

---

## Phase 5: Production Rollout

After successful pilot evaluation:

```
[ ] Review and approve evaluation report with client
[ ] Set BIOS operating mode on all target nodes (if vendor requires it)
[ ] Deploy agent to remaining nodes
[ ] Configure monitoring dashboards
[ ] Set up alerting (thermal events, agent down, profile rejection)
[ ] Establish maintenance schedule (firmware updates, profile refresh)
[ ] Hand off runbook to client operations team
[ ] Schedule 30-day post-deployment review
```

---

## Quick Reference: Minimum Required Information

If pressed for time, these are the **absolute minimum** questions to answer before deployment:

1. **Server vendor + model** (determines entire fan control approach)
2. **BMC IP + credentials** (can't control fans without this)
3. **BMC firmware version** (fan control capabilities vary per version)
4. **Current BIOS thermal/operating mode** (may need to change before optimization works)
5. **GPU type and count** (sets power thresholds)
6. **Fan count and rated max RPM** (configures bootstrap floor and ceiling)
7. **SSH access to the host OS** (for agent deployment)
8. **Can we get a control node for A/B testing?** (proves ROI)

---

## Lessons Learned (from Field Deployments)

1. **Always verify fan commands actually move fans.** IPMI returning success ≠ fans moved. Test with before/after RPM readings.

2. **BIOS operating mode matters enormously.** Lenovo Efficiency_FavorPower silently ignores all external fan commands. Changing to CustomMode dropped baseline 24%.

3. **Stale calibration profiles cause regression.** Always delete profiles on agent restart during initial deployment. The profile rejection guard (idle/ceiling > 0.90) catches most stale profiles.

4. **Calibration circular reference.** If the optimizer is controlling fans during calibration, the calibrator measures the optimizer's output, not the true thermal baseline. The idle stepping fix (force to bootstrap floor when GPU idle during observation) breaks this loop.

5. **Dual-GPU nodes need dual Ollama instances.** One instance per GPU, separate ports, separate CUDA_VISIBLE_DEVICES. Otherwise only one GPU generates load.

6. **The workload seed matters.** Both pilot and control must use `--seed 42` (or same value) so the pseudo-random workload phases (chaos_burst, random_walk, storm) are identical across nodes.

7. **Expect scripts break on special characters.** Passwords with `!`, `$`, or brackets need careful escaping. Prefer SCP + shell scripts over complex expect one-liners.

8. **Redfish API surface varies wildly.** Don't assume endpoints exist — probe first. Some BMC firmwares return 405 (Method Not Allowed) on PATCH to Thermal, others support it fully.
