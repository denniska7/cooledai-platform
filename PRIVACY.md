# CooledAI Privacy Policy — Data Boundary Documentation

## What CooledAI Collects

CooledAI collects **thermal and power telemetry only** — the physical
measurements that describe how hot your hardware is running and how much
power it is consuming.

| Data Point | Source | Used For |
|---|---|---|
| GPU temperature (C) | nvidia-smi --query-gpu=temperature.gpu | Fan curve optimization |
| GPU power draw (W) | nvidia-smi --query-gpu=power.draw | Efficiency tracking |
| GPU utilization (%) | nvidia-smi --query-gpu=utilization.gpu | Idle detection |
| GPU memory utilization (%) | nvidia-smi --query-gpu=utilization.memory | Memory-bound detection |
| CPU temperature (C) | /sys/class/hwmon/ | Thermal safety |
| Fan RPM | ipmitool sdr type Fan (BMC) | Fan health monitoring |
| Fan duty cycle (%) | Redfish /Thermal (BMC) | Optimization feedback |
| Ambient temperature (C) | Redfish /Thermal (BMC) | Environmental baseline |

## What CooledAI Does NOT Collect

CooledAI **never** accesses, reads, stores, or transmits:

- **Running processes or applications** — no `ps`, `top`, or process lists
- **GPU compute applications** — no `--query-compute-apps`
- **Memory contents** — no reads of GPU VRAM or system RAM
- **Network traffic** — no packet capture, no connection monitoring
- **Filesystem contents** — no file reads outside hardware monitoring paths
- **Container or VM information** — no Docker, Kubernetes, or hypervisor queries
- **Application logs** — no journalctl, dmesg, or /var/log reads
- **User data** — no access to /home, /tmp, or any user directories

## How This Is Enforced

1. **Hardware Interface Whitelist** (`core/security/interface_policy.py`):
   Every external command is validated against a strict whitelist before
   execution. Commands outside the whitelist raise `PrivacyBoundaryViolation`
   and halt the agent tick.

2. **Append-Only Audit Log** (`core/security/audit_log.py`): Every hardware
   command is recorded with timestamp, type, and arguments before execution.
   The audit log cannot be modified or deleted.

3. **Security Attestation** (`GET /api/v1/security/attestation`): Generates
   a signed report that any security auditor can independently verify.

4. **Network Isolation**: The agent connects only to the BMC management
   network and the CooledAI gateway. No data plane access.

## BMC/IPMI Architecture

CooledAI operates at the **Baseboard Management Controller (BMC)** layer,
which is a separate microprocessor running independently of the main CPU.
The BMC has its own network interface (out-of-band management) and cannot
access the host operating system, hypervisor, or any running workloads.

This is an architectural guarantee, not a policy decision. The BMC
physically cannot see your data.

## Data Retention

- Telemetry data is transmitted to the CooledAI cloud platform via HTTPS
- Raw telemetry retained for 7 days
- Aggregated metrics retained per your contract terms
- All data encrypted in transit (TLS 1.3) and at rest (AES-256)
- Data deletion available on request per your service agreement

## Contact

For privacy questions: privacy@cooledai.com
