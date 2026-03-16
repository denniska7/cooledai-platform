# ❄️ CooledAI: Physics-Informed Thermal Orchestration

**Targeting 40% Energy Reduction in AI Data Centers via Predictive Control.**

CooledAI is a Physics-Informed Neural Network (PINN) platform designed to solve the "Thermal Inertia" problem in high-density GPU clusters. While traditional cooling is reactive, CooledAI is proactive—ingesting job schedules and telemetry to optimize cooling before the heat hits.

---

## 🚀 Core Technology Pillars

### 1. The Physics-Informed Brain (PINN)

Unlike "Black Box" AI, CooledAI embeds the laws of thermodynamics directly into its loss function.

- **Fan Affinity Laws:** Optimization based on the cubic relationship between RPM and Power (\$P \propto N^3\$).
- **FOPDT Models:** First-Order Plus Dead Time models that self-calibrate per rack to account for "Thermal Mass" and airflow lag.
- **Superposition Principle:** Accurately predicts the cumulative cooling effect of multiple fans across overlapping zones.

### 2. Universal Protocol Gateway

Brand-agnostic control through industry standards:

- **Redfish & IPMI:** Native support for Dell, HPE, Supermicro, and NVIDIA DGX clusters.
- **Discovery Agent:** Automatic network scanning and "Thermal Ping" tests to build a site-wide Influence Map without manual configuration.

### 3. Workload-Aware Proactive Cooling

Direct integration with Slurm and Kubernetes. By observing the job queue, CooledAI "pre-cools" specific racks 60 seconds before a GPU-heavy training job begins, eliminating thermal spikes.

---

## 🛡️ Enterprise Safety & Resilience

- **Watchdog Sidecar:** A standalone "fail-safe" script that forces hardware to 70% cooling if the primary API loses heartbeat.
- **Shadow Mode:** Run the AI in "Audit Only" mode to generate ROI Reports and verify savings before flipping the switch to live control.
- **Deterministic Guardrails:** Hard limits on slew rates, anti-short-cycle logic, and ASHRAE thermal safety bounds.

---

## 🛠️ Setup & Development

### 1. Clone & Configure

```bash
git clone https://github.com/your-repo/cooledai-simulator.git
cd cooledai-simulator
cp .env.example .env  # Fill in your Clerk & System keys
```

### 2. Run the Digital Twin (Stress Test)

Verify the PINN logic in a physics-accurate simulation:

```bash
python scripts/run_digital_twin_stress_test.py
```

### 3. Start the Ecosystem

```bash
# Terminal 1: Backend API
cd backend && uvicorn main:app --reload

# Terminal 2: Frontend Dashboard
cd frontend && npm run dev
```
