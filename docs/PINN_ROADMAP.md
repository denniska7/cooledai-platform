# PINN vs production stack — narrative for “learns and adapts”

## What ships today

- **FOPDT + telemetry** (`core/models/topology.py`): per-rack τ, θ, gain from real step responses — this **is** adaptive physics on new hardware, without retraining a big NN.  
- **Optimizer + safety policy**: efficiency vs guardrails, spike hold, floors.  
- **Predictor**: short-horizon extrapolation + confidence, fed by history.  

That combination **does** learn from what it sees; it is not the same artifact as the **research PINN** checkpoints under `models/`.

## What the PINN adds (selling point / R&D)

- A **differentiable surrogate** for richer scenarios (spatial effects, transients) and eventually **fleet-scale** what-if.  
- To be **hardware-agnostic** at inference, the PINN should take **explicit parameters** (TDP, mass, max cooling, τ priors) as inputs, or use **meta-learning / few-shot** on early telemetry — not a single frozen checkpoint tuned to one lab box.

## How we keep improving

1. **More real telemetry** into calibration and (optionally) PINN training — survey → standardized schema.  
2. **Parameter-conditioned PINN** — same network, different hardware vector.  
3. **Online residual** — PINN prediction + small correction from last N minutes of errors.  
4. **Uncertainty** — MC dropout / ensembles already sketched in training scripts; surface UQ in API for GUARD_MODE.  

**Pitch:** “The product **adapts now** via FOPDT + policies; the PINN is the **acceleration layer** for accuracy and multi-physics as data grows.”
