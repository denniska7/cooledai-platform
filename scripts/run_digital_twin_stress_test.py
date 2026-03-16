#!/usr/bin/env python3
"""
CooledAI Digital Twin Stress Test

Runs OptimizationBrain against the physics-based digital twin.
No hardware required - use for RL agent development and validation.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.simulation.digital_twin import DigitalTwin, DigitalTwinConfig
from core.optimization.optimization_brain import OptimizationBrain


def main() -> None:
    config = DigitalTwinConfig(
        num_racks=10,
        noise_std_c=0.5,
        power_range_w=(30_000.0, 70_000.0),
    )
    twin = DigitalTwin(config=config, seed=42)
    brain = OptimizationBrain(target_temp=65.0, shadow_mode=True)

    print("Running 100-step stress test with load spike at step 50...")
    result = twin.stress_test_loop(
        brain=brain,
        num_steps=100,
        dt=1.0,
        induce_load_spike_at=50,
        load_spike_factor=1.4,
    )

    print(f"\n--- Results ---")
    print(f"Max temp reached: {result['max_temp_reached']:.1f}°C")
    print(f"Min efficiency score: {result['min_efficiency_score']:.1f}")
    print(f"Avg efficiency score: {result['avg_efficiency_score']:.1f}")
    print(f"Steps: {result['num_steps']}")

    final_temps = result["temps_history"][-1]
    print(f"\nFinal rack temps (°C): {[f'{t:.1f}' for t in final_temps]}")
    print("Done.")


if __name__ == "__main__":
    main()
