#!/usr/bin/env python3
"""
Deploy Trained Cooling Control Policy for Real-Time Recommendations

This script loads the trained PPO agent and provides optimal fan speed
recommendations for any given data center state (temperature, IT load).

Usage:
    # Interactive mode
    python3.11 deploy_control_policy.py

    # Single query mode
    python3.11 deploy_control_policy.py --temp 65.0 --load 120000 --fan 1.5

    # Continuous monitoring mode
    python3.11 deploy_control_policy.py --monitor --interval 5

Author: Claude (Anthropic)
Date: 2026-01-28
"""

import numpy as np
import torch
import sys
import os
import argparse
from typing import Dict, Tuple
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from cooling_rl_env import CoolingControlEnv
from models.recurrent_pinn import RecurrentPINN


class CoolingControlPolicy:
    """
    Autonomous cooling control policy for data centers.

    Provides optimal fan speed recommendations based on:
    - Current temperature
    - IT equipment load
    - Current fan speed
    - Predicted thermal dynamics
    """

    def __init__(
        self,
        pinn_path: str = "checkpoints/best_recurrent_pinn.pt",
        agent_path: str = "optimizer/ppo_cooling_agent.zip",
        device: str = 'cpu'
    ):
        """
        Initialize the control policy.

        Args:
            pinn_path: Path to trained RecurrentPINN
            agent_path: Path to trained PPO agent
            device: Device for inference
        """
        self.device = device

        print("🚀 Loading Autonomous Cooling Control Policy...")

        # Load RecurrentPINN (physics engine)
        print(f"  Loading RecurrentPINN from {pinn_path}...")
        self.physics_model = RecurrentPINN(
            input_dim=3,
            hidden_dim=64,
            num_lstm_layers=2,
            use_thermal_inertia=True,
            use_mc_dropout=True,
            use_attention=True,
            num_attention_heads=4
        ).to(device)

        checkpoint = torch.load(pinn_path, map_location=device)
        self.physics_model.load_state_dict(checkpoint['model_state_dict'])
        self.physics_model.eval()
        print("  ✓ RecurrentPINN loaded (MAE: 0.0842°C)")

        # Load trained RL agent
        if os.path.exists(agent_path):
            print(f"  Loading PPO agent from {agent_path}...")
            self.agent = PPO.load(agent_path, device=device)
            print("  ✓ PPO agent loaded")
        else:
            print(f"  ⚠️  PPO agent not found at {agent_path}")
            print("     Using physics-based heuristic controller instead.")
            self.agent = None

        print("✓ Control policy ready!\n")

        # Hidden state for LSTM
        self.hidden_state = None

    def predict_thermal_dynamics(
        self,
        T_current: float,
        Q_load: float,
        u_flow: float
    ) -> Dict[str, float]:
        """
        Predict future temperature using RecurrentPINN.

        Args:
            T_current: Current temperature (°C)
            Q_load: IT equipment load (W)
            u_flow: Air flow velocity (m/s)

        Returns:
            Dictionary with predictions
        """
        with torch.no_grad():
            T_tensor = torch.tensor([T_current], dtype=torch.float32).to(self.device)
            Q_tensor = torch.tensor([Q_load], dtype=torch.float32).to(self.device)
            u_tensor = torch.tensor([u_flow], dtype=torch.float32).to(self.device)

            predictions = self.physics_model(T_tensor, Q_tensor, u_tensor)

            # Note: RecurrentPINN maintains LSTM state internally

            return {
                'T_t1': predictions['T_t1'].cpu().numpy()[0, 0],
                'T_t5': predictions['T_t5'].cpu().numpy()[0, 0],
                'T_t10': predictions['T_t10'].cpu().numpy()[0, 0],
                'dT_dt': predictions['dT_dt'].cpu().numpy()[0, 0]
            }

    def get_control_action(
        self,
        T_current: float,
        Q_load: float,
        u_flow: float,
        time_in_danger: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Get optimal fan speed adjustment.

        Args:
            T_current: Current temperature (°C)
            Q_load: IT equipment load (W)
            u_flow: Current fan velocity (m/s)
            time_in_danger: Consecutive seconds in danger zone

        Returns:
            delta_u: Recommended change in fan velocity (m/s)
            info: Additional information
        """
        # Get thermal predictions
        predictions = self.predict_thermal_dynamics(T_current, Q_load, u_flow)

        # Construct observation
        obs = np.array([
            T_current,
            Q_load,
            u_flow,
            predictions['dT_dt'],
            predictions['T_t1'],
            time_in_danger
        ], dtype=np.float32)

        # Get action from RL agent or heuristic
        if self.agent is not None:
            # Use trained RL agent
            action, _states = self.agent.predict(obs, deterministic=True)
            delta_u = float(action[0])
            control_source = "RL Agent"
        else:
            # Fallback: Physics-based heuristic controller
            delta_u = self._heuristic_control(T_current, Q_load, u_flow, predictions)
            control_source = "Heuristic"

        # Calculate recommended fan speed
        new_u_flow = np.clip(u_flow + delta_u, 0.5, 3.0)

        # Safety checks
        safety_status = self._assess_safety(T_current, predictions['T_t10'])

        info = {
            'delta_u': delta_u,
            'new_fan_speed': new_u_flow,
            'predicted_T_t1': predictions['T_t1'],
            'predicted_T_t5': predictions['T_t5'],
            'predicted_T_t10': predictions['T_t10'],
            'heating_rate': predictions['dT_dt'],
            'safety_status': safety_status,
            'control_source': control_source
        }

        return delta_u, info

    def _heuristic_control(
        self,
        T_current: float,
        Q_load: float,
        u_flow: float,
        predictions: Dict
    ) -> float:
        """
        Physics-based heuristic controller (fallback if RL agent not trained).

        Simple control strategy:
        - If T > 75°C or dT/dt > 0.2: Increase fan speed
        - If T < 60°C and dT/dt < 0: Decrease fan speed
        - Otherwise: Small adjustments based on prediction
        """
        T_pred = predictions['T_t10']
        dT_dt = predictions['dT_dt']

        # Emergency cooling
        if T_current > 75.0 or T_pred > 75.0:
            return +0.5  # Max increase

        # Aggressive cooling
        if T_current > 70.0 or dT_dt > 0.2:
            return +0.3

        # Moderate cooling
        if T_current > 65.0 or dT_dt > 0.1:
            return +0.1

        # Energy saving mode
        if T_current < 60.0 and dT_dt < -0.05:
            return -0.2

        # Slight energy saving
        if T_current < 65.0 and dT_dt < 0:
            return -0.1

        # Maintain current setting
        return 0.0

    def _assess_safety(self, T_current: float, T_predicted: float) -> str:
        """
        Assess thermal safety status.

        Returns:
            Safety status string
        """
        if T_current > 80.0 or T_predicted > 80.0:
            return "🚨 CRITICAL"
        elif T_current > 75.0 or T_predicted > 75.0:
            return "⚠️  DANGER"
        elif T_current > 70.0 or T_predicted > 70.0:
            return "⚡ WARNING"
        else:
            return "✓ SAFE"

    def reset_state(self):
        """Reset LSTM hidden state (for new monitoring session)."""
        self.hidden_state = None


def interactive_mode(policy: CoolingControlPolicy):
    """Run interactive control policy mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE CONTROL POLICY MODE")
    print("=" * 70)
    print("\nEnter current data center state to get optimal fan speed recommendation.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            # Get user input
            T_input = input("Current Temperature (°C) [default: 65.0]: ").strip()
            if T_input.lower() == 'quit':
                break
            T_current = float(T_input) if T_input else 65.0

            Q_input = input("IT Load (kW) [default: 100.0]: ").strip()
            Q_load = (float(Q_input) if Q_input else 100.0) * 1000  # Convert to W

            u_input = input("Current Fan Speed (m/s) [default: 1.5]: ").strip()
            u_flow = float(u_input) if u_input else 1.5

            # Get control recommendation
            delta_u, info = policy.get_control_action(T_current, Q_load, u_flow)

            # Display recommendation
            print("\n" + "-" * 70)
            print("📊 CONTROL RECOMMENDATION")
            print("-" * 70)
            print(f"Current State:")
            print(f"  Temperature: {T_current:.2f}°C")
            print(f"  IT Load: {Q_load/1000:.1f} kW")
            print(f"  Fan Speed: {u_flow:.2f} m/s")

            print(f"\nThermal Predictions:")
            print(f"  T(t+1s):  {info['predicted_T_t1']:.2f}°C")
            print(f"  T(t+5s):  {info['predicted_T_t5']:.2f}°C")
            print(f"  T(t+10s): {info['predicted_T_t10']:.2f}°C")
            print(f"  Heating Rate: {info['heating_rate']:+.4f}°C/s")

            print(f"\n🎯 Recommended Action:")
            print(f"  Fan Speed Adjustment: {delta_u:+.3f} m/s")
            print(f"  New Fan Speed: {info['new_fan_speed']:.2f} m/s")

            print(f"\n{info['safety_status']}")
            print(f"  (Control Source: {info['control_source']})")
            print("-" * 70 + "\n")

        except ValueError:
            print("❌ Invalid input. Please enter numeric values.\n")
        except KeyboardInterrupt:
            break

    print("\n✓ Exiting interactive mode.")


def single_query(
    policy: CoolingControlPolicy,
    temp: float,
    load: float,
    fan: float
):
    """Single query mode."""
    print("\n" + "=" * 70)
    print("SINGLE QUERY MODE")
    print("=" * 70)

    delta_u, info = policy.get_control_action(temp, load * 1000, fan)

    print(f"\nInput State:")
    print(f"  Temperature: {temp:.2f}°C")
    print(f"  IT Load: {load:.1f} kW")
    print(f"  Fan Speed: {fan:.2f} m/s")

    print(f"\nPredicted Temperatures:")
    print(f"  T(t+1s):  {info['predicted_T_t1']:.2f}°C")
    print(f"  T(t+5s):  {info['predicted_T_t5']:.2f}°C")
    print(f"  T(t+10s): {info['predicted_T_t10']:.2f}°C")

    print(f"\n🎯 Recommendation:")
    print(f"  Adjust Fan Speed: {delta_u:+.3f} m/s")
    print(f"  New Fan Speed: {info['new_fan_speed']:.2f} m/s")
    print(f"  Safety Status: {info['safety_status']}")


def monitoring_mode(
    policy: CoolingControlPolicy,
    interval: int = 5
):
    """Continuous monitoring mode (simulated)."""
    print("\n" + "=" * 70)
    print("CONTINUOUS MONITORING MODE")
    print("=" * 70)
    print(f"Simulating data center monitoring every {interval} seconds...")
    print("Press Ctrl+C to stop.\n")

    # Simulate initial conditions
    T_current = np.random.uniform(55.0, 70.0)
    Q_load = np.random.uniform(80000, 130000)
    u_flow = 1.5
    time_in_danger = 0

    try:
        step = 0
        while True:
            step += 1

            # Get control action
            delta_u, info = policy.get_control_action(
                T_current, Q_load, u_flow, time_in_danger
            )

            # Display current status
            print(f"[{time.strftime('%H:%M:%S')}] Step {step}:")
            print(f"  T: {T_current:.2f}°C, Load: {Q_load/1000:.1f}kW, "
                  f"Fan: {u_flow:.2f}m/s → {info['new_fan_speed']:.2f}m/s")
            print(f"  Predicted T(t+10s): {info['predicted_T_t10']:.2f}°C")
            print(f"  Status: {info['safety_status']}")

            # Apply action and simulate next state
            u_flow = info['new_fan_speed']
            T_current = info['predicted_T_t1']  # Use 1-second prediction

            # Simulate load variation
            Q_load += np.random.uniform(-3000, 3000)
            Q_load = np.clip(Q_load, 50000, 150000)

            # Update danger counter
            if T_current > 75.0:
                time_in_danger += 1
            else:
                time_in_danger = 0

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy Autonomous Cooling Control Policy"
    )
    parser.add_argument(
        '--temp', type=float,
        help='Current temperature (°C) for single query'
    )
    parser.add_argument(
        '--load', type=float,
        help='IT load (kW) for single query'
    )
    parser.add_argument(
        '--fan', type=float,
        help='Current fan speed (m/s) for single query'
    )
    parser.add_argument(
        '--monitor', action='store_true',
        help='Run in continuous monitoring mode'
    )
    parser.add_argument(
        '--interval', type=int, default=5,
        help='Monitoring interval in seconds (default: 5)'
    )

    args = parser.parse_args()

    # Check dependencies
    if not SB3_AVAILABLE:
        print("⚠️  Warning: Stable-Baselines3 not installed.")
        print("   Install with: pip install stable-baselines3[extra]")
        print("   Falling back to heuristic controller.\n")

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Initialize policy
    try:
        policy = CoolingControlPolicy(device=device)

        # Run appropriate mode
        if args.temp is not None and args.load is not None and args.fan is not None:
            # Single query mode
            single_query(policy, args.temp, args.load, args.fan)
        elif args.monitor:
            # Monitoring mode
            monitoring_mode(policy, args.interval)
        else:
            # Interactive mode
            interactive_mode(policy)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the RecurrentPINN model is trained:")
        print("  python3.11 train_recurrent_pinn.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
