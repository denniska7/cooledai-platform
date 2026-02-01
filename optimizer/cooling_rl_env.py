#!/usr/bin/env python3
"""
Reinforcement Learning Environment for Data Center Cooling Control

This environment uses the trained RecurrentPINN as a physics engine to
simulate thermal dynamics and trains an RL agent to optimize cooling
while maintaining safe temperatures.

Author: Claude (Anthropic)
Date: 2026-01-28
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.recurrent_pinn import RecurrentPINN


class CoolingControlEnv(gym.Env):
    """
    Custom Gym Environment for Data Center Cooling Control

    The agent controls fan velocity to maintain safe temperatures while
    minimizing energy consumption. The RecurrentPINN acts as the physics
    engine, predicting thermal dynamics with < 0.1°C accuracy.

    State Space:
        - T_current: Current temperature (°C)
        - Q_load: IT equipment heat load (W)
        - u_flow: Air flow velocity (m/s)
        - dT_dt: Heating rate (°C/s)
        - T_t1_pred: 1-second ahead prediction (°C)
        - time_in_danger: Consecutive seconds in danger zone

    Action Space:
        - delta_u: Change in fan velocity (continuous, -0.5 to +0.5 m/s)

    Reward Function:
        - Energy savings: -0.1 * u_flow^3 (cubic power law)
        - Thermal penalty: -1000 if T > 80°C (critical)
        - Danger zone penalty: -10 * (T - 75)^2 if T > 75°C
        - Safe zone bonus: +1 if 50°C < T < 70°C
        - Stability bonus: +0.5 if |dT/dt| < 0.1°C/s

    Done Condition:
        - Temperature exceeds 80°C (thermal runaway)
        - Episode time limit reached (1000 steps = ~16 minutes)
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        max_steps: int = 1000,
        initial_temp_range: Tuple[float, float] = (45.0, 65.0),
        load_range: Tuple[float, float] = (50000.0, 150000.0),
        fan_range: Tuple[float, float] = (0.5, 3.0),
        dt: float = 1.0,  # Time step in seconds
        enable_domain_randomization: bool = False,
        thermal_mass_variation: float = 0.2,  # ±20% variation
        ambient_temp_variation: float = 5.0,  # ±5°C variation
        action_smoothing_penalty: float = 1.0,  # Penalty weight for large actions
        enable_stability_guard: bool = True,  # Enable physical safety override
        ttf_threshold: float = 120.0,  # Time-to-Failure threshold (seconds)
        critical_temp: float = 85.0  # Critical temperature for TTF calculation
    ):
        """
        Initialize the cooling control environment.

        Args:
            model_path: Path to trained RecurrentPINN checkpoint
            device: Device for model inference ('cpu', 'cuda', 'mps')
            max_steps: Maximum steps per episode
            initial_temp_range: Range for initial temperature (°C)
            load_range: Range for IT load variation (W)
            fan_range: Range for fan velocity (m/s)
            dt: Time step in seconds
            enable_domain_randomization: Enable domain randomization for robustness
            thermal_mass_variation: Thermal mass randomization (±fraction)
            ambient_temp_variation: Ambient temperature variation (±°C)
            action_smoothing_penalty: Penalty weight for large fan speed changes
            enable_stability_guard: Enable physical safety override (Phase 4.4)
            ttf_threshold: Time-to-Failure threshold for override (seconds)
            critical_temp: Critical temperature threshold (°C)
        """
        super(CoolingControlEnv, self).__init__()

        # Environment parameters
        self.device = device
        self.max_steps = max_steps
        self.initial_temp_range = initial_temp_range
        self.load_range = load_range
        self.fan_range = fan_range
        self.dt = dt

        # Phase 4.2: Domain Randomization
        self.enable_domain_randomization = enable_domain_randomization
        self.thermal_mass_variation = thermal_mass_variation
        self.ambient_temp_variation = ambient_temp_variation

        # Phase 4.2: Action Smoothing
        self.action_smoothing_penalty = action_smoothing_penalty

        # Phase 4.4: Stability Guard
        self.enable_stability_guard = enable_stability_guard
        self.ttf_threshold = ttf_threshold
        self.critical_temp = critical_temp

        # Load trained RecurrentPINN (physics engine)
        print(f"Loading RecurrentPINN from {model_path}...")
        self.physics_model = RecurrentPINN(
            input_dim=3,
            hidden_dim=64,
            num_lstm_layers=2,
            use_thermal_inertia=True,
            use_mc_dropout=True,
            use_attention=True,
            num_attention_heads=4
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        self.physics_model.load_state_dict(checkpoint['model_state_dict'])
        self.physics_model.eval()
        print("RecurrentPINN loaded successfully!")

        # Define action space: Change in fan velocity
        # Continuous action: delta_u in [-0.5, +0.5] m/s
        self.action_space = spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(1,),
            dtype=np.float32
        )

        # Define observation space
        # [T_current, Q_load, u_flow, dT_dt, T_t1_pred, time_in_danger]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.5, -10.0, 0.0, 0.0]),
            high=np.array([100.0, 200000.0, 3.0, 10.0, 100.0, 100.0]),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.T_current = 0.0
        self.Q_load = 0.0
        self.u_flow = 0.0
        self.u_flow_prev = 0.0  # Phase 4.2: Track previous fan speed
        self.dT_dt = 0.0
        self.T_t1_pred = 0.0
        self.time_in_danger = 0

        # Episode statistics
        self.total_reward = 0.0
        self.total_energy = 0.0
        self.max_temp = 0.0
        self.safety_violations = 0

        # LSTM hidden state (for RecurrentPINN)
        self.hidden_state = None

        # Phase 4.2: Domain Randomization parameters (per episode)
        self.thermal_mass_scale = 1.0
        self.ambient_temp_offset = 0.0

        # Phase 4.4: Stability Guard tracking
        self.safety_override_log = []  # Log of all safety overrides
        self.ttf_current = float('inf')  # Current Time-to-Failure estimate

        # Tier 3 Safety: Automated Load Shedding
        self.shutdown_state = False  # Emergency shutdown triggered
        self.max_fan_duration = 0  # Seconds at 100% fan speed
        self.emergency_shutdown_log = []  # Log of emergency shutdowns
        self.critical_temp_threshold = 85.0  # Critical temperature (°C)
        self.max_fan_timeout = 5  # Seconds at max fan before shutdown

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Phase 4.2: Domain Randomization (if enabled)
        if self.enable_domain_randomization:
            # Randomize thermal mass (±20% by default)
            self.thermal_mass_scale = np.random.uniform(
                1.0 - self.thermal_mass_variation,
                1.0 + self.thermal_mass_variation
            )
            # Randomize ambient temperature offset (±5°C by default)
            self.ambient_temp_offset = np.random.uniform(
                -self.ambient_temp_variation,
                self.ambient_temp_variation
            )
        else:
            self.thermal_mass_scale = 1.0
            self.ambient_temp_offset = 0.0

        # Sample initial conditions
        self.T_current = np.random.uniform(*self.initial_temp_range)
        self.Q_load = np.random.uniform(*self.load_range)
        self.u_flow = np.random.uniform(1.0, 2.0)  # Start with moderate fan speed
        self.u_flow_prev = self.u_flow  # Initialize previous fan speed

        # Get initial prediction from physics model
        with torch.no_grad():
            T_tensor = torch.tensor([self.T_current], dtype=torch.float32).to(self.device)
            Q_tensor = torch.tensor([self.Q_load], dtype=torch.float32).to(self.device)
            u_tensor = torch.tensor([self.u_flow], dtype=torch.float32).to(self.device)

            predictions = self.physics_model(T_tensor, Q_tensor, u_tensor)
            self.T_t1_pred = predictions['T_t1'].cpu().numpy()[0, 0]
            self.dT_dt = predictions['dT_dt'].cpu().numpy()[0, 0]

            # Note: RecurrentPINN maintains LSTM state internally
            # No need to pass hidden state manually

        # Reset tracking variables
        self.time_in_danger = 0
        self.total_reward = 0.0
        self.total_energy = 0.0
        self.max_temp = self.T_current
        self.safety_violations = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Change in fan velocity (delta_u)

        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode ended (thermal runaway)
            truncated: Whether episode reached time limit
            info: Additional information
        """
        self.current_step += 1

        # Phase 4.2: Store previous fan speed for smoothing penalty
        self.u_flow_prev = self.u_flow

        # Simulate IT load variation FIRST (before action application)
        load_change = np.random.uniform(-5000, 5000)  # ±5kW variation
        self.Q_load = np.clip(
            self.Q_load + load_change,
            self.load_range[0],
            self.load_range[1]
        )

        # Get current thermal predictions for stability guard
        with torch.no_grad():
            T_tensor = torch.tensor([self.T_current], dtype=torch.float32).to(self.device)
            Q_tensor = torch.tensor([self.Q_load], dtype=torch.float32).to(self.device)
            u_tensor_current = torch.tensor([self.u_flow], dtype=torch.float32).to(self.device)

            predictions_current = self.physics_model(T_tensor, Q_tensor, u_tensor_current)
            current_dT_dt = predictions_current['dT_dt'].cpu().numpy()[0, 0]

        # Calculate current Time-to-Failure
        self.ttf_current = self._calculate_ttf(self.T_current, current_dT_dt, self.u_flow)

        # Apply action: Adjust fan velocity
        delta_u = float(action[0])
        new_u_flow = np.clip(
            self.u_flow + delta_u,
            self.fan_range[0],
            self.fan_range[1]
        )

        # Phase 4.4: Stability Guard - Physical Override
        override_applied = False
        override_reason = ""

        if self.enable_stability_guard:
            # Check if TTF is critically low and agent is trying to reduce cooling
            if self.ttf_current < self.ttf_threshold and new_u_flow < self.u_flow:
                # OVERRIDE: Force fans to maximum speed
                override_applied = True
                override_reason = (
                    f"TTF={self.ttf_current:.1f}s < {self.ttf_threshold}s threshold. "
                    f"AI wanted u={new_u_flow:.2f} m/s, OVERRIDDEN to MAX (3.0 m/s)"
                )
                new_u_flow = 3.0  # Maximum fan speed

                # Log the override
                self.safety_override_log.append({
                    'step': self.current_step,
                    'T_current': self.T_current,
                    'dT_dt': current_dT_dt,
                    'ttf': self.ttf_current,
                    'ai_action': self.u_flow + delta_u,
                    'override_action': 3.0,
                    'reason': override_reason
                })

        self.u_flow = new_u_flow

        # Use RecurrentPINN to predict next temperature with applied action
        with torch.no_grad():
            T_tensor = torch.tensor([self.T_current], dtype=torch.float32).to(self.device)
            Q_tensor = torch.tensor([self.Q_load], dtype=torch.float32).to(self.device)
            u_tensor = torch.tensor([self.u_flow], dtype=torch.float32).to(self.device)

            predictions = self.physics_model(T_tensor, Q_tensor, u_tensor)

            # Update state based on prediction
            self.T_t1_pred = predictions['T_t1'].cpu().numpy()[0, 0]
            self.dT_dt = predictions['dT_dt'].cpu().numpy()[0, 0]

            # Phase 4.2: Apply domain randomization
            if self.enable_domain_randomization:
                # Scale heating rate by thermal mass (lower mass = faster heating)
                self.dT_dt = self.dT_dt / self.thermal_mass_scale
                # Adjust temperature prediction based on modified heating rate
                self.T_t1_pred = self.T_current + (self.dT_dt * self.dt)
                # Add ambient temperature offset
                self.T_t1_pred += self.ambient_temp_offset

            # Update temperature (physics engine prediction)
            self.T_current = self.T_t1_pred

            # Note: RecurrentPINN maintains LSTM state internally for hysteresis

        # Tier 3 Safety: Automated Load Shedding
        # Track duration at max fan speed
        if abs(self.u_flow - 3.0) < 0.01:  # Fan at 100% (3.0 m/s)
            self.max_fan_duration += 1
        else:
            self.max_fan_duration = 0

        # Emergency shutdown logic
        if (self.T_current > self.critical_temp_threshold and
            self.max_fan_duration >= self.max_fan_timeout and
            not self.shutdown_state):
            # TRIGGER SHUTDOWN
            import datetime
            shutdown_timestamp = datetime.datetime.now().isoformat()

            self.emergency_shutdown_log.append({
                'timestamp': shutdown_timestamp,
                'step': self.current_step,
                'T_critical': self.T_current,
                'fan_speed': self.u_flow,
                'max_fan_duration': self.max_fan_duration,
                'Q_load_before': self.Q_load,
                'reason': f'EMERGENCY: T={self.T_current:.2f}°C > {self.critical_temp_threshold}°C with max fan for {self.max_fan_duration}s'
            })

            # Cut power to rack
            self.Q_load = 0.0
            self.shutdown_state = True

            print(f"🚨 EMERGENCY LOAD SHEDDING at t={self.current_step}s: T={self.T_current:.2f}°C")
            print(f"   Rack power cut to prevent hardware damage. Log: {shutdown_timestamp}")

        # Update tracking
        self.max_temp = max(self.max_temp, self.T_current)

        # Check if in danger zone
        if self.T_current > 75.0:
            self.time_in_danger += 1
        else:
            self.time_in_danger = 0

        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward

        # Calculate energy consumption (fan power ~ u^3)
        energy = 0.1 * (self.u_flow ** 3)
        self.total_energy += energy

        # Check termination conditions
        terminated = False
        if self.T_current > 80.0:
            # Thermal runaway - episode failed
            terminated = True
            reward -= 10000  # Massive penalty
            self.safety_violations += 1

        # Check truncation (time limit)
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_ttf(self, T_current: float, dT_dt: float, u_flow: float) -> float:
        """
        Calculate Time-to-Failure (TTF) in seconds.

        Estimates how long until temperature reaches critical threshold
        if current heating rate continues.

        Args:
            T_current: Current temperature (°C)
            dT_dt: Current heating rate (°C/s)
            u_flow: Current fan velocity (m/s)

        Returns:
            TTF in seconds (infinity if cooling or stable)
        """
        if dT_dt <= 0:
            # Temperature is stable or decreasing - no failure imminent
            return float('inf')

        # Calculate seconds until critical temperature
        temp_margin = self.critical_temp - T_current

        if temp_margin <= 0:
            # Already at or above critical temperature!
            return 0.0

        # TTF = (T_critical - T_current) / dT_dt
        ttf = temp_margin / dT_dt

        return max(0.0, ttf)

    def _calculate_reward(self) -> float:
        """
        Calculate reward for current state.

        Reward components:
        1. Energy savings: Lower fan speed = higher reward
        2. Safety penalty: Temperature above safe threshold
        3. Stability bonus: Smooth temperature control
        4. Safe zone bonus: Operating in optimal range
        5. Action smoothing penalty: Penalize large fan speed changes (Phase 4.2)
        """
        reward = 0.0

        # 1. Energy cost (negative reward for high fan speed)
        # Fan power ~ u^3 (cubic relationship)
        energy_cost = -0.1 * (self.u_flow ** 3)
        reward += energy_cost

        # 2. Thermal safety penalties
        if self.T_current > 80.0:
            # Critical: Immediate failure
            reward -= 1000.0
        elif self.T_current > 75.0:
            # Danger zone: Quadratic penalty
            overheat = self.T_current - 75.0
            reward -= 10.0 * (overheat ** 2)
        elif self.T_current > 70.0:
            # Warning zone: Linear penalty
            warm = self.T_current - 70.0
            reward -= 2.0 * warm
        else:
            # Safe zone bonus
            if 50.0 < self.T_current < 70.0:
                reward += 1.0

        # 3. Stability bonus (smooth control)
        if abs(self.dT_dt) < 0.1:
            reward += 0.5

        # 4. Penalty for prolonged danger
        if self.time_in_danger > 10:
            reward -= 5.0 * (self.time_in_danger - 10)

        # 5. Efficiency bonus (low temp with low fan speed)
        if self.T_current < 65.0 and self.u_flow < 1.5:
            reward += 2.0

        # 6. Phase 4.2: Action Continuity (smoothing penalty)
        # Penalize large changes in fan speed to encourage smooth control
        fan_speed_change = abs(self.u_flow - self.u_flow_prev)
        smoothing_penalty = -self.action_smoothing_penalty * fan_speed_change
        reward += smoothing_penalty

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        return np.array([
            self.T_current,
            self.Q_load,
            self.u_flow,
            self.dT_dt,
            self.T_t1_pred,
            self.time_in_danger
        ], dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            'step': self.current_step,
            'temperature': self.T_current,
            'load': self.Q_load,
            'fan_speed': self.u_flow,
            'heating_rate': self.dT_dt,
            'max_temp': self.max_temp,
            'total_reward': self.total_reward,
            'total_energy': self.total_energy,
            'safety_violations': self.safety_violations,
            'time_in_danger': self.time_in_danger,
            # Phase 4.4: Stability Guard info
            'ttf': self.ttf_current,
            'safety_override_count': len(self.safety_override_log),
            'safety_override_log': self.safety_override_log.copy(),
            # Tier 3 Safety: Emergency Shutdown
            'shutdown_state': self.shutdown_state,
            'max_fan_duration': self.max_fan_duration,
            'emergency_shutdown_count': len(self.emergency_shutdown_log),
            'emergency_shutdown_log': self.emergency_shutdown_log.copy()
        }

    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Temperature: {self.T_current:.2f}°C")
            print(f"IT Load: {self.Q_load/1000:.1f} kW")
            print(f"Fan Speed: {self.u_flow:.2f} m/s")
            print(f"Heating Rate: {self.dT_dt:.4f}°C/s")
            print(f"Predicted T(t+1): {self.T_t1_pred:.2f}°C")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"Total Energy: {self.total_energy:.2f}")

            # Safety status
            if self.T_current > 80.0:
                print("⚠️  STATUS: CRITICAL - THERMAL RUNAWAY")
            elif self.T_current > 75.0:
                print("⚠️  STATUS: DANGER ZONE")
            elif self.T_current > 70.0:
                print("⚡ STATUS: WARNING")
            else:
                print("✓ STATUS: SAFE")

    def close(self):
        """Clean up resources."""
        pass


# Test the environment
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING COOLING CONTROL ENVIRONMENT")
    print("=" * 70)

    # Create environment
    model_path = "checkpoints/best_recurrent_pinn.pt"

    try:
        env = CoolingControlEnv(
            model_path=model_path,
            device='cpu',
            max_steps=100
        )

        print("\n✓ Environment created successfully!")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Test random episode
        print("\n" + "=" * 70)
        print("RUNNING RANDOM EPISODE (100 steps)")
        print("=" * 70)

        obs, info = env.reset()
        print(f"\nInitial state:")
        print(f"  Temperature: {obs[0]:.2f}°C")
        print(f"  IT Load: {obs[1]/1000:.1f} kW")
        print(f"  Fan Speed: {obs[2]:.2f} m/s")

        for step in range(10):  # Show first 10 steps
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if step < 5:  # Print first 5 steps
                print(f"\nStep {step + 1}:")
                print(f"  Action (Δu): {action[0]:+.3f} m/s")
                print(f"  New fan speed: {obs[2]:.2f} m/s")
                print(f"  Temperature: {obs[0]:.2f}°C")
                print(f"  Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"\nEpisode ended at step {step + 1}")
                break

        print("\n" + "=" * 70)
        print("EPISODE SUMMARY")
        print("=" * 70)
        print(f"  Total steps: {info['step']}")
        print(f"  Final temperature: {info['temperature']:.2f}°C")
        print(f"  Max temperature: {info['max_temp']:.2f}°C")
        print(f"  Total reward: {info['total_reward']:.2f}")
        print(f"  Total energy: {info['total_energy']:.2f}")
        print(f"  Safety violations: {info['safety_violations']}")

        print("\n✓ Environment test completed successfully!")

    except FileNotFoundError:
        print(f"\n❌ Error: Model checkpoint not found at {model_path}")
        print("   Please ensure the RecurrentPINN model is trained first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
