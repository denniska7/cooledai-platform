#!/usr/bin/env python3
"""
Multi-Rack Reinforcement Learning Environment for Data Center Cooling Control

Scales from single-rack to 10-rack edge data center cluster with:
- Independent thermal dynamics per rack
- Thermal bleed (5%) between adjacent racks
- Collective stability guard with per-rack TTF monitoring
- Aggregate energy savings tracking

Author: Claude (Anthropic)
Date: 2026-01-28
Phase: 5 - Fleet Management
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.recurrent_pinn import RecurrentPINN


class MultiRackCoolingEnv(gym.Env):
    """
    10-Rack Data Center Cooling Environment

    Each rack has independent:
    - Temperature (T_current)
    - IT Load (Q_load)
    - Fan Speed (u_flow)
    - Heating rate (dT_dt)
    - Time-to-Failure (TTF)

    Racks are thermally coupled:
    - 5% thermal bleed to adjacent racks
    - Rack 0: only affects Rack 1
    - Rack 1-8: affect both neighbors
    - Rack 9: only affects Rack 8

    State Space:
        For each rack i (i=0..9):
        - T_current_i: Current temperature (°C)
        - Q_load_i: IT equipment heat load (W)
        - u_flow_i: Air flow velocity (m/s)
        - dT_dt_i: Heating rate (°C/s)
        - T_t1_pred_i: 1-second ahead prediction (°C)
        - time_in_danger_i: Consecutive seconds in danger zone
        Total: 60 values (6 per rack × 10 racks)

    Action Space:
        - delta_u_i: Change in fan velocity for each rack i (continuous, -0.5 to +0.5 m/s)
        Total: 10 actions (one per rack)

    Reward Function:
        - Aggregate energy savings across all racks
        - Per-rack thermal penalties
        - Collective stability bonuses

    Done Condition:
        - Any rack exceeds 80°C (thermal runaway)
        - Episode time limit reached
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        model_path: str,
        num_racks: int = 10,
        device: str = 'cpu',
        max_steps: int = 1000,
        initial_temp_range: Tuple[float, float] = (45.0, 65.0),
        load_range: Tuple[float, float] = (50000.0, 150000.0),
        fan_range: Tuple[float, float] = (0.5, 3.0),
        dt: float = 1.0,
        thermal_bleed_factor: float = 0.05,  # 5% thermal coupling
        enable_domain_randomization: bool = False,
        thermal_mass_variation: float = 0.2,
        ambient_temp_variation: float = 5.0,
        action_smoothing_penalty: float = 1.0,
        enable_stability_guard: bool = True,
        ttf_threshold: float = 120.0,
        critical_temp: float = 85.0
    ):
        """
        Initialize the multi-rack cooling control environment.

        Args:
            model_path: Path to trained RecurrentPINN checkpoint
            num_racks: Number of racks in the cluster (default: 10)
            device: Device for model inference ('cpu', 'cuda', 'mps')
            max_steps: Maximum steps per episode
            initial_temp_range: Range for initial temperature (°C)
            load_range: Range for IT load variation (W)
            fan_range: Range for fan velocity (m/s)
            dt: Time step in seconds
            thermal_bleed_factor: Fraction of heat that bleeds to adjacent racks (0.05 = 5%)
            enable_domain_randomization: Enable domain randomization
            thermal_mass_variation: Thermal mass randomization (±fraction)
            ambient_temp_variation: Ambient temperature variation (±°C)
            action_smoothing_penalty: Penalty weight for large fan speed changes
            enable_stability_guard: Enable physical safety override
            ttf_threshold: Time-to-Failure threshold for override (seconds)
            critical_temp: Critical temperature threshold (°C)
        """
        super(MultiRackCoolingEnv, self).__init__()

        # Environment parameters
        self.device = device
        self.num_racks = num_racks
        self.max_steps = max_steps
        self.initial_temp_range = initial_temp_range
        self.load_range = load_range
        self.fan_range = fan_range
        self.dt = dt
        self.thermal_bleed_factor = thermal_bleed_factor

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
        print(f"RecurrentPINN loaded successfully for {num_racks}-rack cluster!")

        # Define action space: Change in fan velocity for each rack
        # Shape: (num_racks,) continuous actions in [-0.5, +0.5] m/s
        self.action_space = spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(num_racks,),
            dtype=np.float32
        )

        # Define observation space: 6 values per rack
        # [T_current, Q_load, u_flow, dT_dt, T_t1_pred, time_in_danger] × num_racks
        obs_low = np.tile([0.0, 0.0, 0.5, -10.0, 0.0, 0.0], num_racks)
        obs_high = np.tile([100.0, 200000.0, 3.0, 10.0, 100.0, 100.0], num_racks)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # State variables (arrays for num_racks)
        self.current_step = 0
        self.T_current_racks = np.zeros(num_racks)
        self.Q_load_racks = np.zeros(num_racks)
        self.u_flow_racks = np.zeros(num_racks)
        self.u_flow_prev_racks = np.zeros(num_racks)
        self.dT_dt_racks = np.zeros(num_racks)
        self.T_t1_pred_racks = np.zeros(num_racks)
        self.time_in_danger_racks = np.zeros(num_racks)

        # Episode statistics
        self.total_reward = 0.0
        self.total_energy = 0.0
        self.max_temp_racks = np.zeros(num_racks)
        self.safety_violations_racks = np.zeros(num_racks)

        # Phase 4.2: Domain Randomization parameters (per episode)
        self.thermal_mass_scale = 1.0
        self.ambient_temp_offset = 0.0

        # Phase 4.4: Stability Guard tracking (per rack)
        self.safety_override_log = []  # List of override events
        self.ttf_current_racks = np.full(num_racks, float('inf'))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state (60 values for 10 racks)
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Phase 4.2: Domain Randomization (if enabled)
        if self.enable_domain_randomization:
            self.thermal_mass_scale = np.random.uniform(
                1.0 - self.thermal_mass_variation,
                1.0 + self.thermal_mass_variation
            )
            self.ambient_temp_offset = np.random.uniform(
                -self.ambient_temp_variation,
                self.ambient_temp_variation
            )
        else:
            self.thermal_mass_scale = 1.0
            self.ambient_temp_offset = 0.0

        # Sample initial conditions for each rack
        for i in range(self.num_racks):
            self.T_current_racks[i] = np.random.uniform(*self.initial_temp_range)
            self.Q_load_racks[i] = np.random.uniform(*self.load_range)
            self.u_flow_racks[i] = np.random.uniform(1.0, 2.0)  # Moderate fan speed
            self.u_flow_prev_racks[i] = self.u_flow_racks[i]

        # Get initial predictions for all racks (batch inference)
        with torch.no_grad():
            T_tensor = torch.tensor(self.T_current_racks, dtype=torch.float32).to(self.device)
            Q_tensor = torch.tensor(self.Q_load_racks, dtype=torch.float32).to(self.device)
            u_tensor = torch.tensor(self.u_flow_racks, dtype=torch.float32).to(self.device)

            predictions = self.physics_model(T_tensor, Q_tensor, u_tensor)
            self.T_t1_pred_racks = predictions['T_t1'].cpu().numpy()[:, 0]
            self.dT_dt_racks = predictions['dT_dt'].cpu().numpy()[:, 0]

        # Reset tracking variables
        self.time_in_danger_racks = np.zeros(self.num_racks)
        self.total_reward = 0.0
        self.total_energy = 0.0
        self.max_temp_racks = self.T_current_racks.copy()
        self.safety_violations_racks = np.zeros(self.num_racks)
        self.safety_override_log = []
        self.ttf_current_racks = np.full(self.num_racks, float('inf'))

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Change in fan velocity for each rack (delta_u array, shape: [num_racks])

        Returns:
            observation: New state (60 values)
            reward: Aggregate reward across all racks
            terminated: Whether any rack failed
            truncated: Whether episode reached time limit
            info: Additional information
        """
        self.current_step += 1

        # Store previous fan speeds for smoothing penalty
        self.u_flow_prev_racks = self.u_flow_racks.copy()

        # Simulate IT load variation for each rack
        load_changes = np.random.uniform(-5000, 5000, self.num_racks)
        self.Q_load_racks = np.clip(
            self.Q_load_racks + load_changes,
            self.load_range[0],
            self.load_range[1]
        )

        # Get current thermal predictions for stability guard (batch inference)
        with torch.no_grad():
            T_tensor = torch.tensor(self.T_current_racks, dtype=torch.float32).to(self.device)
            Q_tensor = torch.tensor(self.Q_load_racks, dtype=torch.float32).to(self.device)
            u_tensor_current = torch.tensor(self.u_flow_racks, dtype=torch.float32).to(self.device)

            predictions_current = self.physics_model(T_tensor, Q_tensor, u_tensor_current)
            current_dT_dt_racks = predictions_current['dT_dt'].cpu().numpy()[:, 0]

        # Calculate current Time-to-Failure for each rack
        for i in range(self.num_racks):
            self.ttf_current_racks[i] = self._calculate_ttf(
                self.T_current_racks[i],
                current_dT_dt_racks[i],
                self.u_flow_racks[i]
            )

        # Apply actions: Adjust fan velocity for each rack
        delta_u_racks = action  # Shape: (num_racks,)
        new_u_flow_racks = np.clip(
            self.u_flow_racks + delta_u_racks,
            self.fan_range[0],
            self.fan_range[1]
        )

        # Phase 5: Collective Stability Guard - Per-rack physical override
        if self.enable_stability_guard:
            for i in range(self.num_racks):
                # Check if this rack's TTF is critically low AND agent wants to reduce cooling
                if self.ttf_current_racks[i] < self.ttf_threshold and new_u_flow_racks[i] < self.u_flow_racks[i]:
                    # OVERRIDE: Force this rack's fans to maximum speed
                    override_reason = (
                        f"Rack {i}: TTF={self.ttf_current_racks[i]:.1f}s < {self.ttf_threshold}s. "
                        f"AI wanted u={new_u_flow_racks[i]:.2f} m/s, OVERRIDDEN to MAX (3.0 m/s)"
                    )
                    new_u_flow_racks[i] = 3.0  # Maximum fan speed

                    # Log the override
                    self.safety_override_log.append({
                        'step': self.current_step,
                        'rack_id': i,
                        'T_current': self.T_current_racks[i],
                        'dT_dt': current_dT_dt_racks[i],
                        'ttf': self.ttf_current_racks[i],
                        'ai_action': self.u_flow_racks[i] + delta_u_racks[i],
                        'override_action': 3.0,
                        'reason': override_reason
                    })

        self.u_flow_racks = new_u_flow_racks

        # Use RecurrentPINN to predict next temperature for all racks (batch inference)
        with torch.no_grad():
            T_tensor = torch.tensor(self.T_current_racks, dtype=torch.float32).to(self.device)
            Q_tensor = torch.tensor(self.Q_load_racks, dtype=torch.float32).to(self.device)
            u_tensor = torch.tensor(self.u_flow_racks, dtype=torch.float32).to(self.device)

            predictions = self.physics_model(T_tensor, Q_tensor, u_tensor)

            # Update state based on predictions
            self.T_t1_pred_racks = predictions['T_t1'].cpu().numpy()[:, 0]
            self.dT_dt_racks = predictions['dT_dt'].cpu().numpy()[:, 0]

            # Phase 4.2: Apply domain randomization
            if self.enable_domain_randomization:
                self.dT_dt_racks = self.dT_dt_racks / self.thermal_mass_scale
                self.T_t1_pred_racks = self.T_current_racks + (self.dT_dt_racks * self.dt)
                self.T_t1_pred_racks += self.ambient_temp_offset

            # Update temperatures
            self.T_current_racks = self.T_t1_pred_racks

        # Phase 5: Apply thermal bleed between adjacent racks
        self._apply_thermal_bleed()

        # Update tracking
        self.max_temp_racks = np.maximum(self.max_temp_racks, self.T_current_racks)

        # Check if each rack is in danger zone
        for i in range(self.num_racks):
            if self.T_current_racks[i] > 75.0:
                self.time_in_danger_racks[i] += 1
            else:
                self.time_in_danger_racks[i] = 0

        # Calculate aggregate reward across all racks
        reward = self._calculate_reward()
        self.total_reward += reward

        # Calculate total energy consumption (sum across all racks)
        energy_racks = 0.1 * (self.u_flow_racks ** 3)
        total_energy_this_step = np.sum(energy_racks)
        self.total_energy += total_energy_this_step

        # Check termination conditions
        terminated = False
        for i in range(self.num_racks):
            if self.T_current_racks[i] > 80.0:
                # Thermal runaway in at least one rack - episode failed
                terminated = True
                reward -= 10000  # Massive penalty
                self.safety_violations_racks[i] += 1

        # Check truncation (time limit)
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _apply_thermal_bleed(self):
        """
        Apply thermal coupling between adjacent racks.

        Each rack's temperature is influenced by 5% of the temperature
        difference with adjacent racks.

        Heat flow: Q_bleed = thermal_bleed_factor × (T_neighbor - T_current)
        """
        T_original = self.T_current_racks.copy()

        for i in range(self.num_racks):
            bleed = 0.0

            # Left neighbor (rack i-1)
            if i > 0:
                bleed += self.thermal_bleed_factor * (T_original[i-1] - T_original[i])

            # Right neighbor (rack i+1)
            if i < self.num_racks - 1:
                bleed += self.thermal_bleed_factor * (T_original[i+1] - T_original[i])

            # Apply thermal bleed
            self.T_current_racks[i] += bleed

    def _calculate_ttf(self, T_current: float, dT_dt: float, u_flow: float) -> float:
        """
        Calculate Time-to-Failure (TTF) in seconds for a single rack.

        Args:
            T_current: Current temperature (°C)
            dT_dt: Current heating rate (°C/s)
            u_flow: Current fan velocity (m/s)

        Returns:
            TTF in seconds (infinity if cooling or stable)
        """
        if dT_dt <= 0:
            return float('inf')

        temp_margin = self.critical_temp - T_current

        if temp_margin <= 0:
            return 0.0

        ttf = temp_margin / dT_dt
        return max(0.0, ttf)

    def _calculate_reward(self) -> float:
        """
        Calculate aggregate reward across all racks.

        Components:
        1. Energy savings (sum across racks)
        2. Safety penalties (per-rack)
        3. Stability bonuses (per-rack)
        4. Action smoothing penalties (per-rack)
        """
        reward = 0.0

        # Iterate through all racks
        for i in range(self.num_racks):
            # 1. Energy cost (negative reward for high fan speed)
            energy_cost = -0.1 * (self.u_flow_racks[i] ** 3)
            reward += energy_cost

            # 2. Thermal safety penalties
            T = self.T_current_racks[i]
            if T > 80.0:
                reward -= 1000.0
            elif T > 75.0:
                overheat = T - 75.0
                reward -= 10.0 * (overheat ** 2)
            elif T > 70.0:
                warm = T - 70.0
                reward -= 2.0 * warm
            else:
                if 50.0 < T < 70.0:
                    reward += 1.0

            # 3. Stability bonus
            if abs(self.dT_dt_racks[i]) < 0.1:
                reward += 0.5

            # 4. Penalty for prolonged danger
            if self.time_in_danger_racks[i] > 10:
                reward -= 5.0 * (self.time_in_danger_racks[i] - 10)

            # 5. Efficiency bonus
            if T < 65.0 and self.u_flow_racks[i] < 1.5:
                reward += 2.0

            # 6. Action Continuity (smoothing penalty)
            fan_speed_change = abs(self.u_flow_racks[i] - self.u_flow_prev_racks[i])
            smoothing_penalty = -self.action_smoothing_penalty * fan_speed_change
            reward += smoothing_penalty

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation vector.

        Returns:
            Flattened array of shape (60,) containing:
            [T1, Q1, u1, dT_dt1, T_t11, time1, ..., T10, Q10, u10, dT_dt10, T_t110, time10]
        """
        obs_list = []
        for i in range(self.num_racks):
            obs_list.extend([
                self.T_current_racks[i],
                self.Q_load_racks[i],
                self.u_flow_racks[i],
                self.dT_dt_racks[i],
                self.T_t1_pred_racks[i],
                self.time_in_danger_racks[i]
            ])
        return np.array(obs_list, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            'step': self.current_step,
            'temperatures': self.T_current_racks.copy(),
            'loads': self.Q_load_racks.copy(),
            'fan_speeds': self.u_flow_racks.copy(),
            'heating_rates': self.dT_dt_racks.copy(),
            'max_temps': self.max_temp_racks.copy(),
            'total_reward': self.total_reward,
            'total_energy': self.total_energy,
            'safety_violations': self.safety_violations_racks.copy(),
            'time_in_danger': self.time_in_danger_racks.copy(),
            # Phase 5: Multi-rack metrics
            'avg_temperature': np.mean(self.T_current_racks),
            'max_temperature': np.max(self.T_current_racks),
            'min_temperature': np.min(self.T_current_racks),
            'total_load': np.sum(self.Q_load_racks),
            'avg_fan_speed': np.mean(self.u_flow_racks),
            # Phase 5: Collective Stability Guard
            'ttf_racks': self.ttf_current_racks.copy(),
            'min_ttf': np.min(self.ttf_current_racks),
            'safety_override_count': len(self.safety_override_log),
            'safety_override_log': self.safety_override_log.copy()
        }

    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} - 10-Rack Cluster ===")
            print(f"Cluster Stats:")
            print(f"  Avg Temp: {np.mean(self.T_current_racks):.2f}°C")
            print(f"  Max Temp: {np.max(self.T_current_racks):.2f}°C (Rack {np.argmax(self.T_current_racks)})")
            print(f"  Min Temp: {np.min(self.T_current_racks):.2f}°C (Rack {np.argmin(self.T_current_racks)})")
            print(f"  Total Load: {np.sum(self.Q_load_racks)/1000:.1f} kW")
            print(f"  Avg Fan Speed: {np.mean(self.u_flow_racks):.2f} m/s")
            print(f"  Total Energy: {self.total_energy:.2f}")
            print(f"  Total Reward: {self.total_reward:.2f}")

            print(f"\nPer-Rack Status:")
            for i in range(self.num_racks):
                status = "✓" if self.T_current_racks[i] < 75.0 else "⚠️"
                print(f"  Rack {i}: {status} T={self.T_current_racks[i]:.1f}°C  "
                      f"Q={self.Q_load_racks[i]/1000:.1f}kW  u={self.u_flow_racks[i]:.2f}m/s")

    def close(self):
        """Clean up resources."""
        pass


# Test the environment
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING 10-RACK COOLING CONTROL ENVIRONMENT")
    print("=" * 70)

    # Create environment
    model_path = "../checkpoints/best_recurrent_pinn.pt"

    try:
        env = MultiRackCoolingEnv(
            model_path=model_path,
            num_racks=10,
            device='cpu',
            max_steps=100,
            thermal_bleed_factor=0.05  # 5% thermal coupling
        )

        print("\n✓ Multi-rack environment created successfully!")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Number of racks: {env.num_racks}")
        print(f"  Thermal bleed factor: {env.thermal_bleed_factor * 100}%")

        # Test random episode
        print("\n" + "=" * 70)
        print("RUNNING RANDOM EPISODE (10 steps)")
        print("=" * 70)

        obs, info = env.reset()
        print(f"\nInitial cluster state:")
        print(f"  Avg Temperature: {info['avg_temperature']:.2f}°C")
        print(f"  Total Load: {info['total_load']/1000:.1f} kW")
        print(f"  Avg Fan Speed: {info['avg_fan_speed']:.2f} m/s")

        for step in range(10):
            # Random action for all racks
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if step < 3:  # Print first 3 steps
                print(f"\nStep {step + 1}:")
                print(f"  Actions: {action}")
                print(f"  Avg Temp: {info['avg_temperature']:.2f}°C")
                print(f"  Max Temp: {info['max_temperature']:.2f}°C (Rack {np.argmax(info['temperatures'])})")
                print(f"  Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"\nEpisode ended at step {step + 1}")
                break

        print("\n" + "=" * 70)
        print("EPISODE SUMMARY")
        print("=" * 70)
        print(f"  Total steps: {info['step']}")
        print(f"  Avg temperature: {info['avg_temperature']:.2f}°C")
        print(f"  Max temperature: {info['max_temperature']:.2f}°C")
        print(f"  Total reward: {info['total_reward']:.2f}")
        print(f"  Total energy: {info['total_energy']:.2f}")
        print(f"  Safety overrides: {info['safety_override_count']}")

        print("\n✓ Multi-rack environment test completed successfully!")

    except FileNotFoundError:
        print(f"\n❌ Error: Model checkpoint not found at {model_path}")
        print("   Please ensure the RecurrentPINN model is trained first.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
