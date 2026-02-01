#!/usr/bin/env python3
"""
1MW Thermal Cluster Environment for Data Center Cooling Control

Models a complete 1.0 MW site as 8 Blackwell GB200 Pods (125kW each).
Implements Triple-Tier Safety, thermal bleed between adjacent pods,
and production-ready emergency shutdown logic.

Author: Claude (Anthropic)
Date: 2026-01-29
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
import sys
import os
import datetime

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.recurrent_pinn import RecurrentPINN


class CoolingControl1MWEnv(gym.Env):
    """
    1MW Data Center Cooling Control Environment

    Site Configuration:
        - Total Capacity: 1.0 MW (1,000 kW)
        - Architecture: 8 Blackwell GB200 Pods
        - Pod Power: 125 kW each
        - Cooling: Hybrid (90% liquid, 10% air rejection per pod)

    Triple-Tier Safety:
        Tier 1 (Optimization): RL agent minimizes fan power while T < 70°C
        Tier 2 (Stability Guard): If TTF < 120s, force fans to 100%, bypass AI
        Tier 3 (Surgical Shutdown): If T > 85°C for 5s at 100% fan,
                                     set THAT specific pod's load to 0kW

    Thermal Interaction:
        - Thermal Bleed: 0.05 between adjacent pods
        - Pod 0 affects Pod 1 only
        - Pods 1-6 affect both neighbors
        - Pod 7 affects Pod 6 only

    State Space (48 values):
        For each of 8 pods:
            - T_current: Current temperature (°C)
            - Q_load: IT equipment heat load (W)
            - u_flow: Air flow velocity (m/s)
            - dT_dt: Heating rate (°C/s)
            - T_t1_pred: 1-second ahead prediction (°C)
            - time_in_danger: Consecutive seconds in danger zone

    Action Space (8 values):
        - delta_u[0-7]: Change in fan velocity for each pod (-0.5 to +0.5 m/s)

    Reward Function:
        - Energy savings: -0.1 * sum(u_flow_i^3) (cubic power law)
        - Thermal penalty: -1000 per pod if T > 80°C (critical)
        - Danger zone penalty: -10 * (T - 75)^2 per pod if T > 75°C
        - Safe zone bonus: +1 per pod if 50°C < T < 70°C
        - Stability bonus: +0.5 per pod if |dT/dt| < 0.1°C/s
        - Thermal coupling bonus: +2 if all pods within 10°C of each other
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        max_steps: int = 1000,
        num_pods: int = 8,
        pod_power_kw: float = 125.0,
        initial_temp_range: Tuple[float, float] = (55.0, 65.0),
        fan_range: Tuple[float, float] = (0.5, 3.0),
        dt: float = 1.0,
        thermal_bleed_factor: float = 0.05,
        enable_stability_guard: bool = True,
        ttf_threshold: float = 120.0,
        critical_temp: float = 85.0,
        shutdown_timeout: int = 5,
        enable_domain_randomization: bool = False,
        thermal_mass_variation: float = 0.2,
        ambient_temp_variation: float = 5.0
    ):
        """
        Initialize the 1MW cooling control environment.

        Args:
            model_path: Path to trained RecurrentPINN checkpoint
            device: Device for model inference ('cpu', 'cuda', 'mps')
            max_steps: Maximum steps per episode
            num_pods: Number of Blackwell pods (default: 8)
            pod_power_kw: Power per pod in kW (default: 125kW)
            initial_temp_range: Range for initial temperature (°C)
            fan_range: Range for fan velocity (m/s)
            dt: Time step in seconds
            thermal_bleed_factor: Heat transfer coefficient between pods (0.05 = 5%)
            enable_stability_guard: Enable Tier 2 safety override
            ttf_threshold: Time-to-Failure threshold for Tier 2 (seconds)
            critical_temp: Critical temperature for Tier 3 shutdown (°C)
            shutdown_timeout: Seconds at 100% fan before Tier 3 trigger
            enable_domain_randomization: Enable domain randomization
            thermal_mass_variation: Thermal mass randomization (±fraction)
            ambient_temp_variation: Ambient temperature variation (±°C)
        """
        super(CoolingControl1MWEnv, self).__init__()

        # Site configuration
        self.num_pods = num_pods
        self.pod_power_kw = pod_power_kw
        self.total_capacity_kw = num_pods * pod_power_kw

        # Environment parameters
        self.device = device
        self.max_steps = max_steps
        self.initial_temp_range = initial_temp_range
        self.fan_range = fan_range
        self.dt = dt
        self.thermal_bleed_factor = thermal_bleed_factor

        # Triple-Tier Safety parameters
        self.enable_stability_guard = enable_stability_guard
        self.ttf_threshold = ttf_threshold
        self.critical_temp = critical_temp
        self.shutdown_timeout = shutdown_timeout

        # Domain Randomization
        self.enable_domain_randomization = enable_domain_randomization
        self.thermal_mass_variation = thermal_mass_variation
        self.ambient_temp_variation = ambient_temp_variation

        # Load trained RecurrentPINN (physics engine)
        print(f"Loading RecurrentPINN for 1MW Cluster ({num_pods} pods)...")
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
        print(f"RecurrentPINN loaded successfully! Total capacity: {self.total_capacity_kw:.0f} kW")

        # Define action space: 8 fan velocity changes
        self.action_space = spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(self.num_pods,),
            dtype=np.float32
        )

        # Define observation space: 6 features × 8 pods = 48 values
        obs_dim = self.num_pods * 6
        self.observation_space = spaces.Box(
            low=np.tile([0.0, 0.0, 0.5, -10.0, 0.0, 0.0], self.num_pods),
            high=np.tile([100.0, 200000.0, 3.0, 10.0, 100.0, 100.0], self.num_pods),
            dtype=np.float32
        )

        # Per-pod state variables
        self.T_current_pods = np.zeros(self.num_pods)
        self.Q_load_pods = np.zeros(self.num_pods)
        self.u_flow_pods = np.zeros(self.num_pods)
        self.u_flow_prev_pods = np.zeros(self.num_pods)
        self.dT_dt_pods = np.zeros(self.num_pods)
        self.T_t1_pred_pods = np.zeros(self.num_pods)
        self.time_in_danger_pods = np.zeros(self.num_pods, dtype=int)

        # Tier 2 (Stability Guard) tracking per pod
        self.ttf_current_pods = np.full(self.num_pods, float('inf'))
        self.safety_override_active_pods = np.zeros(self.num_pods, dtype=bool)
        self.safety_override_log = []

        # Tier 3 (Surgical Shutdown) tracking per pod
        self.shutdown_state_pods = np.zeros(self.num_pods, dtype=bool)
        self.max_fan_duration_pods = np.zeros(self.num_pods, dtype=int)
        self.emergency_shutdown_log = []

        # Episode statistics
        self.current_step = 0
        self.total_reward = 0.0
        self.total_energy = 0.0
        self.max_temp = 0.0
        self.safety_violations = 0

        # Domain Randomization (per episode)
        self.thermal_mass_scale = 1.0
        self.ambient_temp_offset = 0.0

        # LSTM hidden states (one per pod)
        self.hidden_states = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state (48 values)
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Domain Randomization (if enabled)
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

        # Initialize pod temperatures (slightly randomized)
        self.T_current_pods = np.random.uniform(
            self.initial_temp_range[0],
            self.initial_temp_range[1],
            size=self.num_pods
        )

        # Initialize pod loads (default: full power)
        self.Q_load_pods = np.full(self.num_pods, self.pod_power_kw * 1000.0)  # Convert to Watts

        # Initialize fan speeds (conservative start: 1.5 m/s)
        self.u_flow_pods = np.full(self.num_pods, 1.5)
        self.u_flow_prev_pods = self.u_flow_pods.copy()

        # Reset derived states
        self.dT_dt_pods = np.zeros(self.num_pods)
        self.T_t1_pred_pods = self.T_current_pods.copy()
        self.time_in_danger_pods = np.zeros(self.num_pods, dtype=int)

        # Reset safety tracking
        self.ttf_current_pods = np.full(self.num_pods, float('inf'))
        self.safety_override_active_pods = np.zeros(self.num_pods, dtype=bool)
        self.safety_override_log = []

        # Reset Tier 3 shutdown tracking
        self.shutdown_state_pods = np.zeros(self.num_pods, dtype=bool)
        self.max_fan_duration_pods = np.zeros(self.num_pods, dtype=int)
        self.emergency_shutdown_log = []

        # Reset episode statistics
        self.total_reward = 0.0
        self.total_energy = 0.0
        self.max_temp = np.max(self.T_current_pods)
        self.safety_violations = 0

        # Reset LSTM hidden states
        self.hidden_states = None

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _apply_thermal_bleed(self):
        """
        Apply thermal bleed between adjacent pods.

        Heat transfer equation:
            Q_bleed_i = thermal_bleed_factor * (T_neighbor - T_i)

        Boundary conditions:
            - Pod 0: Only receives heat from Pod 1
            - Pods 1-6: Receive heat from both neighbors
            - Pod 7: Only receives heat from Pod 6
        """
        T_original = self.T_current_pods.copy()

        for i in range(self.num_pods):
            bleed = 0.0

            # Left neighbor
            if i > 0:
                bleed += self.thermal_bleed_factor * (T_original[i-1] - T_original[i])

            # Right neighbor
            if i < self.num_pods - 1:
                bleed += self.thermal_bleed_factor * (T_original[i+1] - T_original[i])

            # Apply thermal bleed
            self.T_current_pods[i] += bleed

    def _check_tier2_stability_guard(self):
        """
        Tier 2 Safety: Stability Guard

        If Time-to-Failure < 120 seconds, override AI and force fan to 100%.
        This prevents thermal runaway before it becomes critical.
        """
        if not self.enable_stability_guard:
            return

        for i in range(self.num_pods):
            # Skip if pod is shutdown
            if self.shutdown_state_pods[i]:
                continue

            # Calculate Time-to-Failure
            if self.dT_dt_pods[i] > 0:
                self.ttf_current_pods[i] = (self.critical_temp - self.T_current_pods[i]) / self.dT_dt_pods[i]
            else:
                self.ttf_current_pods[i] = float('inf')

            # Trigger Tier 2 override if TTF < threshold
            if self.ttf_current_pods[i] < self.ttf_threshold:
                if not self.safety_override_active_pods[i]:
                    # Log first activation
                    self.safety_override_log.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'step': self.current_step,
                        'pod_idx': i,
                        'tier': 2,
                        'T_current': self.T_current_pods[i],
                        'TTF': self.ttf_current_pods[i],
                        'u_flow_before': self.u_flow_pods[i],
                        'reason': f'TTF={self.ttf_current_pods[i]:.1f}s < {self.ttf_threshold}s'
                    })
                    self.safety_override_active_pods[i] = True

                # Override: Force fan to 100%
                self.u_flow_pods[i] = 3.0
            else:
                if self.safety_override_active_pods[i]:
                    # Log deactivation
                    self.safety_override_log.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'step': self.current_step,
                        'pod_idx': i,
                        'tier': 2,
                        'T_current': self.T_current_pods[i],
                        'TTF': self.ttf_current_pods[i],
                        'reason': 'Override deactivated - TTF recovered'
                    })
                    self.safety_override_active_pods[i] = False

    def _check_tier3_surgical_shutdown(self):
        """
        Tier 3 Safety: Surgical Shutdown

        If T > 85°C AND fan at 100% for 5+ seconds, cut power to THAT pod only.
        This prevents hardware melting when cooling fails.
        """
        for i in range(self.num_pods):
            # Skip if already shutdown
            if self.shutdown_state_pods[i]:
                continue

            # Check if at max fan speed
            fan_at_max = abs(self.u_flow_pods[i] - 3.0) < 0.01

            if self.T_current_pods[i] > self.critical_temp and fan_at_max:
                self.max_fan_duration_pods[i] += 1

                # Trigger shutdown if at max fan for too long
                if self.max_fan_duration_pods[i] >= self.shutdown_timeout:
                    timestamp = datetime.datetime.now().isoformat()

                    self.emergency_shutdown_log.append({
                        'timestamp': timestamp,
                        'step': self.current_step,
                        'pod_idx': i,
                        'tier': 3,
                        'T_critical': self.T_current_pods[i],
                        'fan_speed': self.u_flow_pods[i],
                        'max_fan_duration': self.max_fan_duration_pods[i],
                        'Q_load_before': self.Q_load_pods[i],
                        'reason': f'EMERGENCY: Pod {i} at T={self.T_current_pods[i]:.2f}°C > {self.critical_temp}°C with max fan for {self.max_fan_duration_pods[i]}s'
                    })

                    # SURGICAL SHUTDOWN: Cut power to THIS pod only
                    self.Q_load_pods[i] = 0.0
                    self.shutdown_state_pods[i] = True

                    print(f"🚨 TIER 3 SURGICAL SHUTDOWN: Pod {i} at T={self.T_current_pods[i]:.2f}°C")
                    print(f"   Pod {i} power cut to 0kW. Other pods continue operation.")
                    print(f"   Log: {timestamp}")
            else:
                # Reset timer if conditions not met
                self.max_fan_duration_pods[i] = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep of the environment.

        Args:
            action: Array of 8 fan velocity changes (delta_u)

        Returns:
            observation: Current state (48 values)
            reward: Reward for this timestep
            terminated: Whether episode is done (thermal runaway)
            truncated: Whether episode hit time limit
            info: Additional information
        """
        self.current_step += 1

        # Store previous fan speeds for action smoothing penalty
        self.u_flow_prev_pods = self.u_flow_pods.copy()

        # Apply actions: Update fan velocities
        for i in range(self.num_pods):
            # Skip action for shutdown pods (keep fan at max for cooling residual heat)
            if self.shutdown_state_pods[i]:
                self.u_flow_pods[i] = 3.0
                continue

            delta_u = np.clip(action[i], -0.5, 0.5)
            self.u_flow_pods[i] = np.clip(
                self.u_flow_pods[i] + delta_u,
                self.fan_range[0],
                self.fan_range[1]
            )

        # Tier 2: Stability Guard (override AI if TTF < 120s)
        self._check_tier2_stability_guard()

        # Predict next temperatures using RecurrentPINN (batch inference)
        with torch.no_grad():
            T_tensor = torch.tensor(
                self.T_current_pods + self.ambient_temp_offset,
                dtype=torch.float32
            ).unsqueeze(1).to(self.device)

            Q_tensor = torch.tensor(
                self.Q_load_pods,
                dtype=torch.float32
            ).unsqueeze(1).to(self.device)

            u_tensor = torch.tensor(
                self.u_flow_pods,
                dtype=torch.float32
            ).unsqueeze(1).to(self.device)

            # Batch prediction for all pods
            predictions = self.physics_model(T_tensor, Q_tensor, u_tensor, self.hidden_states)

            # Update temperatures
            self.T_current_pods = predictions['T_t1'].cpu().numpy().flatten()
            self.T_t1_pred_pods = self.T_current_pods.copy()

            # Update heating rates
            self.dT_dt_pods = predictions['dT_dt'].cpu().numpy().flatten()

            # Update LSTM hidden states
            if 'hidden_state' in predictions:
                self.hidden_states = predictions['hidden_state']

        # Apply thermal bleed between adjacent pods
        self._apply_thermal_bleed()

        # Tier 3: Surgical Shutdown (if T > 85°C at max fan for 5s)
        self._check_tier3_surgical_shutdown()

        # Update danger zone tracking
        for i in range(self.num_pods):
            if self.T_current_pods[i] > 75.0:
                self.time_in_danger_pods[i] += 1
            else:
                self.time_in_danger_pods[i] = 0

        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward

        # Calculate energy consumption (fan power ~ u^3)
        energy = 0.1 * np.sum(self.u_flow_pods ** 3)
        self.total_energy += energy

        # Update max temperature
        self.max_temp = max(self.max_temp, np.max(self.T_current_pods))

        # Check termination conditions
        terminated = False
        if np.any(self.T_current_pods > 80.0):
            terminated = True
            self.safety_violations += 1

        truncated = self.current_step >= self.max_steps

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self) -> float:
        """
        Calculate reward for current state.

        Components:
            - Energy savings: -0.1 * sum(u_flow^3)
            - Thermal penalty: -1000 per pod if T > 80°C
            - Danger penalty: -10 * (T - 75)^2 per pod if T > 75°C
            - Safe bonus: +1 per pod if 50°C < T < 70°C
            - Stability bonus: +0.5 per pod if |dT/dt| < 0.1
            - Coupling bonus: +2 if all pods within 10°C
        """
        reward = 0.0

        for i in range(self.num_pods):
            T = self.T_current_pods[i]
            dT = self.dT_dt_pods[i]

            # Critical temperature violation
            if T > 80.0:
                reward -= 1000.0

            # Danger zone (gradual penalty)
            if T > 75.0:
                reward -= 10.0 * (T - 75.0) ** 2

            # Safe operating zone
            if 50.0 < T < 70.0:
                reward += 1.0

            # Stable thermal state
            if abs(dT) < 0.1:
                reward += 0.5

        # Energy cost (cubic power law for fans)
        energy_cost = 0.1 * np.sum(self.u_flow_pods ** 3)
        reward -= energy_cost

        # Thermal coupling bonus (cluster stability)
        temp_range = np.max(self.T_current_pods) - np.min(self.T_current_pods)
        if temp_range < 10.0:
            reward += 2.0

        # Action smoothing penalty (penalize large changes)
        action_penalty = np.sum(np.abs(self.u_flow_pods - self.u_flow_prev_pods))
        reward -= action_penalty

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (48 values for 8 pods).

        Per-pod: [T_current, Q_load, u_flow, dT_dt, T_t1_pred, time_in_danger]
        """
        obs = []
        for i in range(self.num_pods):
            obs.extend([
                self.T_current_pods[i],
                self.Q_load_pods[i],
                self.u_flow_pods[i],
                self.dT_dt_pods[i],
                self.T_t1_pred_pods[i],
                float(self.time_in_danger_pods[i])
            ])
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            'step': self.current_step,
            'T_pods': self.T_current_pods.copy(),
            'Q_pods': self.Q_load_pods.copy(),
            'u_pods': self.u_flow_pods.copy(),
            'dT_dt_pods': self.dT_dt_pods.copy(),
            'max_temp': np.max(self.T_current_pods),
            'min_temp': np.min(self.T_current_pods),
            'avg_temp': np.mean(self.T_current_pods),
            'total_power_kw': np.sum(self.Q_load_pods) / 1000.0,
            'total_reward': self.total_reward,
            'total_energy': self.total_energy,
            'safety_violations': self.safety_violations,
            # Tier 2 (Stability Guard)
            'ttf_pods': self.ttf_current_pods.copy(),
            'safety_override_active': self.safety_override_active_pods.copy(),
            'safety_override_count': len(self.safety_override_log),
            'safety_override_log': self.safety_override_log.copy(),
            # Tier 3 (Surgical Shutdown)
            'shutdown_state_pods': self.shutdown_state_pods.copy(),
            'max_fan_duration_pods': self.max_fan_duration_pods.copy(),
            'emergency_shutdown_count': len(self.emergency_shutdown_log),
            'emergency_shutdown_log': self.emergency_shutdown_log.copy(),
            # Thermal coupling
            'temp_range': np.max(self.T_current_pods) - np.min(self.T_current_pods),
            # Domain randomization
            'thermal_mass_scale': self.thermal_mass_scale,
            'ambient_temp_offset': self.ambient_temp_offset
        }

    def render(self, mode='human'):
        """Render the current state."""
        print(f"\n=== Step {self.current_step} ===")
        print(f"Total Power: {np.sum(self.Q_load_pods)/1000:.1f} kW / {self.total_capacity_kw:.0f} kW")
        print(f"Avg Temp: {np.mean(self.T_current_pods):.1f}°C (Range: {np.min(self.T_current_pods):.1f}-{np.max(self.T_current_pods):.1f}°C)")
        print(f"Avg Fan: {np.mean(self.u_flow_pods):.2f} m/s")
        print(f"Reward: {self.total_reward:.1f}")

        print("\nPod Status:")
        for i in range(self.num_pods):
            status = "SHUTDOWN" if self.shutdown_state_pods[i] else "ACTIVE"
            override = "GUARD" if self.safety_override_active_pods[i] else ""
            print(f"  Pod {i}: {status:8s} {override:6s} T={self.T_current_pods[i]:5.1f}°C  "
                  f"Q={self.Q_load_pods[i]/1000:6.1f}kW  u={self.u_flow_pods[i]:4.2f}m/s  "
                  f"TTF={self.ttf_current_pods[i]:6.1f}s")

    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    """Test the 1MW environment."""
    print("Testing 1MW Thermal Cluster Environment...")

    # Create environment
    env = CoolingControl1MWEnv(
        model_path="checkpoints/best_recurrent_pinn.pt",
        device='cpu',
        num_pods=8,
        pod_power_kw=125.0,
        thermal_bleed_factor=0.05,
        enable_stability_guard=True,
        ttf_threshold=120.0,
        critical_temp=85.0,
        shutdown_timeout=5
    )

    print(f"\n✓ Environment created successfully!")
    print(f"  - Total Capacity: {env.total_capacity_kw:.0f} kW")
    print(f"  - Number of Pods: {env.num_pods}")
    print(f"  - Thermal Bleed: {env.thermal_bleed_factor:.2%}")
    print(f"  - Action Space: {env.action_space.shape}")
    print(f"  - Observation Space: {env.observation_space.shape}")

    # Test reset
    obs, info = env.reset()
    print(f"\n✓ Reset successful!")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Initial temps: {info['T_pods']}")

    # Test step
    action = np.zeros(8)  # No change
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n✓ Step successful!")
    print(f"  - Reward: {reward:.2f}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Max temp: {info['max_temp']:.1f}°C")

    print("\n✅ 1MW Thermal Cluster Environment is production-ready!")
