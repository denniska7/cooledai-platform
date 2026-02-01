#!/usr/bin/env python3
"""
Train Reinforcement Learning Agent for Autonomous Cooling Control

Uses Proximal Policy Optimization (PPO) to train an agent that:
1. Minimizes energy consumption (lower fan speeds)
2. Maintains safe temperatures (< 75°C optimal, < 80°C critical)
3. Provides smooth, stable control

The trained RecurrentPINN serves as the physics engine, enabling
sample-efficient training without real hardware testing.

Author: Claude (Anthropic)
Date: 2026-01-28
"""

import numpy as np
import torch
import sys
import os
from typing import Dict
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️  Stable-Baselines3 not installed. Will provide installation instructions.")
    SB3_AVAILABLE = False

from cooling_rl_env import CoolingControlEnv


class ThermalSafetyCallback(BaseCallback):
    """
    Custom callback to monitor thermal safety during training.

    Logs:
    - Average temperature
    - Max temperature
    - Safety violations
    - Energy consumption
    - Episode rewards
    """

    def __init__(self, verbose=0):
        super(ThermalSafetyCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_temps = []
        self.episode_violations = []
        self.episode_energies = []

    def _on_step(self) -> bool:
        # Check if episode just ended
        if self.locals.get('dones') is not None and self.locals['dones'][0]:
            # Get episode info
            info = self.locals['infos'][0]

            self.episode_rewards.append(info.get('total_reward', 0))
            self.episode_lengths.append(info.get('step', 0))
            self.episode_temps.append(info.get('max_temp', 0))
            self.episode_violations.append(info.get('safety_violations', 0))
            self.episode_energies.append(info.get('total_energy', 0))

            # Log every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_temp = np.mean(self.episode_temps[-10:])
                avg_violations = np.mean(self.episode_violations[-10:])
                avg_energy = np.mean(self.episode_energies[-10:])

                print(f"\n📊 Episode {len(self.episode_rewards)}:")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Max Temp: {avg_temp:.2f}°C")
                print(f"  Avg Violations: {avg_violations:.2f}")
                print(f"  Avg Energy: {avg_energy:.2f}")

        return True

    def _on_training_end(self) -> None:
        """Save training statistics."""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_temps': self.episode_temps,
            'episode_violations': self.episode_violations,
            'episode_energies': self.episode_energies,
            'total_episodes': len(self.episode_rewards)
        }

        # Save statistics
        stats_path = 'optimizer/training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Training statistics saved to {stats_path}")


def train_ppo_agent(
    model_path: str = "checkpoints/best_recurrent_pinn.pt",
    device: str = 'cpu',
    total_timesteps: int = 100000,
    save_path: str = "optimizer/ppo_cooling_agent"
):
    """
    Train PPO agent for cooling control.

    Args:
        model_path: Path to trained RecurrentPINN checkpoint
        device: Device for inference ('cpu', 'cuda', 'mps')
        total_timesteps: Total training timesteps
        save_path: Path to save trained agent
    """
    print("=" * 70)
    print("TRAINING AUTONOMOUS COOLING CONTROL AGENT")
    print("=" * 70)

    # Create environment
    print(f"\n📦 Creating environment...")
    env = CoolingControlEnv(
        model_path=model_path,
        device=device,
        max_steps=1000,  # ~16 minutes per episode
        # Phase 4.2: Enable domain randomization for robust training
        enable_domain_randomization=True,
        thermal_mass_variation=0.2,  # ±20% thermal mass variation
        ambient_temp_variation=5.0,   # ±5°C ambient temperature variation
        action_smoothing_penalty=1.0  # Penalty weight for large actions
    )

    # Check environment validity
    print("✓ Checking environment validity...")
    check_env(env, warn=True)

    # Wrap environment for monitoring
    env = Monitor(env)

    print("\n✓ Environment created successfully!")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Create PPO agent
    print(f"\n🤖 Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log=None  # Disabled (tensorboard not installed)
    )

    print("✓ PPO agent initialized!")
    print(f"  Policy network: MLP (Multi-Layer Perceptron)")
    print(f"  Learning rate: 3e-4")
    print(f"  Total timesteps: {total_timesteps:,}")

    # Create callbacks
    thermal_callback = ThermalSafetyCallback(verbose=1)

    # Train agent
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Training for {total_timesteps:,} timesteps...")
    print("This will take approximately 30-60 minutes depending on hardware.\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=thermal_callback,
            progress_bar=False  # Disabled (tqdm/rich compatibility)
        )

        # Save trained model
        print(f"\n💾 Saving trained agent to {save_path}...")
        model.save(save_path)
        print("✓ Agent saved successfully!")

        # Final evaluation
        print("\n" + "=" * 70)
        print("FINAL EVALUATION")
        print("=" * 70)

        evaluate_agent(model, env, n_episodes=10)

        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\nTrained agent saved to: {save_path}.zip")
        print(f"Training stats saved to: optimizer/training_stats.json")
        print("\nNext steps:")
        print("  1. Run 'python3.11 optimizer/deploy_control_policy.py' to test the agent")
        print("  2. Use the agent for real-time cooling control")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print(f"Saving current model to {save_path}_interrupted.zip...")
        model.save(f"{save_path}_interrupted")
        print("✓ Model saved. You can resume training later.")

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()


def evaluate_agent(model, env, n_episodes: int = 10):
    """
    Evaluate trained agent performance.

    Args:
        model: Trained PPO model
        env: Environment
        n_episodes: Number of evaluation episodes
    """
    print(f"\nEvaluating agent over {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []
    max_temps = []
    safety_violations = []
    total_energies = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        max_temps.append(info['max_temp'])
        safety_violations.append(info['safety_violations'])
        total_energies.append(info['total_energy'])

        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"MaxTemp={info['max_temp']:.2f}°C, "
              f"Violations={info['safety_violations']}")

    # Print summary statistics
    print("\n📊 Evaluation Summary:")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Length: {np.mean(episode_lengths):.0f} steps")
    print(f"  Average Max Temp: {np.mean(max_temps):.2f}°C ± {np.std(max_temps):.2f}")
    print(f"  Average Violations: {np.mean(safety_violations):.2f}")
    print(f"  Average Energy: {np.mean(total_energies):.2f}")
    print(f"  Success Rate: {100 * (1 - np.mean(safety_violations) / n_episodes):.1f}%")


if __name__ == "__main__":
    if not SB3_AVAILABLE:
        print("\n" + "=" * 70)
        print("INSTALLATION REQUIRED: Stable-Baselines3")
        print("=" * 70)
        print("\nStable-Baselines3 is not installed. Install it with:")
        print("\n  pip install stable-baselines3[extra]")
        print("\nOr if you prefer conda:")
        print("\n  conda install -c conda-forge stable-baselines3")
        print("\nAfter installation, run this script again.")
        sys.exit(1)

    # Check if model exists
    model_path = "checkpoints/best_recurrent_pinn.pt"
    if not os.path.exists(model_path):
        print(f"\n❌ Error: RecurrentPINN model not found at {model_path}")
        print("   Please train the RecurrentPINN first by running:")
        print("   python3.11 train_recurrent_pinn.py")
        sys.exit(1)

    # Determine device
    # Note: Using CPU for RL training due to Stable-Baselines3 compatibility
    # (SB3 uses float64, which MPS doesn't support)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'  # Force CPU for RL training

    print(f"\n🖥️  Using device: {device.upper()}")
    print("   (RL training uses CPU for stability; RecurrentPINN uses MPS)")

    # Train agent
    train_ppo_agent(
        model_path=model_path,
        device='cpu',  # RecurrentPINN inference will use CPU for compatibility
        total_timesteps=100000,  # 100k timesteps (~2-3 hours training)
        save_path="optimizer/ppo_cooling_agent"
    )
