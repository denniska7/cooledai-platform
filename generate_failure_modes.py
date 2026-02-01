#!/usr/bin/env python3
"""
Step 2.3: Transient Failure Modeling - Thermal Runaway Simulation

This script simulates cooling loss events for data center racks and generates
thermal runaway curves for training a Physics-Informed Neural Network (PINN)
to predict Time-to-Failure.

Key Physics:
    dT/dt = Q / (m · c_p)

Where:
    - T: Temperature (°C)
    - t: Time (seconds)
    - Q: Heat generation rate (W)
    - m: Thermal mass (kg of air in room)
    - c_p: Specific heat capacity of air (1005 J/(kg·K))

Failure Scenarios:
    1. Complete Cooling Loss (instant CRAC failure)
    2. Partial Cooling Loss (reduced airflow)
    3. Gradual Degradation (fan bearing failure)
    4. Intermittent Cooling (compressor cycling)
    5. Hybrid Cooling Failure (liquid cooling loss)

Critical Temperature Thresholds:
    - 65°C: Warning threshold (start throttling)
    - 75°C: High temperature alert
    - 85°C: Critical threshold (emergency throttling)
    - 95°C: Emergency shutdown
    - 100°C: Hardware damage risk

Author: CoolingAI Simulator
Date: 2026-01-27
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import yaml
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Air properties (at ~25°C)
RHO_AIR = 1.184  # kg/m³ (density at 25°C)
C_P_AIR = 1005.0  # J/(kg·K) (specific heat)

# Data center room dimensions (from previous config)
ROOM_LENGTH = 10.0  # meters
ROOM_WIDTH = 8.0    # meters
ROOM_HEIGHT = 3.0   # meters
ROOM_VOLUME = ROOM_LENGTH * ROOM_WIDTH * ROOM_HEIGHT  # 240 m³

# Thermal mass (air in room)
THERMAL_MASS = RHO_AIR * ROOM_VOLUME  # kg

# Critical temperature thresholds (°C)
T_NOMINAL = 22.0      # Normal operating temperature
T_WARNING = 65.0      # Start throttling workload
T_HIGH = 75.0         # High temperature alert
T_CRITICAL = 85.0     # Emergency throttling
T_SHUTDOWN = 95.0     # Emergency shutdown
T_DAMAGE = 100.0      # Hardware damage risk

# Simulation parameters
SIMULATION_TIME = 60.0  # seconds
TIME_STEP = 0.1         # seconds
TIME_POINTS = np.arange(0, SIMULATION_TIME + TIME_STEP, TIME_STEP)

# Blackwell GB200 rack parameters
RACK_POWER_MAX = 120000.0  # 120 kW (air rejection component)
RACK_POWER_MIN = 10000.0   # 10 kW (idle)


# ============================================================================
# THERMAL RUNAWAY SIMULATOR
# ============================================================================

class ThermalRunawaySimulator:
    """
    Simulates thermal runaway events during cooling system failures.

    Uses the transient heat equation:
        dT/dt = Q / (m · c_p)

    Where the effective thermal mass includes both air mass and thermal
    inertia from equipment, racks, and building structure.
    """

    def __init__(
        self,
        room_volume: float = ROOM_VOLUME,
        thermal_mass_multiplier: float = 3.0,
        initial_temperature: float = T_NOMINAL
    ):
        """
        Initialize the thermal runaway simulator.

        Args:
            room_volume: Volume of data center room (m³)
            thermal_mass_multiplier: Factor to account for thermal inertia
                                    of equipment, racks, floor, etc.
                                    Typical value: 2-5x air mass
            initial_temperature: Starting temperature (°C)
        """
        self.room_volume = room_volume
        self.air_mass = RHO_AIR * room_volume

        # Effective thermal mass includes air + equipment thermal inertia
        self.thermal_mass = self.air_mass * thermal_mass_multiplier

        self.c_p = C_P_AIR
        self.T_initial = initial_temperature

        # Thermal capacity (J/K)
        self.thermal_capacity = self.thermal_mass * self.c_p

    def heat_balance_ode(
        self,
        T: float,
        t: float,
        Q_gen: callable,
        Q_removal: callable
    ) -> float:
        """
        Ordinary differential equation for heat balance.

        dT/dt = (Q_generation - Q_removal) / (m · c_p)

        Args:
            T: Current temperature (°C)
            t: Current time (s)
            Q_gen: Function that returns heat generation rate Q(t, T) in Watts
            Q_removal: Function that returns heat removal rate Q(t, T) in Watts

        Returns:
            dT/dt: Rate of temperature change (°C/s or K/s)
        """
        # Heat generation (from IT equipment)
        Q_generated = Q_gen(t, T)

        # Heat removal (from cooling system)
        Q_removed = Q_removal(t, T)

        # Net heat accumulation
        Q_net = Q_generated - Q_removed

        # Temperature rate of change
        dT_dt = Q_net / self.thermal_capacity

        return dT_dt

    def simulate_thermal_runaway(
        self,
        Q_it_load: float,
        cooling_failure_scenario: Dict,
        time_points: np.ndarray = TIME_POINTS
    ) -> Dict:
        """
        Simulate thermal runaway for a given failure scenario.

        Args:
            Q_it_load: IT equipment heat load (W)
            cooling_failure_scenario: Dictionary describing the failure:
                - 'type': 'instant', 'gradual', 'partial', 'intermittent', 'hybrid'
                - 'failure_time': When cooling loss occurs (s)
                - 'remaining_capacity': Fraction of cooling remaining (0-1)
                - 'degradation_rate': Rate of gradual failure (1/s)
            time_points: Array of time points for simulation (s)

        Returns:
            Dictionary with simulation results
        """
        failure_type = cooling_failure_scenario.get('type', 'instant')
        failure_time = cooling_failure_scenario.get('failure_time', 5.0)
        remaining_capacity = cooling_failure_scenario.get('remaining_capacity', 0.0)
        degradation_rate = cooling_failure_scenario.get('degradation_rate', 0.1)

        # Initial cooling capacity (assume cooling matches IT load initially)
        Q_cooling_nominal = Q_it_load

        # Define heat generation function (could include workload throttling)
        def Q_generation(t, T):
            """Heat generation from IT equipment with thermal throttling."""
            # Throttling starts at T_WARNING, full throttle at T_CRITICAL
            if T < T_WARNING:
                throttle_factor = 1.0
            elif T < T_CRITICAL:
                # Linear throttling from 100% to 50%
                throttle_factor = 1.0 - 0.5 * (T - T_WARNING) / (T_CRITICAL - T_WARNING)
            else:
                # Emergency throttle to 50%
                throttle_factor = 0.5

            return Q_it_load * throttle_factor

        # Define heat removal function based on failure scenario
        def Q_removal(t, T):
            """Heat removal from cooling system with failure dynamics."""
            if t < failure_time:
                # Normal operation: cooling matches load
                return Q_cooling_nominal

            elif failure_type == 'instant':
                # Complete instant failure
                return Q_cooling_nominal * remaining_capacity

            elif failure_type == 'gradual':
                # Gradual degradation (exponential decay)
                time_since_failure = t - failure_time
                decay = np.exp(-degradation_rate * time_since_failure)
                current_capacity = remaining_capacity + (1.0 - remaining_capacity) * decay
                return Q_cooling_nominal * current_capacity

            elif failure_type == 'partial':
                # Instant drop to partial capacity
                return Q_cooling_nominal * remaining_capacity

            elif failure_type == 'intermittent':
                # Cycling on/off (compressor issues)
                cycle_period = 10.0  # 10 second cycles
                phase = (t - failure_time) % cycle_period
                on_time = cycle_period * remaining_capacity
                if phase < on_time:
                    return Q_cooling_nominal
                else:
                    return 0.0

            elif failure_type == 'hybrid':
                # Hybrid cooling: liquid loss + air remains
                # Liquid cooling was handling 90%, now only air (10%)
                return Q_cooling_nominal * 0.1

            else:
                return Q_cooling_nominal * remaining_capacity

        # Solve ODE using scipy
        temperatures = odeint(
            self.heat_balance_ode,
            self.T_initial,
            time_points,
            args=(Q_generation, Q_removal)
        ).flatten()

        # Calculate instantaneous heating rate
        dT_dt = np.gradient(temperatures, time_points)

        # Calculate time-to-failure for each threshold
        time_to_warning = self._time_to_threshold(time_points, temperatures, T_WARNING)
        time_to_critical = self._time_to_threshold(time_points, temperatures, T_CRITICAL)
        time_to_shutdown = self._time_to_threshold(time_points, temperatures, T_SHUTDOWN)
        time_to_damage = self._time_to_threshold(time_points, temperatures, T_DAMAGE)

        # Calculate peak temperature and time
        peak_temp = np.max(temperatures)
        peak_time = time_points[np.argmax(temperatures)]

        # Calculate average heating rate in failure region
        failure_mask = time_points >= failure_time
        avg_heating_rate = np.mean(dT_dt[failure_mask]) if np.any(failure_mask) else 0.0

        return {
            'time': time_points,
            'temperature': temperatures,
            'dT_dt': dT_dt,
            'Q_it_load': Q_it_load,
            'failure_scenario': cooling_failure_scenario,
            'time_to_warning': time_to_warning,
            'time_to_critical': time_to_critical,
            'time_to_shutdown': time_to_shutdown,
            'time_to_damage': time_to_damage,
            'peak_temperature': peak_temp,
            'peak_time': peak_time,
            'avg_heating_rate': avg_heating_rate,
            'initial_temperature': self.T_initial,
            'thermal_mass': self.thermal_mass,
        }

    def _time_to_threshold(
        self,
        time_points: np.ndarray,
        temperatures: np.ndarray,
        threshold: float
    ) -> Optional[float]:
        """
        Calculate time to reach a temperature threshold.

        Returns:
            Time to threshold (s), or None if threshold never reached
        """
        exceeds = temperatures >= threshold
        if np.any(exceeds):
            idx = np.where(exceeds)[0][0]
            return time_points[idx]
        else:
            return None


# ============================================================================
# FAILURE SCENARIO GENERATOR
# ============================================================================

def generate_failure_scenarios() -> List[Dict]:
    """
    Generate a comprehensive set of cooling failure scenarios.

    Returns:
        List of failure scenario dictionaries
    """
    scenarios = []

    # 1. Complete Cooling Loss (instant CRAC failure)
    for failure_time in [2.0, 5.0, 10.0]:
        scenarios.append({
            'type': 'instant',
            'name': 'Complete CRAC Failure',
            'failure_time': failure_time,
            'remaining_capacity': 0.0,
            'description': f'Instant complete cooling loss at t={failure_time}s'
        })

    # 2. Partial Cooling Loss (reduced airflow)
    for remaining in [0.25, 0.5, 0.75]:
        scenarios.append({
            'type': 'partial',
            'name': f'Partial Cooling ({int(remaining*100)}%)',
            'failure_time': 5.0,
            'remaining_capacity': remaining,
            'description': f'Instant drop to {int(remaining*100)}% cooling capacity'
        })

    # 3. Gradual Degradation (fan bearing failure)
    for deg_rate in [0.05, 0.1, 0.2]:
        scenarios.append({
            'type': 'gradual',
            'name': f'Gradual Degradation (rate={deg_rate})',
            'failure_time': 5.0,
            'remaining_capacity': 0.1,
            'degradation_rate': deg_rate,
            'description': f'Exponential decay with rate {deg_rate}/s to 10%'
        })

    # 4. Intermittent Cooling (compressor cycling)
    for duty_cycle in [0.3, 0.5, 0.7]:
        scenarios.append({
            'type': 'intermittent',
            'name': f'Intermittent ({int(duty_cycle*100)}% duty)',
            'failure_time': 5.0,
            'remaining_capacity': duty_cycle,
            'description': f'Cycling on/off with {int(duty_cycle*100)}% duty cycle'
        })

    # 5. Hybrid Cooling Failure (liquid cooling loss, air remains)
    scenarios.append({
        'type': 'hybrid',
        'name': 'Liquid Cooling Loss (90% loss)',
        'failure_time': 5.0,
        'remaining_capacity': 0.1,  # Only 10% air cooling remains
        'description': 'Liquid cooling fails, only air cooling (10%) remains'
    })

    return scenarios


# ============================================================================
# DATASET GENERATOR
# ============================================================================

def generate_failure_dataset(
    num_scenarios: int = 500,
    save_path: str = 'data/processed/failure_modes_v1.csv',
    save_detailed: bool = True
) -> pd.DataFrame:
    """
    Generate comprehensive failure mode dataset for PINN training.

    Args:
        num_scenarios: Number of failure scenarios to simulate
        save_path: Path to save the summary dataset
        save_detailed: If True, save detailed time-series data

    Returns:
        DataFrame with failure mode data
    """
    print("=" * 70)
    print("THERMAL RUNAWAY SIMULATOR - Failure Mode Dataset Generation")
    print("=" * 70)
    print(f"\nData Center Configuration:")
    print(f"  Room volume: {ROOM_VOLUME:.1f} m³")
    print(f"  Thermal mass: {THERMAL_MASS:.1f} kg (air only)")
    print(f"  Effective thermal mass: {THERMAL_MASS * 3.0:.1f} kg (with equipment)")
    print(f"  Thermal capacity: {THERMAL_MASS * 3.0 * C_P_AIR / 1e6:.2f} MJ/K")
    print(f"\nCritical Temperature Thresholds:")
    print(f"  Warning: {T_WARNING}°C (throttling starts)")
    print(f"  Critical: {T_CRITICAL}°C (emergency throttling)")
    print(f"  Shutdown: {T_SHUTDOWN}°C (emergency shutdown)")
    print(f"  Damage: {T_DAMAGE}°C (hardware damage risk)")
    print(f"\nSimulation Parameters:")
    print(f"  Duration: {SIMULATION_TIME:.0f} seconds")
    print(f"  Time step: {TIME_STEP:.2f} seconds")
    print(f"  Target scenarios: {num_scenarios}")
    print()

    # Create simulator
    simulator = ThermalRunawaySimulator(
        room_volume=ROOM_VOLUME,
        thermal_mass_multiplier=3.0,  # Account for equipment thermal inertia
        initial_temperature=T_NOMINAL
    )

    # Get base failure scenarios
    base_scenarios = generate_failure_scenarios()
    print(f"Base failure scenario templates: {len(base_scenarios)}")

    # Generate IT load variations
    # Focus on high loads where failures are most critical
    Q_loads = np.concatenate([
        np.linspace(50000, 80000, num_scenarios // 4),    # 50-80 kW
        np.linspace(80000, 110000, num_scenarios // 4),   # 80-110 kW
        np.linspace(110000, 140000, num_scenarios // 4),  # 110-140 kW
        np.linspace(140000, 170000, num_scenarios // 4),  # 140-170 kW (10% air rejection from 1.7MW rack)
    ])
    np.random.shuffle(Q_loads)

    # Generate initial temperature variations
    T_initials = np.random.uniform(18.0, 30.0, num_scenarios)

    # Storage for results
    summary_data = []
    detailed_data = []

    print(f"\nGenerating {num_scenarios} thermal runaway scenarios...")
    print()

    for idx in tqdm(range(num_scenarios), desc="Simulating failures"):
        # Select random scenario and parameters
        base_scenario = base_scenarios[idx % len(base_scenarios)]
        Q_load = Q_loads[idx]
        T_init = T_initials[idx]

        # Update simulator initial temperature
        simulator.T_initial = T_init

        # Add some randomization to failure parameters
        scenario = base_scenario.copy()
        if scenario['type'] == 'gradual':
            scenario['degradation_rate'] *= np.random.uniform(0.8, 1.2)
        elif scenario['type'] == 'partial':
            scenario['remaining_capacity'] *= np.random.uniform(0.9, 1.1)
            scenario['remaining_capacity'] = np.clip(scenario['remaining_capacity'], 0.0, 1.0)

        # Run simulation
        result = simulator.simulate_thermal_runaway(
            Q_it_load=Q_load,
            cooling_failure_scenario=scenario,
            time_points=TIME_POINTS
        )

        # Store summary data (one row per scenario)
        summary_row = {
            'scenario_id': idx,
            'failure_type': scenario['type'],
            'failure_name': scenario['name'],
            'Q_it_load_W': Q_load,
            'Q_it_load_kW': Q_load / 1000.0,
            'initial_temperature_C': T_init,
            'failure_time_s': scenario['failure_time'],
            'remaining_capacity': scenario.get('remaining_capacity', 0.0),
            'degradation_rate': scenario.get('degradation_rate', 0.0),
            'peak_temperature_C': result['peak_temperature'],
            'peak_time_s': result['peak_time'],
            'avg_heating_rate_C_per_s': result['avg_heating_rate'],
            'time_to_warning_s': result['time_to_warning'] if result['time_to_warning'] else -1,
            'time_to_critical_s': result['time_to_critical'] if result['time_to_critical'] else -1,
            'time_to_shutdown_s': result['time_to_shutdown'] if result['time_to_shutdown'] else -1,
            'time_to_damage_s': result['time_to_damage'] if result['time_to_damage'] else -1,
            'thermal_mass_kg': simulator.thermal_mass,
            'thermal_capacity_J_per_K': simulator.thermal_capacity,
        }
        summary_data.append(summary_row)

        # Store detailed time-series data (multiple rows per scenario)
        if save_detailed:
            for t_idx, (t, T, dTdt) in enumerate(zip(
                result['time'],
                result['temperature'],
                result['dT_dt']
            )):
                detailed_row = {
                    'scenario_id': idx,
                    'time_s': t,
                    'temperature_C': T,
                    'dT_dt_C_per_s': dTdt,
                    'Q_it_load_W': Q_load,
                    'failure_type': scenario['type'],
                    'is_after_failure': 1 if t >= scenario['failure_time'] else 0,
                }
                detailed_data.append(detailed_row)

    # Create DataFrames
    df_summary = pd.DataFrame(summary_data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save summary dataset
    df_summary.to_csv(save_path, index=False)
    print(f"\n✓ Saved summary dataset: {save_path}")
    print(f"  Size: {os.path.getsize(save_path) / 1024:.1f} KB")
    print(f"  Rows: {len(df_summary)}")
    print(f"  Columns: {len(df_summary.columns)}")

    # Save detailed time-series dataset
    if save_detailed and detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        detailed_path = save_path.replace('.csv', '_detailed.csv')
        df_detailed.to_csv(detailed_path, index=False)
        print(f"\n✓ Saved detailed time-series dataset: {detailed_path}")
        print(f"  Size: {os.path.getsize(detailed_path) / 1024:.1f} KB")
        print(f"  Rows: {len(df_detailed)}")
        print(f"  Columns: {len(df_detailed.columns)}")

    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    print("\nFailure Scenario Distribution:")
    print(df_summary['failure_type'].value_counts().to_string())

    print("\n\nIT Load Statistics:")
    print(f"  Min: {df_summary['Q_it_load_kW'].min():.1f} kW")
    print(f"  Max: {df_summary['Q_it_load_kW'].max():.1f} kW")
    print(f"  Mean: {df_summary['Q_it_load_kW'].mean():.1f} kW")
    print(f"  Std: {df_summary['Q_it_load_kW'].std():.1f} kW")

    print("\nPeak Temperature Statistics:")
    print(f"  Min: {df_summary['peak_temperature_C'].min():.1f} °C")
    print(f"  Max: {df_summary['peak_temperature_C'].max():.1f} °C")
    print(f"  Mean: {df_summary['peak_temperature_C'].mean():.1f} °C")
    print(f"  Std: {df_summary['peak_temperature_C'].std():.1f} °C")

    print("\nTime-to-Critical Statistics (when reached):")
    critical_reached = df_summary[df_summary['time_to_critical_s'] > 0]
    if len(critical_reached) > 0:
        print(f"  Scenarios reaching critical: {len(critical_reached)} ({len(critical_reached)/len(df_summary)*100:.1f}%)")
        print(f"  Min time: {critical_reached['time_to_critical_s'].min():.1f} s")
        print(f"  Max time: {critical_reached['time_to_critical_s'].max():.1f} s")
        print(f"  Mean time: {critical_reached['time_to_critical_s'].mean():.1f} s")
    else:
        print("  No scenarios reached critical threshold")

    print("\nTime-to-Shutdown Statistics (when reached):")
    shutdown_reached = df_summary[df_summary['time_to_shutdown_s'] > 0]
    if len(shutdown_reached) > 0:
        print(f"  Scenarios reaching shutdown: {len(shutdown_reached)} ({len(shutdown_reached)/len(df_summary)*100:.1f}%)")
        print(f"  Min time: {shutdown_reached['time_to_shutdown_s'].min():.1f} s")
        print(f"  Max time: {shutdown_reached['time_to_shutdown_s'].max():.1f} s")
        print(f"  Mean time: {shutdown_reached['time_to_shutdown_s'].mean():.1f} s")
    else:
        print("  No scenarios reached shutdown threshold")

    print("\nAverage Heating Rate Statistics:")
    print(f"  Min: {df_summary['avg_heating_rate_C_per_s'].min():.4f} °C/s")
    print(f"  Max: {df_summary['avg_heating_rate_C_per_s'].max():.4f} °C/s")
    print(f"  Mean: {df_summary['avg_heating_rate_C_per_s'].mean():.4f} °C/s")

    print("\n" + "=" * 70)
    print("PHYSICS VALIDATION")
    print("=" * 70)

    # Validate physics: check if heating rates match theoretical prediction
    # For complete cooling loss: dT/dt = Q / (m·c_p)
    instant_failures = df_summary[df_summary['failure_type'] == 'instant']
    if len(instant_failures) > 0:
        sample = instant_failures.iloc[0]
        theoretical_rate = sample['Q_it_load_W'] / simulator.thermal_capacity
        observed_rate = sample['avg_heating_rate_C_per_s']
        error = abs(theoretical_rate - observed_rate) / theoretical_rate * 100

        print(f"\nTheoretical vs Observed Heating Rate (complete failure):")
        print(f"  Theoretical: {theoretical_rate:.4f} °C/s")
        print(f"  Observed: {observed_rate:.4f} °C/s")
        print(f"  Error: {error:.2f}%")
        if error < 10:
            print("  ✓ Physics validation PASSED")
        else:
            print("  ✗ Physics validation FAILED")

    print("\n" + "=" * 70)
    print("✓ Thermal runaway dataset generation complete!")
    print("=" * 70)

    print("\nDataset Files:")
    print(f"  Summary: {save_path}")
    if save_detailed:
        print(f"  Detailed: {save_path.replace('.csv', '_detailed.csv')}")

    print("\nNext Steps:")
    print("  1. Inspect the data: pandas.read_csv('{}')".format(save_path))
    print("  2. Train PINN to predict temperature trajectories")
    print("  3. Build Time-to-Failure prediction model")
    print("  4. Integrate with real-time monitoring for failure prediction")
    print("  5. Deploy as early warning system")

    return df_summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Generate comprehensive failure mode dataset
    df = generate_failure_dataset(
        num_scenarios=500,
        save_path='data/processed/failure_modes_v1.csv',
        save_detailed=True
    )

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print("=" * 70)
    print("\nYou now have:")
    print("  • 500 failure scenarios with Time-to-Failure data")
    print("  • Complete thermal runaway curves")
    print("  • Multiple failure modes (instant, gradual, partial, etc.)")
    print("  • Ready for PINN training!")
    print("\nThis data will enable the most robust failure-prediction engine in the market! 🚀")
