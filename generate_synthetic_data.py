"""
Synthetic Data Generator for Data Center Thermal Analysis

This script generates 1,000 synthetic data points by varying:
- IT Load (Q): 10 kW to 150 kW
- Fan Velocity (u): 0.5 m/s to 3.0 m/s

And computing the resulting steady-state temperature distribution.

Output: data/synthetic_thermal_v1.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import yaml
from tqdm import tqdm


class DataCenterThermalModel:
    """
    Simplified analytical model for steady-state temperature in a data center.

    Uses energy balance and convection-diffusion principles.
    """

    def __init__(self, config: Dict):
        """
        Initialize thermal model.

        Args:
            config: Configuration dictionary with geometry and thermal properties
        """
        # Geometry
        self.length = config['geometry']['length']  # m
        self.width = config['geometry']['width']    # m
        self.height = config['geometry']['height']  # m
        self.volume = self.length * self.width * self.height  # m³
        self.floor_area = self.length * self.width  # m²

        # Thermal properties
        self.rho = config['thermal']['density']  # kg/m³
        self.c_p = config['thermal']['specific_heat']  # J/(kg·K)
        self.k = config['thermal']['thermal_conductivity']  # W/(m·K)

        # Boundary conditions
        self.T_inlet = config['boundaries']['inlet_temp']  # °C
        self.T_initial = config['boundaries']['initial_temp']  # °C

    def compute_mass_flow_rate(self, velocity: float) -> float:
        """
        Compute air mass flow rate from velocity.

        Args:
            velocity: Air velocity [m/s]

        Returns:
            Mass flow rate [kg/s]
        """
        # Cross-sectional area (inlet face)
        area = self.width * self.height  # m²

        # Volumetric flow rate
        volumetric_flow = velocity * area  # m³/s

        # Mass flow rate
        mass_flow = self.rho * volumetric_flow  # kg/s

        return mass_flow

    def compute_steady_state_temperature(
        self,
        Q_total: float,
        velocity: float,
        position: Tuple[float, float, float] = None
    ) -> float:
        """
        Compute steady-state temperature at a point using energy balance.

        For steady-state: Heat input = Heat removed by convection

        Args:
            Q_total: Total IT load [W]
            velocity: Air velocity [m/s]
            position: (x, y, z) position in room [m] (optional)

        Returns:
            Temperature [°C]
        """
        # Mass flow rate
        m_dot = self.compute_mass_flow_rate(velocity)

        # Heat capacity rate
        C_dot = m_dot * self.c_p  # W/K

        # Temperature rise from energy balance
        # Q = m_dot * c_p * (T_out - T_in)
        # T_out = T_in + Q / (m_dot * c_p)
        if C_dot > 0:
            delta_T = Q_total / C_dot
        else:
            delta_T = 0.0

        # If position is given, compute spatial variation
        if position is not None:
            x, y, z = position

            # Normalized position along flow direction
            x_norm = x / self.length  # 0 at inlet, 1 at outlet

            # Temperature increases linearly along flow direction
            T = self.T_inlet + delta_T * x_norm

            # Add vertical stratification (hot air rises)
            z_norm = z / self.height
            stratification = 0.5 * delta_T * z_norm  # Up to 50% increase near ceiling

            # Add lateral mixing effect (temperature smoothing)
            y_norm = abs(y - self.width/2) / (self.width/2)
            lateral_effect = -0.1 * delta_T * y_norm  # Slightly cooler at sides

            T = T + stratification + lateral_effect

        else:
            # Return average outlet temperature
            T = self.T_inlet + delta_T

        return T

    def compute_temperature_field(
        self,
        Q_total: float,
        velocity: float,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute temperature at multiple points in the room.

        Args:
            Q_total: Total IT load [W]
            velocity: Air velocity [m/s]
            n_points: Number of points to sample

        Returns:
            x, y, z, T arrays
        """
        # Random sampling
        x = np.random.uniform(0, self.length, n_points)
        y = np.random.uniform(0, self.width, n_points)
        z = np.random.uniform(0, self.height, n_points)

        T = np.zeros(n_points)
        for i in range(n_points):
            T[i] = self.compute_steady_state_temperature(
                Q_total, velocity, position=(x[i], y[i], z[i])
            )

        return x, y, z, T

    def compute_average_temperature(
        self,
        Q_total: float,
        velocity: float
    ) -> float:
        """
        Compute volume-averaged temperature.

        Args:
            Q_total: Total IT load [W]
            velocity: Air velocity [m/s]

        Returns:
            Average temperature [°C]
        """
        # Sample many points
        x, y, z, T = self.compute_temperature_field(Q_total, velocity, n_points=1000)

        return np.mean(T)

    def compute_max_temperature(
        self,
        Q_total: float,
        velocity: float
    ) -> float:
        """
        Compute maximum temperature (hot spot).

        Args:
            Q_total: Total IT load [W]
            velocity: Air velocity [m/s]

        Returns:
            Maximum temperature [°C]
        """
        # Hot spot is typically at outlet, near ceiling
        # x = length (outlet), y = center, z = height (ceiling)
        T_max = self.compute_steady_state_temperature(
            Q_total, velocity,
            position=(self.length * 0.95, self.width/2, self.height * 0.9)
        )

        return T_max

    def compute_outlet_temperature(
        self,
        Q_total: float,
        velocity: float
    ) -> float:
        """
        Compute outlet temperature (hot aisle).

        Args:
            Q_total: Total IT load [W]
            velocity: Air velocity [m/s]

        Returns:
            Outlet temperature [°C]
        """
        return self.compute_steady_state_temperature(Q_total, velocity, position=None)


def generate_synthetic_dataset(
    config_path: str,
    n_samples: int = 1000,
    Q_range: Tuple[float, float] = (10000, 150000),  # W
    velocity_range: Tuple[float, float] = (0.5, 3.0),  # m/s
    noise_std: float = 0.5,  # °C
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic thermal data by varying IT load and fan velocity.

    Args:
        config_path: Path to physics configuration file
        n_samples: Number of data points to generate
        Q_range: (Q_min, Q_max) IT load range [W]
        velocity_range: (u_min, u_max) velocity range [m/s]
        noise_std: Standard deviation of measurement noise [°C]
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(random_seed)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize thermal model
    model = DataCenterThermalModel(config)

    print(f"Data Center Configuration:")
    print(f"  Dimensions: {model.length}m × {model.width}m × {model.height}m")
    print(f"  Volume: {model.volume:.1f} m³")
    print(f"  Floor area: {model.floor_area:.1f} m²")
    print(f"  Inlet temperature: {model.T_inlet}°C")

    print(f"\nGenerating {n_samples} synthetic data points...")
    print(f"  IT Load range: {Q_range[0]/1000:.1f} - {Q_range[1]/1000:.1f} kW")
    print(f"  Fan velocity range: {velocity_range[0]:.1f} - {velocity_range[1]:.1f} m/s")

    # Generate samples
    data = []

    # Use Latin Hypercube Sampling for better coverage
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2, seed=random_seed)
    samples = sampler.random(n=n_samples)

    # Scale to desired ranges
    Q_samples = samples[:, 0] * (Q_range[1] - Q_range[0]) + Q_range[0]
    velocity_samples = samples[:, 1] * (velocity_range[1] - velocity_range[0]) + velocity_range[0]

    for i in tqdm(range(n_samples), desc="Generating samples"):
        Q_total = Q_samples[i]
        velocity = velocity_samples[i]

        # Compute temperatures at different locations
        # 1. Average room temperature
        T_avg = model.compute_average_temperature(Q_total, velocity)

        # 2. Maximum temperature (hot spot)
        T_max = model.compute_max_temperature(Q_total, velocity)

        # 3. Outlet temperature
        T_outlet = model.compute_outlet_temperature(Q_total, velocity)

        # 4. Inlet temperature (known)
        T_inlet = model.T_inlet

        # 5. Sample a random point in the room
        x_sample = np.random.uniform(0, model.length)
        y_sample = np.random.uniform(0, model.width)
        z_sample = np.random.uniform(0, model.height)
        T_sample = model.compute_steady_state_temperature(
            Q_total, velocity,
            position=(x_sample, y_sample, z_sample)
        )

        # Add realistic measurement noise
        noise = np.random.normal(0, noise_std)
        T_avg_noisy = T_avg + noise
        T_max_noisy = T_max + np.abs(noise)  # Max is always >= avg
        T_outlet_noisy = T_outlet + noise
        T_sample_noisy = T_sample + noise

        # Compute derived quantities
        mass_flow = model.compute_mass_flow_rate(velocity)
        delta_T = T_outlet - T_inlet

        # Power calculation check
        Q_computed = mass_flow * model.c_p * delta_T
        efficiency = min(Q_computed / Q_total, 1.0) if Q_total > 0 else 0

        # Store data point
        data.append({
            # Input parameters
            'Q_total_W': Q_total,
            'Q_total_kW': Q_total / 1000,
            'velocity_m_s': velocity,
            'mass_flow_kg_s': mass_flow,
            'volumetric_flow_m3_s': mass_flow / model.rho,

            # Output temperatures
            'T_inlet_C': T_inlet,
            'T_outlet_C': T_outlet_noisy,
            'T_avg_C': T_avg_noisy,
            'T_max_C': T_max_noisy,
            'T_sample_C': T_sample_noisy,

            # Sample position
            'x_sample_m': x_sample,
            'y_sample_m': y_sample,
            'z_sample_m': z_sample,

            # Derived quantities
            'delta_T_C': delta_T,
            'Q_computed_W': Q_computed,
            'cooling_efficiency': efficiency,

            # Dimensionless numbers
            'reynolds_number': velocity * model.length / 1.5e-5,
            'peclet_number': velocity * model.length / 2.2e-5,
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    print(f"\n✓ Generated {len(df)} data points")

    return df


def analyze_dataset(df: pd.DataFrame):
    """Print statistics about the generated dataset."""
    print("\n" + "=" * 70)
    print("SYNTHETIC DATASET STATISTICS")
    print("=" * 70)

    print("\nInput Parameters:")
    print(f"  IT Load (Q):")
    print(f"    Min: {df['Q_total_kW'].min():.1f} kW")
    print(f"    Max: {df['Q_total_kW'].max():.1f} kW")
    print(f"    Mean: {df['Q_total_kW'].mean():.1f} kW")
    print(f"    Std: {df['Q_total_kW'].std():.1f} kW")

    print(f"\n  Fan Velocity (u):")
    print(f"    Min: {df['velocity_m_s'].min():.2f} m/s")
    print(f"    Max: {df['velocity_m_s'].max():.2f} m/s")
    print(f"    Mean: {df['velocity_m_s'].mean():.2f} m/s")
    print(f"    Std: {df['velocity_m_s'].std():.2f} m/s")

    print("\nOutput Temperatures:")
    print(f"  Inlet Temperature:")
    print(f"    Constant: {df['T_inlet_C'].iloc[0]:.1f} °C")

    print(f"\n  Outlet Temperature:")
    print(f"    Min: {df['T_outlet_C'].min():.1f} °C")
    print(f"    Max: {df['T_outlet_C'].max():.1f} °C")
    print(f"    Mean: {df['T_outlet_C'].mean():.1f} °C")

    print(f"\n  Average Room Temperature:")
    print(f"    Min: {df['T_avg_C'].min():.1f} °C")
    print(f"    Max: {df['T_avg_C'].max():.1f} °C")
    print(f"    Mean: {df['T_avg_C'].mean():.1f} °C")

    print(f"\n  Maximum Temperature (Hot Spot):")
    print(f"    Min: {df['T_max_C'].min():.1f} °C")
    print(f"    Max: {df['T_max_C'].max():.1f} °C")
    print(f"    Mean: {df['T_max_C'].mean():.1f} °C")

    print("\nDerived Quantities:")
    print(f"  Temperature Rise (ΔT):")
    print(f"    Min: {df['delta_T_C'].min():.1f} °C")
    print(f"    Max: {df['delta_T_C'].max():.1f} °C")
    print(f"    Mean: {df['delta_T_C'].mean():.1f} °C")

    print(f"\n  Cooling Efficiency:")
    print(f"    Min: {df['cooling_efficiency'].min():.2%}")
    print(f"    Max: {df['cooling_efficiency'].max():.2%}")
    print(f"    Mean: {df['cooling_efficiency'].mean():.2%}")

    print(f"\n  Péclet Number (Convection/Diffusion):")
    print(f"    Min: {df['peclet_number'].min():.0f}")
    print(f"    Max: {df['peclet_number'].max():.0f}")
    print(f"    Mean: {df['peclet_number'].mean():.0f}")

    print("\nPhysical Validation:")
    # Check that temperatures are in reasonable range
    temp_reasonable = (
        (df['T_avg_C'] >= 18) & (df['T_avg_C'] <= 60) &
        (df['T_max_C'] >= 18) & (df['T_max_C'] <= 80)
    ).all()
    print(f"  Temperature in reasonable range: {'✓' if temp_reasonable else '✗'}")

    # Check that hot spot is hotter than average
    hotspot_valid = (df['T_max_C'] >= df['T_avg_C']).all()
    print(f"  Hot spot ≥ average temp: {'✓' if hotspot_valid else '✗'}")

    # Check that outlet is hotter than inlet
    outlet_valid = (df['T_outlet_C'] >= df['T_inlet_C']).all()
    print(f"  Outlet ≥ inlet temp: {'✓' if outlet_valid else '✗'}")

    # Check energy balance
    energy_balance_error = np.abs(df['Q_computed_W'] - df['Q_total_W']) / df['Q_total_W']
    print(f"  Energy balance error: {energy_balance_error.mean():.2%} (should be small)")


def visualize_dataset(df: pd.DataFrame, save_dir: Path):
    """Create visualizations of the synthetic dataset."""
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. IT Load vs Temperature
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Outlet temperature vs IT load (colored by velocity)
    scatter = axes[0, 0].scatter(
        df['Q_total_kW'], df['T_outlet_C'],
        c=df['velocity_m_s'], cmap='coolwarm', alpha=0.6, s=20
    )
    axes[0, 0].set_xlabel('IT Load (kW)')
    axes[0, 0].set_ylabel('Outlet Temperature (°C)')
    axes[0, 0].set_title('Outlet Temperature vs IT Load')
    plt.colorbar(scatter, ax=axes[0, 0], label='Velocity (m/s)')
    axes[0, 0].grid(True, alpha=0.3)

    # Average temperature vs IT load
    scatter = axes[0, 1].scatter(
        df['Q_total_kW'], df['T_avg_C'],
        c=df['velocity_m_s'], cmap='coolwarm', alpha=0.6, s=20
    )
    axes[0, 1].set_xlabel('IT Load (kW)')
    axes[0, 1].set_ylabel('Average Temperature (°C)')
    axes[0, 1].set_title('Average Temperature vs IT Load')
    plt.colorbar(scatter, ax=axes[0, 1], label='Velocity (m/s)')
    axes[0, 1].grid(True, alpha=0.3)

    # Temperature rise vs velocity (colored by IT load)
    scatter = axes[1, 0].scatter(
        df['velocity_m_s'], df['delta_T_C'],
        c=df['Q_total_kW'], cmap='viridis', alpha=0.6, s=20
    )
    axes[1, 0].set_xlabel('Fan Velocity (m/s)')
    axes[1, 0].set_ylabel('Temperature Rise ΔT (°C)')
    axes[1, 0].set_title('Temperature Rise vs Fan Velocity')
    plt.colorbar(scatter, ax=axes[1, 0], label='IT Load (kW)')
    axes[1, 0].grid(True, alpha=0.3)

    # Hot spot temperature distribution
    axes[1, 1].hist(df['T_max_C'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Maximum Temperature (°C)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Hot Spot Temperature Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'synthetic_data_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_dir / 'synthetic_data_analysis.png'}")
    plt.close()


def main():
    """Main data generation pipeline."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATOR - Data Center Thermal Analysis")
    print("=" * 70)

    # Paths
    project_root = Path(__file__).parent
    config_path = project_root / 'configs' / 'physics_config.yaml'
    output_dir = project_root / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'synthetic_thermal_v1.csv'
    viz_dir = project_root / 'results'

    # Generate synthetic data
    df = generate_synthetic_dataset(
        config_path=str(config_path),
        n_samples=1000,
        Q_range=(10000, 150000),  # 10 kW to 150 kW
        velocity_range=(0.5, 3.0),  # 0.5 to 3.0 m/s
        noise_std=0.5,  # 0.5°C measurement noise
        random_seed=42
    )

    # Analyze dataset
    analyze_dataset(df)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved dataset: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Create visualizations
    print("\nGenerating visualizations...")
    try:
        visualize_dataset(df, viz_dir)
    except ImportError:
        print("⚠ matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"⚠ Could not create visualizations: {e}")

    # Print column info
    print("\n" + "=" * 70)
    print("DATASET COLUMNS")
    print("=" * 70)
    print("\nInput Variables (controlled):")
    print("  - Q_total_W, Q_total_kW: IT load")
    print("  - velocity_m_s: Fan/air velocity")
    print("  - mass_flow_kg_s: Air mass flow rate")

    print("\nOutput Variables (measured):")
    print("  - T_inlet_C: Inlet temperature (cold aisle)")
    print("  - T_outlet_C: Outlet temperature (hot aisle)")
    print("  - T_avg_C: Average room temperature")
    print("  - T_max_C: Maximum temperature (hot spot)")
    print("  - T_sample_C: Temperature at random point")

    print("\nSpatial Information:")
    print("  - x_sample_m, y_sample_m, z_sample_m: Random sample location")

    print("\nDerived Quantities:")
    print("  - delta_T_C: Temperature rise")
    print("  - cooling_efficiency: Heat removal efficiency")
    print("  - reynolds_number, peclet_number: Dimensionless numbers")

    print("\n" + "=" * 70)
    print("✓ Data generation complete!")
    print("=" * 70)

    print(f"\nNext steps:")
    print(f"  1. Inspect the data: pandas.read_csv('{output_file}')")
    print(f"  2. Use for PINN training")
    print(f"  3. Validate PINN predictions against this synthetic data")
    print(f"  4. Explore relationships between Q, u, and T")

    return df


if __name__ == "__main__":
    main()
