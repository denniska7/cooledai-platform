"""
Hardware profiling for modern data center equipment

This module provides classes and functions for:
- GPU and server specifications (NVIDIA Blackwell GB200)
- Hybrid cooling models (90% liquid, 10% air)
- Power and thermal calculations
- Rack-level heat generation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml
import numpy as np
from pathlib import Path


@dataclass
class GPUSpecifications:
    """Specifications for a GPU accelerator."""
    model: str
    architecture: str
    power_per_gpu: float  # Watts
    thermal_design_power: float  # Watts
    max_junction_temp: float  # °C
    idle_power: float  # Watts


@dataclass
class LiquidCoolingConfig:
    """Liquid cooling configuration."""
    enabled: bool
    heat_capture_percentage: float  # 0-100
    components: List[str]
    coolant_type: str
    inlet_temp: float  # °C
    outlet_temp: float  # °C
    flow_rate: float  # L/min per server
    pressure: float  # bar


@dataclass
class AirCoolingConfig:
    """Air cooling configuration."""
    enabled: bool
    heat_rejection_percentage: float  # 0-100
    components: List[str]
    inlet_temp: float  # °C
    outlet_temp: float  # °C
    airflow_cfm: float  # CFM per server


@dataclass
class RackConfiguration:
    """Data center rack configuration."""
    gpus_per_server: int
    servers_per_rack: int
    total_gpus_per_rack: int
    rack_power_compute: float  # Watts
    rack_power_overhead: float  # percentage (e.g., 0.10 for 10%)
    total_rack_power: float  # Watts

    @property
    def rack_power_kw(self) -> float:
        """Get rack power in kilowatts."""
        return self.total_rack_power / 1000.0


class HardwareProfile:
    """
    Complete hardware profile for data center equipment.

    Supports modern AI infrastructure with hybrid cooling.
    """

    def __init__(self, config_dict: Dict):
        """
        Initialize hardware profile from configuration dictionary.

        Args:
            config_dict: Configuration from YAML file
        """
        self.name = config_dict['name']
        self.generation = config_dict.get('generation', 'Unknown')
        self.release_year = config_dict.get('release_year', 2024)

        # GPU specs
        gpu_cfg = config_dict['gpu']
        self.gpu = GPUSpecifications(
            model=gpu_cfg['model'],
            architecture=gpu_cfg['architecture'],
            power_per_gpu=gpu_cfg['power_per_gpu'],
            thermal_design_power=gpu_cfg['thermal_design_power'],
            max_junction_temp=gpu_cfg['max_junction_temp'],
            idle_power=gpu_cfg['idle_power']
        )

        # Rack configuration
        rack_cfg = config_dict['rack']
        self.rack = RackConfiguration(
            gpus_per_server=rack_cfg['gpus_per_server'],
            servers_per_rack=rack_cfg['servers_per_rack'],
            total_gpus_per_rack=rack_cfg['total_gpus_per_rack'],
            rack_power_compute=rack_cfg['rack_power_compute'],
            rack_power_overhead=rack_cfg['rack_power_overhead'] / 100.0 if isinstance(rack_cfg['rack_power_overhead'], (int, float)) else 0.10,
            total_rack_power=rack_cfg['total_rack_power']
        )

        # Cooling configuration
        cooling_cfg = config_dict['cooling']
        self.cooling_type = cooling_cfg['type']

        if cooling_cfg['liquid']['enabled']:
            liquid_cfg = cooling_cfg['liquid']
            self.liquid_cooling = LiquidCoolingConfig(
                enabled=True,
                heat_capture_percentage=liquid_cfg['heat_capture_percentage'],
                components=liquid_cfg['components'],
                coolant_type=liquid_cfg['coolant_type'],
                inlet_temp=liquid_cfg['inlet_temp'],
                outlet_temp=liquid_cfg['outlet_temp'],
                flow_rate=liquid_cfg['flow_rate'],
                pressure=liquid_cfg['pressure']
            )
        else:
            self.liquid_cooling = None

        if cooling_cfg['air']['enabled']:
            air_cfg = cooling_cfg['air']
            self.air_cooling = AirCoolingConfig(
                enabled=True,
                heat_rejection_percentage=air_cfg['heat_rejection_percentage'],
                components=air_cfg['components'],
                inlet_temp=air_cfg['inlet_temp'],
                outlet_temp=air_cfg['outlet_temp'],
                airflow_cfm=air_cfg['airflow_cfm']
            )
        else:
            self.air_cooling = None

        # Thermal characteristics
        thermal_cfg = config_dict['thermal']
        self.heat_output_liquid = thermal_cfg['heat_output_liquid']
        self.heat_output_air = thermal_cfg['heat_output_air']
        self.heat_flux_per_rack = thermal_cfg['heat_flux_per_rack']

    def calculate_heat_generation(
        self,
        utilization: float = 0.70
    ) -> Tuple[float, float, float]:
        """
        Calculate heat generation at given utilization.

        Args:
            utilization: GPU utilization (0.0 to 1.0)

        Returns:
            (total_heat, liquid_heat, air_heat) in Watts
        """
        # Power scaling (approximately cubic for GPUs)
        # P(u) = P_idle + (P_max - P_idle) * u^3
        gpu_power = (
            self.gpu.idle_power +
            (self.gpu.power_per_gpu - self.gpu.idle_power) * (utilization ** 3)
        )

        # Total power per rack
        total_power = (
            gpu_power * self.rack.total_gpus_per_rack *
            (1 + self.rack.rack_power_overhead)
        )

        # Split between liquid and air
        if self.liquid_cooling:
            liquid_heat = total_power * (self.liquid_cooling.heat_capture_percentage / 100.0)
        else:
            liquid_heat = 0.0

        if self.air_cooling:
            air_heat = total_power * (self.air_cooling.heat_rejection_percentage / 100.0)
        else:
            air_heat = 0.0

        return total_power, liquid_heat, air_heat

    def calculate_coolant_flow(self) -> Tuple[float, float]:
        """
        Calculate required coolant flow rate and heat removal.

        Returns:
            (flow_rate_total, heat_capacity) in (L/min, kW)
        """
        if not self.liquid_cooling:
            return 0.0, 0.0

        # Flow rate per rack
        flow_rate_per_rack = (
            self.liquid_cooling.flow_rate * self.rack.servers_per_rack
        )

        # Heat capacity calculation
        # Q = ṁ · c_p · ΔT
        # For water-glycol: c_p ≈ 3.8 kJ/(kg·K), ρ ≈ 1.05 kg/L
        c_p = 3.8  # kJ/(kg·K)
        density = 1.05  # kg/L
        delta_T = self.liquid_cooling.outlet_temp - self.liquid_cooling.inlet_temp

        # Mass flow rate (kg/min)
        mass_flow = flow_rate_per_rack * density

        # Heat removal (kW)
        heat_capacity = (mass_flow * c_p * delta_T) / 60.0  # Convert to kW

        return flow_rate_per_rack, heat_capacity

    def calculate_air_cfm(self) -> float:
        """
        Calculate total air flow requirement for rack.

        Returns:
            Total CFM for the rack
        """
        if not self.air_cooling:
            return 0.0

        return self.air_cooling.airflow_cfm * self.rack.servers_per_rack

    def validate_cooling_capacity(self) -> Dict[str, bool]:
        """
        Validate that cooling capacity matches heat generation.

        Returns:
            Dictionary of validation checks
        """
        checks = {}

        # Total heat vs cooling capacity
        total_power, liquid_heat, air_heat = self.calculate_heat_generation(1.0)

        checks['total_cooling_matches'] = abs(
            (liquid_heat + air_heat) - total_power
        ) < 0.01 * total_power

        # Liquid cooling capacity
        if self.liquid_cooling:
            _, heat_capacity_kw = self.calculate_coolant_flow()
            checks['liquid_capacity_sufficient'] = (
                heat_capacity_kw >= (liquid_heat / 1000.0)
            )

            checks['liquid_capture_in_range'] = (
                85 <= self.liquid_cooling.heat_capture_percentage <= 95
            )

        # Air cooling
        if self.air_cooling:
            checks['air_rejection_in_range'] = (
                5 <= self.air_cooling.heat_rejection_percentage <= 15
            )

        return checks

    def get_summary(self) -> str:
        """Get human-readable summary of hardware profile."""
        lines = [
            f"Hardware Profile: {self.name}",
            f"Generation: {self.generation} ({self.release_year})",
            f"",
            f"GPU Specifications:",
            f"  Model: {self.gpu.model}",
            f"  Power per GPU: {self.gpu.power_per_gpu}W",
            f"  Max Junction Temp: {self.gpu.max_junction_temp}°C",
            f"",
            f"Rack Configuration:",
            f"  GPUs per rack: {self.rack.total_gpus_per_rack}",
            f"  Servers per rack: {self.rack.servers_per_rack}",
            f"  Total rack power: {self.rack.rack_power_kw:.1f} kW",
            f"",
            f"Cooling Type: {self.cooling_type.upper()}",
        ]

        if self.liquid_cooling:
            lines.extend([
                f"  Liquid Cooling: {self.liquid_cooling.heat_capture_percentage}% heat capture",
                f"    Inlet: {self.liquid_cooling.inlet_temp}°C",
                f"    Outlet: {self.liquid_cooling.outlet_temp}°C",
                f"    Flow: {self.liquid_cooling.flow_rate} L/min per server",
            ])

        if self.air_cooling:
            lines.extend([
                f"  Air Cooling: {self.air_cooling.heat_rejection_percentage}% heat rejection",
                f"    Inlet: {self.air_cooling.inlet_temp}°C",
                f"    Outlet: {self.air_cooling.outlet_temp}°C",
                f"    Airflow: {self.air_cooling.airflow_cfm} CFM per server",
            ])

        return "\n".join(lines)


class DataCenterFloor:
    """
    Represents a data center floor with multiple racks.
    """

    def __init__(
        self,
        hardware_profile: HardwareProfile,
        num_racks: int,
        floor_dimensions: Tuple[float, float, float]
    ):
        """
        Initialize data center floor.

        Args:
            hardware_profile: Hardware profile for racks
            num_racks: Number of racks on the floor
            floor_dimensions: (length, width, height) in meters
        """
        self.hardware_profile = hardware_profile
        self.num_racks = num_racks
        self.floor_dimensions = floor_dimensions

        # Calculate total power
        self.total_it_power = (
            hardware_profile.rack.total_rack_power * num_racks / 1000.0  # kW
        )

    def calculate_heat_distribution(
        self,
        rack_positions: List[Tuple[float, float]],
        utilization: float = 0.70
    ) -> np.ndarray:
        """
        Calculate heat source distribution for PINN.

        Args:
            rack_positions: List of (x, y) positions for racks
            utilization: Average utilization

        Returns:
            Heat source array for collocation points
        """
        _, liquid_heat, air_heat = self.hardware_profile.calculate_heat_generation(
            utilization
        )

        # We only model AIR heat rejection in the PINN
        # (Liquid cooling is handled separately by CDU)
        air_heat_per_rack = air_heat  # Watts

        return rack_positions, air_heat_per_rack

    def calculate_cooling_load(self) -> Dict[str, float]:
        """
        Calculate cooling system requirements.

        Returns:
            Dictionary with cooling loads (liquid and air)
        """
        total_power, liquid_heat, air_heat = (
            self.hardware_profile.calculate_heat_generation(1.0)
        )

        # Total for all racks
        total_liquid_heat = liquid_heat * self.num_racks / 1000.0  # kW
        total_air_heat = air_heat * self.num_racks / 1000.0  # kW

        # Coolant flow
        flow_per_rack, _ = self.hardware_profile.calculate_coolant_flow()
        total_coolant_flow = flow_per_rack * self.num_racks  # L/min

        # Air CFM
        cfm_per_rack = self.hardware_profile.calculate_air_cfm()
        total_air_cfm = cfm_per_rack * self.num_racks

        return {
            'total_it_power_kw': self.total_it_power,
            'liquid_cooling_load_kw': total_liquid_heat,
            'air_cooling_load_kw': total_air_heat,
            'coolant_flow_lpm': total_coolant_flow,
            'air_flow_cfm': total_air_cfm
        }


def load_hardware_profile(profile_name: str = 'nvidia_blackwell_gb200') -> HardwareProfile:
    """
    Load hardware profile from configuration file.

    Args:
        profile_name: Name of the profile to load

    Returns:
        HardwareProfile object
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'hardware_profiles.yaml'

    with open(config_path, 'r') as f:
        all_profiles = yaml.safe_load(f)

    if profile_name not in all_profiles:
        raise ValueError(f"Profile '{profile_name}' not found in configuration")

    profile_config = all_profiles[profile_name]
    return HardwareProfile(profile_config)


def calculate_pue(
    it_power_kw: float,
    cooling_power_kw: float,
    ups_losses_kw: float = 0.0,
    lighting_kw: float = 0.0,
    other_kw: float = 0.0
) -> float:
    """
    Calculate Power Usage Effectiveness (PUE).

    PUE = Total Facility Power / IT Equipment Power

    Args:
        it_power_kw: IT equipment power (servers, storage, networking)
        cooling_power_kw: Cooling system power (chillers, pumps, CRAC)
        ups_losses_kw: UPS and PDU losses
        lighting_kw: Lighting power
        other_kw: Other facility power

    Returns:
        PUE value (lower is better, ideal is 1.0)
    """
    total_facility_power = (
        it_power_kw +
        cooling_power_kw +
        ups_losses_kw +
        lighting_kw +
        other_kw
    )

    pue = total_facility_power / it_power_kw if it_power_kw > 0 else float('inf')

    return pue


# Example usage and validation
if __name__ == "__main__":
    print("Loading NVIDIA Blackwell GB200 hardware profile...")

    # Load profile
    profile = load_hardware_profile('nvidia_blackwell_gb200')

    # Print summary
    print("\n" + "=" * 60)
    print(profile.get_summary())
    print("=" * 60)

    # Calculate heat at different utilizations
    print("\nHeat Generation at Different Utilizations:")
    print("-" * 60)

    for util in [0.3, 0.5, 0.7, 0.9, 1.0]:
        total, liquid, air = profile.calculate_heat_generation(util)
        print(f"Utilization: {util*100:3.0f}% | "
              f"Total: {total/1000:6.1f} kW | "
              f"Liquid: {liquid/1000:6.1f} kW ({liquid/total*100:.1f}%) | "
              f"Air: {air/1000:6.1f} kW ({air/total*100:.1f}%)")

    # Validate cooling capacity
    print("\nCooling Capacity Validation:")
    print("-" * 60)
    checks = profile.validate_cooling_capacity()
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")

    # Data center floor example
    print("\nData Center Floor Example:")
    print("-" * 60)
    floor = DataCenterFloor(
        hardware_profile=profile,
        num_racks=32,
        floor_dimensions=(20.0, 16.0, 3.0)
    )

    cooling_loads = floor.calculate_cooling_load()
    print(f"Number of racks: {floor.num_racks}")
    print(f"Total IT Power: {cooling_loads['total_it_power_kw']:.1f} kW")
    print(f"Liquid Cooling Load: {cooling_loads['liquid_cooling_load_kw']:.1f} kW")
    print(f"Air Cooling Load: {cooling_loads['air_cooling_load_kw']:.1f} kW")
    print(f"Coolant Flow: {cooling_loads['coolant_flow_lpm']:.1f} L/min")
    print(f"Air Flow: {cooling_loads['air_flow_cfm']:.0f} CFM")

    # PUE calculation example
    print("\nPUE Calculation Example:")
    print("-" * 60)

    # Assume efficient cooling with hybrid approach
    cooling_power = cooling_loads['liquid_cooling_load_kw'] * 0.05  # 5% of heat load (very efficient)
    cooling_power += cooling_loads['air_cooling_load_kw'] * 0.10  # 10% of heat load (CRAC)

    pue = calculate_pue(
        it_power_kw=cooling_loads['total_it_power_kw'],
        cooling_power_kw=cooling_power,
        ups_losses_kw=cooling_loads['total_it_power_kw'] * 0.05,  # 5% UPS losses
        lighting_kw=10.0,
        other_kw=5.0
    )

    print(f"IT Power: {cooling_loads['total_it_power_kw']:.1f} kW")
    print(f"Cooling Power: {cooling_power:.1f} kW")
    print(f"Total Facility Power: {cooling_loads['total_it_power_kw'] + cooling_power + 15:.1f} kW")
    print(f"PUE: {pue:.3f}")

    if pue <= 1.20:
        print("✓ Excellent PUE (world-class efficiency)")
    elif pue <= 1.30:
        print("✓ Good PUE (modern data center)")
    elif pue <= 1.50:
        print("⚠ Acceptable PUE (room for improvement)")
    else:
        print("✗ Poor PUE (needs optimization)")
