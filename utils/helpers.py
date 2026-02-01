"""
Utility functions for CoolingAI Simulator

This module provides helper functions for:
- Data conversion and preprocessing
- Coordinate transformations
- Physics calculations
- File I/O
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save file
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def numpy_to_torch(
    *arrays: np.ndarray,
    device: str = 'cpu',
    requires_grad: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Convert numpy arrays to torch tensors.

    Args:
        *arrays: Variable number of numpy arrays
        device: Device to place tensors on
        requires_grad: Whether to enable gradient computation

    Returns:
        Tuple of torch tensors
    """
    tensors = []
    for arr in arrays:
        tensor = torch.tensor(arr, dtype=torch.float32, device=device)
        if tensor.dim() == 1:
            tensor = tensor.reshape(-1, 1)
        tensor.requires_grad = requires_grad
        tensors.append(tensor)

    return tuple(tensors) if len(tensors) > 1 else tensors[0]


def torch_to_numpy(*tensors: torch.Tensor) -> Tuple[np.ndarray, ...]:
    """
    Convert torch tensors to numpy arrays.

    Args:
        *tensors: Variable number of torch tensors

    Returns:
        Tuple of numpy arrays
    """
    arrays = []
    for tensor in tensors:
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.flatten()
        arrays.append(arr)

    return tuple(arrays) if len(arrays) > 1 else arrays[0]


def create_meshgrid_3d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 3D meshgrid for visualization.

    Args:
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        z_range: (z_min, z_max)
        resolution: Number of points per dimension

    Returns:
        X, Y, Z meshgrid arrays
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    return X, Y, Z


def calculate_thermal_diffusivity(
    thermal_conductivity: float,
    density: float,
    specific_heat: float
) -> float:
    """
    Calculate thermal diffusivity α = k / (ρ * c_p)

    Args:
        thermal_conductivity: k [W/(m·K)]
        density: ρ [kg/m³]
        specific_heat: c_p [J/(kg·K)]

    Returns:
        Thermal diffusivity [m²/s]
    """
    return thermal_conductivity / (density * specific_heat)


def calculate_heat_flux(
    temperature_gradient: np.ndarray,
    thermal_conductivity: float
) -> np.ndarray:
    """
    Calculate heat flux using Fourier's law: q = -k∇T

    Args:
        temperature_gradient: ∇T [K/m]
        thermal_conductivity: k [W/(m·K)]

    Returns:
        Heat flux [W/m²]
    """
    return -thermal_conductivity * temperature_gradient


def calculate_reynolds_number(
    velocity: float,
    characteristic_length: float,
    kinematic_viscosity: float = 1.5e-5
) -> float:
    """
    Calculate Reynolds number: Re = vL/ν

    Args:
        velocity: Flow velocity [m/s]
        characteristic_length: Characteristic length [m]
        kinematic_viscosity: Kinematic viscosity [m²/s]

    Returns:
        Reynolds number (dimensionless)
    """
    return velocity * characteristic_length / kinematic_viscosity


def calculate_prandtl_number(
    kinematic_viscosity: float = 1.5e-5,
    thermal_diffusivity: float = 2.2e-5
) -> float:
    """
    Calculate Prandtl number: Pr = ν/α

    Args:
        kinematic_viscosity: ν [m²/s]
        thermal_diffusivity: α [m²/s]

    Returns:
        Prandtl number (dimensionless)
    """
    return kinematic_viscosity / thermal_diffusivity


def dimensionalize_temperature(
    T_normalized: np.ndarray,
    T_ref: float,
    T_scale: float
) -> np.ndarray:
    """
    Convert normalized temperature back to physical units.

    Args:
        T_normalized: Normalized temperature
        T_ref: Reference temperature [°C]
        T_scale: Temperature scale [°C]

    Returns:
        Physical temperature [°C]
    """
    return T_normalized * T_scale + T_ref


def create_server_rack_layout(
    num_racks: int,
    room_length: float,
    room_width: float,
    rack_width: float,
    rack_depth: float,
    aisle_width: float = 1.0,
    seed: Optional[int] = 42
) -> List[Tuple[float, float]]:
    """
    Generate server rack positions in cold aisle/hot aisle configuration.

    Args:
        num_racks: Number of server racks
        room_length: Room length [m]
        room_width: Room width [m]
        rack_width: Rack width [m]
        rack_depth: Rack depth [m]
        aisle_width: Aisle width between racks [m]
        seed: Random seed for reproducibility

    Returns:
        List of (x, y) positions for rack centers
    """
    if seed is not None:
        np.random.seed(seed)

    positions = []

    # Simple row-based layout
    rows = int(np.sqrt(num_racks))
    cols = int(np.ceil(num_racks / rows))

    spacing_x = room_length / (cols + 1)
    spacing_y = room_width / (rows + 1)

    for i in range(rows):
        for j in range(cols):
            if len(positions) >= num_racks:
                break

            x = spacing_x * (j + 1)
            y = spacing_y * (i + 1)
            positions.append((x, y))

    return positions[:num_racks]


def calculate_cooling_efficiency(
    inlet_temp: float,
    outlet_temp: float,
    power_dissipated: float,
    air_flow_rate: float,
    air_density: float = 1.2,
    specific_heat: float = 1005
) -> float:
    """
    Calculate cooling efficiency.

    Args:
        inlet_temp: Inlet air temperature [°C]
        outlet_temp: Outlet air temperature [°C]
        power_dissipated: Total heat dissipation [W]
        air_flow_rate: Volumetric air flow rate [m³/s]
        air_density: Air density [kg/m³]
        specific_heat: Specific heat of air [J/(kg·K)]

    Returns:
        Cooling efficiency (0 to 1)
    """
    # Heat removal capacity
    mass_flow_rate = air_flow_rate * air_density
    heat_removed = mass_flow_rate * specific_heat * (outlet_temp - inlet_temp)

    # Efficiency
    efficiency = heat_removed / power_dissipated if power_dissipated > 0 else 0

    return min(efficiency, 1.0)


def calculate_pue(
    total_facility_power: float,
    it_equipment_power: float
) -> float:
    """
    Calculate Power Usage Effectiveness (PUE).

    PUE = Total Facility Power / IT Equipment Power

    Args:
        total_facility_power: Total facility power [W]
        it_equipment_power: IT equipment power [W]

    Returns:
        PUE value (ideally close to 1.0)
    """
    return total_facility_power / it_equipment_power if it_equipment_power > 0 else float('inf')


def compute_error_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Compute error metrics between predicted and true values.

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        Dictionary of error metrics
    """
    # Absolute errors
    abs_error = np.abs(y_pred - y_true)
    mae = np.mean(abs_error)
    max_error = np.max(abs_error)

    # Relative errors
    rel_error = abs_error / (np.abs(y_true) + 1e-8)
    mape = np.mean(rel_error) * 100

    # RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'mae': mae,
        'max_error': max_error,
        'mape': mape,
        'rmse': rmse,
        'r2': r2
    }


def export_temperature_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    T: np.ndarray,
    save_path: str,
    format: str = 'csv'
):
    """
    Export temperature field to file.

    Args:
        x, y, z: Coordinate arrays
        T: Temperature array
        save_path: Path to save file
        format: Export format ('csv', 'npy', 'vtk')
    """
    if format == 'csv':
        data = np.column_stack([x.flatten(), y.flatten(), z.flatten(), T.flatten()])
        np.savetxt(save_path, data, delimiter=',',
                   header='x,y,z,temperature', comments='')

    elif format == 'npy':
        np.save(save_path, {'x': x, 'y': y, 'z': z, 'T': T})

    elif format == 'vtk':
        # Basic VTK export (requires external library for full support)
        print("VTK export requires pyvista or similar library")

    else:
        raise ValueError(f"Unknown format: {format}")


def celsius_to_kelvin(temp_celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return temp_celsius + 273.15


def kelvin_to_celsius(temp_kelvin: float) -> float:
    """Convert Kelvin to Celsius."""
    return temp_kelvin - 273.15


def fahrenheit_to_celsius(temp_fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_fahrenheit - 32) * 5/9


def celsius_to_fahrenheit(temp_celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_celsius * 9/5 + 32
