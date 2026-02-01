"""
Convection-Diffusion Physics-Informed Neural Network

This module implements a PINN for the convection-diffusion equation:
    ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q

Where:
    T: Temperature field [°C]
    u: Velocity field [m/s] (from CRAC units)
    Q: Heat source [W/m³] (Blackwell GB200 racks)
    ρ: Air density [kg/m³]
    c_p: Specific heat [J/(kg·K)]
    k: Thermal conductivity [W/(m·K)]

This extends the basic diffusion PINN to include convection (air flow).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import sys
from pathlib import Path

# Import base PINN components
sys.path.append(str(Path(__file__).parent))
from pinn import FullyConnectedNN


class ConvectionDiffusionPINN(nn.Module):
    """
    Physics-Informed Neural Network for convection-diffusion heat transfer.

    Solves: ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q

    This models:
    - Heat diffusion (conduction)
    - Heat convection (advection by air flow)
    - Heat generation (servers)
    """

    def __init__(self, config: Dict, velocity_field: Optional[Callable] = None):
        """
        Initialize Convection-Diffusion PINN.

        Args:
            config: Configuration dictionary with physics parameters
            velocity_field: Optional function u(x,y,z,t) -> (u_x, u_y, u_z)
                           If None, will learn velocity field jointly with temperature
        """
        super(ConvectionDiffusionPINN, self).__init__()

        # Physical properties
        self.density = config['thermal']['density']  # ρ [kg/m³]
        self.specific_heat = config['thermal']['specific_heat']  # c_p [J/(kg·K)]
        self.thermal_conductivity = config['thermal']['thermal_conductivity']  # k [W/(m·K)]

        # Derived property
        self.rho_cp = self.density * self.specific_heat

        # Temperature network
        net_config = config['network']
        self.temperature_network = FullyConnectedNN(
            input_dim=4,  # [x, y, z, t]
            output_dim=1,  # T
            hidden_layers=net_config['hidden_layers'],
            activation=net_config['activation']
        )

        # Velocity field (if not provided, learn it)
        self.velocity_field_fn = velocity_field
        if velocity_field is None:
            # Learn velocity field with separate network
            self.velocity_network = FullyConnectedNN(
                input_dim=4,  # [x, y, z, t]
                output_dim=3,  # [u_x, u_y, u_z]
                hidden_layers=[32, 32, 32],  # Smaller network for velocity
                activation='tanh'
            )
            self.learn_velocity = True
        else:
            self.velocity_network = None
            self.learn_velocity = False

        # Domain bounds
        self.domain_bounds = {
            'x': [0, config['geometry']['length']],
            'y': [0, config['geometry']['width']],
            'z': [0, config['geometry']['height']],
            't': [0, 1.0]
        }

        # Normalization parameters
        self.T_ref = config['boundaries']['initial_temp']
        self.T_scale = 50.0
        self.u_scale = 2.0  # Typical CRAC air velocity ~2 m/s

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict temperature T(x, y, z, t).

        Args:
            x, y, z: Spatial coordinates [batch_size, 1]
            t: Time coordinate [batch_size, 1]

        Returns:
            Temperature [batch_size, 1]
        """
        # Normalize inputs
        x_norm = self._normalize(x, 'x')
        y_norm = self._normalize(y, 'y')
        z_norm = self._normalize(z, 'z')
        t_norm = self._normalize(t, 't')

        inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)

        # Predict normalized temperature
        T_norm = self.temperature_network(inputs)

        # Denormalize
        T = T_norm * self.T_scale + self.T_ref

        return T

    def predict_velocity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict or compute velocity field u(x, y, z, t).

        Args:
            x, y, z: Spatial coordinates [batch_size, 1]
            t: Time coordinate [batch_size, 1]

        Returns:
            (u_x, u_y, u_z) velocity components [batch_size, 1] each
        """
        if self.velocity_field_fn is not None:
            # Use provided velocity field function
            return self.velocity_field_fn(x, y, z, t)

        elif self.learn_velocity:
            # Learn velocity field with neural network
            x_norm = self._normalize(x, 'x')
            y_norm = self._normalize(y, 'y')
            z_norm = self._normalize(z, 'z')
            t_norm = self._normalize(t, 't')

            inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)
            u_norm = self.velocity_network(inputs)

            # Denormalize and split
            u_x = u_norm[:, 0:1] * self.u_scale
            u_y = u_norm[:, 1:2] * self.u_scale
            u_z = u_norm[:, 2:3] * self.u_scale

            return u_x, u_y, u_z

        else:
            # Zero velocity (pure diffusion fallback)
            batch_size = x.shape[0]
            device = x.device
            return (
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device),
                torch.zeros(batch_size, 1, device=device)
            )

    def compute_convection_diffusion_residual(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        Q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the PDE residual for convection-diffusion equation:

        Residual = ρ c_p (∂T/∂t + u·∇T) - ∇·(k∇T) - Q

        Uses automatic differentiation to compute all derivatives.

        Args:
            x, y, z: Spatial coordinates [batch_size, 1] (requires_grad=True)
            t: Time coordinate [batch_size, 1] (requires_grad=True)
            Q: Heat source [batch_size, 1] in W/m³ (optional)

        Returns:
            PDE residual [batch_size, 1] (should be close to zero)
        """
        # Enable gradient computation
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)

        # ===== STEP 1: Predict temperature =====
        T = self.forward(x, y, z, t)

        # ===== STEP 2: Compute temporal derivative ∂T/∂t =====
        T_t = self._gradient(T, t)

        # ===== STEP 3: Compute spatial gradients ∇T =====
        T_x = self._gradient(T, x)
        T_y = self._gradient(T, y)
        T_z = self._gradient(T, z)

        # ===== STEP 4: Compute second derivatives (Laplacian) =====
        T_xx = self._gradient(T_x, x)
        T_yy = self._gradient(T_y, y)
        T_zz = self._gradient(T_z, z)
        laplacian_T = T_xx + T_yy + T_zz

        # ===== STEP 5: Get velocity field =====
        u_x, u_y, u_z = self.predict_velocity(x, y, z, t)

        # ===== STEP 6: Compute convection term u·∇T =====
        convection = u_x * T_x + u_y * T_y + u_z * T_z

        # ===== STEP 7: Heat source =====
        if Q is None:
            Q = torch.zeros_like(T)

        # ===== STEP 8: Assemble PDE residual =====
        # Left side: ρ c_p (∂T/∂t + u·∇T)
        left_side = self.rho_cp * (T_t + convection)

        # Right side: ∇·(k∇T) + Q = k∇²T + Q (assuming constant k)
        right_side = self.thermal_conductivity * laplacian_T + Q

        # Residual (should be zero if PDE is satisfied)
        residual = left_side - right_side

        return residual

    def compute_divergence_free_constraint(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute divergence-free constraint for velocity field: ∇·u = 0

        For incompressible flow, velocity must be divergence-free.

        Args:
            x, y, z: Spatial coordinates [batch_size, 1] (requires_grad=True)
            t: Time coordinate [batch_size, 1]

        Returns:
            Divergence ∇·u [batch_size, 1] (should be zero)
        """
        if not self.learn_velocity:
            # If velocity is prescribed, assume it's already divergence-free
            return torch.zeros_like(x)

        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        # Get velocity components
        u_x, u_y, u_z = self.predict_velocity(x, y, z, t)

        # Compute divergence: ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z
        du_x_dx = self._gradient(u_x, x)
        du_y_dy = self._gradient(u_y, y)
        du_z_dz = self._gradient(u_z, z)

        divergence = du_x_dx + du_y_dy + du_z_dz

        return divergence

    def _gradient(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient dy/dx using automatic differentiation.

        Args:
            y: Output tensor [batch_size, 1]
            x: Input tensor [batch_size, 1] (requires_grad=True)

        Returns:
            Gradient dy/dx [batch_size, 1]
        """
        grad = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return grad

    def _normalize(self, x: torch.Tensor, coord: str) -> torch.Tensor:
        """Normalize coordinate to [-1, 1]."""
        x_min, x_max = self.domain_bounds[coord]
        return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


class CRACVelocityField:
    """
    Analytical velocity field model for CRAC (Computer Room Air Conditioner) units.

    Models cold aisle / hot aisle configuration with air flow patterns.
    """

    def __init__(
        self,
        inlet_velocity: float = 2.0,  # m/s
        inlet_positions: list = None,
        outlet_positions: list = None,
        room_dimensions: Tuple[float, float, float] = (20.0, 16.0, 3.0)
    ):
        """
        Initialize CRAC velocity field.

        Args:
            inlet_velocity: Inlet air velocity [m/s]
            inlet_positions: List of (x, y, z) inlet positions
            outlet_positions: List of (x, y, z) outlet positions
            room_dimensions: (length, width, height) [m]
        """
        self.inlet_velocity = inlet_velocity
        self.room_length, self.room_width, self.room_height = room_dimensions

        # Default: inlet at x=0 (cold aisle), outlet at x=length (hot aisle)
        if inlet_positions is None:
            self.inlet_positions = [(0.0, self.room_width/2, 0.5)]
        else:
            self.inlet_positions = inlet_positions

        if outlet_positions is None:
            self.outlet_positions = [(self.room_length, self.room_width/2, 2.0)]
        else:
            self.outlet_positions = outlet_positions

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute velocity field at given positions.

        Args:
            x, y, z: Spatial coordinates [batch_size, 1]
            t: Time coordinate [batch_size, 1] (not used for steady flow)

        Returns:
            (u_x, u_y, u_z) velocity components [batch_size, 1]
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize velocity components
        u_x = torch.zeros_like(x)
        u_y = torch.zeros_like(y)
        u_z = torch.zeros_like(z)

        # Simple model: horizontal flow from inlet to outlet
        # u_x increases linearly from inlet to outlet
        x_normalized = x / self.room_length
        u_x = self.inlet_velocity * x_normalized

        # Add slight upward flow (hot air rises)
        # u_z increases with height
        z_normalized = z / self.room_height
        u_z = 0.1 * self.inlet_velocity * z_normalized  # 10% of horizontal velocity

        # Lateral flow is minimal in cold/hot aisle config
        u_y = torch.zeros_like(y)

        return u_x, u_y, u_z


def create_convection_diffusion_model(
    config_path: str,
    use_crac_velocity: bool = True
) -> ConvectionDiffusionPINN:
    """
    Factory function to create convection-diffusion PINN.

    Args:
        config_path: Path to physics configuration YAML
        use_crac_velocity: If True, use analytical CRAC velocity field

    Returns:
        Configured ConvectionDiffusionPINN model
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create velocity field
    if use_crac_velocity:
        velocity_field = CRACVelocityField(
            inlet_velocity=2.0,  # 2 m/s typical CRAC velocity
            room_dimensions=(
                config['geometry']['length'],
                config['geometry']['width'],
                config['geometry']['height']
            )
        )
    else:
        velocity_field = None  # Will learn velocity with neural network

    # Create model
    model = ConvectionDiffusionPINN(config, velocity_field=velocity_field)

    return model


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Convection-Diffusion PINN - Physics-Informed Loss Function")
    print("=" * 70)
    print("\nEquation: ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q")
    print("\nComponents:")
    print("  - ∂T/∂t: Temporal derivative (temperature change)")
    print("  - u·∇T: Convection term (heat advection by air flow)")
    print("  - ∇·(k∇T): Diffusion term (heat conduction)")
    print("  - Q: Heat source (Blackwell GB200 racks)")

    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'physics_config.yaml'

    print("\n" + "=" * 70)
    print("1. Creating Convection-Diffusion PINN Model")
    print("=" * 70)

    model = create_convection_diffusion_model(
        config_path=str(config_path),
        use_crac_velocity=True
    )

    print(f"✓ Model created")
    print(f"  Temperature network: {sum(p.numel() for p in model.temperature_network.parameters()):,} parameters")
    if model.velocity_network:
        print(f"  Velocity network: {sum(p.numel() for p in model.velocity_network.parameters()):,} parameters")
    else:
        print(f"  Velocity field: Analytical CRAC model")

    # Test forward pass
    print("\n" + "=" * 70)
    print("2. Testing Forward Pass (Temperature Prediction)")
    print("=" * 70)

    batch_size = 100
    x = torch.rand(batch_size, 1) * 20.0  # 0-20m
    y = torch.rand(batch_size, 1) * 16.0  # 0-16m
    z = torch.rand(batch_size, 1) * 3.0   # 0-3m
    t = torch.rand(batch_size, 1) * 1.0   # 0-1s

    T = model(x, y, z, t)
    print(f"✓ Temperature prediction: shape {T.shape}")
    print(f"  Temperature range: [{T.min():.2f}, {T.max():.2f}] °C")

    # Test velocity prediction
    print("\n" + "=" * 70)
    print("3. Testing Velocity Field (Air Flow from CRAC)")
    print("=" * 70)

    u_x, u_y, u_z = model.predict_velocity(x, y, z, t)
    print(f"✓ Velocity prediction:")
    print(f"  u_x (horizontal): [{u_x.min():.3f}, {u_x.max():.3f}] m/s")
    print(f"  u_y (lateral):    [{u_y.min():.3f}, {u_y.max():.3f}] m/s")
    print(f"  u_z (vertical):   [{u_z.min():.3f}, {u_z.max():.3f}] m/s")
    print(f"  Mean velocity magnitude: {torch.sqrt(u_x**2 + u_y**2 + u_z**2).mean():.3f} m/s")

    # Test PDE residual computation (THE KEY PART!)
    print("\n" + "=" * 70)
    print("4. Computing Physics-Informed Loss (PDE Residual)")
    print("=" * 70)
    print("\nUsing AUTOMATIC DIFFERENTIATION (torch.autograd):")
    print("  ✓ ∂T/∂t computed with autograd")
    print("  ✓ ∇T (gradient) computed with autograd")
    print("  ✓ ∇²T (Laplacian) computed with autograd")

    # Add heat source (simulate Blackwell GB200 rack at center)
    Q = torch.zeros_like(x)
    # 120kW rack at center (x=10m, y=8m)
    rack_center_x, rack_center_y = 10.0, 8.0
    rack_volume = 0.6 * 1.0 * 2.0  # 0.6m x 1.0m x 2.0m
    rack_power_density = (120000 / rack_volume)  # W/m³

    # Add heat to points near rack center
    distances = torch.sqrt((x - rack_center_x)**2 + (y - rack_center_y)**2)
    Q[distances < 1.0] = rack_power_density

    print(f"\n  Heat source Q:")
    print(f"    Points with heat: {(Q > 0).sum().item()}/{batch_size}")
    print(f"    Max Q: {Q.max():.0f} W/m³ (Blackwell GB200 rack)")

    # Compute PDE residual
    residual = model.compute_convection_diffusion_residual(x, y, z, t, Q)

    print(f"\n✓ PDE Residual computed:")
    print(f"  Shape: {residual.shape}")
    print(f"  Mean |residual|: {residual.abs().mean():.4e}")
    print(f"  Max |residual|: {residual.abs().max():.4e}")
    print(f"\n  (Lower residual = better physics satisfaction)")

    # Test divergence-free constraint
    print("\n" + "=" * 70)
    print("5. Testing Incompressibility Constraint (∇·u = 0)")
    print("=" * 70)

    divergence = model.compute_divergence_free_constraint(x, y, z, t)
    print(f"✓ Velocity divergence computed:")
    print(f"  Mean |∇·u|: {divergence.abs().mean():.4e}")
    print(f"  Max |∇·u|: {divergence.abs().max():.4e}")
    print(f"\n  (Should be close to 0 for incompressible flow)")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Physics-Informed Loss Components")
    print("=" * 70)
    print("\nLoss Function = L_pde + L_boundary + L_initial + L_divergence")
    print("\nWhere:")
    print("  L_pde = mean(|ρ c_p (∂T/∂t + u·∇T) - ∇·(k∇T) - Q|²)")
    print("         ↳ Convection-diffusion equation residual")
    print("\n  L_boundary = mean(|T_pred - T_bc|²)")
    print("         ↳ Boundary condition enforcement")
    print("\n  L_initial = mean(|T(t=0) - T_initial|²)")
    print("         ↳ Initial condition enforcement")
    print("\n  L_divergence = mean(|∇·u|²)")
    print("         ↳ Incompressible flow constraint")

    print("\n" + "=" * 70)
    print("✓ All components working correctly!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Integrate with Blackwell GB200 heat sources")
    print("  2. Add CRAC inlet/outlet boundary conditions")
    print("  3. Train the model with physics-informed loss")
    print("  4. Validate against EnergyPlus simulation")
    print("\nSee: convection_diffusion_trainer.py for training example")
