"""
Loss Functions for Convection-Diffusion PINN

Implements physics-informed loss components for:
- Convection-diffusion PDE residual
- Boundary conditions (inlet/outlet)
- Initial conditions
- Incompressibility constraint (∇·u = 0)
- Data matching (if sensor data available)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class ConvectionDiffusionLoss(nn.Module):
    """
    Complete loss function for convection-diffusion PINN.

    Total Loss = λ₁·L_pde + λ₂·L_boundary + λ₃·L_initial + λ₄·L_divergence + λ₅·L_data

    Where:
        L_pde: Convection-diffusion equation residual
        L_boundary: Boundary condition enforcement
        L_initial: Initial condition enforcement
        L_divergence: Velocity divergence constraint (∇·u = 0)
        L_data: Data matching (optional)
    """

    def __init__(self, config: Dict):
        """
        Initialize loss function.

        Args:
            config: Configuration dictionary with loss weights
        """
        super(ConvectionDiffusionLoss, self).__init__()

        # Loss weights
        weights = config['training']['loss_weights']
        self.w_pde = weights.get('pde_loss', 1.0)
        self.w_boundary = weights.get('boundary_loss', 100.0)
        self.w_initial = weights.get('initial_loss', 100.0)
        self.w_divergence = weights.get('divergence_loss', 10.0)  # NEW: incompressibility
        self.w_data = weights.get('data_loss', 10.0)

        # MSE loss
        self.mse = nn.MSELoss()

    def forward(
        self,
        # PDE residual
        pde_residual: torch.Tensor,

        # Boundary conditions
        boundary_pred: Optional[torch.Tensor] = None,
        boundary_true: Optional[torch.Tensor] = None,

        # Initial conditions
        initial_pred: Optional[torch.Tensor] = None,
        initial_true: Optional[torch.Tensor] = None,

        # Velocity divergence (incompressibility)
        velocity_divergence: Optional[torch.Tensor] = None,

        # Data matching
        data_pred: Optional[torch.Tensor] = None,
        data_true: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.

        Args:
            pde_residual: Convection-diffusion residual [N_collocation, 1]
            boundary_pred: Predicted T at boundaries [N_boundary, 1]
            boundary_true: True boundary conditions [N_boundary, 1]
            initial_pred: Predicted T at t=0 [N_initial, 1]
            initial_true: Initial conditions [N_initial, 1]
            velocity_divergence: ∇·u [N_collocation, 1]
            data_pred: Predicted T at sensor locations [N_data, 1]
            data_true: Measured temperatures [N_data, 1]

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        # 1. PDE Loss (convection-diffusion equation)
        loss_pde = torch.mean(pde_residual ** 2)

        # 2. Boundary condition loss
        if boundary_pred is not None and boundary_true is not None:
            loss_boundary = self.mse(boundary_pred, boundary_true)
        else:
            loss_boundary = torch.tensor(0.0, device=pde_residual.device)

        # 3. Initial condition loss
        if initial_pred is not None and initial_true is not None:
            loss_initial = self.mse(initial_pred, initial_true)
        else:
            loss_initial = torch.tensor(0.0, device=pde_residual.device)

        # 4. Divergence-free constraint (incompressible flow)
        if velocity_divergence is not None:
            loss_divergence = torch.mean(velocity_divergence ** 2)
        else:
            loss_divergence = torch.tensor(0.0, device=pde_residual.device)

        # 5. Data loss (sensor measurements)
        if data_pred is not None and data_true is not None:
            loss_data = self.mse(data_pred, data_true)
        else:
            loss_data = torch.tensor(0.0, device=pde_residual.device)

        # Total weighted loss
        total_loss = (
            self.w_pde * loss_pde +
            self.w_boundary * loss_boundary +
            self.w_initial * loss_initial +
            self.w_divergence * loss_divergence +
            self.w_data * loss_data
        )

        # Create loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'boundary': loss_boundary.item() if isinstance(loss_boundary, torch.Tensor) else 0.0,
            'initial': loss_initial.item() if isinstance(loss_initial, torch.Tensor) else 0.0,
            'divergence': loss_divergence.item() if isinstance(loss_divergence, torch.Tensor) else 0.0,
            'data': loss_data.item() if isinstance(loss_data, torch.Tensor) else 0.0
        }

        return total_loss, loss_dict


class PhysicsMetrics:
    """
    Calculate physics-based validation metrics.

    Useful for monitoring training and validating against physical laws.
    """

    @staticmethod
    def reynolds_number(velocity: float, length: float, viscosity: float = 1.5e-5) -> float:
        """
        Calculate Reynolds number: Re = u·L/ν

        Args:
            velocity: Characteristic velocity [m/s]
            length: Characteristic length [m]
            viscosity: Kinematic viscosity [m²/s]

        Returns:
            Reynolds number (dimensionless)
        """
        return velocity * length / viscosity

    @staticmethod
    def peclet_number(
        velocity: float,
        length: float,
        thermal_diffusivity: float = 2.2e-5
    ) -> float:
        """
        Calculate Péclet number: Pe = u·L/α

        Ratio of convective to diffusive transport.
        - Pe >> 1: Convection dominated
        - Pe << 1: Diffusion dominated

        Args:
            velocity: Characteristic velocity [m/s]
            length: Characteristic length [m]
            thermal_diffusivity: Thermal diffusivity [m²/s]

        Returns:
            Péclet number (dimensionless)
        """
        return velocity * length / thermal_diffusivity

    @staticmethod
    def energy_balance_error(
        heat_generated: float,
        heat_removed_convection: float,
        heat_removed_diffusion: float
    ) -> float:
        """
        Calculate energy balance error.

        For steady state: Heat generated = Heat removed

        Args:
            heat_generated: Total heat from sources [W]
            heat_removed_convection: Heat removed by convection [W]
            heat_removed_diffusion: Heat removed by diffusion [W]

        Returns:
            Relative error (0 = perfect balance)
        """
        total_removed = heat_removed_convection + heat_removed_diffusion
        if heat_generated > 0:
            error = abs(heat_generated - total_removed) / heat_generated
        else:
            error = 0.0
        return error

    @staticmethod
    def compute_nusselt_number(
        heat_flux: float,
        temperature_difference: float,
        length: float,
        thermal_conductivity: float = 0.0257
    ) -> float:
        """
        Calculate Nusselt number: Nu = (q·L)/(k·ΔT)

        Ratio of convective to conductive heat transfer.

        Args:
            heat_flux: Heat flux [W/m²]
            temperature_difference: T_hot - T_cold [K]
            length: Characteristic length [m]
            thermal_conductivity: Thermal conductivity [W/(m·K)]

        Returns:
            Nusselt number (dimensionless)
        """
        if temperature_difference > 0:
            return (heat_flux * length) / (thermal_conductivity * temperature_difference)
        return 0.0


def validate_physics_satisfaction(
    model,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    z_test: torch.Tensor,
    t_test: torch.Tensor,
    Q_test: torch.Tensor,
    threshold: float = 1e-3
) -> Dict[str, bool]:
    """
    Validate that the trained model satisfies physical laws.

    Args:
        model: Trained ConvectionDiffusionPINN
        x_test, y_test, z_test, t_test: Test coordinates
        Q_test: Heat source at test points
        threshold: Acceptable residual threshold

    Returns:
        Dictionary of validation checks
    """
    model.eval()

    with torch.no_grad():
        # Compute PDE residual
        residual = model.compute_convection_diffusion_residual(
            x_test, y_test, z_test, t_test, Q_test
        )

        # Compute velocity divergence
        divergence = model.compute_divergence_free_constraint(
            x_test, y_test, z_test, t_test
        )

        # Validation checks
        checks = {
            'pde_satisfied': residual.abs().mean().item() < threshold,
            'pde_mean_residual': residual.abs().mean().item(),
            'pde_max_residual': residual.abs().max().item(),

            'divergence_free': divergence.abs().mean().item() < threshold,
            'divergence_mean': divergence.abs().mean().item(),
            'divergence_max': divergence.abs().max().item(),

            'temperature_physical': True,  # Check if T in reasonable range
        }

        # Check temperature range
        T = model(x_test, y_test, z_test, t_test)
        T_min, T_max = T.min().item(), T.max().item()
        checks['temperature_range'] = (T_min, T_max)
        checks['temperature_physical'] = (T_min > -50 and T_max < 100)  # Reasonable for data center

    return checks


# Example: Demonstrate loss computation
if __name__ == "__main__":
    print("=" * 70)
    print("Convection-Diffusion Loss Functions")
    print("=" * 70)

    # Create dummy config
    config = {
        'training': {
            'loss_weights': {
                'pde_loss': 1.0,
                'boundary_loss': 100.0,
                'initial_loss': 100.0,
                'divergence_loss': 10.0,
                'data_loss': 10.0
            }
        }
    }

    # Initialize loss function
    criterion = ConvectionDiffusionLoss(config)

    print("\n✓ Loss function initialized")
    print(f"  PDE weight: {criterion.w_pde}")
    print(f"  Boundary weight: {criterion.w_boundary}")
    print(f"  Initial weight: {criterion.w_initial}")
    print(f"  Divergence weight: {criterion.w_divergence} (NEW - incompressibility)")
    print(f"  Data weight: {criterion.w_data}")

    # Create dummy residuals
    print("\n" + "=" * 70)
    print("Example Loss Computation")
    print("=" * 70)

    batch_size = 1000
    pde_residual = torch.randn(batch_size, 1) * 0.1
    boundary_pred = torch.randn(100, 1) * 25.0 + 20.0
    boundary_true = torch.ones(100, 1) * 18.0
    initial_pred = torch.randn(200, 1) * 23.0 + 22.0
    initial_true = torch.ones(200, 1) * 22.0
    velocity_divergence = torch.randn(batch_size, 1) * 0.01

    # Compute loss
    total_loss, loss_dict = criterion(
        pde_residual=pde_residual,
        boundary_pred=boundary_pred,
        boundary_true=boundary_true,
        initial_pred=initial_pred,
        initial_true=initial_true,
        velocity_divergence=velocity_divergence
    )

    print("\n✓ Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key:12s}: {value:.6f}")

    # Physics metrics
    print("\n" + "=" * 70)
    print("Physics Validation Metrics")
    print("=" * 70)

    metrics = PhysicsMetrics()

    # Reynolds number
    u = 2.0  # m/s (CRAC velocity)
    L = 20.0  # m (room length)
    Re = metrics.reynolds_number(u, L)
    print(f"\n✓ Reynolds number: {Re:.0f}")
    if Re < 2300:
        print("  → Laminar flow")
    elif Re < 4000:
        print("  → Transitional flow")
    else:
        print("  → Turbulent flow")

    # Péclet number
    Pe = metrics.peclet_number(u, L)
    print(f"\n✓ Péclet number: {Pe:.0f}")
    if Pe > 10:
        print("  → Convection dominated (air flow is important!)")
    elif Pe < 0.1:
        print("  → Diffusion dominated")
    else:
        print("  → Mixed convection-diffusion")

    # Energy balance
    heat_gen = 120000  # W (Blackwell GB200 rack)
    heat_conv = 108000  # W (90% by convection)
    heat_diff = 12000   # W (10% by diffusion)
    error = metrics.energy_balance_error(heat_gen, heat_conv, heat_diff)
    print(f"\n✓ Energy balance error: {error*100:.2f}%")

    print("\n" + "=" * 70)
    print("Key Insights for Data Center:")
    print("=" * 70)
    print("\n1. Convection is DOMINANT (Pe >> 1)")
    print("   → Air flow from CRAC units is critical")
    print("   → Cannot ignore velocity field u")
    print("\n2. Need to model:")
    print("   → Temperature: T(x,y,z,t)")
    print("   → Velocity: u(x,y,z,t)")
    print("   → Heat sources: Q(x,y,z) from Blackwell racks")
    print("\n3. Physics constraints:")
    print("   → Convection-diffusion: ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q")
    print("   → Incompressibility: ∇·u = 0")
    print("   → Boundary conditions: T_inlet = 18°C, u_inlet = 2 m/s")
    print("\n✓ All components ready for training!")
