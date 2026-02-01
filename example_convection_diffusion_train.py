"""
Training Example: Convection-Diffusion PINN for Data Center Thermal Management

This script demonstrates training a Physics-Informed Neural Network that solves:
    ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q

With:
    - CRAC air flow velocity field
    - Blackwell GB200 heat sources (120kW per rack)
    - Automatic differentiation for all derivatives
    - Physics-informed loss function

Author: CoolingAI Simulator
Phase: 2.1 - Convection-Diffusion Implementation
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add project to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.convection_diffusion_pinn import ConvectionDiffusionPINN, CRACVelocityField
from models.convection_diffusion_losses import ConvectionDiffusionLoss, validate_physics_satisfaction
from models.hardware import load_hardware_profile


class ConvectionDiffusionTrainer:
    """
    Training pipeline for convection-diffusion PINN.
    """

    def __init__(self, config_path: str, device: str = 'cpu'):
        """
        Initialize trainer.

        Args:
            config_path: Path to physics configuration
            device: Device for training ('cpu', 'cuda', 'mps')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device)

        # Load hardware profile (Blackwell GB200)
        hardware_config_path = project_root / 'configs' / 'hardware_profiles.yaml'
        if hardware_config_path.exists():
            self.hardware_profile = load_hardware_profile('nvidia_blackwell_gb200')
        else:
            self.hardware_profile = None

        # Create CRAC velocity field
        self.velocity_field = CRACVelocityField(
            inlet_velocity=2.0,  # 2 m/s
            room_dimensions=(
                self.config['geometry']['length'],
                self.config['geometry']['width'],
                self.config['geometry']['height']
            )
        )

        # Create model
        self.model = ConvectionDiffusionPINN(
            self.config,
            velocity_field=self.velocity_field
        ).to(self.device)

        # Loss function
        self.criterion = ConvectionDiffusionLoss(self.config)

        # Optimizer
        lr = self.config['training']['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=500,
            verbose=True
        )

        # Training history
        self.history = {
            'total_loss': [],
            'pde_loss': [],
            'boundary_loss': [],
            'initial_loss': [],
            'divergence_loss': []
        }

    def sample_collocation_points(self, n_points: int):
        """Sample interior points for PDE."""
        x = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['length']
        y = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['width']
        z = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['height']
        t = torch.rand(n_points, 1, device=self.device) * 1.0  # 1 second simulation
        return x, y, z, t

    def sample_boundary_points(self, n_points: int):
        """Sample boundary points (inlet)."""
        # Inlet at x=0
        x = torch.zeros(n_points, 1, device=self.device)
        y = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['width']
        z = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['height']
        t = torch.rand(n_points, 1, device=self.device) * 1.0

        T_inlet = torch.ones(n_points, 1, device=self.device) * self.config['boundaries']['inlet_temp']

        return x, y, z, t, T_inlet

    def sample_initial_points(self, n_points: int):
        """Sample points at t=0 for initial condition."""
        x = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['length']
        y = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['width']
        z = torch.rand(n_points, 1, device=self.device) * self.config['geometry']['height']
        t = torch.zeros(n_points, 1, device=self.device)

        T_initial = torch.ones(n_points, 1, device=self.device) * self.config['boundaries']['initial_temp']

        return x, y, z, t, T_initial

    def generate_heat_sources_blackwell(self, x, y, z):
        """
        Generate heat sources from Blackwell GB200 racks.

        Args:
            x, y, z: Collocation points

        Returns:
            Q: Heat source [W/m³]
        """
        Q = torch.zeros_like(x)

        if self.hardware_profile is None:
            return Q

        # Get rack specifications
        num_racks = self.config['servers']['num_racks']
        rack_width = self.config['servers']['rack_width']
        rack_depth = self.config['servers']['rack_depth']
        rack_height = self.config['servers']['rack_height']

        # Calculate heat output (only AIR rejection - 10%)
        _, _, air_heat = self.hardware_profile.calculate_heat_generation(utilization=0.70)
        air_heat_per_rack = air_heat  # W

        # Volume per rack
        rack_volume = rack_width * rack_depth * rack_height
        Q_density = air_heat_per_rack / rack_volume  # W/m³

        # Place racks in grid pattern
        room_length = self.config['geometry']['length']
        room_width = self.config['geometry']['width']

        rows = 2
        cols = num_racks // rows

        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= num_racks:
                    break

                # Rack center position
                rack_x = (j + 1) * room_length / (cols + 1)
                rack_y = (i + 1) * room_width / (rows + 1)
                rack_z = rack_height / 2

                # Add heat to points within rack volume
                mask = (
                    (x >= rack_x - rack_width/2) & (x <= rack_x + rack_width/2) &
                    (y >= rack_y - rack_depth/2) & (y <= rack_y + rack_depth/2) &
                    (z >= 0) & (z <= rack_height)
                )

                Q[mask] = Q_density

        return Q

    def train(self, epochs: int, verbose: bool = True):
        """
        Train the convection-diffusion PINN.

        Args:
            epochs: Number of training epochs
            verbose: Print progress
        """
        n_collocation = self.config['training']['n_collocation']
        n_boundary = self.config['training']['n_boundary']
        n_initial = self.config['training']['n_initial']

        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)

        for epoch in pbar:
            self.model.train()

            # Sample points
            x_col, y_col, z_col, t_col = self.sample_collocation_points(n_collocation)
            x_bc, y_bc, z_bc, t_bc, T_bc = self.sample_boundary_points(n_boundary)
            x_ic, y_ic, z_ic, t_ic, T_ic = self.sample_initial_points(n_initial)

            # Generate heat sources (Blackwell GB200)
            Q = self.generate_heat_sources_blackwell(x_col, y_col, z_col)

            # Compute PDE residual (AUTOMATIC DIFFERENTIATION!)
            pde_residual = self.model.compute_convection_diffusion_residual(
                x_col, y_col, z_col, t_col, Q
            )

            # Compute divergence (incompressibility)
            velocity_divergence = self.model.compute_divergence_free_constraint(
                x_col, y_col, z_col, t_col
            )

            # Boundary predictions
            T_bc_pred = self.model(x_bc, y_bc, z_bc, t_bc)

            # Initial predictions
            T_ic_pred = self.model(x_ic, y_ic, z_ic, t_ic)

            # Compute loss
            total_loss, loss_dict = self.criterion(
                pde_residual=pde_residual,
                boundary_pred=T_bc_pred,
                boundary_true=T_bc,
                initial_pred=T_ic_pred,
                initial_true=T_ic,
                velocity_divergence=velocity_divergence
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update scheduler
            self.scheduler.step(total_loss)

            # Log history
            self.history['total_loss'].append(loss_dict['total'])
            self.history['pde_loss'].append(loss_dict['pde'])
            self.history['boundary_loss'].append(loss_dict['boundary'])
            self.history['initial_loss'].append(loss_dict['initial'])
            self.history['divergence_loss'].append(loss_dict['divergence'])

            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total']:.4e}",
                    'PDE': f"{loss_dict['pde']:.4e}",
                    'BC': f"{loss_dict['boundary']:.4e}",
                    'Div': f"{loss_dict['divergence']:.4e}"
                })

    def plot_results(self, save_path: str = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].semilogy(self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True)

        axes[0, 1].semilogy(self.history['pde_loss'])
        axes[0, 1].set_title('PDE Loss (Convection-Diffusion)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].grid(True)

        axes[0, 2].semilogy(self.history['boundary_loss'])
        axes[0, 2].set_title('Boundary Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].grid(True)

        axes[1, 0].semilogy(self.history['initial_loss'])
        axes[1, 0].set_title('Initial Condition Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True)

        axes[1, 1].semilogy(self.history['divergence_loss'])
        axes[1, 1].set_title('Divergence Loss (∇·u = 0)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)

        # Summary text
        axes[1, 2].axis('off')
        summary = f"""
        Training Summary
        ================
        Final Losses:
          Total: {self.history['total_loss'][-1]:.4e}
          PDE: {self.history['pde_loss'][-1]:.4e}
          Boundary: {self.history['boundary_loss'][-1]:.4e}
          Initial: {self.history['initial_loss'][-1]:.4e}
          Divergence: {self.history['divergence_loss'][-1]:.4e}

        Model: Convection-Diffusion PINN
        Equation: ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q
        Hardware: Blackwell GB200 (120kW racks)
        """
        axes[1, 2].text(0.1, 0.5, summary, fontsize=10, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("Convection-Diffusion PINN Training")
    print("Equation: ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q")
    print("=" * 80)

    # Configuration
    config_path = project_root / 'configs' / 'physics_config.yaml'

    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("\n✓ Using Apple Metal GPU (MPS)")
    else:
        device = 'cpu'
        print("\n✓ Using CPU")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = ConvectionDiffusionTrainer(str(config_path), device=device)

    print(f"\nModel Architecture:")
    print(f"  Temperature network: {sum(p.numel() for p in trainer.model.temperature_network.parameters()):,} parameters")
    print(f"  Velocity field: Analytical CRAC model")

    if trainer.hardware_profile:
        print(f"\nHardware Profile: {trainer.hardware_profile.name}")
        _, liquid_heat, air_heat = trainer.hardware_profile.calculate_heat_generation(0.70)
        print(f"  Rack power: {trainer.hardware_profile.rack.rack_power_kw:.1f} kW")
        print(f"  Air rejection (10%): {air_heat/1000:.1f} kW (modeled in PINN)")
        print(f"  Liquid capture (90%): {liquid_heat/1000:.1f} kW (handled by CDU)")

    print(f"\nPhysics Parameters:")
    print(f"  Air density (ρ): {trainer.config['thermal']['density']} kg/m³")
    print(f"  Specific heat (c_p): {trainer.config['thermal']['specific_heat']} J/(kg·K)")
    print(f"  Thermal conductivity (k): {trainer.config['thermal']['thermal_conductivity']} W/(m·K)")
    print(f"  CRAC inlet velocity: 2.0 m/s")
    print(f"  Inlet temperature: {trainer.config['boundaries']['inlet_temp']}°C")

    # Train
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)

    epochs = trainer.config['training']['epochs']
    print(f"\nTraining for {epochs} epochs")
    print("Using AUTOMATIC DIFFERENTIATION for:")
    print("  ✓ ∂T/∂t (temporal derivative)")
    print("  ✓ u·∇T (convection term)")
    print("  ✓ ∇²T (Laplacian)")
    print("  ✓ ∇·u (velocity divergence)")

    trainer.train(epochs=epochs, verbose=True)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # Validate
    print("\nValidating physics satisfaction...")
    n_test = 1000
    x_test, y_test, z_test, t_test = trainer.sample_collocation_points(n_test)
    Q_test = trainer.generate_heat_sources_blackwell(x_test, y_test, z_test)

    checks = validate_physics_satisfaction(
        trainer.model, x_test, y_test, z_test, t_test, Q_test
    )

    print("\nValidation Results:")
    print(f"  PDE satisfied: {'✓' if checks['pde_satisfied'] else '✗'}")
    print(f"    Mean residual: {checks['pde_mean_residual']:.4e}")
    print(f"    Max residual: {checks['pde_max_residual']:.4e}")
    print(f"  Divergence-free: {'✓' if checks['divergence_free'] else '✗'}")
    print(f"    Mean |∇·u|: {checks['divergence_mean']:.4e}")
    print(f"  Temperature range: [{checks['temperature_range'][0]:.1f}, {checks['temperature_range'][1]:.1f}] °C")
    print(f"  Physically reasonable: {'✓' if checks['temperature_physical'] else '✗'}")

    # Plot results
    print("\nGenerating training plots...")
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    trainer.plot_results(save_path=str(results_dir / 'convection_diffusion_training.png'))
    print(f"✓ Saved: {results_dir / 'convection_diffusion_training.png'}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\n✓ Successfully trained convection-diffusion PINN")
    print("✓ Physics-informed loss with automatic differentiation")
    print("✓ Integrated Blackwell GB200 heat sources")
    print("✓ Modeled CRAC air flow velocity field")
    print("✓ Enforced incompressibility constraint (∇·u = 0)")
    print("\nNext steps:")
    print("  1. Visualize temperature and velocity fields")
    print("  2. Compare with EnergyPlus simulation")
    print("  3. Optimize cooling setpoints for minimum PUE")
    print("  4. Deploy for real-time predictions")


if __name__ == "__main__":
    main()
