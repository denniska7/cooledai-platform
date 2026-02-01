"""
Training pipeline for Physics-Informed Neural Network

This module handles:
- Data sampling (collocation, boundary, initial condition points)
- Heat source generation
- Training loop with logging
- Model checkpointing
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from .pinn import HeatPINN
from .losses import PINNLoss


class DataSampler:
    """
    Generates training points for PINN.

    Types of points:
    1. Collocation points (interior domain) - for PDE residual
    2. Boundary points - for boundary conditions
    3. Initial condition points - for t=0
    4. Heat source points - server rack locations
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.geometry = config['geometry']
        self.boundaries = config['boundaries']
        self.servers = config['servers']

        # Domain bounds
        self.x_bounds = [0, self.geometry['length']]
        self.y_bounds = [0, self.geometry['width']]
        self.z_bounds = [0, self.geometry['height']]
        self.t_bounds = [0, 1.0]  # Will be updated based on simulation time

    def set_time_bounds(self, t_max: float):
        """Set maximum simulation time."""
        self.t_bounds[1] = t_max

    def sample_collocation_points(
        self,
        n_points: int,
        method: str = 'random'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample interior domain points for PDE residual.

        Args:
            n_points: Number of points to sample
            method: Sampling method ('random', 'grid', 'lhs')

        Returns:
            x, y, z, t tensors of shape [n_points, 1]
        """
        if method == 'random':
            x = self._random_uniform(n_points, self.x_bounds)
            y = self._random_uniform(n_points, self.y_bounds)
            z = self._random_uniform(n_points, self.z_bounds)
            t = self._random_uniform(n_points, self.t_bounds)

        elif method == 'grid':
            # Create uniform grid
            n_per_dim = int(np.ceil(n_points ** (1/4)))
            x_grid = np.linspace(*self.x_bounds, n_per_dim)
            y_grid = np.linspace(*self.y_bounds, n_per_dim)
            z_grid = np.linspace(*self.z_bounds, n_per_dim)
            t_grid = np.linspace(*self.t_bounds, n_per_dim)

            X, Y, Z, T = np.meshgrid(x_grid, y_grid, z_grid, t_grid)
            x = torch.tensor(X.flatten()[:n_points], dtype=torch.float32).reshape(-1, 1)
            y = torch.tensor(Y.flatten()[:n_points], dtype=torch.float32).reshape(-1, 1)
            z = torch.tensor(Z.flatten()[:n_points], dtype=torch.float32).reshape(-1, 1)
            t = torch.tensor(T.flatten()[:n_points], dtype=torch.float32).reshape(-1, 1)

        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return x, y, z, t

    def sample_boundary_points(
        self,
        n_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points on domain boundaries.

        Returns:
            x, y, z, t, boundary_values tensors
        """
        points_per_boundary = n_points // 6  # 6 faces

        x_list, y_list, z_list, t_list, T_bc_list = [], [], [], [], []

        # Inlet face (x=0) - cold air supply
        x = torch.zeros(points_per_boundary, 1)
        y = self._random_uniform(points_per_boundary, self.y_bounds)
        z = self._random_uniform(points_per_boundary, self.z_bounds)
        t = self._random_uniform(points_per_boundary, self.t_bounds)
        T_bc = torch.full((points_per_boundary, 1), self.boundaries['inlet_temp'])

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        t_list.append(t)
        T_bc_list.append(T_bc)

        # Other boundaries (Neumann - handled separately, or fixed temp)
        # For simplicity, we'll focus on the inlet boundary
        # You can extend this for outlet, walls, floor, ceiling

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        z = torch.cat(z_list, dim=0)
        t = torch.cat(t_list, dim=0)
        T_bc = torch.cat(T_bc_list, dim=0)

        return x, y, z, t, T_bc

    def sample_initial_points(
        self,
        n_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points at t=0 for initial condition.

        Returns:
            x, y, z, t, T_initial tensors
        """
        x = self._random_uniform(n_points, self.x_bounds)
        y = self._random_uniform(n_points, self.y_bounds)
        z = self._random_uniform(n_points, self.z_bounds)
        t = torch.zeros(n_points, 1)
        T_initial = torch.full((n_points, 1), self.boundaries['initial_temp'])

        return x, y, z, t, T_initial

    def generate_heat_sources(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate heat source term Q(x,y,z) from server racks.

        Args:
            x, y, z: Spatial coordinates [N, 1]

        Returns:
            Heat source Q in W/m³ [N, 1]
        """
        Q = torch.zeros_like(x)

        # Generate random server rack positions
        num_racks = self.servers['num_racks']
        power_per_rack = self.servers['power_per_rack']

        rack_width = self.servers['rack_width']
        rack_depth = self.servers['rack_depth']
        rack_height = self.servers['rack_height']
        rack_volume = rack_width * rack_depth * rack_height

        # Volumetric heat generation
        Q_rack = power_per_rack / rack_volume

        # Random rack positions (simplified - in practice, use actual layout)
        np.random.seed(42)  # For reproducibility
        rack_positions = []

        for i in range(num_racks):
            rack_x = np.random.uniform(1.0, self.geometry['length'] - 1.0)
            rack_y = np.random.uniform(1.0, self.geometry['width'] - 1.0)
            rack_z = 0.0  # Floor level

            rack_positions.append((rack_x, rack_y, rack_z))

        # Add heat sources at rack locations
        for rack_x, rack_y, rack_z in rack_positions:
            # Points within rack volume
            mask = (
                (x >= rack_x - rack_width/2) & (x <= rack_x + rack_width/2) &
                (y >= rack_y - rack_depth/2) & (y <= rack_y + rack_depth/2) &
                (z >= rack_z) & (z <= rack_z + rack_height)
            )

            Q[mask] = Q_rack

        return Q

    def _random_uniform(
        self,
        n_points: int,
        bounds: List[float]
    ) -> torch.Tensor:
        """Sample uniformly in range [bounds[0], bounds[1]]."""
        return torch.rand(n_points, 1) * (bounds[1] - bounds[0]) + bounds[0]


class PINNTrainer:
    """
    Training pipeline for Heat PINN.
    """

    def __init__(self, config_path: str, device: str = 'cpu'):
        """
        Args:
            config_path: Path to YAML configuration file
            device: Device for training ('cpu', 'cuda', 'mps')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device)

        # Initialize model
        self.model = HeatPINN(self.config).to(self.device)

        # Initialize loss function
        self.criterion = PINNLoss(self.config)

        # Initialize optimizer
        lr = self.config['training']['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=500,
            verbose=True
        )

        # Data sampler
        self.sampler = DataSampler(self.config)
        self.sampler.set_time_bounds(1.0)  # 1 second simulation

        # Training history
        self.history = {
            'total_loss': [],
            'pde_loss': [],
            'boundary_loss': [],
            'initial_loss': []
        }

    def train(
        self,
        epochs: int,
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Train the PINN model.

        Args:
            epochs: Number of training epochs
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
        """
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        n_collocation = self.config['training']['n_collocation']
        n_boundary = self.config['training']['n_boundary']
        n_initial = self.config['training']['n_initial']

        # Progress bar
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)

        for epoch in pbar:
            self.model.train()

            # Sample training points
            x_col, y_col, z_col, t_col = self.sampler.sample_collocation_points(
                n_collocation
            )
            x_bc, y_bc, z_bc, t_bc, T_bc = self.sampler.sample_boundary_points(
                n_boundary
            )
            x_ic, y_ic, z_ic, t_ic, T_ic = self.sampler.sample_initial_points(
                n_initial
            )

            # Move to device
            x_col = x_col.to(self.device)
            y_col = y_col.to(self.device)
            z_col = z_col.to(self.device)
            t_col = t_col.to(self.device)

            x_bc = x_bc.to(self.device)
            y_bc = y_bc.to(self.device)
            z_bc = z_bc.to(self.device)
            t_bc = t_bc.to(self.device)
            T_bc = T_bc.to(self.device)

            x_ic = x_ic.to(self.device)
            y_ic = y_ic.to(self.device)
            z_ic = z_ic.to(self.device)
            t_ic = t_ic.to(self.device)
            T_ic = T_ic.to(self.device)

            # Generate heat sources
            Q = self.sampler.generate_heat_sources(x_col, y_col, z_col)
            Q = Q.to(self.device)

            # Compute PDE residual
            pde_residual = self.model.compute_pde_residual(
                x_col, y_col, z_col, t_col, Q
            )

            # Boundary predictions
            T_bc_pred = self.model(x_bc, y_bc, z_bc, t_bc)

            # Initial condition predictions
            T_ic_pred = self.model(x_ic, y_ic, z_ic, t_ic)

            # Compute loss
            total_loss, loss_dict = self.criterion(
                pde_residual=pde_residual,
                boundary_pred=T_bc_pred,
                boundary_true=T_bc,
                initial_pred=T_ic_pred,
                initial_true=T_ic
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

            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total']:.4e}",
                    'PDE': f"{loss_dict['pde']:.4e}",
                    'BC': f"{loss_dict['boundary']:.4e}",
                    'IC': f"{loss_dict['initial']:.4e}"
                })

            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % 1000 == 0:
                self.save_checkpoint(
                    checkpoint_path / f"checkpoint_epoch_{epoch+1}.pt",
                    epoch
                )

        # Save final model
        if checkpoint_dir:
            self.save_checkpoint(
                checkpoint_path / "final_model.pt",
                epochs
            )

    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']

    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].semilogy(self.history['total_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)

        axes[0, 1].semilogy(self.history['pde_loss'])
        axes[0, 1].set_title('PDE Residual Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)

        axes[1, 0].semilogy(self.history['boundary_loss'])
        axes[1, 0].set_title('Boundary Condition Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)

        axes[1, 1].semilogy(self.history['initial_loss'])
        axes[1, 1].set_title('Initial Condition Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()
