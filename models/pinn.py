"""
Physics-Informed Neural Network for Heat Distribution Simulation

This module implements a PINN that solves the 3D transient heat equation:
    ∂T/∂t = α∇²T + Q/(ρ·c_p)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class FullyConnectedNN(nn.Module):
    """
    Fully connected neural network with customizable architecture.

    Args:
        input_dim: Dimension of input (typically 4 for [x, y, z, t])
        output_dim: Dimension of output (typically 1 for temperature)
        hidden_layers: List of hidden layer sizes
        activation: Activation function ('tanh', 'relu', 'gelu', 'swish')
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 1,
        hidden_layers: list = [64, 64, 64, 64, 64],
        activation: str = 'tanh'
    ):
        super(FullyConnectedNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network layers
        layers = []
        in_features = input_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self._get_activation(activation))
            in_features = hidden_size

        # Output layer
        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.Tanh())

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return self.network(x)


class HeatPINN(nn.Module):
    """
    Physics-Informed Neural Network for heat distribution in data centers.

    Solves: ∂T/∂t = α∇²T + Q/(ρ·c_p)

    Args:
        config: Dictionary containing physics and network configuration
    """

    def __init__(self, config: Dict):
        super(HeatPINN, self).__init__()

        # Extract configuration
        self.thermal_diffusivity = config['thermal']['diffusivity']
        self.density = config['thermal']['density']
        self.specific_heat = config['thermal']['specific_heat']

        # Network architecture
        net_config = config['network']
        self.network = FullyConnectedNN(
            input_dim=net_config['input_dim'],
            output_dim=net_config['output_dim'],
            hidden_layers=net_config['hidden_layers'],
            activation=net_config['activation']
        )

        # Domain bounds for normalization
        self.domain_bounds = {
            'x': [0, config['geometry']['length']],
            'y': [0, config['geometry']['width']],
            'z': [0, config['geometry']['height']],
            't': [0, 1.0]  # Will be set during training
        }

        # Reference temperature for normalization
        self.T_ref = config['boundaries']['initial_temp']
        self.T_scale = 50.0  # Temperature scale for normalization

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute temperature T(x, y, z, t).

        Args:
            x, y, z: Spatial coordinates [batch_size, 1]
            t: Time coordinate [batch_size, 1]

        Returns:
            Temperature field [batch_size, 1]
        """
        # Normalize inputs
        x_norm = self._normalize(x, 'x')
        y_norm = self._normalize(y, 'y')
        z_norm = self._normalize(z, 'z')
        t_norm = self._normalize(t, 't')

        # Concatenate inputs
        inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)

        # Network prediction (normalized temperature)
        T_norm = self.network(inputs)

        # Denormalize temperature
        T = T_norm * self.T_scale + self.T_ref

        return T

    def _normalize(self, x: torch.Tensor, coord: str) -> torch.Tensor:
        """Normalize coordinate to [-1, 1]."""
        x_min, x_max = self.domain_bounds[coord]
        return 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    def compute_pde_residual(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        Q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the PDE residual: f = ∂T/∂t - α∇²T - Q/(ρ·c_p)

        Args:
            x, y, z: Spatial coordinates [batch_size, 1] (requires_grad=True)
            t: Time coordinate [batch_size, 1] (requires_grad=True)
            Q: Heat source term [batch_size, 1] in W/m³ (optional)

        Returns:
            PDE residual [batch_size, 1]
        """
        # Ensure gradients are enabled
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)

        # Forward pass
        T = self.forward(x, y, z, t)

        # Compute first-order derivatives
        T_t = self._gradient(T, t)
        T_x = self._gradient(T, x)
        T_y = self._gradient(T, y)
        T_z = self._gradient(T, z)

        # Compute second-order derivatives (Laplacian)
        T_xx = self._gradient(T_x, x)
        T_yy = self._gradient(T_y, y)
        T_zz = self._gradient(T_z, z)

        laplacian = T_xx + T_yy + T_zz

        # Heat source term
        if Q is None:
            Q = torch.zeros_like(T)

        Q_term = Q / (self.density * self.specific_heat)

        # PDE residual
        residual = T_t - self.thermal_diffusivity * laplacian - Q_term

        return residual

    def _gradient(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient dy/dx using automatic differentiation.

        Args:
            y: Output tensor [batch_size, 1]
            x: Input tensor [batch_size, 1]

        Returns:
            Gradient tensor [batch_size, 1]
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

    def predict_temperature(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Predict temperature at given coordinates (inference mode).

        Args:
            x, y, z, t: Numpy arrays of coordinates

        Returns:
            Temperature predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            z_t = torch.tensor(z, dtype=torch.float32).reshape(-1, 1)
            t_t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

            T = self.forward(x_t, y_t, z_t, t_t)

        return T.cpu().numpy().flatten()


class AdaptiveWeightedLoss:
    """
    Adaptive loss weighting strategy for balancing multiple loss terms.

    Uses a simple exponential moving average to track loss magnitudes
    and adjust weights accordingly.
    """

    def __init__(self, initial_weights: Dict[str, float], alpha: float = 0.9):
        """
        Args:
            initial_weights: Dictionary of initial loss weights
            alpha: EMA smoothing factor (0 < alpha < 1)
        """
        self.weights = initial_weights.copy()
        self.loss_ema = {key: 1.0 for key in initial_weights.keys()}
        self.alpha = alpha

    def update(self, losses: Dict[str, float]):
        """Update EMA and weights based on current losses."""
        for key in losses.keys():
            if key in self.loss_ema:
                # Update EMA
                self.loss_ema[key] = (
                    self.alpha * self.loss_ema[key] +
                    (1 - self.alpha) * losses[key]
                )

    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights


def initialize_weights_custom(layer, gain=1.0):
    """
    Custom weight initialization for better convergence.

    Args:
        layer: Neural network layer
        gain: Scaling factor for initialization
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
