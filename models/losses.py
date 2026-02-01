"""
Loss functions for Physics-Informed Neural Networks

This module defines various loss components:
- PDE residual loss
- Boundary condition loss
- Initial condition loss
- Data loss (if measurements available)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class PINNLoss(nn.Module):
    """
    Combined loss function for Physics-Informed Neural Network.

    Total loss = λ₁·L_pde + λ₂·L_boundary + λ₃·L_initial + λ₄·L_data
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary containing loss weights
        """
        super(PINNLoss, self).__init__()

        # Loss weights
        weights = config['training']['loss_weights']
        self.w_pde = weights['pde_loss']
        self.w_boundary = weights['boundary_loss']
        self.w_initial = weights['initial_loss']
        self.w_data = weights.get('data_loss', 10.0)

        # MSE loss function
        self.mse = nn.MSELoss()

    def forward(
        self,
        pde_residual: torch.Tensor,
        boundary_pred: Optional[torch.Tensor] = None,
        boundary_true: Optional[torch.Tensor] = None,
        initial_pred: Optional[torch.Tensor] = None,
        initial_true: Optional[torch.Tensor] = None,
        data_pred: Optional[torch.Tensor] = None,
        data_true: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual loss components.

        Args:
            pde_residual: PDE residual [N_collocation, 1]
            boundary_pred: Predicted temperature at boundaries [N_boundary, 1]
            boundary_true: True boundary conditions [N_boundary, 1]
            initial_pred: Predicted temperature at t=0 [N_initial, 1]
            initial_true: Initial conditions [N_initial, 1]
            data_pred: Predicted temperature at measurement points [N_data, 1]
            data_true: Measured temperature data [N_data, 1]

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary containing individual loss values
        """
        # PDE loss (physics residual)
        loss_pde = torch.mean(pde_residual ** 2)

        # Boundary condition loss
        if boundary_pred is not None and boundary_true is not None:
            loss_boundary = self.mse(boundary_pred, boundary_true)
        else:
            loss_boundary = torch.tensor(0.0, device=pde_residual.device)

        # Initial condition loss
        if initial_pred is not None and initial_true is not None:
            loss_initial = self.mse(initial_pred, initial_true)
        else:
            loss_initial = torch.tensor(0.0, device=pde_residual.device)

        # Data loss (if measurements available)
        if data_pred is not None and data_true is not None:
            loss_data = self.mse(data_pred, data_true)
        else:
            loss_data = torch.tensor(0.0, device=pde_residual.device)

        # Total weighted loss
        total_loss = (
            self.w_pde * loss_pde +
            self.w_boundary * loss_boundary +
            self.w_initial * loss_initial +
            self.w_data * loss_data
        )

        # Create loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'boundary': loss_boundary.item() if isinstance(loss_boundary, torch.Tensor) else 0.0,
            'initial': loss_initial.item() if isinstance(loss_initial, torch.Tensor) else 0.0,
            'data': loss_data.item() if isinstance(loss_data, torch.Tensor) else 0.0
        }

        return total_loss, loss_dict


class RelativeLoss(nn.Module):
    """
    Relative L2 loss: ||pred - true||₂ / ||true||₂

    Useful when values span multiple orders of magnitude.
    """

    def __init__(self, epsilon: float = 1e-8):
        super(RelativeLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Compute relative L2 loss.

        Args:
            pred: Predicted values
            true: True values

        Returns:
            Relative L2 loss
        """
        diff = pred - true
        numerator = torch.sqrt(torch.mean(diff ** 2))
        denominator = torch.sqrt(torch.mean(true ** 2)) + self.epsilon
        return numerator / denominator


class CausalLoss(nn.Module):
    """
    Causal loss for time-dependent problems.

    Weights the loss more heavily for earlier time steps to enforce
    temporal causality in the solution.
    """

    def __init__(self, decay_rate: float = 0.9):
        """
        Args:
            decay_rate: Exponential decay rate for time weighting (0 < decay_rate < 1)
        """
        super(CausalLoss, self).__init__()
        self.decay_rate = decay_rate

    def forward(
        self,
        residual: torch.Tensor,
        t: torch.Tensor,
        t_max: float
    ) -> torch.Tensor:
        """
        Compute causally-weighted loss.

        Args:
            residual: PDE residual [N, 1]
            t: Time coordinates [N, 1]
            t_max: Maximum time value for normalization

        Returns:
            Weighted loss
        """
        # Normalize time to [0, 1]
        t_norm = t / t_max

        # Compute exponential weights (higher weight for earlier times)
        weights = torch.exp(-self.decay_rate * t_norm)

        # Weighted MSE
        weighted_residual = weights * (residual ** 2)
        return torch.mean(weighted_residual)


class AdaptiveLossBalancing:
    """
    Adaptive loss balancing using gradient statistics.

    Balances different loss terms by monitoring their gradient magnitudes.
    Based on the method from "Gradient Surgery for Multi-Task Learning".
    """

    def __init__(self, num_losses: int, alpha: float = 0.16):
        """
        Args:
            num_losses: Number of loss terms to balance
            alpha: Update rate for exponential moving average
        """
        self.num_losses = num_losses
        self.alpha = alpha
        self.loss_weights = torch.ones(num_losses)
        self.grad_norms = torch.ones(num_losses)

    def update_weights(self, losses: list, model: nn.Module):
        """
        Update loss weights based on gradient norms.

        Args:
            losses: List of individual loss tensors
            model: Neural network model
        """
        # Compute gradient norms for each loss
        grad_norms = []

        for i, loss in enumerate(losses):
            # Zero gradients
            model.zero_grad()

            # Backward pass for this loss
            loss.backward(retain_graph=True)

            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            grad_norms.append(total_norm)

        grad_norms = torch.tensor(grad_norms)

        # Update exponential moving average
        self.grad_norms = (
            self.alpha * grad_norms +
            (1 - self.alpha) * self.grad_norms
        )

        # Compute weights inversely proportional to gradient norms
        mean_norm = torch.mean(self.grad_norms)
        self.loss_weights = mean_norm / (self.grad_norms + 1e-8)

        # Normalize weights to sum to num_losses
        self.loss_weights = (
            self.loss_weights * self.num_losses / torch.sum(self.loss_weights)
        )

        return self.loss_weights


class HardConstraintLoss:
    """
    Helper functions for hard constraint enforcement.

    Instead of soft penalties, modifies the network output to exactly
    satisfy boundary/initial conditions.
    """

    @staticmethod
    def enforce_initial_condition(
        network_output: torch.Tensor,
        t: torch.Tensor,
        T_initial: float
    ) -> torch.Tensor:
        """
        Enforce initial condition: T(x,y,z,0) = T_initial

        Modified output: T_modified = T_initial + t * network_output

        Args:
            network_output: Raw network output
            t: Time coordinate [N, 1]
            T_initial: Initial temperature

        Returns:
            Modified output satisfying T(t=0) = T_initial
        """
        return T_initial + t * network_output

    @staticmethod
    def enforce_dirichlet_boundary(
        network_output: torch.Tensor,
        x: torch.Tensor,
        x_boundary: float,
        T_boundary: float,
        L: float
    ) -> torch.Tensor:
        """
        Enforce Dirichlet boundary condition at x = x_boundary.

        Args:
            network_output: Raw network output
            x: Spatial coordinate [N, 1]
            x_boundary: Boundary location
            T_boundary: Boundary temperature
            L: Domain length for normalization

        Returns:
            Modified output satisfying boundary condition
        """
        # Distance from boundary (normalized)
        distance = torch.abs(x - x_boundary) / L

        # Modified output
        return T_boundary + distance * network_output


def compute_gradient_penalty(
    model: nn.Module,
    x: torch.Tensor,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """
    Compute gradient penalty for smoothness regularization.

    Encourages smoother temperature fields by penalizing large gradients.

    Args:
        model: PINN model
        x: Input coordinates [N, 4]
        lambda_gp: Weight for gradient penalty

    Returns:
        Gradient penalty term
    """
    # Forward pass
    output = model(x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4])

    # Compute gradients w.r.t input
    gradients = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]

    # Gradient penalty (L2 norm)
    penalty = lambda_gp * torch.mean(gradients ** 2)

    return penalty
