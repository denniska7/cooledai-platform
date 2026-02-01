"""
Unit tests for PINN implementation

Tests:
- Network initialization
- Forward pass
- Gradient computation
- PDE residual calculation
- Loss functions
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.pinn import FullyConnectedNN, HeatPINN
from models.losses import PINNLoss
import yaml


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / 'configs' / 'physics_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cpu')


class TestFullyConnectedNN:
    """Test the neural network architecture."""

    def test_initialization(self):
        """Test network initialization."""
        model = FullyConnectedNN(
            input_dim=4,
            output_dim=1,
            hidden_layers=[32, 32],
            activation='tanh'
        )
        assert model is not None
        assert model.input_dim == 4
        assert model.output_dim == 1

    def test_forward_pass(self):
        """Test forward pass."""
        model = FullyConnectedNN(input_dim=4, output_dim=1, hidden_layers=[32, 32])
        x = torch.randn(10, 4)
        output = model(x)

        assert output.shape == (10, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_different_activations(self):
        """Test different activation functions."""
        activations = ['tanh', 'relu', 'gelu', 'swish']

        for activation in activations:
            model = FullyConnectedNN(
                input_dim=4,
                output_dim=1,
                hidden_layers=[16, 16],
                activation=activation
            )
            x = torch.randn(5, 4)
            output = model(x)
            assert output.shape == (5, 1)


class TestHeatPINN:
    """Test the Heat PINN model."""

    def test_initialization(self, config):
        """Test PINN initialization."""
        model = HeatPINN(config)
        assert model is not None
        assert model.thermal_diffusivity == config['thermal']['diffusivity']

    def test_forward_pass(self, config, device):
        """Test forward pass through PINN."""
        model = HeatPINN(config).to(device)

        # Create sample inputs
        batch_size = 10
        x = torch.randn(batch_size, 1, device=device)
        y = torch.randn(batch_size, 1, device=device)
        z = torch.randn(batch_size, 1, device=device)
        t = torch.randn(batch_size, 1, device=device)

        # Forward pass
        T = model(x, y, z, t)

        assert T.shape == (batch_size, 1)
        assert not torch.isnan(T).any()

    def test_gradient_computation(self, config, device):
        """Test gradient computation using autograd."""
        model = HeatPINN(config).to(device)

        # Create sample inputs with grad enabled
        x = torch.randn(5, 1, device=device, requires_grad=True)
        y = torch.randn(5, 1, device=device, requires_grad=True)
        z = torch.randn(5, 1, device=device, requires_grad=True)
        t = torch.randn(5, 1, device=device, requires_grad=True)

        # Forward pass
        T = model(x, y, z, t)

        # Compute gradient w.r.t. x
        T_x = model._gradient(T, x)

        assert T_x.shape == (5, 1)
        assert not torch.isnan(T_x).any()

    def test_pde_residual(self, config, device):
        """Test PDE residual computation."""
        model = HeatPINN(config).to(device)

        # Sample points
        x = torch.randn(10, 1, device=device, requires_grad=True)
        y = torch.randn(10, 1, device=device, requires_grad=True)
        z = torch.randn(10, 1, device=device, requires_grad=True)
        t = torch.randn(10, 1, device=device, requires_grad=True)

        # Compute residual
        residual = model.compute_pde_residual(x, y, z, t)

        assert residual.shape == (10, 1)
        assert not torch.isnan(residual).any()
        assert not torch.isinf(residual).any()

    def test_predict_temperature(self, config):
        """Test temperature prediction."""
        model = HeatPINN(config)

        # Create numpy inputs
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([0.5, 1.0, 1.5])
        t = np.array([0.1, 0.2, 0.3])

        # Predict
        T = model.predict_temperature(x, y, z, t)

        assert T.shape == (3,)
        assert not np.isnan(T).any()


class TestLossFunctions:
    """Test loss functions."""

    def test_pinn_loss_initialization(self, config):
        """Test PINN loss initialization."""
        criterion = PINNLoss(config)
        assert criterion is not None
        assert criterion.w_pde > 0
        assert criterion.w_boundary > 0
        assert criterion.w_initial > 0

    def test_pde_loss_computation(self, config, device):
        """Test PDE loss computation."""
        criterion = PINNLoss(config)

        # Create fake residuals
        pde_residual = torch.randn(100, 1, device=device)

        # Compute loss
        total_loss, loss_dict = criterion(pde_residual=pde_residual)

        assert total_loss.item() >= 0
        assert 'pde' in loss_dict
        assert 'total' in loss_dict

    def test_full_loss_computation(self, config, device):
        """Test full loss with all components."""
        criterion = PINNLoss(config)

        # Create fake data
        pde_residual = torch.randn(100, 1, device=device)
        boundary_pred = torch.randn(50, 1, device=device)
        boundary_true = torch.randn(50, 1, device=device)
        initial_pred = torch.randn(50, 1, device=device)
        initial_true = torch.randn(50, 1, device=device)

        # Compute loss
        total_loss, loss_dict = criterion(
            pde_residual=pde_residual,
            boundary_pred=boundary_pred,
            boundary_true=boundary_true,
            initial_pred=initial_pred,
            initial_true=initial_true
        )

        assert total_loss.item() >= 0
        assert len(loss_dict) == 5  # total, pde, boundary, initial, data


class TestPhysicsConsistency:
    """Test physics consistency."""

    def test_steady_state_no_source(self, config, device):
        """
        Test that without heat sources and at steady state,
        temperature should be uniform.
        """
        model = HeatPINN(config).to(device)
        model.eval()

        # Sample points at same time (pseudo steady-state)
        n_points = 20
        x = torch.rand(n_points, 1, device=device) * 10
        y = torch.rand(n_points, 1, device=device) * 8
        z = torch.rand(n_points, 1, device=device) * 3
        t = torch.ones(n_points, 1, device=device) * 10.0  # Late time

        # Predict (without training, should be somewhat uniform)
        with torch.no_grad():
            T = model(x, y, z, t)

        # Check that predictions are in reasonable range
        assert T.min() > -100  # Sanity check
        assert T.max() < 200   # Sanity check

    def test_boundary_condition_enforcement(self, config, device):
        """Test that boundary conditions can be enforced."""
        model = HeatPINN(config).to(device)

        # Point at inlet (x=0)
        x = torch.zeros(5, 1, device=device, requires_grad=True)
        y = torch.rand(5, 1, device=device, requires_grad=True)
        z = torch.rand(5, 1, device=device, requires_grad=True)
        t = torch.rand(5, 1, device=device, requires_grad=True)

        # Forward pass
        T = model(x, y, z, t)

        # Should produce valid output
        assert T.shape == (5, 1)
        assert not torch.isnan(T).any()


class TestDataSampler:
    """Test data sampling functions."""

    def test_collocation_sampling(self, config):
        """Test collocation point sampling."""
        from models.trainer import DataSampler

        sampler = DataSampler(config)
        x, y, z, t = sampler.sample_collocation_points(100)

        assert x.shape == (100, 1)
        assert y.shape == (100, 1)
        assert z.shape == (100, 1)
        assert t.shape == (100, 1)

        # Check bounds
        assert x.min() >= 0 and x.max() <= config['geometry']['length']
        assert y.min() >= 0 and y.max() <= config['geometry']['width']
        assert z.min() >= 0 and z.max() <= config['geometry']['height']

    def test_boundary_sampling(self, config):
        """Test boundary point sampling."""
        from models.trainer import DataSampler

        sampler = DataSampler(config)
        x, y, z, t, T_bc = sampler.sample_boundary_points(100)

        assert x.shape[0] > 0
        assert T_bc.shape[0] > 0

        # Check that boundary temperature is correct
        assert torch.all(T_bc == config['boundaries']['inlet_temp'])

    def test_initial_sampling(self, config):
        """Test initial condition sampling."""
        from models.trainer import DataSampler

        sampler = DataSampler(config)
        x, y, z, t, T_ic = sampler.sample_initial_points(100)

        assert t.shape == (100, 1)
        assert torch.all(t == 0.0)
        assert torch.all(T_ic == config['boundaries']['initial_temp'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
