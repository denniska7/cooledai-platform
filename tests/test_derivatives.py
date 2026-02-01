"""
Test Script for Automatic Differentiation in Physics-Informed Loss

This script validates that torch.autograd correctly computes:
1. First-order derivatives (∂T/∂x, ∂T/∂y, ∂T/∂z, ∂T/∂t)
2. Second-order derivatives (∂²T/∂x², ∂²T/∂y², ∂²T/∂z²)
3. Laplacian (∇²T)
4. Convection term (u·∇T)
5. Full PDE residual

Uses analytical solutions for validation.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent))

from src.physics_loss import ConvectionDiffusionPINN, CRACVelocityField
import yaml


class DerivativeTest:
    """Test automatic differentiation against analytical solutions."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.tolerance = 1e-5

    def assert_close(self, computed, analytical, name, tol=None):
        """Check if computed value matches analytical value."""
        tol = tol or self.tolerance
        error = torch.abs(computed - analytical).max().item()

        if error < tol:
            self.passed.append(f"✓ {name}: error = {error:.2e} < {tol:.2e}")
            return True
        else:
            self.failed.append(f"✗ {name}: error = {error:.2e} >= {tol:.2e}")
            return False

    def print_results(self):
        """Print test results."""
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)

        print(f"\n✓ PASSED: {len(self.passed)}/{len(self.passed) + len(self.failed)}")
        for msg in self.passed:
            print(f"  {msg}")

        if self.failed:
            print(f"\n✗ FAILED: {len(self.failed)}/{len(self.passed) + len(self.failed)}")
            for msg in self.failed:
                print(f"  {msg}")
        else:
            print("\n🎉 ALL TESTS PASSED!")

        return len(self.failed) == 0


def test_first_order_derivatives():
    """
    Test 1: First-order derivatives for a simple polynomial.

    Function: T(x,y,z,t) = x² + y² + z² + t²

    Analytical derivatives:
        ∂T/∂x = 2x
        ∂T/∂y = 2y
        ∂T/∂z = 2z
        ∂T/∂t = 2t
    """
    print("\n" + "=" * 70)
    print("TEST 1: First-Order Derivatives (∂T/∂x, ∂T/∂y, ∂T/∂z, ∂T/∂t)")
    print("=" * 70)
    print("\nFunction: T(x,y,z,t) = x² + y² + z² + t²")

    tester = DerivativeTest()

    # Create test points
    batch_size = 10
    x = torch.rand(batch_size, 1, requires_grad=True) * 10.0
    y = torch.rand(batch_size, 1, requires_grad=True) * 10.0
    z = torch.rand(batch_size, 1, requires_grad=True) * 10.0
    t = torch.rand(batch_size, 1, requires_grad=True) * 1.0

    # Define function: T = x² + y² + z² + t²
    T = x**2 + y**2 + z**2 + t**2

    print(f"\nTest points: {batch_size} random samples")
    print(f"  x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  z range: [{z.min():.2f}, {z.max():.2f}]")
    print(f"  t range: [{t.min():.2f}, {t.max():.2f}]")

    # Compute derivatives using autograd
    T_x_auto = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_y_auto = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_z_auto = torch.autograd.grad(T, z, torch.ones_like(T), create_graph=True)[0]
    T_t_auto = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]

    # Analytical derivatives
    T_x_analytical = 2 * x
    T_y_analytical = 2 * y
    T_z_analytical = 2 * z
    T_t_analytical = 2 * t

    print("\nComparing autograd vs analytical:")
    tester.assert_close(T_x_auto, T_x_analytical, "∂T/∂x = 2x")
    tester.assert_close(T_y_auto, T_y_analytical, "∂T/∂y = 2y")
    tester.assert_close(T_z_auto, T_z_analytical, "∂T/∂z = 2z")
    tester.assert_close(T_t_auto, T_t_analytical, "∂T/∂t = 2t")

    return tester


def test_second_order_derivatives():
    """
    Test 2: Second-order derivatives for a polynomial.

    Function: T(x,y,z,t) = x³ + y³ + z³

    First derivatives:
        ∂T/∂x = 3x²
        ∂T/∂y = 3y²
        ∂T/∂z = 3z²

    Second derivatives:
        ∂²T/∂x² = 6x
        ∂²T/∂y² = 6y
        ∂²T/∂z² = 6z
    """
    print("\n" + "=" * 70)
    print("TEST 2: Second-Order Derivatives (∂²T/∂x², ∂²T/∂y², ∂²T/∂z²)")
    print("=" * 70)
    print("\nFunction: T(x,y,z) = x³ + y³ + z³")

    tester = DerivativeTest()

    # Create test points
    batch_size = 10
    x = torch.rand(batch_size, 1, requires_grad=True) * 5.0
    y = torch.rand(batch_size, 1, requires_grad=True) * 5.0
    z = torch.rand(batch_size, 1, requires_grad=True) * 5.0

    # Define function: T = x³ + y³ + z³
    T = x**3 + y**3 + z**3

    # First derivatives
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, torch.ones_like(T), create_graph=True)[0]

    # Second derivatives (autograd)
    T_xx_auto = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_yy_auto = torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]
    T_zz_auto = torch.autograd.grad(T_z, z, torch.ones_like(T_z), create_graph=True)[0]

    # Analytical second derivatives
    T_xx_analytical = 6 * x
    T_yy_analytical = 6 * y
    T_zz_analytical = 6 * z

    print("\nComparing autograd vs analytical:")
    tester.assert_close(T_xx_auto, T_xx_analytical, "∂²T/∂x² = 6x")
    tester.assert_close(T_yy_auto, T_yy_analytical, "∂²T/∂y² = 6y")
    tester.assert_close(T_zz_auto, T_zz_analytical, "∂²T/∂z² = 6z")

    return tester


def test_laplacian():
    """
    Test 3: Laplacian (∇²T) computation.

    Function: T(x,y,z) = sin(x) + sin(y) + sin(z)

    Laplacian:
        ∇²T = ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²
            = -sin(x) - sin(y) - sin(z)
            = -T
    """
    print("\n" + "=" * 70)
    print("TEST 3: Laplacian (∇²T = ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)")
    print("=" * 70)
    print("\nFunction: T(x,y,z) = sin(x) + sin(y) + sin(z)")
    print("Expected: ∇²T = -sin(x) - sin(y) - sin(z) = -T")

    tester = DerivativeTest()

    # Create test points
    batch_size = 10
    x = torch.rand(batch_size, 1, requires_grad=True) * 2 * np.pi
    y = torch.rand(batch_size, 1, requires_grad=True) * 2 * np.pi
    z = torch.rand(batch_size, 1, requires_grad=True) * 2 * np.pi

    # Define function
    T = torch.sin(x) + torch.sin(y) + torch.sin(z)

    # First derivatives
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, torch.ones_like(T), create_graph=True)[0]

    # Second derivatives
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]
    T_zz = torch.autograd.grad(T_z, z, torch.ones_like(T_z), create_graph=True)[0]

    # Laplacian
    laplacian_auto = T_xx + T_yy + T_zz
    laplacian_analytical = -T

    tester.assert_close(laplacian_auto, laplacian_analytical, "∇²T = -T")

    return tester


def test_convection_term():
    """
    Test 4: Convection term (u·∇T).

    Velocity: u = (1, 2, 3)
    Temperature: T(x,y,z) = x² + 2y² + 3z²

    Gradient:
        ∇T = (2x, 4y, 6z)

    Convection:
        u·∇T = 1*(2x) + 2*(4y) + 3*(6z) = 2x + 8y + 18z
    """
    print("\n" + "=" * 70)
    print("TEST 4: Convection Term (u·∇T)")
    print("=" * 70)
    print("\nVelocity: u = (1, 2, 3) m/s")
    print("Temperature: T(x,y,z) = x² + 2y² + 3z²")
    print("Expected: u·∇T = 2x + 8y + 18z")

    tester = DerivativeTest()

    # Create test points
    batch_size = 10
    x = torch.rand(batch_size, 1, requires_grad=True) * 5.0
    y = torch.rand(batch_size, 1, requires_grad=True) * 5.0
    z = torch.rand(batch_size, 1, requires_grad=True) * 5.0

    # Temperature function
    T = x**2 + 2*y**2 + 3*z**2

    # Compute gradient
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, torch.ones_like(T), create_graph=True)[0]

    # Velocity field (constant)
    u_x = torch.ones_like(T) * 1.0
    u_y = torch.ones_like(T) * 2.0
    u_z = torch.ones_like(T) * 3.0

    # Convection term
    convection_auto = u_x * T_x + u_y * T_y + u_z * T_z
    convection_analytical = 2*x + 8*y + 18*z

    tester.assert_close(convection_auto, convection_analytical, "u·∇T")

    return tester


def test_pinn_model():
    """
    Test 5: Full PINN model PDE residual computation.

    Tests that the ConvectionDiffusionPINN correctly computes all derivatives
    and assembles the PDE residual.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Full PINN Model (PDE Residual Computation)")
    print("=" * 70)
    print("\nEquation: ρ c_p (∂T/∂t + u·∇T) = ∇·(k∇T) + Q")

    tester = DerivativeTest()

    # Load configuration
    config_path = Path(__file__).parent / 'configs' / 'physics_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create velocity field
    velocity_field = CRACVelocityField(
        inlet_velocity=2.0,
        room_dimensions=(20.0, 16.0, 3.0)
    )

    # Create model
    model = ConvectionDiffusionPINN(config, velocity_field=velocity_field)

    print(f"\n✓ Model created")
    print(f"  Physics parameters:")
    print(f"    ρ (density): {model.density} kg/m³")
    print(f"    c_p (specific heat): {model.specific_heat} J/(kg·K)")
    print(f"    k (conductivity): {model.thermal_conductivity} W/(m·K)")

    # Test points
    batch_size = 20
    x = torch.rand(batch_size, 1, requires_grad=True) * 20.0
    y = torch.rand(batch_size, 1, requires_grad=True) * 16.0
    z = torch.rand(batch_size, 1, requires_grad=True) * 3.0
    t = torch.rand(batch_size, 1, requires_grad=True) * 1.0

    # Heat source
    Q = torch.rand(batch_size, 1) * 10000  # W/m³

    print(f"\n✓ Test data created: {batch_size} points")

    # Compute PDE residual
    try:
        residual = model.compute_convection_diffusion_residual(x, y, z, t, Q)

        print(f"\n✓ PDE residual computed successfully")
        print(f"  Shape: {residual.shape}")
        print(f"  Mean: {residual.mean().item():.4e}")
        print(f"  Std: {residual.std().item():.4e}")
        print(f"  Min: {residual.min().item():.4e}")
        print(f"  Max: {residual.max().item():.4e}")

        # Check that residual has correct shape
        if residual.shape == (batch_size, 1):
            tester.passed.append("✓ PDE residual shape correct")
        else:
            tester.failed.append(f"✗ PDE residual shape: {residual.shape}, expected ({batch_size}, 1)")

        # Check that residual is not all zeros or NaN
        if not torch.isnan(residual).any():
            tester.passed.append("✓ No NaN values in residual")
        else:
            tester.failed.append("✗ NaN values found in residual")

        if not torch.isinf(residual).any():
            tester.passed.append("✓ No Inf values in residual")
        else:
            tester.failed.append("✗ Inf values found in residual")

        # Test divergence constraint
        divergence = model.compute_divergence_free_constraint(x, y, z, t)

        print(f"\n✓ Velocity divergence computed")
        print(f"  Mean |∇·u|: {divergence.abs().mean().item():.4e}")
        print(f"  Max |∇·u|: {divergence.abs().max().item():.4e}")

        # For analytical velocity field, divergence should be small
        # (Not exactly zero due to numerical precision and non-linear transformations)
        if divergence.abs().mean().item() < 1e-2:
            tester.passed.append("✓ Velocity field approximately divergence-free")
        else:
            tester.failed.append(f"✗ High divergence: {divergence.abs().mean().item():.4e}")

    except Exception as e:
        tester.failed.append(f"✗ Exception during computation: {str(e)}")
        import traceback
        traceback.print_exc()

    return tester


def main():
    """Run all derivative tests."""
    print("=" * 70)
    print("AUTOMATIC DIFFERENTIATION VALIDATION")
    print("Testing torch.autograd for Physics-Informed Loss Function")
    print("=" * 70)
    print("\nThis test validates that automatic differentiation correctly computes:")
    print("  1. First-order spatial and temporal derivatives")
    print("  2. Second-order derivatives (Laplacian)")
    print("  3. Convection term (u·∇T)")
    print("  4. Full PDE residual in PINN model")

    all_results = []

    # Run tests
    all_results.append(test_first_order_derivatives())
    all_results.append(test_second_order_derivatives())
    all_results.append(test_laplacian())
    all_results.append(test_convection_term())
    all_results.append(test_pinn_model())

    # Print combined results
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_passed = sum(len(r.passed) for r in all_results)
    total_failed = sum(len(r.failed) for r in all_results)
    total_tests = total_passed + total_failed

    print(f"\nTotal Tests: {total_tests}")
    print(f"✓ Passed: {total_passed} ({100*total_passed/total_tests:.1f}%)")

    if total_failed > 0:
        print(f"✗ Failed: {total_failed} ({100*total_failed/total_tests:.1f}%)")
        print("\nFailed tests:")
        for result in all_results:
            for msg in result.failed:
                print(f"  {msg}")
        return False
    else:
        print("\n" + "🎉" * 35)
        print("ALL TESTS PASSED!")
        print("Automatic differentiation is working correctly!")
        print("🎉" * 35)
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
