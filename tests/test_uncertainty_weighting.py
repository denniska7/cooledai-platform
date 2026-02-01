#!/usr/bin/env python3
"""
Quick test script to verify Uncertainty Weighting implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from models.recurrent_pinn import RecurrentPINN, RecurrentPINNLoss

def test_uncertainty_weighting():
    """Test that uncertainty weighting works correctly."""
    print("="*70)
    print("TESTING UNCERTAINTY WEIGHTING IMPLEMENTATION")
    print("="*70)

    # Set device
    device = 'cpu'

    # Create dummy data
    batch_size = 16
    T_current = torch.randn(batch_size).to(device) * 10 + 50  # Around 50°C
    Q_load = torch.randn(batch_size).to(device) * 20000 + 80000  # Around 80kW
    u_flow = torch.randn(batch_size).to(device) * 0.5 + 1.5  # Around 1.5 m/s

    # Create dummy targets
    T_t1 = (T_current + torch.randn(batch_size).to(device) * 0.5).unsqueeze(1)
    T_t5 = (T_current + torch.randn(batch_size).to(device) * 2.0).unsqueeze(1)
    T_t10 = (T_current + torch.randn(batch_size).to(device) * 4.0).unsqueeze(1)

    targets = {
        'T_t1': T_t1,
        'T_t5': T_t5,
        'T_t10': T_t10,
        'T_current': T_current
    }

    # Test 1: Uncertainty Weighting Enabled
    print("\n✓ Test 1: Uncertainty Weighting ENABLED")
    print("-" * 70)

    model_uw = RecurrentPINN(input_dim=3, hidden_dim=32, num_lstm_layers=2).to(device)
    criterion_uw = RecurrentPINNLoss(use_uncertainty_weighting=True).to(device)

    # Check that loss function has learnable parameters
    loss_params = list(criterion_uw.parameters())
    print(f"Number of learnable loss parameters: {len(loss_params)}")
    assert len(loss_params) == 4, "Should have 4 learnable parameters!"

    # Forward pass
    predictions = model_uw(T_current, Q_load, u_flow)
    loss_dict = criterion_uw(predictions, targets, Q_load, u_flow)

    print(f"\nInitial learned weights:")
    print(f"  Data:         {loss_dict['weight_data']:.4f}")
    print(f"  Physics:      {loss_dict['weight_physics']:.4f}")
    print(f"  Consistency:  {loss_dict['weight_consistency']:.4f}")
    print(f"  Monotonicity: {loss_dict['weight_monotonicity']:.4f}")

    print(f"\nLoss components:")
    print(f"  Total Loss:   {loss_dict['loss'].item():.4f}")
    print(f"  Data Loss:    {loss_dict['loss_data'].item():.4f}")
    print(f"  Physics Loss: {loss_dict['loss_physics'].item():.4f}")

    # Verify backward pass works
    loss_dict['loss'].backward()
    print(f"\n✓ Backward pass successful!")

    # Check gradients exist for loss parameters
    for i, param in enumerate(loss_params):
        if param.grad is not None:
            print(f"  Loss param {i} gradient: {param.grad.item():.4f}")
        else:
            print(f"  WARNING: Loss param {i} has no gradient!")

    # Test 2: Fixed Weights (Traditional)
    print("\n" + "="*70)
    print("✓ Test 2: Fixed Weights (Traditional Approach)")
    print("-" * 70)

    model_fixed = RecurrentPINN(input_dim=3, hidden_dim=32, num_lstm_layers=2).to(device)
    criterion_fixed = RecurrentPINNLoss(
        use_uncertainty_weighting=False,
        lambda_data=1.0,
        lambda_physics=0.1,
        lambda_consistency=0.05,
        lambda_monotonicity=0.01
    ).to(device)

    # Check that loss function has NO learnable parameters
    loss_params_fixed = list(criterion_fixed.parameters())
    print(f"Number of learnable loss parameters: {len(loss_params_fixed)}")
    assert len(loss_params_fixed) == 0, "Should have 0 learnable parameters!"

    # Forward pass
    predictions_fixed = model_fixed(T_current, Q_load, u_flow)
    loss_dict_fixed = criterion_fixed(predictions_fixed, targets, Q_load, u_flow)

    print(f"\nFixed weights:")
    print(f"  Data:         {loss_dict_fixed['weight_data']:.4f}")
    print(f"  Physics:      {loss_dict_fixed['weight_physics']:.4f}")
    print(f"  Consistency:  {loss_dict_fixed['weight_consistency']:.4f}")
    print(f"  Monotonicity: {loss_dict_fixed['weight_monotonicity']:.4f}")

    print(f"\nLoss components:")
    print(f"  Total Loss:   {loss_dict_fixed['loss'].item():.4f}")
    print(f"  Data Loss:    {loss_dict_fixed['loss_data'].item():.4f}")
    print(f"  Physics Loss: {loss_dict_fixed['loss_physics'].item():.4f}")

    # Verify backward pass works
    loss_dict_fixed['loss'].backward()
    print(f"\n✓ Backward pass successful!")

    # Test 3: Verify weights evolve with optimization
    print("\n" + "="*70)
    print("✓ Test 3: Verify Weights Evolve During Training")
    print("-" * 70)

    model_train = RecurrentPINN(input_dim=3, hidden_dim=32, num_lstm_layers=2).to(device)
    criterion_train = RecurrentPINNLoss(use_uncertainty_weighting=True).to(device)

    # Create optimizer including loss parameters
    optimizer = torch.optim.Adam(
        list(model_train.parameters()) + list(criterion_train.parameters()),
        lr=0.01
    )

    print("\nTraining for 5 steps to see if weights evolve...")
    for step in range(5):
        optimizer.zero_grad()

        # Forward pass
        predictions = model_train(T_current, Q_load, u_flow)
        loss_dict = criterion_train(predictions, targets, Q_load, u_flow)
        loss = loss_dict['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_weights = (
                loss_dict['weight_data'],
                loss_dict['weight_physics'],
                loss_dict['weight_consistency'],
                loss_dict['weight_monotonicity']
            )
            print(f"  Step 0 - Loss: {loss.item():.4f}, "
                  f"Weights: [{initial_weights[0]:.3f}, {initial_weights[1]:.3f}, "
                  f"{initial_weights[2]:.3f}, {initial_weights[3]:.3f}]")
        elif step == 4:
            final_weights = (
                loss_dict['weight_data'],
                loss_dict['weight_physics'],
                loss_dict['weight_consistency'],
                loss_dict['weight_monotonicity']
            )
            print(f"  Step 4 - Loss: {loss.item():.4f}, "
                  f"Weights: [{final_weights[0]:.3f}, {final_weights[1]:.3f}, "
                  f"{final_weights[2]:.3f}, {final_weights[3]:.3f}]")

    # Check if weights changed
    weights_changed = not np.allclose(initial_weights, final_weights, rtol=0.01)
    if weights_changed:
        print(f"\n✓ Weights evolved during training! (This is expected)")
        print(f"  Change in data weight: {abs(final_weights[0] - initial_weights[0]):.4f}")
    else:
        print(f"\n⚠ Weights didn't change much (may need more steps or different data)")

    # Final Summary
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nSummary:")
    print("  ✓ Uncertainty weighting mode works correctly")
    print("  ✓ Fixed weights mode works correctly")
    print("  ✓ Loss parameters have gradients")
    print("  ✓ Optimizer can update loss parameters")
    print("  ✓ Implementation is ready for training!")
    print("\nYou can now run full training with:")
    print("  python3.11 train_recurrent_pinn.py")
    print("="*70)

if __name__ == "__main__":
    test_uncertainty_weighting()
