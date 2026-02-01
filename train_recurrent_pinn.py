#!/usr/bin/env python3
"""
Training Script for Recurrent Physics-Informed Neural Network (Recurrent PINN)

This script trains a recurrent PINN to predict temperature evolution during
data center cooling failures using the thermal runaway dataset.

Key Features:
    - Multi-horizon temperature prediction (t+1, t+5, t+10 seconds)
    - Physics-informed loss function
    - Time-to-Failure estimation
    - Comprehensive evaluation metrics
    - Model checkpointing and visualization

Dataset:
    - data/failure_modes_v1_detailed.csv (300,500 time points from 500 scenarios)

Author: CoolingAI Simulator
Date: 2026-01-27
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Import our Recurrent PINN
from models.recurrent_pinn import (
    RecurrentPINN,
    RecurrentPINNLoss,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
    print_model_summary
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration."""

    # Data
    data_path = 'data/processed/failure_modes_v1_detailed.csv'
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15

    # Model
    input_dim = 3
    hidden_dim = 64
    num_lstm_layers = 2
    num_dense_layers = 3
    dropout = 0.1

    # Step 3.2: Thermal Inertia Modeling
    use_thermal_inertia = True  # Enable learnable thermal mass and hysteresis logic
    use_mc_dropout = True       # Enable MC Dropout for uncertainty quantification

    # Step 3.3: Spatial-Temporal Attention (Final push to < 0.1°C MAE)
    use_attention = True         # Enable self-attention on LSTM output
    num_attention_heads = 4      # Number of attention heads
    use_stratification = True    # Enable height-dependent stratification bias
    stratification_coefficient = 0.01  # Weight for stratification loss

    # Training (100 epochs for hyper-fine tuning)
    batch_size = 128
    num_epochs = 100  # Increased from 50 for final accuracy push
    learning_rate = 0.001
    weight_decay = 1e-5
    grad_clip = 1.0

    # Loss configuration
    use_uncertainty_weighting = True  # Use learned uncertainty weighting (automatic balancing)

    # Loss weights (only used if use_uncertainty_weighting=False)
    lambda_data = 1.0
    lambda_physics = 0.1
    lambda_consistency = 0.05
    lambda_monotonicity = 0.01

    # Optimization
    optimizer_type = 'adam'  # 'adam' or 'adamw'
    scheduler_type = 'plateau'  # 'plateau' or 'cosine'
    patience = 10  # For plateau scheduler

    # Device (prioritize MPS for Apple Silicon, then CUDA for NVIDIA, then CPU)
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Checkpointing
    checkpoint_dir = 'checkpoints'
    save_every = 5  # Save checkpoint every N epochs
    best_model_path = 'checkpoints/best_recurrent_pinn.pt'

    # Logging
    log_every = 10  # Log every N batches
    results_dir = 'results'

    # Random seed
    seed = 42


# ============================================================================
# DATASET
# ============================================================================

class FailureModeDataset(Dataset):
    """
    PyTorch Dataset for thermal failure mode time-series data.

    Loads data from failure_modes_v1_detailed.csv and creates samples
    with current state and multi-horizon future targets.

    Each sample contains:
        Input: [T_current, Q_load, u_flow] at time t
        Target: [T_t+1, T_t+5, T_t+10] at times t+1, t+5, t+10
    """

    def __init__(
        self,
        csv_path: str,
        sequence_length: int = 1,
        prediction_horizons: List[int] = [1, 5, 10],
        subsample_rate: int = 1
    ):
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV file
            sequence_length: Number of past time steps to include (for LSTM context)
            prediction_horizons: Future time steps to predict (in seconds)
            subsample_rate: Subsample every Nth time step (for efficiency)
        """
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons
        self.subsample_rate = subsample_rate

        # Load data
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)

        # Subsample if requested
        if subsample_rate > 1:
            self.df = self.df[::subsample_rate].reset_index(drop=True)

        print(f"  Loaded {len(self.df)} time points from {self.df['scenario_id'].nunique()} scenarios")

        # Group by scenario for efficient lookup
        self.scenarios = {}
        for scenario_id, group in self.df.groupby('scenario_id'):
            self.scenarios[scenario_id] = group.reset_index(drop=True)

        # Create sample indices (scenario_id, time_index)
        self.samples = []
        max_horizon = max(prediction_horizons)

        for scenario_id, scenario_df in self.scenarios.items():
            n_timesteps = len(scenario_df)

            # For each valid time point (must have future data for all horizons)
            for t in range(sequence_length - 1, n_timesteps - max_horizon):
                self.samples.append((scenario_id, t))

        print(f"  Created {len(self.samples)} training samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - T_current: Current temperature
                - Q_load: IT heat load
                - u_flow: Air flow velocity
                - T_t1, T_t5, T_t10: Future temperatures
        """
        scenario_id, t = self.samples[idx]
        scenario_df = self.scenarios[scenario_id]

        # Current state (time t)
        current = scenario_df.iloc[t]
        T_current = torch.tensor(current['temperature_C'], dtype=torch.float32)
        Q_load = torch.tensor(current['Q_it_load_W'], dtype=torch.float32)

        # u_flow: assume reduced during failure (simplified)
        # In detailed scenario, we'd load actual velocity field
        # For now, use binary: 2.0 m/s before failure, 0.2 m/s after
        u_flow = torch.tensor(
            2.0 if current['is_after_failure'] == 0 else 0.2,
            dtype=torch.float32
        )

        # Future targets
        targets = {}
        for horizon in self.prediction_horizons:
            future_idx = t + horizon
            if future_idx < len(scenario_df):
                T_future = scenario_df.iloc[future_idx]['temperature_C']
            else:
                # Extrapolate using last known heating rate
                last_temp = scenario_df.iloc[-1]['temperature_C']
                last_dTdt = scenario_df.iloc[-1]['dT_dt_C_per_s']
                extrapolation_time = (future_idx - len(scenario_df) + 1) * 0.1
                T_future = last_temp + last_dTdt * extrapolation_time

            targets[f'T_t{horizon}'] = torch.tensor(T_future, dtype=torch.float32)

        return {
            'T_current': T_current,
            'Q_load': Q_load,
            'u_flow': u_flow,
            'T_t1': targets['T_t1'],
            'T_t5': targets['T_t5'],
            'T_t10': targets['T_t10'],
        }


def create_dataloaders(
    csv_path: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 128,
    num_workers: int = 0,
    subsample_rate: int = 1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        csv_path: Path to CSV file
        train_split, val_split, test_split: Data split ratios
        batch_size: Batch size
        num_workers: Number of worker processes
        subsample_rate: Subsample every Nth time step

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    dataset = FailureModeDataset(
        csv_path=csv_path,
        sequence_length=1,
        prediction_horizons=[1, 5, 10],
        subsample_rate=subsample_rate
    )

    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.seed)
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples ({train_split*100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({val_split*100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({test_split*100:.0f}%)")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if Config.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if Config.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if Config.device == 'cuda' else False
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(
    model: RecurrentPINN,
    train_loader: DataLoader,
    criterion: RecurrentPINNLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_loss_data = 0.0
    total_loss_physics = 0.0
    total_loss_consistency = 0.0
    total_loss_monotonicity = 0.0
    total_weight_data = 0.0
    total_weight_physics = 0.0
    total_weight_consistency = 0.0
    total_weight_monotonicity = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        T_current = batch['T_current'].to(device)
        Q_load = batch['Q_load'].to(device)
        u_flow = batch['u_flow'].to(device)
        T_t1 = batch['T_t1'].to(device).unsqueeze(1)
        T_t5 = batch['T_t5'].to(device).unsqueeze(1)
        T_t10 = batch['T_t10'].to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(T_current, Q_load, u_flow)

        # Prepare targets
        targets = {
            'T_t1': T_t1,
            'T_t5': T_t5,
            'T_t10': T_t10,
            'T_current': T_current
        }

        # Compute loss
        loss_dict = criterion(predictions, targets, Q_load, u_flow)
        loss = loss_dict['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_loss_data += loss_dict['loss_data'].item()
        total_loss_physics += loss_dict['loss_physics'].item()
        total_loss_consistency += loss_dict['loss_consistency'].item()
        total_loss_monotonicity += loss_dict['loss_monotonicity'].item()

        # Accumulate learned weights
        total_weight_data += loss_dict['weight_data']
        total_weight_physics += loss_dict['weight_physics']
        total_weight_consistency += loss_dict['weight_consistency']
        total_weight_monotonicity += loss_dict['weight_monotonicity']

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'data': f"{loss_dict['loss_data'].item():.4f}",
            'physics': f"{loss_dict['loss_physics'].item():.4f}"
        })

    # Average losses and weights
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'loss_data': total_loss_data / n_batches,
        'loss_physics': total_loss_physics / n_batches,
        'loss_consistency': total_loss_consistency / n_batches,
        'loss_monotonicity': total_loss_monotonicity / n_batches,
        'weight_data': total_weight_data / n_batches,
        'weight_physics': total_weight_physics / n_batches,
        'weight_consistency': total_weight_consistency / n_batches,
        'weight_monotonicity': total_weight_monotonicity / n_batches,
    }


def validate(
    model: RecurrentPINN,
    val_loader: DataLoader,
    criterion: RecurrentPINNLoss,
    device: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_loss_data = 0.0
    total_loss_physics = 0.0

    all_predictions = {'T_t1': [], 'T_t5': [], 'T_t10': []}
    all_targets = {'T_t1': [], 'T_t5': [], 'T_t10': []}

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            # Move to device
            T_current = batch['T_current'].to(device)
            Q_load = batch['Q_load'].to(device)
            u_flow = batch['u_flow'].to(device)
            T_t1 = batch['T_t1'].to(device).unsqueeze(1)
            T_t5 = batch['T_t5'].to(device).unsqueeze(1)
            T_t10 = batch['T_t10'].to(device).unsqueeze(1)

            # Forward pass
            predictions = model(T_current, Q_load, u_flow)

            # Prepare targets
            targets = {
                'T_t1': T_t1,
                'T_t5': T_t5,
                'T_t10': T_t10,
                'T_current': T_current
            }

            # Compute loss
            loss_dict = criterion(predictions, targets, Q_load, u_flow)

            # Accumulate losses
            total_loss += loss_dict['loss'].item()
            total_loss_data += loss_dict['loss_data'].item()
            total_loss_physics += loss_dict['loss_physics'].item()

            # Store predictions and targets for metrics
            all_predictions['T_t1'].append(predictions['T_t1'].cpu())
            all_predictions['T_t5'].append(predictions['T_t5'].cpu())
            all_predictions['T_t10'].append(predictions['T_t10'].cpu())
            all_targets['T_t1'].append(T_t1.cpu())
            all_targets['T_t5'].append(T_t5.cpu())
            all_targets['T_t10'].append(T_t10.cpu())

    # Concatenate all batches
    for key in all_predictions:
        all_predictions[key] = torch.cat(all_predictions[key], dim=0)
        all_targets[key] = torch.cat(all_targets[key], dim=0)

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)

    # Average losses
    n_batches = len(val_loader)
    losses = {
        'loss': total_loss / n_batches,
        'loss_data': total_loss_data / n_batches,
        'loss_physics': total_loss_physics / n_batches,
    }

    return losses, metrics


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_recurrent_pinn():
    """Main training function."""

    # Set random seeds
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    # Create directories
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.results_dir, exist_ok=True)

    print("=" * 70)
    print("RECURRENT PINN TRAINING")
    print("=" * 70)

    # Device
    device = torch.device(Config.device)
    print(f"\nDevice: {device}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=Config.data_path,
        train_split=Config.train_split,
        val_split=Config.val_split,
        test_split=Config.test_split,
        batch_size=Config.batch_size,
        subsample_rate=2  # Use every other time step for efficiency
    )

    # Create model
    print("\nCreating model...")
    model = RecurrentPINN(
        input_dim=Config.input_dim,
        hidden_dim=Config.hidden_dim,
        num_lstm_layers=Config.num_lstm_layers,
        num_dense_layers=Config.num_dense_layers,
        dropout=Config.dropout,
        use_thermal_inertia=Config.use_thermal_inertia,  # Step 3.2
        use_mc_dropout=Config.use_mc_dropout,            # Step 3.2
        use_attention=Config.use_attention,              # Step 3.3
        num_attention_heads=Config.num_attention_heads   # Step 3.3
    ).to(device)

    print_model_summary(model)

    # Create loss function
    criterion = RecurrentPINNLoss(
        use_uncertainty_weighting=Config.use_uncertainty_weighting,
        lambda_data=Config.lambda_data,
        lambda_physics=Config.lambda_physics,
        lambda_consistency=Config.lambda_consistency,
        lambda_monotonicity=Config.lambda_monotonicity,
        use_stratification=Config.use_stratification,              # Step 3.3
        stratification_coefficient=Config.stratification_coefficient  # Step 3.3
    ).to(device)

    # Create optimizer (include loss parameters if using uncertainty weighting)
    if Config.use_uncertainty_weighting:
        # Include both model and loss function parameters for optimization
        optimizer_params = list(model.parameters()) + list(criterion.parameters())
        print(f"\n✓ Using Uncertainty Weighting (automatic loss balancing)")
        print(f"  - Learnable loss weights will be optimized during training")
    else:
        optimizer_params = model.parameters()
        print(f"\n✓ Using Fixed Loss Weights")
        print(f"  - Data: {Config.lambda_data}, Physics: {Config.lambda_physics}")
        print(f"  - Consistency: {Config.lambda_consistency}, Monotonicity: {Config.lambda_monotonicity}")

    if Config.optimizer_type == 'adam':
        optimizer = optim.Adam(
            optimizer_params,
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay
        )
    elif Config.optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            optimizer_params,
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {Config.optimizer_type}")

    # Create learning rate scheduler
    if Config.scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=Config.patience,
            verbose=True
        )
    elif Config.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=Config.num_epochs,
            eta_min=1e-6
        )
    else:
        scheduler = None

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    # Training loop
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch+1}/{Config.num_epochs}")
        print("-" * 70)

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_losses, val_metrics = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        if scheduler is not None:
            if Config.scheduler_type == 'plateau':
                scheduler.step(val_losses['loss'])
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_losses['loss']:.4f} "
              f"(data: {train_losses['loss_data']:.4f}, "
              f"physics: {train_losses['loss_physics']:.4f})")
        print(f"  Val Loss:   {val_losses['loss']:.4f}")
        print(f"  Val MAE:    {val_metrics['mae_avg']:.4f}°C")
        print(f"  Val RMSE:   {val_metrics['rmse_avg']:.4f}°C")
        print(f"  LR:         {current_lr:.6f}")

        # Print learned weights if using uncertainty weighting
        if Config.use_uncertainty_weighting:
            print(f"  Learned Weights: data={train_losses['weight_data']:.3f}, "
                  f"physics={train_losses['weight_physics']:.3f}, "
                  f"consistency={train_losses['weight_consistency']:.3f}, "
                  f"monotonicity={train_losses['weight_monotonicity']:.3f}")

        # Update history
        history['train_loss'].append(train_losses['loss'])
        history['val_loss'].append(val_losses['loss'])
        history['val_mae'].append(val_metrics['mae_avg'])
        history['val_rmse'].append(val_metrics['rmse_avg'])
        history['learning_rate'].append(current_lr)

        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            print(f"  ✓ New best model! (val_loss: {best_val_loss:.4f})")
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, Config.best_model_path
            )

        # Save periodic checkpoint
        if (epoch + 1) % Config.save_every == 0:
            checkpoint_path = f"{Config.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch, val_losses['loss'], checkpoint_path)

    # Save final model
    final_path = f"{Config.checkpoint_dir}/final_model.pt"
    save_checkpoint(model, optimizer, Config.num_epochs - 1, val_losses['loss'], final_path)

    # Save training history
    history_path = f"{Config.results_dir}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {Config.best_model_path}")
    print(f"Training history saved to: {history_path}")

    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    # Load best model
    load_checkpoint(model, None, Config.best_model_path, device=Config.device)

    # Evaluate on test set
    test_losses, test_metrics = validate(model, test_loader, criterion, device)

    print(f"\nTest Set Results:")
    print(f"  Loss: {test_losses['loss']:.4f}")
    print(f"  MAE (avg):  {test_metrics['mae_avg']:.4f}°C")
    print(f"  RMSE (avg): {test_metrics['rmse_avg']:.4f}°C")
    print(f"  MAPE (avg): {test_metrics['mape_avg']:.2f}%")
    print(f"\nPer-Horizon Metrics:")
    print(f"  t+1:  MAE={test_metrics['mae_t1']:.4f}°C, RMSE={test_metrics['rmse_t1']:.4f}°C")
    print(f"  t+5:  MAE={test_metrics['mae_t5']:.4f}°C, RMSE={test_metrics['rmse_t5']:.4f}°C")
    print(f"  t+10: MAE={test_metrics['mae_t10']:.4f}°C, RMSE={test_metrics['rmse_t10']:.4f}°C")

    # Save test results
    test_results = {
        'test_loss': test_losses['loss'],
        'test_metrics': test_metrics
    }
    test_results_path = f"{Config.results_dir}/test_results.json"
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\nTest results saved to: {test_results_path}")

    print("\n" + "=" * 70)
    print("🚀 Recurrent PINN is ready for Time-to-Failure prediction!")
    print("=" * 70)

    return model, history, test_metrics


if __name__ == "__main__":
    # Train model
    model, history, metrics = train_recurrent_pinn()

    print("\n✓ Training script completed successfully!")
