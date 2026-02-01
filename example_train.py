"""
Example training script for CoolingAI Simulator PINN

This script demonstrates:
1. Loading configuration
2. Training the PINN model
3. Visualizing results
4. Saving the trained model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.trainer import PINNTrainer
from visualization.plotter import HeatVisualizer


def main():
    """Main training pipeline."""

    print("=" * 60)
    print("CoolingAI Simulator - PINN Training")
    print("=" * 60)

    # Configuration
    config_path = "configs/physics_config.yaml"
    checkpoint_dir = "checkpoints"
    results_dir = "results"

    # Create directories
    Path(checkpoint_dir).mkdir(exist_ok=True)
    Path(results_dir).mkdir(exist_ok=True)

    # Check device availability
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
    print(f"\nInitializing PINN trainer...")
    trainer = PINNTrainer(config_path, device=device)

    # Print model architecture
    print(f"\nModel Architecture:")
    print(f"  Input dimension: 4 (x, y, z, t)")
    print(f"  Hidden layers: {trainer.config['network']['hidden_layers']}")
    print(f"  Output dimension: 1 (Temperature)")
    print(f"  Activation: {trainer.config['network']['activation']}")
    print(f"  Total parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Print domain information
    print(f"\nData Center Domain:")
    geom = trainer.config['geometry']
    print(f"  Length: {geom['length']} m")
    print(f"  Width: {geom['width']} m")
    print(f"  Height: {geom['height']} m")
    print(f"  Volume: {geom['length'] * geom['width'] * geom['height']:.1f} m³")

    # Print training configuration
    print(f"\nTraining Configuration:")
    train_cfg = trainer.config['training']
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Learning rate: {train_cfg['learning_rate']}")
    print(f"  Collocation points: {train_cfg['n_collocation']}")
    print(f"  Boundary points: {train_cfg['n_boundary']}")
    print(f"  Initial condition points: {train_cfg['n_initial']}")

    # Train model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")

    trainer.train(
        epochs=train_cfg['epochs'],
        verbose=True,
        checkpoint_dir=checkpoint_dir
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Plot training history
    print("\nGenerating training history plots...")
    trainer.plot_training_history(save_path=f"{results_dir}/training_history.png")
    print(f"✓ Saved: {results_dir}/training_history.png")

    # Visualization
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60 + "\n")

    visualizer = HeatVisualizer(trainer.model, trainer.config)

    # 2D slice at mid-height
    print("1. Creating 2D slice (XY plane at mid-height)...")
    mid_height = geom['height'] / 2
    visualizer.plot_2d_slice(
        axis='z',
        slice_value=mid_height,
        t=0.5,
        resolution=150,
        save_path=f"{results_dir}/temperature_slice_xy.png"
    )
    print(f"   ✓ Saved: {results_dir}/temperature_slice_xy.png")

    # 2D slice (XZ plane)
    print("2. Creating 2D slice (XZ plane at mid-width)...")
    mid_width = geom['width'] / 2
    visualizer.plot_2d_slice(
        axis='y',
        slice_value=mid_width,
        t=0.5,
        resolution=150,
        save_path=f"{results_dir}/temperature_slice_xz.png"
    )
    print(f"   ✓ Saved: {results_dir}/temperature_slice_xz.png")

    # Multiple slices
    print("3. Creating multiple Z slices...")
    visualizer.plot_multiple_slices(
        t=0.5,
        n_slices=4,
        axis='z',
        resolution=100,
        save_path=f"{results_dir}/temperature_multiple_slices.png"
    )
    print(f"   ✓ Saved: {results_dir}/temperature_multiple_slices.png")

    # Time evolution at a point
    print("4. Creating time evolution plot...")
    # Sample point in the middle of the data center
    sample_point = (geom['length']/2, geom['width']/2, geom['height']/2)
    visualizer.plot_time_evolution(
        point=sample_point,
        t_range=(0, 1.0),
        n_points=100,
        save_path=f"{results_dir}/temperature_evolution.png"
    )
    print(f"   ✓ Saved: {results_dir}/temperature_evolution.png")

    # 3D visualization (interactive HTML)
    print("5. Creating 3D interactive visualization...")
    try:
        visualizer.plot_3d_volume(
            t=0.5,
            resolution=20,  # Lower resolution for 3D
            save_path=f"{results_dir}/temperature_3d.html"
        )
        print(f"   ✓ Saved: {results_dir}/temperature_3d.html")
    except Exception as e:
        print(f"   ⚠ Could not create 3D visualization: {e}")

    # Sample predictions
    print("\n" + "=" * 60)
    print("Sample Temperature Predictions")
    print("=" * 60 + "\n")

    sample_points = [
        (1.0, 1.0, 0.5, 0.5, "Near inlet, low height"),
        (5.0, 4.0, 1.5, 0.5, "Center, mid-height"),
        (9.0, 7.0, 2.5, 0.5, "Far from inlet, high"),
    ]

    for x, y, z, t, description in sample_points:
        T = trainer.model.predict_temperature(
            np.array([x]), np.array([y]), np.array([z]), np.array([t])
        )[0]
        print(f"Location: ({x:.1f}, {y:.1f}, {z:.1f}) m at t={t}s")
        print(f"Description: {description}")
        print(f"Temperature: {T:.2f} °C\n")

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n✓ Model trained successfully")
    print(f"✓ Final checkpoint saved to: {checkpoint_dir}/final_model.pt")
    print(f"✓ Visualizations saved to: {results_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review training history plot to check convergence")
    print(f"  2. Examine temperature distributions in the visualizations")
    print(f"  3. Validate against analytical solutions or experimental data")
    print(f"  4. Fine-tune hyperparameters if needed")
    print(f"  5. Experiment with different data center configurations")
    print(f"\nTo load the trained model:")
    print(f"  trainer = PINNTrainer('{config_path}', device='{device}')")
    print(f"  trainer.load_checkpoint('{checkpoint_dir}/final_model.pt')")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
