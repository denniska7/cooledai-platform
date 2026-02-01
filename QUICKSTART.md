# Quick Start Guide - CoolingAI Simulator

## Installation

### 1. Set up Python environment

```bash
cd coolingai_simulator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running Your First Simulation

### Option 1: Run the example training script

```bash
python example_train.py
```

This will:
- Train a PINN on a sample data center configuration
- Generate visualizations of temperature distribution
- Save the trained model and plots to `checkpoints/` and `results/`

Expected runtime: 15-30 minutes on CPU, 5-10 minutes on GPU

### Option 2: Use Jupyter Notebook (Interactive)

```bash
jupyter notebook notebooks/interactive_demo.ipynb
```

This provides an interactive environment to:
- Adjust parameters in real-time
- Visualize results interactively
- Experiment with different configurations

## Understanding the Output

After training, you'll find:

### 1. Training History (`results/training_history.png`)
- Shows convergence of different loss components
- PDE loss should decrease and stabilize
- Boundary and initial condition losses should approach zero

### 2. Temperature Distributions
- `temperature_slice_xy.png`: Top-down view (birds-eye)
- `temperature_slice_xz.png`: Side view
- `temperature_multiple_slices.png`: Multiple horizontal slices
- `temperature_3d.html`: Interactive 3D visualization (open in browser)

### 3. Time Evolution (`temperature_evolution.png`)
- Shows how temperature changes over time at a specific point

## Customizing Your Simulation

### Modify Data Center Configuration

Edit `configs/physics_config.yaml`:

```yaml
geometry:
  length: 10.0   # Change room dimensions
  width: 8.0
  height: 3.0

servers:
  num_racks: 8          # Number of server racks
  power_per_rack: 5000  # Heat output per rack (Watts)

boundaries:
  inlet_temp: 18.0      # Cooling air temperature (°C)
  initial_temp: 22.0    # Initial room temperature (°C)
```

### Adjust Training Parameters

```yaml
training:
  epochs: 10000           # More epochs = better accuracy
  learning_rate: 1e-3     # Learning rate
  n_collocation: 10000    # More points = higher accuracy

  loss_weights:
    pde_loss: 1.0         # Weight for physics equation
    boundary_loss: 100.0  # Weight for boundary conditions
    initial_loss: 100.0   # Weight for initial conditions
```

### Modify Network Architecture

```yaml
network:
  hidden_layers: [64, 64, 64, 64, 64]  # Add/remove layers
  activation: "tanh"                    # Try 'relu', 'gelu', 'swish'
```

## Using the Trained Model

### Load a saved model

```python
from models.trainer import PINNTrainer

trainer = PINNTrainer('configs/physics_config.yaml', device='cpu')
trainer.load_checkpoint('checkpoints/final_model.pt')
```

### Make predictions

```python
import numpy as np

# Predict temperature at specific location
x = np.array([5.0])  # 5 meters from inlet
y = np.array([4.0])  # 4 meters from side wall
z = np.array([1.5])  # 1.5 meters height
t = np.array([0.5])  # 0.5 seconds

temperature = trainer.model.predict_temperature(x, y, z, t)
print(f"Temperature: {temperature[0]:.2f} °C")
```

### Visualize results

```python
from visualization.plotter import HeatVisualizer

visualizer = HeatVisualizer(trainer.model, trainer.config)

# Create a 2D slice
visualizer.plot_2d_slice(
    axis='z',
    slice_value=1.5,  # Height of 1.5m
    t=0.5,            # At t=0.5s
    resolution=200    # High resolution
)
```

## Validation and Testing

### 1. Check Loss Convergence
- All loss components should decrease
- Final PDE loss < 1e-4 is good
- Boundary/initial losses < 1e-5 is excellent

### 2. Physical Consistency
- Temperature should be hottest near server racks
- Temperature should decrease near inlet (cooling air)
- Temperature should increase over time (if servers are on)

### 3. Conservation Laws
- Total energy should be conserved
- Heat flow should follow physical principles

## Troubleshooting

### Training is slow
- Reduce `n_collocation`, `n_boundary`, `n_initial`
- Use GPU if available
- Reduce number of epochs for initial testing

### Loss not converging
- Increase `boundary_loss` and `initial_loss` weights
- Try different activation function (tanh usually works best)
- Reduce learning rate
- Check that boundary conditions are physically reasonable

### Temperature predictions are unrealistic
- Ensure proper normalization in the model
- Check that thermal properties are correct
- Validate heat source locations and magnitudes
- Try training longer (more epochs)

### Out of memory errors
- Reduce batch size
- Reduce number of collocation points
- Reduce network size (fewer/smaller hidden layers)

## Next Steps

1. **Experiment with different geometries**: Try different room sizes and layouts
2. **Add more complex boundary conditions**: Implement time-varying inlet temperature
3. **Incorporate air flow**: Extend to include convection (velocity field)
4. **Real-time prediction**: Deploy model for fast inference
5. **Optimization**: Use PINN to find optimal cooling strategies

## Resources

- **Physics Documentation**: See `docs_physics.md` for detailed equations
- **Code Documentation**: Check docstrings in each module
- **Example Notebook**: `notebooks/interactive_demo.ipynb`

## Support

For issues and questions:
1. Check that all dependencies are installed correctly
2. Review the example scripts and notebooks
3. Validate your configuration file syntax
4. Ensure your data center parameters are physically reasonable

Happy simulating!
