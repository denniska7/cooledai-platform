# CoolingAI Simulator

A Physics-Informed Neural Network (PINN) based system for simulating heat distribution in data centers.

## Overview

CoolingAI Simulator uses deep learning combined with physics-based constraints to predict temperature distribution in data center environments. By incorporating the heat equation and boundary conditions directly into the loss function, the model ensures physically consistent predictions.

## Project Structure

```
coolingai_simulator/
├── backend/         # FastAPI API, safety modules, protocol adapters (Modbus, SNMP)
├── frontend/        # Streamlit dashboards (dashboard.py, dashboard_thermal_lag.py)
├── core/            # Universal HAL, OptimizationBrain, SyntheticSensor, DataIngestor
├── data/
│   ├── raw/         # Raw sensor logs (e.g., Log-013026.csv)
│   └── processed/   # Generated datasets (synthetic_thermal, failure_modes)
├── models/          # PINN architecture and neural network models
├── utils/           # Helper functions and utilities
├── visualization/   # Plotting and visualization tools
├── configs/         # Configuration files for physics parameters
├── notebooks/       # Jupyter notebooks for research and experimentation
├── tests/           # Unit tests and safety validation
└── requirements.txt # Python dependencies
```

## Physics Background

The system solves the 3D heat equation:

```
∂T/∂t = α∇²T + Q
```

Where:
- T: Temperature field
- α: Thermal diffusivity
- Q: Heat source term (from servers, cooling systems)
- ∇²: Laplacian operator

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Current Status

**Phase:** Research and Development

## Key Features (Planned)

- Physics-informed neural network for heat distribution
- Support for complex data center geometries
- Real-time temperature prediction
- Cooling optimization recommendations
- Interactive visualization dashboard

## License

TBD
