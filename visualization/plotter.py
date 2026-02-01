"""
Visualization tools for heat distribution results.

This module provides functions to visualize:
- 2D slices of temperature field
- 3D temperature distribution
- Time evolution animations
- Comparison with ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Tuple, List
import torch


class HeatVisualizer:
    """
    Visualization tools for PINN heat distribution results.
    """

    def __init__(self, model, config: dict):
        """
        Args:
            model: Trained HeatPINN model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.geometry = config['geometry']

    def plot_2d_slice(
        self,
        axis: str = 'z',
        slice_value: float = 1.5,
        t: float = 0.0,
        resolution: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D slice of temperature field.

        Args:
            axis: Axis perpendicular to slice ('x', 'y', or 'z')
            slice_value: Position of slice along the axis
            t: Time value
            resolution: Grid resolution
            save_path: Path to save figure
        """
        self.model.eval()

        if axis == 'z':
            # XY plane at fixed z
            x_grid = np.linspace(0, self.geometry['length'], resolution)
            y_grid = np.linspace(0, self.geometry['width'], resolution)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = np.full_like(X, slice_value)
            T_time = np.full_like(X, t)

            # Predict temperature
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            t_flat = T_time.flatten()

            T_pred = self.model.predict_temperature(x_flat, y_flat, z_flat, t_flat)
            T_pred = T_pred.reshape(X.shape)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = ax.contourf(X, Y, T_pred, levels=20, cmap='hot')
            ax.contour(X, Y, T_pred, levels=10, colors='black', linewidths=0.5, alpha=0.3)
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Temperature (°C)', fontsize=12)

            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title(f'Temperature at z={slice_value}m, t={t}s', fontsize=14)
            ax.set_aspect('equal')

        elif axis == 'y':
            # XZ plane at fixed y
            x_grid = np.linspace(0, self.geometry['length'], resolution)
            z_grid = np.linspace(0, self.geometry['height'], resolution)
            X, Z = np.meshgrid(x_grid, z_grid)
            Y = np.full_like(X, slice_value)
            T_time = np.full_like(X, t)

            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            t_flat = T_time.flatten()

            T_pred = self.model.predict_temperature(x_flat, y_flat, z_flat, t_flat)
            T_pred = T_pred.reshape(X.shape)

            fig, ax = plt.subplots(figsize=(10, 6))
            contour = ax.contourf(X, Z, T_pred, levels=20, cmap='hot')
            ax.contour(X, Z, T_pred, levels=10, colors='black', linewidths=0.5, alpha=0.3)
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Temperature (°C)', fontsize=12)

            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Z (m)', fontsize=12)
            ax.set_title(f'Temperature at y={slice_value}m, t={t}s', fontsize=14)

        elif axis == 'x':
            # YZ plane at fixed x
            y_grid = np.linspace(0, self.geometry['width'], resolution)
            z_grid = np.linspace(0, self.geometry['height'], resolution)
            Y, Z = np.meshgrid(y_grid, z_grid)
            X = np.full_like(Y, slice_value)
            T_time = np.full_like(Y, t)

            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            t_flat = T_time.flatten()

            T_pred = self.model.predict_temperature(x_flat, y_flat, z_flat, t_flat)
            T_pred = T_pred.reshape(Y.shape)

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(Y, Z, T_pred, levels=20, cmap='hot')
            ax.contour(Y, Z, T_pred, levels=10, colors='black', linewidths=0.5, alpha=0.3)
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Temperature (°C)', fontsize=12)

            ax.set_xlabel('Y (m)', fontsize=12)
            ax.set_ylabel('Z (m)', fontsize=12)
            ax.set_title(f'Temperature at x={slice_value}m, t={t}s', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_3d_volume(
        self,
        t: float = 0.0,
        resolution: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Plot 3D volume rendering of temperature field using Plotly.

        Args:
            t: Time value
            resolution: Grid resolution
            save_path: Path to save HTML figure
        """
        self.model.eval()

        # Create 3D grid
        x_grid = np.linspace(0, self.geometry['length'], resolution)
        y_grid = np.linspace(0, self.geometry['width'], resolution)
        z_grid = np.linspace(0, self.geometry['height'], resolution)

        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        # Flatten
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        t_flat = np.full_like(x_flat, t)

        # Predict
        T_pred = self.model.predict_temperature(x_flat, y_flat, z_flat, t_flat)
        T_pred = T_pred.reshape(X.shape)

        # Create isosurface plot
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=T_pred.flatten(),
            isomin=T_pred.min(),
            isomax=T_pred.max(),
            surface_count=10,
            colorscale='Hot',
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title='Temperature (°C)')
        ))

        fig.update_layout(
            title=f'3D Temperature Distribution at t={t}s',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=900,
            height=700
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_time_evolution(
        self,
        point: Tuple[float, float, float],
        t_range: Tuple[float, float] = (0, 1.0),
        n_points: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot temperature evolution at a specific point over time.

        Args:
            point: (x, y, z) coordinates
            t_range: Time range (t_start, t_end)
            n_points: Number of time points
            save_path: Path to save figure
        """
        self.model.eval()

        x, y, z = point
        t_values = np.linspace(t_range[0], t_range[1], n_points)

        x_array = np.full(n_points, x)
        y_array = np.full(n_points, y)
        z_array = np.full(n_points, z)

        T_pred = self.model.predict_temperature(x_array, y_array, z_array, t_values)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_values, T_pred, linewidth=2, color='red')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'Temperature Evolution at ({x:.1f}, {y:.1f}, {z:.1f})', fontsize=14)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_multiple_slices(
        self,
        t: float = 0.0,
        n_slices: int = 4,
        axis: str = 'z',
        resolution: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot multiple 2D slices in a grid.

        Args:
            t: Time value
            n_slices: Number of slices
            axis: Axis perpendicular to slices
            resolution: Grid resolution
            save_path: Path to save figure
        """
        if axis == 'z':
            slice_values = np.linspace(0, self.geometry['height'], n_slices + 2)[1:-1]
        elif axis == 'y':
            slice_values = np.linspace(0, self.geometry['width'], n_slices + 2)[1:-1]
        elif axis == 'x':
            slice_values = np.linspace(0, self.geometry['length'], n_slices + 2)[1:-1]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for i, slice_val in enumerate(slice_values):
            ax = axes[i]
            self.model.eval()

            if axis == 'z':
                x_grid = np.linspace(0, self.geometry['length'], resolution)
                y_grid = np.linspace(0, self.geometry['width'], resolution)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = np.full_like(X, slice_val)
                T_time = np.full_like(X, t)

                T_pred = self.model.predict_temperature(
                    X.flatten(), Y.flatten(), Z.flatten(), T_time.flatten()
                ).reshape(X.shape)

                contour = ax.contourf(X, Y, T_pred, levels=20, cmap='hot')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'z = {slice_val:.2f} m')

            plt.colorbar(contour, ax=ax)
            ax.set_aspect('equal')

        plt.suptitle(f'Temperature Distribution at t={t}s', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_error_distribution(
        self,
        T_true: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot error distribution compared to ground truth.

        Args:
            T_true: True temperature values
            x, y, z, t: Coordinates
            save_path: Path to save figure
        """
        self.model.eval()

        # Predict
        T_pred = self.model.predict_temperature(x, y, z, t)

        # Compute errors
        absolute_error = np.abs(T_pred - T_true)
        relative_error = absolute_error / (np.abs(T_true) + 1e-8) * 100

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Absolute error histogram
        axes[0].hist(absolute_error, bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Absolute Error (°C)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Absolute Error Distribution', fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Relative error histogram
        axes[1].hist(relative_error, bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Relative Error (%)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Relative Error Distribution', fontsize=14)
        axes[1].grid(True, alpha=0.3)

        # Add statistics
        mean_abs = np.mean(absolute_error)
        mean_rel = np.mean(relative_error)
        axes[0].axvline(mean_abs, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_abs:.3f}°C')
        axes[1].axvline(mean_rel, color='blue', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_rel:.2f}%')

        axes[0].legend()
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        # Print statistics
        print("\n=== Error Statistics ===")
        print(f"Mean Absolute Error: {mean_abs:.4f} °C")
        print(f"Max Absolute Error: {np.max(absolute_error):.4f} °C")
        print(f"Mean Relative Error: {mean_rel:.2f} %")
        print(f"Max Relative Error: {np.max(relative_error):.2f} %")


def create_animation_frames(
    model,
    geometry: dict,
    t_range: Tuple[float, float],
    n_frames: int = 30,
    axis: str = 'z',
    slice_value: float = 1.5,
    resolution: int = 100
) -> List[np.ndarray]:
    """
    Generate frames for temperature animation.

    Args:
        model: Trained PINN model
        geometry: Geometry configuration
        t_range: Time range (t_start, t_end)
        n_frames: Number of animation frames
        axis: Slice axis
        slice_value: Slice position
        resolution: Grid resolution

    Returns:
        List of temperature arrays for each frame
    """
    model.eval()
    frames = []

    t_values = np.linspace(t_range[0], t_range[1], n_frames)

    for t in t_values:
        if axis == 'z':
            x_grid = np.linspace(0, geometry['length'], resolution)
            y_grid = np.linspace(0, geometry['width'], resolution)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z = np.full_like(X, slice_value)
            T_time = np.full_like(X, t)

            T_pred = model.predict_temperature(
                X.flatten(), Y.flatten(), Z.flatten(), T_time.flatten()
            ).reshape(X.shape)

        frames.append(T_pred)

    return frames
